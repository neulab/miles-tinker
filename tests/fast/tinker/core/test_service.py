"""TinkerService over a FakeBackend: the protocol properties end to end —
idempotent resubmit, two-phase create, barrier ordering, lease reclamation,
sampler publication, checkpoint sidecars, poisoned windows."""

import asyncio
import os

import pytest
from tests.fast.tinker.harness import (
    ADAM,
    await_settled,
    create_model_payload,
    created_model,
    fb_payload,
    make_service,
    row,
)

from miles.tinker.core.promise import DONE, FAILED
from miles.tinker.core.service import SIDECAR_NAME, read_sidecar
from miles.tinker.core.types import UserInputError


def _optim_payload(model_id: str, seq_id: int) -> dict:
    return {"model_id": model_id, "seq_id": seq_id, "adam_params": dict(ADAM)}


def _sampler_payload(model_id: str, seq_id: int, name: str | None = None, unnamed: bool = False) -> dict:
    payload = {"model_id": model_id, "seq_id": seq_id, "name": name, "sampling_session_seq_id": None}
    if unnamed:
        payload["sampling_session_seq_id"] = 0
    return payload


def _sample_payload(**overrides) -> dict:
    return {
        "num_samples": 1,
        "prompt_tokens": [1, 2],
        "sampling_params": {"max_tokens": 4},
        "prompt_logprobs": False,
        "topk_prompt_logprobs": 0,
        **overrides,
    }


async def _published(service, model_id: str, seq_id: int, name: str | None = None) -> dict:
    request_id = service.submit("tenant", "save_weights_for_sampler", _sampler_payload(model_id, seq_id, name))
    return (await await_settled(service, "tenant", request_id)).result


# -------- streams and barriers --------


async def test_resubmitted_seq_id_reuses_the_promise_and_runs_once(service):
    model_id = await created_model(service)
    payload = fb_payload(model_id, 1, [row()])

    first = service.submit("tenant", "forward_backward", payload)
    second = service.submit("tenant", "forward_backward", payload)
    assert first == second

    promise = await await_settled(service, "tenant", first)
    assert promise.state == DONE
    assert service.submit("tenant", "forward_backward", payload) == first, "a resubmit after completion also dedups"
    assert len(service.backend.named("forward_backward")) == 1, "the resubmit must not run the trainer again"


async def test_out_of_order_chunks_complete_and_the_barrier_waits(service):
    model_id = await created_model(service)

    optim = service.submit("tenant", "optim_step", _optim_payload(model_id, 3))
    late = service.submit("tenant", "forward_backward", fb_payload(model_id, 2, [row()]))
    await asyncio.sleep(0.05)
    assert not service.backend.named("optim_step"), "the barrier must wait for the whole window"

    early = service.submit("tenant", "forward_backward", fb_payload(model_id, 1, [row()]))
    for request_id in (early, late, optim):
        assert (await await_settled(service, "tenant", request_id)).state == DONE

    kinds = [name for name, _ in service.backend.calls if name != "load_slot"]
    assert kinds.index("optim_step") > max(i for i, k in enumerate(kinds) if k == "forward_backward")


async def test_admission_failure_fails_the_promise_not_the_stream(service):
    model_id = await created_model(service)

    oversized = service.submit(
        "tenant", "forward_backward", fb_payload(model_id, 1, [row(service.config.max_tokens_per_datum + 1)])
    )
    assert (await await_settled(service, "tenant", oversized)).state == FAILED

    healthy = service.submit("tenant", "forward_backward", fb_payload(model_id, 2, [row()]))
    assert (await await_settled(service, "tenant", healthy)).state == DONE, "the stream must keep flowing"


async def test_forward_backward_outputs_align_to_datums(service):
    model_id = await created_model(service)
    request_id = service.submit("tenant", "forward_backward", fb_payload(model_id, 1, [row(2), row(5)]))

    promise = await await_settled(service, "tenant", request_id)
    assert [len(output["logprobs"]) for output in promise.result["outputs"]] == [2, 5]


async def test_optim_step_returns_the_backend_metrics(service):
    model_id = await created_model(service)
    slot = service.models[model_id].slot
    service.submit("tenant", "forward_backward", fb_payload(model_id, 1, [row()]))
    request_id = service.submit("tenant", "optim_step", _optim_payload(model_id, 2))

    promise = await await_settled(service, "tenant", request_id)
    assert promise.result["metrics"]["grad_norm"] == 0.5 + slot
    assert promise.result["metrics"]["lr"] == ADAM["learning_rate"], "the backend dict travels verbatim"


async def test_a_failed_unit_poisons_the_window_until_the_next_optim_step(service):
    model_id = await created_model(service)
    slot = service.models[model_id].slot
    service.backend.fail_next = RuntimeError("cuda went away")
    failed = service.submit("tenant", "forward_backward", fb_payload(model_id, 1, [row()]))
    assert (await await_settled(service, "tenant", failed)).error_category == "server"

    optim = service.submit("tenant", "optim_step", _optim_payload(model_id, 2))
    promise = await await_settled(service, "tenant", optim)
    assert (promise.state, promise.error_category) == (FAILED, "user")
    assert "gradients discarded" in promise.error
    assert service.backend.named("discard_grads") == [{"slot": slot}]
    assert not service.backend.named("optim_step"), "a poisoned window must not step"

    clean = service.submit("tenant", "forward_backward", fb_payload(model_id, 3, [row()]))
    await await_settled(service, "tenant", clean)
    optim = service.submit("tenant", "optim_step", _optim_payload(model_id, 4))
    assert (await await_settled(service, "tenant", optim)).state == DONE, "the flag clears with the failed step"


async def test_a_backend_returning_the_wrong_output_count_fails_the_riders_not_the_loop(service):
    model_id = await created_model(service)

    async def short_forward_backward(unit_id, slot_rows, loss_fn, loss_fn_config):
        return [{"loss": 1.0, "logprobs": [0.0]}]  # one output for two rows

    service.backend.forward_backward = short_forward_backward
    broken = service.submit("tenant", "forward_backward", fb_payload(model_id, 1, [row(), row()]))
    promise = await await_settled(service, "tenant", broken)
    assert (promise.state, promise.error_category) == (FAILED, "server") and "2 rows" in promise.error

    service.backend.forward_backward = type(service.backend).forward_backward.__get__(service.backend)
    healthy = service.submit("tenant", "forward_backward", fb_payload(model_id, 2, [row()]))
    assert (await await_settled(service, "tenant", healthy)).state == DONE, "the dispatch loop must survive"


async def test_a_dirty_stream_does_not_block_a_merged_clean_one(service):
    service.config.coalesce_wait_s = 0.0  # keep the two windows in separate units
    dirty = await created_model(service)
    clean = await created_model(service)
    service.backend.fail_next = RuntimeError("boom")
    service.submit("tenant", "forward_backward", fb_payload(dirty, 1, [row()]))
    await asyncio.sleep(0.05)
    service.submit("tenant", "forward_backward", fb_payload(clean, 1, [row()]))
    await asyncio.sleep(0.05)
    dirty_optim = service.submit("tenant", "optim_step", _optim_payload(dirty, 2))
    clean_optim = service.submit("tenant", "optim_step", _optim_payload(clean, 2))

    assert (await await_settled(service, "tenant", dirty_optim)).state == FAILED
    assert (await await_settled(service, "tenant", clean_optim)).state == DONE
    assert list(service.backend.named("optim_step")[0]["adam_params_by_slot"]) == [service.models[clean].slot]


# -------- models --------


async def test_create_model_is_two_phase(service):
    request_id, model_id = service.create_model("tenant", create_model_payload(service))

    promise = await await_settled(service, "tenant", request_id)
    assert promise.result == {"kind": "create_model", "model_id": model_id}
    assert service.backend.named("load_slot")[0]["rank"] == 8
    record = service.models[model_id]
    assert (record.is_lora, record.lora_rank, record.lora_alpha) == (True, 8, 16.0)


async def test_create_model_is_idempotent_per_session_seq(service):
    session_id = service.create_session("tenant", {"sdk_version": "0.27.1"})
    first = service.create_model("tenant", create_model_payload(service, session_id=session_id, model_seq_id=0))
    retry = service.create_model("tenant", create_model_payload(service, session_id=session_id, model_seq_id=0))
    other = service.create_model("tenant", create_model_payload(service, session_id=session_id, model_seq_id=1))

    assert first == retry, "an SDK retry must not allocate a second slot"
    assert other[1] != first[1]
    assert service.models[first[1]].sdk_version == "0.27.1"


async def test_failed_slot_init_returns_the_slot(service):
    free_before = set(service.free_slots)
    service.backend.fail_next = RuntimeError("init blew up")
    request_id, model_id = service.create_model("tenant", create_model_payload(service))

    promise = await await_settled(service, "tenant", request_id)
    assert (promise.state, promise.error_category) == (FAILED, "server")
    assert model_id not in service.models
    assert service.free_slots == free_before


async def test_no_free_slots_is_a_user_error(service):
    for _ in range(service.config.n_slots):
        await created_model(service)
    with pytest.raises(UserInputError, match="no free adapter slots"):
        service.create_model("tenant", create_model_payload(service))


async def test_the_wrong_base_model_is_rejected(service):
    with pytest.raises(UserInputError, match="serves"):
        service.create_model("tenant", {"base_model": "other", "lora_config": {"rank": 8}})


@pytest.mark.parametrize("lora_config", [None, {"rank": 0}, {}])
async def test_a_lora_gateway_needs_a_positive_rank(service, lora_config):
    with pytest.raises(UserInputError, match="rank must be > 0"):
        service.create_model("tenant", {"base_model": "base", "lora_config": lora_config})


async def test_unload_model_frees_the_slot_and_the_adapters(service):
    model_id = await created_model(service)
    slot = service.models[model_id].slot
    await _published(service, model_id, 1)
    stale = service.submit("tenant", "forward_backward", fb_payload(model_id, 3, [row()]))  # gapped

    request_id = service.unload_model("tenant", model_id)
    promise = await await_settled(service, "tenant", request_id)
    assert promise.result == {"kind": "unload_model", "model_id": model_id}
    assert model_id not in service.models and slot in service.free_slots
    assert service.backend.named("unload_adapter") == [{"lora_name": f"{model_id}@1"}]
    assert service.backend.named("unload_slot") == [{"slot": slot}]
    assert service.retrieve("tenant", stale).state == FAILED


async def test_training_run_reports_the_record(service):
    model_id = await created_model(service, user_metadata={"renderer_name": "qwen3"})
    run = service.training_run("tenant", model_id)
    assert (run["training_run_id"], run["model_owner"], run["is_lora"], run["lora_rank"]) == (
        model_id,
        "tenant",
        True,
        8,
    )
    assert run["user_metadata"] == {"renderer_name": "qwen3"}
    with pytest.raises(UserInputError):
        service.training_run("tenant", "model-nope")


# -------- checkpoints --------


async def test_save_then_load_roundtrip_paths(service):
    model_id = await created_model(service)
    save = service.submit("tenant", "save_state", {"model_id": model_id, "seq_id": 1, "name": "ckpt"})
    saved_path = (await await_settled(service, "tenant", save)).result["path"]
    assert saved_path == f"tinker://{model_id}/weights/ckpt"
    sidecar = read_sidecar(f"{service.config.checkpoint_root}/{model_id}/weights/ckpt")
    assert (sidecar["tenant"], sidecar["lora_rank"], sidecar["kind"], sidecar["train_mode"]) == (
        "tenant",
        8,
        "weights",
        "lora",
    )

    load = service.submit(
        "tenant", "load_state", {"model_id": model_id, "seq_id": 2, "path": saved_path, "optimizer": False}
    )
    assert (await await_settled(service, "tenant", load)).state == DONE
    weights_only = service.backend.named("load_slot")[-1]
    assert weights_only["load_optimizer"] is False
    assert weights_only["ckpt_path"].endswith(f"{model_id}/weights/ckpt")


async def test_resume_from_the_sidecar_survives_a_gateway_restart(service):
    model_id = await created_model(service, tenant="owner")
    save = service.submit("owner", "save_state", {"model_id": model_id, "seq_id": 1, "name": "ckpt"})
    saved_path = (await await_settled(service, "owner", save)).result["path"]

    restarted = make_service(checkpoint_root=service.config.checkpoint_root)
    run_task = asyncio.create_task(restarted.run())
    try:
        info = restarted.weights_info("owner", saved_path)
        assert (info["base_model"], info["is_lora"], info["lora_rank"]) == ("base", True, 8)
        with pytest.raises(Exception, match="does not belong"):
            restarted.weights_info("thief", saved_path)

        # the SDK recreates the model with weights_info.lora_rank, then loads at seq 1
        _, new_model = restarted.create_model("owner", {"base_model": "base", "lora_config": {"rank": 8}})
        await asyncio.sleep(0.01)
        load = restarted.submit(
            "owner", "load_state", {"model_id": new_model, "seq_id": 1, "path": saved_path, "optimizer": True}
        )
        assert (await await_settled(restarted, "owner", load)).state == DONE
        assert restarted.backend.named("load_slot")[-1]["ckpt_path"].endswith(f"{model_id}/weights/ckpt")
    finally:
        run_task.cancel()


async def test_load_state_takes_rank_and_alpha_from_the_sidecar(service):
    model_id = await created_model(service)
    save = service.submit("tenant", "save_state", {"model_id": model_id, "seq_id": 1, "name": "ckpt"})
    saved_path = (await await_settled(service, "tenant", save)).result["path"]
    other = await created_model(service, lora_config={"rank": 4})

    load = service.submit(
        "tenant", "load_state", {"model_id": other, "seq_id": 1, "path": saved_path, "optimizer": True}
    )
    assert (await await_settled(service, "tenant", load)).state == DONE
    assert (service.backend.named("load_slot")[-1]["rank"], service.models[other].lora_rank) == (8, 8)


async def test_a_foreign_checkpoint_is_rejected_as_a_user_error(service):
    victim = await created_model(service, tenant="tenant-a")
    thief = await created_model(service, tenant="tenant-b")
    save = service.submit("tenant-a", "save_state", {"model_id": victim, "seq_id": 1, "name": "ckpt"})
    await await_settled(service, "tenant-a", save)

    request_id = service.submit(
        "tenant-b",
        "load_state",
        {"model_id": thief, "seq_id": 1, "path": f"tinker://{victim}/weights/ckpt", "optimizer": True},
    )
    promise = await await_settled(service, "tenant-b", request_id)
    assert (promise.state, promise.error_category) == (FAILED, "user")


async def test_delete_checkpoint_removes_the_dir_and_tolerates_absence(service):
    model_id = await created_model(service)
    save = service.submit("tenant", "save_state", {"model_id": model_id, "seq_id": 1, "name": "ckpt"})
    await await_settled(service, "tenant", save)
    path = f"{service.config.checkpoint_root}/{model_id}/weights/ckpt"
    assert os.path.exists(f"{path}/{SIDECAR_NAME}")

    service.delete_checkpoint("tenant", model_id, "weights/ckpt")
    assert not os.path.exists(path)
    service.delete_checkpoint("tenant", model_id, "weights/ckpt")  # already gone: fine
    with pytest.raises(UserInputError):
        service.delete_checkpoint("tenant", model_id, "bogus")


async def test_checkpoint_names_cannot_escape_the_checkpoint_root(service):
    model_id = await created_model(service)
    escaping = service.submit("tenant", "save_state", {"model_id": model_id, "seq_id": 1, "name": "../escape"})
    promise = await await_settled(service, "tenant", escaping)
    assert (promise.state, promise.error_category) == (FAILED, "user")
    assert not service.backend.named("save_slot")

    for checkpoint_id in ("weights/..", "weights/a/b"):
        with pytest.raises(UserInputError, match="invalid checkpoint path"):
            service.delete_checkpoint("tenant", model_id, checkpoint_id)
    with pytest.raises(UserInputError, match="invalid checkpoint path"):
        service.delete_checkpoint("tenant", "..", "weights/ckpt")
    with pytest.raises(UserInputError, match="invalid checkpoint path"):
        service.weights_info("tenant", "tinker://../weights/..")


# -------- samplers --------


async def test_sampler_save_bumps_the_version_and_pushes(service):
    model_id = await created_model(service)
    for seq_id in (1, 2):
        result = await _published(service, model_id, seq_id)
        assert result["path"] == f"tinker://{model_id}/sampler_weights/{seq_id}"
    assert [push["lora_name"] for push in service.backend.named("push_slot")] == [f"{model_id}@1", f"{model_id}@2"]
    assert service.models[model_id].loaded_versions == {1, 2}


async def test_a_named_sampler_save_keeps_the_client_name(service):
    model_id = await created_model(service)
    result = await _published(service, model_id, 1, name="000010")
    assert result["path"] == f"tinker://{model_id}/sampler_weights/000010"
    assert result["sampling_session_id"] is None
    assert service.backend.named("save_slot")[-1]["path"].endswith("sampler_weights/000010")
    assert service.backend.named("push_slot")[-1]["lora_name"] == f"{model_id}@1", "the adapter name stays versioned"
    assert read_sidecar(f"{service.config.checkpoint_root}/{model_id}/sampler_weights/000010")["sampler_version"] == 1


async def test_an_unnamed_sampler_save_mints_a_sampling_session(service):
    session_id = service.create_session("tenant", {})
    model_id = await created_model(service, session_id=session_id)
    request_id = service.submit("tenant", "save_weights_for_sampler", _sampler_payload(model_id, 1, unnamed=True))
    result = (await await_settled(service, "tenant", request_id)).result

    sampling_session = service.sampling_sessions[result["sampling_session_id"]]
    assert sampling_session["model_path"] == result["path"] == f"tinker://{model_id}/sampler_weights/1"
    assert sampling_session["session_id"] == session_id

    sample, _ = service.submit_sample("tenant", _sample_payload(sampling_session_id=result["sampling_session_id"]))
    await await_settled(service, "tenant", sample)
    assert service.backend.named("sample")[0]["lora_name"] == f"{model_id}@1"


async def test_sampling_resolves_against_the_pushed_version(service):
    model_id = await created_model(service)
    sampler_path = (await _published(service, model_id, 1))["path"]

    request_id, sequence_ids = service.submit_sample("tenant", _sample_payload(model_path=sampler_path, num_samples=2))
    assert len(sequence_ids) == 2
    promise = await await_settled(service, "tenant", request_id)
    assert len(promise.result["sequences"]) == 2
    assert service.backend.named("sample")[0]["lora_name"] == f"{model_id}@1"


async def test_sample_resubmits_are_idempotent_per_sampling_seq(service):
    model_id = await created_model(service)
    sampler_path = (await _published(service, model_id, 1))["path"]
    sid = service.create_sampling_session("tenant", {"model_path": sampler_path})

    first = service.submit_sample("tenant", _sample_payload(sampling_session_id=sid, seq_id=0))
    retry = service.submit_sample("tenant", _sample_payload(sampling_session_id=sid, seq_id=0))
    fresh = service.submit_sample("tenant", _sample_payload(sampling_session_id=sid, seq_id=1))
    await await_settled(service, "tenant", fresh[0])
    assert first == retry and fresh != first
    assert len(service.backend.named("sample")) == 2


async def test_unknown_samplers_are_user_errors(service):
    model_id = await created_model(service)
    with pytest.raises(UserInputError, match="unknown sampling session"):
        service.submit_sample("tenant", _sample_payload(sampling_session_id="sampling-nope"))
    with pytest.raises(UserInputError, match="unknown sampler version"):
        service.create_sampling_session("tenant", {"model_path": f"tinker://{model_id}/sampler_weights/7"})
    with pytest.raises(UserInputError, match="unknown model"):
        service.create_sampling_session("tenant", {"model_path": "tinker://model-nope/sampler_weights/1"})
    with pytest.raises(UserInputError, match="serves"):
        service.create_sampling_session("tenant", {"base_model": "other"})


async def test_get_sampler_reports_base_model_and_path(service):
    sid = service.create_sampling_session("tenant", {"base_model": "base"})
    assert service.get_sampler("tenant", sid) == {"sampler_id": sid, "base_model": "base", "model_path": None}


async def test_old_sampler_versions_are_retired(service):
    service.config.sampler_versions_to_keep = 2
    model_id = await created_model(service)
    for seq_id in (1, 2, 3):
        await _published(service, model_id, seq_id)

    assert service.backend.named("unload_adapter") == [{"lora_name": f"{model_id}@1"}]
    assert service.models[model_id].loaded_versions == {2, 3}
    assert not os.path.exists(f"{service.config.checkpoint_root}/{model_id}/sampler_weights/1")
    assert os.path.exists(f"{service.config.checkpoint_root}/{model_id}/sampler_weights/2/{SIDECAR_NAME}")
    with pytest.raises(UserInputError, match="retired"):
        service.submit_sample("tenant", _sample_payload(model_path=f"tinker://{model_id}/sampler_weights/1"))


async def test_a_version_with_a_running_sample_is_not_retired(service):
    service.config.sampler_versions_to_keep = 1
    model_id = await created_model(service)
    sampler_path = (await _published(service, model_id, 1))["path"]
    gate = asyncio.Event()

    async def slow_sample(payload, lora_name):
        await gate.wait()
        return {"sequences": []}

    service.backend.sample = slow_sample
    request_id, _ = service.submit_sample("tenant", _sample_payload(model_path=sampler_path))
    await asyncio.sleep(0.01)
    await _published(service, model_id, 2)
    assert not service.backend.named("unload_adapter"), "an adapter with a sample in flight stays loaded"

    gate.set()
    await await_settled(service, "tenant", request_id)
    await _published(service, model_id, 3)
    assert {u["lora_name"] for u in service.backend.named("unload_adapter")} == {f"{model_id}@1", f"{model_id}@2"}


async def test_a_failed_unload_keeps_the_version_resident(service):
    service.config.sampler_versions_to_keep = 1
    model_id = await created_model(service)
    await _published(service, model_id, 1)

    async def refusing_unload(lora_name):
        raise RuntimeError("engine busy")

    service.backend.unload_adapter = refusing_unload
    await _published(service, model_id, 2)
    assert service.models[model_id].loaded_versions == {1, 2}


# -------- leases --------


async def test_lease_expiry_reclaims_the_session(service):
    session_id = service.create_session("tenant", {})
    model_id = await created_model(service, session_id=session_id)
    slot = service.models[model_id].slot
    queued = service.submit("tenant", "optim_step", _optim_payload(model_id, 1))
    await await_settled(service, "tenant", queued)
    sampler_path = (await _published(service, model_id, 2))["path"]
    sid = service.create_sampling_session("tenant", {"session_id": session_id, "model_path": sampler_path})
    stale = service.submit("tenant", "forward_backward", fb_payload(model_id, 4, [row()]))  # gapped: stays queued

    service.sessions[session_id]["last_heartbeat"] -= service.config.lease_timeout_s + 1
    await service._sweep_once()

    assert model_id not in service.models
    assert slot in service.free_slots
    assert service.backend.named("unload_adapter") == [{"lora_name": f"{model_id}@1"}]
    assert service.backend.named("unload_slot") == [{"slot": slot}]
    assert service.retrieve("tenant", stale).state == FAILED
    assert sid not in service.sampling_sessions and session_id not in service.sessions


async def test_leases_are_per_session_not_per_tenant(service):
    stale_session = service.create_session("tenant", {})
    live_session = service.create_session("tenant", {})
    stale_model = await created_model(service, session_id=stale_session)
    live_model = await created_model(service, session_id=live_session)

    service.sessions[stale_session]["last_heartbeat"] -= service.config.lease_timeout_s + 1
    await service._sweep_once()

    assert stale_model not in service.models, "a dead run's slot is reclaimed while its tenant is still active"
    assert live_model in service.models


async def test_a_stale_sampling_session_cancels_its_samples(service):
    session_id = service.create_session("tenant", {})
    sid = service.create_sampling_session("tenant", {"session_id": session_id, "base_model": "base"})
    gate = asyncio.Event()

    async def slow_sample(payload, lora_name):
        await gate.wait()
        return {"sequences": []}

    service.backend.sample = slow_sample
    request_id, _ = service.submit_sample("tenant", _sample_payload(sampling_session_id=sid))
    service.sessions[session_id]["last_heartbeat"] -= service.config.lease_timeout_s + 1
    await service._sweep_once()

    promise = await await_settled(service, "tenant", request_id)
    assert (promise.state, promise.error) == (FAILED, "cancelled")
    assert sid not in service.sampling_sessions


async def test_a_fresh_heartbeat_keeps_the_model(service):
    session_id = service.create_session("tenant", {})
    model_id = await created_model(service, session_id=session_id)

    service.heartbeat(session_id)
    await service._sweep_once()

    assert model_id in service.models


async def test_run_warms_the_backend_up_once(service):
    await asyncio.sleep(0.01)
    assert service.backend.named("warmup") == [{}]
    assert service.backend.warmed_up


async def test_a_failed_warmup_still_serves():
    svc = make_service()
    svc.backend.fail_next = RuntimeError("no engines yet")
    run_task = asyncio.create_task(svc.run())
    try:
        model_id = await created_model(svc)
        request_id = svc.submit("tenant", "forward_backward", fb_payload(model_id, 1, [row()]))
        assert (await await_settled(svc, "tenant", request_id)).state == DONE
    finally:
        run_task.cancel()
