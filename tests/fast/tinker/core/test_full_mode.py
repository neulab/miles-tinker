"""train_mode="full": one slot, no adapter names, the engine weight version is
the sampler version, and every sampler path resolves to the latest weights."""

import asyncio

import pytest
from tests.fast.tinker.harness import ADAM, await_settled, created_model, make_service

from miles.tinker.core.promise import DONE
from miles.tinker.core.types import UserInputError


@pytest.fixture
async def full_service():
    svc = make_service(train_mode="full", n_slots=1)
    run_task = asyncio.create_task(svc.run())
    yield svc
    for task in (run_task, getattr(svc, "_sweep_task", None)):
        if task is not None:
            task.cancel()


def test_full_mode_serves_exactly_one_slot():
    with pytest.raises(ValueError, match="n_slots"):
        make_service(train_mode="full", n_slots=2)


async def test_create_model_ignores_lora_config_in_full_mode(full_service):
    model_id = await created_model(full_service, lora_config={"rank": 32})
    record = full_service.models[model_id]
    assert (record.is_lora, record.lora_rank, record.lora_alpha) == (False, 0, 0.0)
    assert full_service.backend.named("load_slot")[0]["rank"] == 0
    assert full_service.training_run("tenant", model_id)["is_lora"] is False

    with pytest.raises(UserInputError, match="one model at a time"):
        full_service.create_model("tenant", {"base_model": "base", "lora_config": None})


async def test_the_pushed_weight_version_becomes_the_sampler_version(full_service):
    model_id = await created_model(full_service, lora_config=None)
    full_service.backend.weight_version = 41  # engines already served 41 pushes

    request_id = full_service.submit(
        "tenant",
        "save_weights_for_sampler",
        {"model_id": model_id, "seq_id": 1, "name": None, "sampling_session_seq_id": 0},
    )
    result = (await await_settled(full_service, "tenant", request_id)).result
    assert result["path"] == f"tinker://{model_id}/sampler_weights/42"
    assert full_service.models[model_id].sampler_version == 42
    assert not full_service.backend.named("save_slot"), "FFT sampler saves are a push, not a checkpoint"
    assert not full_service.models[model_id].loaded_versions

    sample, _ = full_service.submit_sample(
        "tenant",
        {
            "sampling_session_id": result["sampling_session_id"],
            "num_samples": 1,
            "prompt_tokens": [1],
            "sampling_params": {"max_tokens": 2},
            "prompt_logprobs": False,
            "topk_prompt_logprobs": 0,
        },
    )
    await await_settled(full_service, "tenant", sample)
    assert full_service.backend.named("sample")[0]["lora_name"] is None, "no adapter name: latest weights"
    assert full_service._resolve_sampler("tenant", result["path"]) is None
    with pytest.raises(UserInputError, match="unknown sampler version"):
        full_service._resolve_sampler("tenant", f"tinker://{model_id}/sampler_weights/7")


async def test_weights_info_reports_a_lora_checkpoint_for_compatibility(full_service):
    model_id = await created_model(full_service, lora_config=None)
    save = full_service.submit("tenant", "save_state", {"model_id": model_id, "seq_id": 1, "name": "ckpt"})
    path = (await await_settled(full_service, "tenant", save)).result["path"]

    info = full_service.weights_info("tenant", path)
    assert (info["is_lora"], info["lora_rank"], info["train_mode"]) == (True, 0, "full")

    load = full_service.submit(
        "tenant", "load_state", {"model_id": model_id, "seq_id": 2, "path": path, "optimizer": True}
    )
    assert (await await_settled(full_service, "tenant", load)).state == DONE


async def test_a_lora_checkpoint_cannot_load_into_a_full_model(full_service):
    lora_gateway = make_service(checkpoint_root=full_service.config.checkpoint_root)
    lora_task = asyncio.create_task(lora_gateway.run())
    try:
        lora_model = await created_model(lora_gateway)
        save = lora_gateway.submit("tenant", "save_state", {"model_id": lora_model, "seq_id": 1, "name": "ckpt"})
        path = (await await_settled(lora_gateway, "tenant", save)).result["path"]
    finally:
        lora_task.cancel()

    model_id = await created_model(full_service, lora_config=None)
    load = full_service.submit(
        "tenant", "load_state", {"model_id": model_id, "seq_id": 1, "path": path, "optimizer": True}
    )
    promise = await await_settled(full_service, "tenant", load)
    assert promise.error_category == "user" and "mode" in promise.error


async def test_optim_step_in_full_mode(full_service):
    model_id = await created_model(full_service, lora_config=None)
    full_service.submit(
        "tenant",
        "forward_backward",
        {
            "model_id": model_id,
            "seq_id": 1,
            "rows": [{"tokens": [1, 2], "target_len": 1}],
            "loss_fn": "cross_entropy",
            "loss_fn_config": {},
        },
    )
    optim = full_service.submit("tenant", "optim_step", {"model_id": model_id, "seq_id": 2, "adam_params": dict(ADAM)})
    assert (await await_settled(full_service, "tenant", optim)).result["metrics"]["grad_norm"] == 0.5
