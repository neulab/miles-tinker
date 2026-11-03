"""DELETE .../checkpoints/{weights|sampler_weights}/{name} in the service: only the caller's own checkpoint, only
inside the checkpoint root, never under a save or load_state that uses it; sampling is never disturbed; the files
are freed off the event loop, and every delete is logged."""

import asyncio
import errno
import json
import logging
import os

import pytest
from tests.fast.tinker.harness import await_settled, created_model, make_service

from miles.tinker.core.checkpoint_delete import TRASH_DIR, CheckpointReaper
from miles.tinker.core.promise import DONE, FAILED
from miles.tinker.core.service import SIDECAR_NAME
from miles.tinker.core.types import ConflictError, NotFoundError, PermissionDeniedError, UserInputError


MALFORMED_IDS = ("bogus", "weights", "weights/", "optimizer/ckpt", "weights/..", "weights/a/b", "weights/a\0b")


@pytest.fixture
async def full_service():
    svc = make_service(train_mode="full", n_slots=1)
    run_task = asyncio.create_task(svc.run())
    yield svc
    for task in (run_task, getattr(svc, "_sweep_task", None)):
        if task is not None:
            task.cancel()


async def _saved(service, model_id: str, name: str | None, *, seq_id: int = 1, tenant: str = "tenant") -> str:
    request_id = service.submit(tenant, "save_state", {"model_id": model_id, "seq_id": seq_id, "name": name})
    assert (await await_settled(service, tenant, request_id)).state == DONE
    return f"{service.config.checkpoint_root}/{model_id}/weights/{name}"


def _sampler_payload(model_id: str, seq_id: int, name: str | None = None) -> dict:
    return {"model_id": model_id, "seq_id": seq_id, "name": name, "sampling_session_seq_id": None if name else 0}


def _sample_payload(sampling_session_id: str) -> dict:
    return {
        "sampling_session_id": sampling_session_id,
        "num_samples": 1,
        "prompt_tokens": [1, 2],
        "sampling_params": {"max_tokens": 4},
    }


def _trash_entries(service) -> list[str]:
    trash = os.path.join(os.path.realpath(service.config.checkpoint_root), TRASH_DIR)
    return sorted(os.listdir(trash)) if os.path.isdir(trash) else []


def _fake_checkpoint(path, tenant: str = "tenant") -> None:
    os.makedirs(path)
    with open(os.path.join(path, SIDECAR_NAME), "w") as handle:
        json.dump({"tenant": tenant}, handle)
    with open(os.path.join(path, "data.bin"), "wb") as handle:
        handle.write(b"\0" * 1024)


# -------- what a delete does --------


async def test_a_delete_takes_the_checkpoint_away_at_once_and_frees_it_in_the_background(service, caplog):
    caplog.set_level(logging.INFO, logger="miles.tinker.core")
    model_id = await created_model(service)
    path = await _saved(service, model_id, "recovery-000005")
    with open(os.path.join(path, "shard.distcp"), "wb") as handle:
        handle.write(b"\0" * 4096)

    service.delete_checkpoint("tenant", model_id, "weights/recovery-000005")
    assert not os.path.exists(path), "the checkpoint leaves the namespace before the request returns"
    assert service.checkpoint_reaper.drain(timeout=10)
    assert _trash_entries(service) == []
    label = f"tinker://{model_id}/weights/recovery-000005"
    assert any(f"{label} deleted" in record.getMessage() for record in caplog.records)
    assert any(f"freed {label}: 2 files" in record.getMessage() for record in caplog.records)

    with pytest.raises(NotFoundError, match="not found"):
        service.delete_checkpoint("tenant", model_id, "weights/recovery-000005")


async def test_a_name_saved_again_after_its_delete_is_a_fresh_checkpoint(service):
    model_id = await created_model(service)
    path = await _saved(service, model_id, "ckpt")
    service.delete_checkpoint("tenant", model_id, "weights/ckpt")
    await _saved(service, model_id, "ckpt", seq_id=2)
    assert service.checkpoint_reaper.drain(timeout=10)
    assert os.path.isfile(os.path.join(path, SIDECAR_NAME)), "the reaper frees only what the delete moved away"


async def test_leftovers_of_an_earlier_gateway_are_freed_by_the_first_delete(service):
    leftover = os.path.join(service.config.checkpoint_root, TRASH_DIR, "0" * 32, "iter_0000001")
    os.makedirs(leftover)
    with open(os.path.join(leftover, "__0_0.distcp"), "wb") as handle:
        handle.write(b"\0" * 1024)
    model_id = await created_model(service)
    await _saved(service, model_id, "ckpt")

    service.delete_checkpoint("tenant", model_id, "weights/ckpt")
    assert service.checkpoint_reaper.drain(timeout=10)
    assert _trash_entries(service) == []


# -------- whose checkpoint --------


async def test_another_tenants_checkpoint_is_not_found_and_kept(service):
    victim = await created_model(service, tenant="tenant-a")
    path = await _saved(service, victim, "ckpt", tenant="tenant-a")

    with pytest.raises(NotFoundError) as foreign:
        service.delete_checkpoint("tenant-b", victim, "weights/ckpt")
    assert os.path.isfile(os.path.join(path, SIDECAR_NAME))
    with pytest.raises(NotFoundError) as missing:
        service.delete_checkpoint("tenant-a", victim, "weights/never-saved")
    assert str(foreign.value).replace("ckpt", "never-saved") == str(missing.value), "no hint that it exists"


async def test_a_finished_runs_checkpoint_is_deleted_by_its_sidecar_tenant(service):
    """After a gateway restart the run is gone from memory; the sidecar still names its owner."""
    root = service.config.checkpoint_root
    _fake_checkpoint(f"{root}/model-earlier/weights/recovery-000010", tenant="tenant")

    with pytest.raises(NotFoundError):
        service.delete_checkpoint("someone-else", "model-earlier", "weights/recovery-000010")
    service.delete_checkpoint("tenant", "model-earlier", "weights/recovery-000010")
    assert not os.path.exists(f"{root}/model-earlier/weights/recovery-000010")


async def test_a_directory_without_sidecar_is_deletable_only_by_its_live_runs_tenant(service):
    root = service.config.checkpoint_root
    os.makedirs(f"{root}/model-unknown/weights/orphan")
    with pytest.raises(NotFoundError):  # no sidecar and no live run: nobody's checkpoint
        service.delete_checkpoint("tenant", "model-unknown", "weights/orphan")
    assert os.path.isdir(f"{root}/model-unknown/weights/orphan")

    model_id = await created_model(service)
    os.makedirs(f"{root}/{model_id}/weights/legacy")  # saved before sidecars existed
    with pytest.raises(NotFoundError):
        service.delete_checkpoint("tenant-b", model_id, "weights/legacy")
    service.delete_checkpoint("tenant", model_id, "weights/legacy")
    assert not os.path.exists(f"{root}/{model_id}/weights/legacy")


async def test_malformed_ids_are_user_errors(service):
    model_id = await created_model(service)
    for checkpoint_id in MALFORMED_IDS:
        with pytest.raises(UserInputError):
            service.delete_checkpoint("tenant", model_id, checkpoint_id)
    for bad_model in ("..", "model\0x", ""):
        with pytest.raises(UserInputError):
            service.delete_checkpoint("tenant", bad_model, "weights/ckpt")
    with pytest.raises(NotFoundError):  # what is being freed is nobody's checkpoint
        service.delete_checkpoint("tenant", TRASH_DIR, "weights/ckpt")


# -------- only inside the checkpoint root --------


async def test_symbolic_links_never_lead_a_delete_out_of_the_root(service, tmp_path):
    root = service.config.checkpoint_root
    outside = tmp_path / "outside"
    _fake_checkpoint(outside / "weights" / "ckpt")
    model_id = await created_model(service)
    inside = await _saved(service, model_id, "inside")

    os.symlink(outside, f"{root}/model-elsewhere")  # the run's directory
    os.makedirs(f"{root}/model-kind")
    os.symlink(outside / "weights", f"{root}/model-kind/weights")  # the kind directory
    os.symlink(outside / "weights" / "ckpt", f"{root}/{model_id}/weights/linked")  # the checkpoint itself
    os.symlink(inside, f"{root}/{model_id}/weights/alias")  # even a link that stays inside
    attempts = [
        ("model-elsewhere", "weights/ckpt"),
        ("model-kind", "weights/ckpt"),
        (model_id, "weights/linked"),
        (model_id, "weights/alias"),
    ]
    for run, checkpoint_id in attempts:
        with pytest.raises(PermissionDeniedError, match="symbolic link"):
            service.delete_checkpoint("tenant", run, checkpoint_id)
    assert sorted(os.listdir(outside / "weights" / "ckpt")) == sorted([SIDECAR_NAME, "data.bin"])
    assert os.path.isfile(os.path.join(inside, SIDECAR_NAME))


async def test_a_linked_checkpoint_root_is_resolved_once(tmp_path):
    real_root = tmp_path / "real"
    os.makedirs(real_root)
    os.symlink(real_root, tmp_path / "root-link")
    service = make_service(checkpoint_root=str(tmp_path / "root-link"))
    run_task = asyncio.create_task(service.run())
    try:
        model_id = await created_model(service)
        await _saved(service, model_id, "ckpt")
        service.delete_checkpoint("tenant", model_id, "weights/ckpt")
        assert service.checkpoint_reaper.drain(timeout=10)
        assert os.listdir(real_root / model_id / "weights") == []
        assert os.listdir(real_root / TRASH_DIR) == []
    finally:
        for task in (run_task, getattr(service, "_sweep_task", None)):
            if task is not None:
                task.cancel()


async def test_a_mount_point_below_the_root_hides_the_checkpoint_beside_itself(service, monkeypatch):
    rename = os.rename

    def no_cross_directory_rename(src, dst, *, src_dir_fd=None, dst_dir_fd=None):
        if src_dir_fd != dst_dir_fd:
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return rename(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    model_id = await created_model(service)
    path = await _saved(service, model_id, "ckpt")
    monkeypatch.setattr(os, "rename", no_cross_directory_rename)
    service.delete_checkpoint("tenant", model_id, "weights/ckpt")
    assert not os.path.exists(path)
    assert service.checkpoint_reaper.drain(timeout=10)
    assert os.listdir(os.path.dirname(path)) == []


async def test_an_unusable_trash_hides_the_checkpoint_beside_itself(service):
    open(os.path.join(service.config.checkpoint_root, TRASH_DIR), "w").close()  # a file where the trash goes
    model_id = await created_model(service)
    path = await _saved(service, model_id, "ckpt")
    service.delete_checkpoint("tenant", model_id, "weights/ckpt")
    assert not os.path.exists(path)
    assert service.checkpoint_reaper.drain(timeout=10)
    assert os.listdir(os.path.dirname(path)) == []


# -------- saves and loads that use it --------


async def test_a_running_save_blocks_the_delete_of_what_it_writes(service):
    model_id = await created_model(service)
    started, release = asyncio.Event(), asyncio.Event()

    async def slow_save(slot, path):
        os.makedirs(path, exist_ok=True)
        started.set()
        await release.wait()

    service.backend.save_slot = slow_save
    request_id = service.submit("tenant", "save_state", {"model_id": model_id, "seq_id": 1, "name": "ckpt"})
    await asyncio.wait_for(started.wait(), 2)
    with pytest.raises(ConflictError, match="a save_state is writing it"):
        service.delete_checkpoint("tenant", model_id, "weights/ckpt")
    with pytest.raises(NotFoundError):  # the save's owner check still runs first
        service.delete_checkpoint("tenant-b", model_id, "weights/ckpt")

    release.set()
    assert (await await_settled(service, "tenant", request_id)).state == DONE
    service.delete_checkpoint("tenant", model_id, "weights/ckpt")


async def test_a_running_load_blocks_the_delete_of_what_it_reads(service):
    model_id = await created_model(service)
    await _saved(service, model_id, "ckpt")
    started, release = asyncio.Event(), asyncio.Event()

    async def slow_load(slot, rank, alpha, ckpt_path=None, load_optimizer=True):
        started.set()
        await release.wait()

    service.backend.load_slot = slow_load
    request_id = service.submit(
        "tenant",
        "load_state",
        {"model_id": model_id, "seq_id": 2, "path": f"tinker://{model_id}/weights/ckpt", "optimizer": True},
    )
    await asyncio.wait_for(started.wait(), 2)
    with pytest.raises(ConflictError, match="a load_state is reading it"):
        service.delete_checkpoint("tenant", model_id, "weights/ckpt")

    release.set()
    assert (await await_settled(service, "tenant", request_id)).state == DONE
    service.delete_checkpoint("tenant", model_id, "weights/ckpt")


async def test_queued_saves_and_loads_block_the_delete_of_what_they_name(service):
    model_id = await created_model(service)
    resumer = await created_model(service)  # a later run resuming from this one's checkpoint
    await _saved(service, model_id, "ckpt", seq_id=1)
    await _saved(service, model_id, "checkpoint-000003", seq_id=2)
    await_these = []
    async with service._backend_lock:  # the dispatch loop cannot start a barrier: everything below stays queued
        await_these.append(
            service.submit(
                "tenant",
                "load_state",
                {"model_id": resumer, "seq_id": 1, "path": f"tinker://{model_id}/weights/ckpt", "optimizer": True},
            )
        )
        with pytest.raises(ConflictError, match="a queued load_state will read it"):
            service.delete_checkpoint("tenant", model_id, "weights/ckpt")
        # an unnamed save_state writes checkpoint-<seq_id>
        await_these.append(service.submit("tenant", "save_state", {"model_id": model_id, "seq_id": 3, "name": None}))
        with pytest.raises(ConflictError, match="a queued save_state will write it"):
            service.delete_checkpoint("tenant", model_id, "weights/checkpoint-000003")
    for request_id in await_these:
        assert (await await_settled(service, "tenant", request_id)).state == DONE
    service.delete_checkpoint("tenant", model_id, "weights/ckpt")
    service.delete_checkpoint("tenant", model_id, "weights/checkpoint-000003")


async def test_a_failed_save_does_not_leave_its_checkpoint_marked_in_use(service):
    model_id = await created_model(service)
    service.backend.fail_next = RuntimeError("disk full")
    request_id = service.submit("tenant", "save_state", {"model_id": model_id, "seq_id": 1, "name": "ckpt"})
    assert (await await_settled(service, "tenant", request_id)).state == FAILED
    assert service._checkpoint_io == {}


# -------- sampling never reads a checkpoint directory --------


async def test_a_lora_sampler_delete_leaves_its_sessions_and_the_resident_adapter_alone(service):
    model_id = await created_model(service)
    request_id = service.submit("tenant", "save_weights_for_sampler", _sampler_payload(model_id, 1, "s1"))
    published = await await_settled(service, "tenant", request_id)
    sampler_path = published.result["path"]
    session = service.create_sampling_session("tenant", {"model_path": sampler_path})
    started, release = asyncio.Event(), asyncio.Event()
    sample = service.backend.sample

    async def slow_sample(payload, lora_name):
        started.set()
        await release.wait()
        return await sample(payload, lora_name)

    service.backend.sample = slow_sample
    in_flight, _ = service.submit_sample("tenant", _sample_payload(session))
    await asyncio.wait_for(started.wait(), 2)

    service.delete_checkpoint("tenant", model_id, "sampler_weights/s1")  # the engines hold the adapter, not the dir
    assert not os.path.exists(f"{service.config.checkpoint_root}/{model_id}/sampler_weights/s1")
    release.set()
    assert (await await_settled(service, "tenant", in_flight)).state == DONE
    service.backend.sample = sample
    after, _ = service.submit_sample("tenant", _sample_payload(session))
    assert (await await_settled(service, "tenant", after)).state == DONE
    assert service.models[model_id].loaded_versions == {1}
    assert not service.backend.named("unload_adapter")


async def test_a_full_fine_tuning_sampler_checkpoint_is_only_its_sidecar(full_service):
    model_id = await created_model(full_service, lora_config=None)
    request_id = full_service.submit("tenant", "save_weights_for_sampler", _sampler_payload(model_id, 1))
    published = await await_settled(full_service, "tenant", request_id)
    name = published.result["path"].rsplit("/", 1)[-1]
    sampler_dir = f"{full_service.config.checkpoint_root}/{model_id}/sampler_weights/{name}"
    assert os.listdir(sampler_dir) == [SIDECAR_NAME]

    full_service.delete_checkpoint("tenant", model_id, f"sampler_weights/{name}")
    request_id, _ = full_service.submit_sample("tenant", _sample_payload(published.result["sampling_session_id"]))
    assert (await await_settled(full_service, "tenant", request_id)).state == DONE


# -------- the reaper --------


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores directory permissions")
def test_the_reaper_logs_what_it_cannot_free_and_keeps_going(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="miles.tinker.core")
    locked = tmp_path / "entry" / "locked"
    os.makedirs(locked)
    (locked / "shard").write_bytes(b"\0" * 10)
    os.chmod(locked, 0o500)
    reaper = CheckpointReaper()
    try:
        reaper.free(str(tmp_path / "entry"), "tinker://model-x/weights/stuck")
        assert reaper.drain(timeout=10)
        assert any("could not be freed" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)
    finally:
        os.chmod(locked, 0o700)
    reaper.free(str(tmp_path / "entry"), "tinker://model-x/weights/stuck")
    assert reaper.drain(timeout=10)
    assert not (tmp_path / "entry").exists()
    assert any("freed tinker://model-x/weights/stuck" in r.getMessage() for r in caplog.records)
