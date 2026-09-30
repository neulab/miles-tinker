"""Retention must follow resumes without delaying training or deleting readers."""

import asyncio
import json
import threading
from pathlib import Path

from tests.fast.tinker.harness import ADAM, await_settled, created_model, make_service

from miles.tinker.core.promise import DONE
from miles.tinker.core.retention import checkpoint_lock, prune_checkpoints
from miles.tinker.core.service import read_sidecar


async def save(service, model, seq, name, tenant="tenant"):
    request = service.submit(tenant, "save_state", {"model_id": model, "seq_id": seq, "name": name})
    result = await await_settled(service, tenant, request)
    assert result.state == DONE
    await asyncio.gather(*service._background_tasks)
    return result.result["path"]


async def step(service, model, seq):
    request = service.submit("tenant", "optim_step", {"model_id": model, "seq_id": seq, "adam_params": ADAM})
    assert (await await_settled(service, "tenant", request)).state == DONE


async def test_retention_follows_restarts_and_prefers_resumed_copy(service):
    model = await created_model(service)
    unrelated = await created_model(service)
    await save(service, unrelated, 1, "unrelated")
    paths = []
    for i in range(4):
        await step(service, model, i * 2 + 1)
        paths.append(await save(service, model, i * 2 + 2, f"step-{i}"))
    root = Path(service.config.checkpoint_root)
    assert not (root / model / "weights/step-0").exists()
    assert len(list((root / model / "weights").glob("*/tinker_meta.json"))) == 3

    resumed = make_service(checkpoint_root=str(root))
    task = asyncio.create_task(resumed.run())
    try:
        newer = await created_model(resumed, tenant="new-token")
        request = resumed.submit(
            "new-token", "load_state", {"model_id": newer, "seq_id": 1, "path": paths[-1], "optimizer": True}
        )
        assert (await await_settled(resumed, "new-token", request)).state == DONE
        await save(resumed, newer, 2, "step-3", tenant="new-token")
        assert not (root / model / "weights/step-3").exists()
        assert (root / newer / "weights/step-3/tinker_meta.json").exists()
        assert (root / model / "weights/step-1/tinker_meta.json").exists()
        assert (root / unrelated / "weights/unrelated/tinker_meta.json").exists()
        meta = read_sidecar(str(root / newer / "weights/step-3"))
        assert meta["checkpoint_lineage"] == model
        assert meta["resume_started_at"] > service.models[model].resume_started_at
    finally:
        task.cancel()
        resumed._sweep_task.cancel()


def checkpoint(root, model, name, **meta):
    path = root / model / "weights" / name
    path.mkdir(parents=True)
    (path / "tinker_meta.json").write_text(json.dumps({"model_id": model, "created_at": 1, **meta}))
    return path


def test_legacy_incomplete_and_pinned_checkpoints(tmp_path):
    old = checkpoint(tmp_path, "old", "one")
    newest = checkpoint(
        tmp_path, "resumed", "one", checkpoint_lineage="old", resume_started_at=1, checkpoint_state=str(old)
    )
    unrelated = checkpoint(tmp_path, "other", "one")
    incomplete = tmp_path / "old/weights/incomplete"
    incomplete.mkdir()
    with checkpoint_lock(str(old)):
        prune_checkpoints(str(tmp_path), "old", ("old",), 1)
        assert old.exists()
    prune_checkpoints(str(tmp_path), "old", ("old",), 1)
    assert not old.exists()
    assert newest.exists() and unrelated.exists() and incomplete.exists()


async def test_slow_deletion_does_not_block_training(service, monkeypatch):
    started, release = threading.Event(), threading.Event()

    def slow_prune(*args):
        started.set()
        assert release.wait(timeout=10)

    monkeypatch.setattr("miles.tinker.core.service.prune_checkpoints", slow_prune)
    model = await created_model(service)
    request = service.submit("tenant", "save_state", {"model_id": model, "seq_id": 1, "name": "saved"})
    try:
        assert (await await_settled(service, "tenant", request)).state == DONE
        assert await asyncio.to_thread(started.wait, 3)
        await asyncio.wait_for(step(service, model, 2), timeout=1)
    finally:
        release.set()
        await asyncio.gather(*service._background_tasks)


async def test_failed_save_and_load_do_not_prune_or_change_lineage(service):
    model = await created_model(service)
    path = await save(service, model, 1, "good")
    service.backend.fail_next = RuntimeError("disk full")
    request = service.submit("tenant", "save_state", {"model_id": model, "seq_id": 2, "name": "failed"})
    assert (await await_settled(service, "tenant", request)).state != DONE
    assert not (Path(service.config.checkpoint_root) / model / "weights/failed/tinker_meta.json").exists()
    newer = await created_model(service)
    service.backend.fail_next = RuntimeError("bad load")
    request = service.submit("tenant", "load_state", {"model_id": newer, "seq_id": 1, "path": path, "optimizer": True})
    assert (await await_settled(service, "tenant", request)).state != DONE
    assert service.models[newer].checkpoint_lineage == newer


def test_latest_resume_wins_even_when_loaded_from_an_older_checkpoint(tmp_path):
    # A fresh resume from an early checkpoint takes priority over a later
    # training branch, regardless of checkpoint names or wall-clock save order.
    previous = checkpoint(
        tmp_path, "previous", "step-100", checkpoint_lineage="root", resume_started_at=10, created_at=100
    )
    latest = checkpoint(tmp_path, "latest", "step-2", checkpoint_lineage="root", resume_started_at=20, created_at=21)
    prune_checkpoints(str(tmp_path), "root", (), 1)
    assert latest.exists()
    assert not previous.exists()


def test_deletion_failure_keeps_newest_checkpoint(tmp_path, monkeypatch, caplog):
    old = checkpoint(tmp_path, "model", "old", created_at=1)
    newest = checkpoint(tmp_path, "model", "new", created_at=2)

    def fail(path):
        raise OSError("NFS unavailable")

    monkeypatch.setattr("miles.tinker.core.retention.shutil.rmtree", fail)
    prune_checkpoints(str(tmp_path), "model", (), 1)
    assert newest.exists() and old.exists()
    assert "Checkpoint retention failed" in caplog.text
