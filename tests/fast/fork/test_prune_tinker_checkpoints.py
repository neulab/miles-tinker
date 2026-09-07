import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "deploy" / "abci" / "clients" / "prune_tinker_checkpoints.py"


@pytest.fixture(scope="module")
def prune():
    spec = importlib.util.spec_from_file_location("prune_tinker_checkpoints", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve `from __future__` annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def record(model: str, name: str, batch: int) -> dict:
    return {
        "name": name,
        "batch": batch,
        "state_path": f"tinker://{model}/weights/{name}",
        "sampler_path": f"tinker://{model}/sampler_weights/{name}",
    }


def dirs_of(root: Path, records: list[dict]) -> dict[Path, float]:
    old = 10_000.0
    existing = {}
    for rec in records:
        existing[root / rec["state_path"].split("tinker://")[1]] = old
        existing[root / rec["sampler_path"].split("tinker://")[1]] = old
    return existing


def rel(root: Path, paths) -> set[str]:
    return {p.relative_to(root).as_posix() for p in paths}


def test_tinker_path_mapping(prune, tmp_path):
    assert prune.tinker_path_to_dir(tmp_path, "tinker://model-abc/weights/000005") == tmp_path / "model-abc" / "weights" / "000005"
    assert prune.tinker_path_to_dir(tmp_path, "tinker://model-abc/sampler_weights/x") == tmp_path / "model-abc" / "sampler_weights" / "x"
    assert prune.tinker_path_to_dir(tmp_path, "tinker://model-abc/other/x") is None
    assert prune.tinker_path_to_dir(tmp_path, "/local/path") is None


def test_recovery_names_are_the_prunable_tier(prune):
    assert prune.is_recovery_name("recovery-000008")
    assert not prune.is_recovery_name("000008")
    assert not prune.is_recovery_name("000008_interrupted")
    assert not prune.is_recovery_name("final")


def test_keeps_newest_recovery_records_across_model_ids_and_deletes_the_older_ones(prune, tmp_path):
    root = tmp_path / "ckpt"
    records = [
        record("model-a", "recovery-000000", 0),  # first launch
        record("model-a", "recovery-000001", 1),
        record("model-a", "000002", 2),  # permanent (checkpoint.every)
        record("model-b", "recovery-000002", 2),  # relaunch -> new model_id
        record("model-b", "recovery-000003", 3),
    ]
    existing = dirs_of(root, records)
    plan = prune.plan_prune([records], existing, root, keep=2, now=100_000.0)
    assert rel(root, plan.delete) == {
        "model-a/weights/recovery-000000",
        "model-a/sampler_weights/recovery-000000",
        "model-a/weights/recovery-000001",
        "model-a/sampler_weights/recovery-000001",
    }
    assert rel(root, (p for p in existing if p in plan.keep)) == {
        "model-a/weights/000002",
        "model-a/sampler_weights/000002",
        "model-b/weights/recovery-000002",
        "model-b/sampler_weights/recovery-000002",
        "model-b/weights/recovery-000003",
        "model-b/sampler_weights/recovery-000003",
    }
    assert plan.reasons[root / "model-a" / "weights" / "000002"] == "permanent checkpoint"


def test_permanent_and_unreferenced_directories_are_never_deleted(prune, tmp_path):
    root = tmp_path / "ckpt"
    records = [record("model-a", "recovery-000000", 0), record("model-a", "recovery-000005", 5)]
    existing = dirs_of(root, records)
    for name in ("000020", "000031_interrupted", "final", "stray"):
        existing[root / "model-a" / "weights" / name] = 10_000.0
    stale_recovery = root / "model-a" / "weights" / "recovery-999999"  # no record at all
    existing[stale_recovery] = 10_000.0
    plan = prune.plan_prune([records], existing, root, keep=1, now=100_000.0)
    assert rel(root, plan.delete) == {
        "model-a/weights/recovery-000000",
        "model-a/sampler_weights/recovery-000000",
        "model-a/weights/recovery-999999",
    }
    for name in ("000020", "000031_interrupted", "final", "stray"):
        assert root / "model-a" / "weights" / name in plan.keep


def test_launch_checkpoint_survives_until_a_later_record_exists(prune, tmp_path):
    root = tmp_path / "ckpt"
    launch = [record("model-a", "recovery-000000", 0)]
    plan = prune.plan_prune([launch], dirs_of(root, launch), root, keep=1, now=100_000.0)
    assert plan.delete == ()
    later = launch + [record("model-a", "recovery-000005", 5)]
    plan = prune.plan_prune([later], dirs_of(root, later), root, keep=1, now=100_000.0)
    assert {p.name for p in plan.delete} == {"recovery-000000"} and len(plan.delete) == 2


def test_last_record_state_path_is_never_deleted_even_with_keep_1_and_more_records(prune, tmp_path):
    root = tmp_path / "ckpt"
    records = [
        record("model-a", "recovery-000000", 0),
        record("model-a", "recovery-000005", 5),
        {"name": "recovery-sampler-only", "sampler_path": "tinker://model-a/sampler_weights/recovery-s1"},
    ]
    existing = dirs_of(root, records[:2])
    existing[root / "model-a" / "sampler_weights" / "recovery-s1"] = 10_000.0
    plan = prune.plan_prune([records], existing, root, keep=1, now=100_000.0)
    assert root / "model-a" / "weights" / "recovery-000005" in plan.keep  # last record with a state_path
    assert root / "model-a" / "sampler_weights" / "recovery-s1" in plan.keep  # the newest recovery record
    assert plan.reasons[root / "model-a" / "weights" / "recovery-000005"] == "last record's state_path"


def test_a_missing_sampler_directory_is_tolerated(prune, tmp_path):
    """In the gateway's full mode a sampler_path has no directory of its own."""
    root = tmp_path / "ckpt"
    records = [record("model-a", "recovery-000000", 0), record("model-a", "recovery-000005", 5)]
    existing = {root / "model-a" / "weights" / name: 10_000.0 for name in ("recovery-000000", "recovery-000005")}
    plan = prune.plan_prune([records], existing, root, keep=1, now=100_000.0)
    assert rel(root, plan.delete) == {"model-a/weights/recovery-000000"}


def test_recent_directories_are_in_progress_saves(prune, tmp_path):
    root = tmp_path / "ckpt"
    records = [record("model-a", "recovery-000000", 0), record("model-a", "recovery-000005", 5)]
    existing = dirs_of(root, records)
    in_progress = root / "model-a" / "weights" / "recovery-000010"
    existing[in_progress] = 100_000.0 - 60  # written a minute ago, no record yet
    plan = prune.plan_prune([records], existing, root, keep=1, now=100_000.0)
    assert in_progress in plan.keep and "grace" in plan.reasons[in_progress]
    assert {p.name for p in plan.delete} == {"recovery-000000"}


def test_keep_must_be_positive(prune, tmp_path):
    with pytest.raises(ValueError):
        prune.plan_prune([[]], {}, tmp_path, keep=0, now=0.0)


def test_cli_deletes_directories_and_dry_run_does_not(tmp_path):
    root = tmp_path / "ckpt"
    log_path = tmp_path / "logs"
    trial = log_path / "exp" / "trial"
    trial.mkdir(parents=True)
    records = [
        record("model-a", "recovery-000000", 0),
        record("model-a", "000003", 3),
        record("model-a", "recovery-000005", 5),
    ]
    trial.joinpath("checkpoints.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records) + "{not json\n")
    for rec in records:
        for key in ("state_path", "sampler_path"):
            d = root / rec[key].split("tinker://")[1]
            d.mkdir(parents=True)
            (d / "shard.pt").write_bytes(b"x")
            os.utime(d, (time.time() - 7200, time.time() - 7200))
    cmd = [sys.executable, str(SCRIPT), "--log-path", str(log_path), "--checkpoint-root", str(root), "--keep", "1"]
    out = subprocess.run(cmd + ["--dry-run"], capture_output=True, text=True, check=True).stdout
    assert "would delete" in out and (root / "model-a" / "weights" / "recovery-000000").is_dir()
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    assert "delete " in out
    assert not (root / "model-a" / "weights" / "recovery-000000").exists()
    assert not (root / "model-a" / "sampler_weights" / "recovery-000000").exists()
    assert (root / "model-a" / "weights" / "recovery-000005" / "shard.pt").exists()
    assert (root / "model-a" / "sampler_weights" / "recovery-000005" / "shard.pt").exists()
    assert (root / "model-a" / "weights" / "000003" / "shard.pt").exists()  # permanent, kept forever
    assert (root / "model-a" / "sampler_weights" / "000003" / "shard.pt").exists()
