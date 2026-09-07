"""Host-side launch and retention checks; no GPU or running Ray cluster needed."""

import json
import os
from pathlib import Path

import pytest

from tools.amd_tinker import BackendConfig, _container_command, _prune, _runtime_env, _server_args


def test_relocated_backend_preserves_paths_and_explicit_mfu(tmp_path):
    config = BackendConfig(root=tmp_path / "runtime with spaces", port=11613, train_gpus=2, rollout_gpus=2)
    args = _server_args(config)
    assert args[args.index("--tinker-checkpoint-root") + 1] == str(config.root / "checkpoints")
    assert args[args.index("--num-gpus-per-node") + 1] == "4"
    assert args[args.index("--tinker-train-mode") + 1] == "full"
    assert args[args.index("--mfu-peak-tflops") + 1] == "2309.6"
    command = _container_command(config, [config.python, "-c", "pass"])
    assert "--rocm" in command
    assert command[-3:] == [str(config.python), "-c", "pass"]
    assert _runtime_env(config)["MILES_SGLANG_LEGACY_WEIGHT_UPDATE"] == "1"


@pytest.mark.parametrize("overrides", [{"train_gpus": 3, "tp": 2}, {"root": Path("/work1/not-local")}, {"peak_tflops": float("nan")}, {"rollout_gpus_per_engine": 3}])
def test_invalid_configuration_fails_before_launch(overrides):
    with pytest.raises(ValueError):
        BackendConfig(**overrides)


def test_pruner_preserves_newest_complete_and_incomplete_saves(tmp_path):
    config = BackendConfig(root=tmp_path)
    base = tmp_path / "checkpoints/model/weights"
    for name, timestamp in (("old", 1), ("new", 2)):
        path = base / name
        path.mkdir(parents=True)
        marker = path / "tinker_meta.json"
        marker.write_text(json.dumps({"name": name}))
        os.utime(marker, (timestamp, timestamp))
    (base / "incomplete").mkdir()
    _prune(config, keep=1, once=True)
    assert not (base / "old").exists()
    assert (base / "new/tinker_meta.json").exists()
    assert (base / "incomplete").exists()
