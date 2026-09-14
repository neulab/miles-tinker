"""Host-side launch and retention checks; no GPU or running Ray cluster needed."""

import json
import os
import shutil
import socket
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


def test_custom_architecture_image_and_external_checkout(tmp_path):
    config = BackendConfig(root=tmp_path, repo=Path("/work/miles"), image_uri="docker://rlsys/miles:MI300-latest")
    command = _container_command(config, [config.python, "-c", "pass"])
    assert "/work/miles:/work/miles" in command
    assert str(config.image) in command
    assert config.image != BackendConfig(root=tmp_path).image
    assert _runtime_env(config)["PYTHONPATH"].startswith("/work/miles:")


def test_optional_aiter_environment_survives_cleanenv(tmp_path, monkeypatch):
    monkeypatch.delenv("CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT", raising=False)
    monkeypatch.delenv("AITER_JIT_DIR", raising=False)
    default = BackendConfig(root=tmp_path)
    assert "CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT" not in _runtime_env(default)
    assert "AITER_JIT_DIR" not in _runtime_env(default)
    monkeypatch.setenv("CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT", "0")
    monkeypatch.setenv("AITER_JIT_DIR", str(tmp_path / "aiter-jit-bf16-rne"))
    command = _container_command(default, [default.python, "-c", "pass"])
    environment = command[command.index("--env") + 1]
    assert "CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=0" in environment.split(",")
    assert f"AITER_JIT_DIR={tmp_path}/aiter-jit-bf16-rne" in environment.split(",")


def test_external_checkpoint_root_is_passed_and_bound(tmp_path):
    checkpoint_root = Path("/work1/checkpoints with spaces")
    config = BackendConfig(root=tmp_path, checkpoint_root=checkpoint_root)
    args = _server_args(config)
    assert args[args.index("--tinker-checkpoint-root") + 1] == str(checkpoint_root)
    command = _container_command(config, [config.python, "-c", "pass"])
    assert f"{checkpoint_root}:{checkpoint_root}" in command


def test_ray_socket_path_fits_with_long_runtime_root_and_large_pid(tmp_path):
    config = BackendConfig(root=tmp_path / ("long-runtime-" * 12))
    ray_root = Path(_runtime_env(config)["RAY_TMPDIR"])
    socket_path = ray_root / "ray/session_2026-09-13_15-03-09_747001_2147483647/sockets/dash_MetricsHead"
    assert len(os.fsencode(socket_path)) <= 107
    ray_root.mkdir(exist_ok=False)
    try:
        socket_path.parent.mkdir(parents=True)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(socket_path))
    finally:
        shutil.rmtree(ray_root)


def test_ray_temp_directory_is_stable_and_scoped_to_runtime(tmp_path, monkeypatch):
    config = BackendConfig(root=tmp_path / "runtime")
    before = _runtime_env(config)["RAY_TMPDIR"]
    monkeypatch.setattr(os, "getpid", lambda: 2147483647)
    assert _runtime_env(config)["RAY_TMPDIR"] == before
    other = BackendConfig(root=tmp_path / "other-runtime")
    assert _runtime_env(other)["RAY_TMPDIR"] != before


@pytest.mark.parametrize(
    "overrides",
    [
        {"train_gpus": 3, "tp": 2},
        {"root": Path("/work1/not-local")},
        {"peak_tflops": float("nan")},
        {"rollout_gpus_per_engine": 3},
    ],
)
def test_invalid_configuration_fails_before_launch(overrides):
    with pytest.raises(ValueError):
        BackendConfig(**overrides)


@pytest.mark.parametrize("external_checkpoints", [False, True])
def test_pruner_preserves_newest_complete_and_incomplete_saves(tmp_path, external_checkpoints):
    checkpoint_root = tmp_path / "persistent-checkpoints" if external_checkpoints else None
    config = BackendConfig(root=tmp_path / "runtime", checkpoint_root=checkpoint_root)
    base = config.checkpoint_root / "model/weights"
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
