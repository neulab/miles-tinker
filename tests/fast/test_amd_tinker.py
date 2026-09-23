"""Host-side launch and retention checks; no GPU or running Ray cluster needed."""

import json
import os
import shutil
import socket
from pathlib import Path

import pytest

import tools.amd_tinker as amd_tinker
from tools.amd_tinker import BackendConfig, _container_command, _prune, _runtime_env, _server_args


def test_setup_retains_native_image_stack(tmp_path, monkeypatch):
    config = BackendConfig(root=tmp_path)
    config.image.parent.mkdir(parents=True)
    config.image.touch()
    (config.root / "deps/aiter").mkdir(parents=True)
    commands = []
    monkeypatch.setattr(amd_tinker, "_run", lambda config, command: commands.append(list(map(str, command))))

    amd_tinker._setup(config)

    installs = [command[command.index("install") + 1 :] for command in commands if "install" in command]
    assert any("tinker==0.27.1" in args for args in installs)
    assert all("--no-deps" in args for args in installs)
    assert not any("nvidia-modelopt==0.44.0" in args or "ray[default]==2.56.0" in args for args in installs)
    assert any("ray[default]" in args for args in installs)
    assert not any("git" in command for command in commands)


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
    assert "MILES_SGLANG_LEGACY_WEIGHT_UPDATE" not in _runtime_env(config)


def test_convert_runs_in_image_and_reuses_completed_checkpoint(tmp_path, monkeypatch):
    config = BackendConfig(root=tmp_path, model_id="Qwen/Qwen3.5-4B", model_type="qwen3.5-4B")
    output = config.root / "models/Qwen3.5-4B_torch_dist"
    commands = []

    def run(config, command):
        commands.append(list(map(str, command)))
        output.mkdir(parents=True)
        (output / "latest_checkpointed_iteration.txt").write_text("release")

    monkeypatch.setattr(amd_tinker, "_run", run)
    amd_tinker._convert(config)
    amd_tinker._convert(config)
    assert len(commands) == 1
    assert commands[0][-4:] == [config.model_type, str(config.model_path), str(output), str(config.context_length)]
    assert str(config.image) in commands[0]


def test_convert_rejects_incomplete_output(tmp_path):
    config = BackendConfig(root=tmp_path, model_id="Qwen/Qwen3.5-4B", model_type="qwen3.5-4B")
    (config.root / "models/Qwen3.5-4B_torch_dist").mkdir(parents=True)
    with pytest.raises(ValueError, match="Incomplete conversion"):
        amd_tinker._convert(config)


def test_pinned_image_and_external_checkout(tmp_path):
    config = BackendConfig(root=tmp_path, repo=Path("/work/miles"))
    command = _container_command(config, [config.python, "-c", "pass"])
    assert "/work/miles:/work/miles" in command
    assert str(config.image) in command
    assert config.image == BackendConfig(root=tmp_path).image
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


def test_new_image_launch_defaults_and_context_override(tmp_path):
    args = _server_args(BackendConfig(root=tmp_path, context_length=65000))
    assert args[args.index("--sglang-context-length") + 1] == "65000"
    assert args[args.index("--sglang-device") + 1] == "cuda"
    assert args[args.index("--megatron-to-hf-mode") + 1] == "bridge"
    assert "--ref-load" not in args


def test_qwen35_selects_converted_text_model(tmp_path):
    config = BackendConfig(root=tmp_path, model_id="Qwen/Qwen3.5-4B", model_type="qwen3.5-4B")
    args = _server_args(config)
    assert args[args.index("--megatron-to-hf-mode") + 1] == "raw"
    assert args[args.index("--ref-load") + 1] == str(config.model_path.with_name("Qwen3.5-4B_torch_dist"))
