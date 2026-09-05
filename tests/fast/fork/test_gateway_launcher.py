import importlib.util
import shlex
from pathlib import Path

import pytest

import miles.utils.external_utils.command_utils as U

LAUNCHER = Path(__file__).resolve().parents[3] / "examples" / "tinker" / "run_qwen3_4b_gateway.py"


@pytest.fixture
def launcher(monkeypatch):
    spec = importlib.util.spec_from_file_location("run_qwen3_4b_gateway", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _serve_argv(module, monkeypatch, **overrides) -> list[str]:
    captured = {}
    monkeypatch.setattr(U, "execute_train", lambda **kwargs: captured.update(kwargs))
    # @U.dataclass_cli turns the ScriptArgs fields into keyword arguments
    module.serve(hf_checkpoint="/x/hf", checkpoint_root="/x/ckpt", **overrides)
    assert captured["train_script"] == "serve_tinker.py"
    return shlex.split(captured["train_args"])


def _flag(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_full_mode_emits_the_train_mode_flag_and_no_lora_flags(launcher, monkeypatch):
    argv = _serve_argv(launcher, monkeypatch, mode="full")
    assert _flag(argv, "--tinker-train-mode") == "full"
    assert "--multi-lora-n-adapters" not in argv and "--lora-rank" not in argv


def test_lora_mode_emits_the_slot_pool(launcher, monkeypatch):
    argv = _serve_argv(launcher, monkeypatch, mode="lora", n_adapters=6)
    assert _flag(argv, "--multi-lora-n-adapters") == "6"
    assert "--tinker-train-mode" not in argv
    assert "--sglang-max-loras-per-batch" not in argv  # shadowed by miles for multi-LoRA; not emitted


def test_weight_push_and_engine_knobs(launcher, monkeypatch):
    argv = _serve_argv(launcher, monkeypatch, mode="lora")
    assert _flag(argv, "--pause-generation-mode") == "in_place"
    assert _flag(argv, "--update-weight-buffer-size") == str(1024**3)
    assert _flag(argv, "--sglang-chunked-prefill-size") == "4096"
    assert _flag(argv, "--sglang-max-running-requests") == "512"
    assert _flag(argv, "--tinker-server-port") == "10613"
    assert _flag(argv, "--tinker-checkpoint-root") == "/x/ckpt"


def test_rejects_unknown_modes(launcher):
    with pytest.raises(AssertionError):
        launcher.ScriptArgs(mode="qlora")
    with pytest.raises(AssertionError):
        launcher.ScriptArgs(pause_generation_mode="freeze")
