"""``examples/tinker/run_qwen3_6_35b_a3b_gateway.py``: the argv it hands ``serve_tinker.py``.

The ``--sglang-*`` allowlist comes from ``tests/fast/fork/data/sglang_server_args_fields.txt``,
a snapshot of ``dataclasses.fields(ServerArgs)`` taken inside the serving image with
``tools/fork/dump_sglang_server_args.py`` (the fast tests have no sglang to ask).
"""

import importlib.util
import shlex
from collections import Counter
from pathlib import Path

import pytest

import miles.utils.external_utils.command_utils as U

REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = REPO_ROOT / "examples" / "tinker" / "run_qwen3_6_35b_a3b_gateway.py"
SERVER_ARGS_FIELDS = REPO_ROOT / "tests" / "fast" / "fork" / "data" / "sglang_server_args_fields.txt"

# --sglang-<flag> flags whose stem is not a ServerArgs field name: miles registers ServerArgs.add_cli_args
# under the prefix, so every other stem must be a field (dashes -> underscores).
_SGLANG_FLAG_TO_FIELD = {"--sglang-ep-size": "ep_size"}


@pytest.fixture
def launcher():
    spec = importlib.util.spec_from_file_location("run_qwen3_6_35b_a3b_gateway", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _serve_argv(module, monkeypatch, **overrides) -> list[str]:
    captured = {}
    monkeypatch.setattr(U, "execute_train", lambda **kwargs: captured.update(kwargs))
    module.serve(hf_checkpoint="/x/hf", checkpoint_root="/x/ckpt", **overrides)
    assert captured["train_script"] == "serve_tinker.py"
    assert captured["megatron_model_type"] == "qwen3.6-35B-A3B"
    return shlex.split(captured["train_args"])


def _flag(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def _flags_multiset(argv: list[str]) -> Counter:
    """``{flag: (values...)}`` counted; a value-less flag maps to ``()``."""
    pairs = []
    index = 0
    while index < len(argv):
        flag = argv[index]
        assert flag.startswith("--"), f"stray value {flag!r} at {index}"
        values = []
        index += 1
        while index < len(argv) and not argv[index].startswith("--"):
            values.append(argv[index])
            index += 1
        pairs.append((flag, tuple(values)))
    return Counter(pairs)


def _server_args_fields() -> set[str]:
    lines = SERVER_ARGS_FIELDS.read_text().splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def _sglang_field(flag: str) -> str:
    return _SGLANG_FLAG_TO_FIELD.get(flag, flag.removeprefix("--sglang-").replace("-", "_"))


_EXPECTED_DEFAULT_ARGV = (
    "--hf-checkpoint /x/hf --megatron-to-hf-mode bridge --freeze-vision-model "
    "--tinker-train-mode full "
    "--tinker-server-port 10613 --tinker-checkpoint-root /x/ckpt "
    "--tinker-base-model apurvaga/Qwen3.6-35B-A3B-preserve-thinking "
    "--tinker-max-tokens-per-datum 65537 --tinker-unit-token-budget 2097184 "
    "--tinker-coalesce-wait-s 5.0 --tinker-coalesce-max-wait-s 120.0 --tinker-max-concurrent-samples 1024 "
    "--tinker-lease-timeout-s 600 --tinker-routing-replay-policy natural "
    "--tinker-routing-cache-max-bytes 137438953472 --tinker-routing-cache-ttl-s 21600 "
    "--tinker-routing-prefix-stride 256 "
    "--use-rollout-routing-replay "
    "--fused-lm-head-logprobs --fused-lm-head-chunk-size 1024 "
    "--optimizer adam --lr 3e-6 --use-distributed-optimizer --accumulate-allreduce-grads-in-fp32 "
    "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 "
    "--expert-model-parallel-size 8 --expert-tensor-parallel-size 1 --sequence-parallel "
    "--use-dynamic-batch-size --max-tokens-per-gpu 65537 --balance-data "
    "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
    "--moe-token-dispatcher-type alltoall "
    "--mtp-num-layers 0 "
    "--rollout-num-gpus-per-engine 8 --sglang-ep-size 8 --sglang-mem-fraction-static 0.75 "
    "--sglang-context-length 65537 --rollout-max-context-len 65536 --sglang-chunked-prefill-size 16384 "
    "--sglang-max-running-requests 128 --sglang-enable-fp32-lm-head "
    "--pause-generation-mode retract --update-weight-buffer-size 2147483648 "
    "--num-gpus-per-node 8 --actor-num-nodes 4 --actor-num-gpus-per-node 8 --rollout-num-gpus 88 "
    "--attention-dropout 0.0 --hidden-dropout 0.0 --attention-softmax-in-fp32 --attention-backend flash"
)


def test_vision_tower_is_frozen_by_default_and_can_be_trained_on_request(launcher, monkeypatch):
    """A text-only RL run must not train (or weight-decay) the VL checkpoint's vision tower."""
    assert "--freeze-vision-model" in _serve_argv(launcher, monkeypatch)
    assert "--freeze-vision-model" not in _serve_argv(launcher, monkeypatch, freeze_vision_model=False)


def test_default_serve_argv_is_the_designed_flag_multiset(launcher, monkeypatch):
    argv = _serve_argv(launcher, monkeypatch)
    assert _flags_multiset(argv) == _flags_multiset(shlex.split(_EXPECTED_DEFAULT_ARGV))


def test_default_argv_pins_the_recipe_decisions(launcher, monkeypatch):
    argv = _serve_argv(launcher, monkeypatch)
    assert _flag(argv, "--mtp-num-layers") == "0" and "--enable-mtp-training" not in argv
    assert "--use-rollout-routing-replay" in argv
    assert "--fused-lm-head-logprobs" in argv and "--log-probs-chunk-size" not in argv
    assert "--balance-data" in argv
    assert _flag(argv, "--expert-model-parallel-size") == "8"
    assert _flag(argv, "--tinker-max-tokens-per-datum") == "65537"
    assert _flag(argv, "--tinker-lease-timeout-s") == "600"
    assert _flag(argv, "--tensor-model-parallel-size") == "1"
    assert not [flag for flag in argv if flag.startswith("--sglang-mamba-")]
    assert "--sglang-max-total-tokens" not in argv
    assert "--check-weight-update-equal" not in argv and "--tinker-r3-check" not in argv
    assert "--use-precision-aware-optimizer" not in argv and "--optimizer-cpu-offload" not in argv
    assert "--overlap-grad-reduce" not in argv and "--overlap-param-gather" not in argv
    assert "--multi-lora-n-adapters" not in argv and "--lora-rank" not in argv


def test_every_sglang_flag_is_a_server_args_field(launcher, monkeypatch):
    fields = _server_args_fields()
    assert len(fields) > 300, "the ServerArgs snapshot looks truncated"
    for overrides in ({}, {"sglang_max_total_tokens": 4_000_000, "sglang_fp32_lm_head": False}):
        argv = _serve_argv(launcher, monkeypatch, **overrides)
        sglang_flags = [flag for flag in argv if flag.startswith("--sglang-")]
        unknown = {flag for flag in sglang_flags if _sglang_field(flag) not in fields}
        assert not unknown, f"not ServerArgs fields in the serving image: {sorted(unknown)}"


def test_the_stale_mamba_scheduler_flag_is_not_a_server_args_field():
    """The reason the allowlist exists: the image only has ``mamba_radix_cache_strategy``."""
    fields = _server_args_fields()
    assert "mamba_radix_cache_strategy" in fields
    assert "mamba_scheduler_strategy" not in fields
    assert {"enable_fp32_lm_head", "max_total_tokens", "enable_return_routed_experts"} <= fields


def test_tp2_fallback_and_optional_knobs(launcher, monkeypatch):
    argv = _serve_argv(
        launcher,
        monkeypatch,
        tp=2,
        check_weight_update_equal=True,
        r3_check=True,
        sglang_max_total_tokens=5_000_000,
        routing_policy="strict",
    )
    assert _flag(argv, "--tensor-model-parallel-size") == "2" and "--sequence-parallel" in argv
    assert "--check-weight-update-equal" in argv
    assert "--tinker-r3-check" in argv
    assert _flag(argv, "--sglang-max-total-tokens") == "5000000"
    assert _flag(argv, "--tinker-routing-replay-policy") == "strict"


def test_head_fallback_and_r3_off(launcher, monkeypatch):
    argv = _serve_argv(launcher, monkeypatch, fused_lm_head=False, routing_replay=False, sglang_fp32_lm_head=False)
    assert "--fused-lm-head-logprobs" not in argv and "--fused-lm-head-chunk-size" not in argv
    assert _flag(argv, "--log-probs-chunk-size") == "2048"
    assert "--use-rollout-routing-replay" not in argv
    assert "--sglang-enable-fp32-lm-head" not in argv


def test_context_length_drives_both_sides(launcher, monkeypatch):
    argv = _serve_argv(launcher, monkeypatch, context_len=32768, max_tokens_per_gpu=32769, max_tokens_per_datum=32769)
    assert _flag(argv, "--sglang-context-length") == "32769"
    assert _flag(argv, "--rollout-max-context-len") == "32768"
    assert _flag(argv, "--max-tokens-per-gpu") == "32769"


def test_topology_flags_follow_the_split(launcher, monkeypatch):
    argv = _serve_argv(launcher, monkeypatch, actor_num_nodes=1, rollout_num_gpus=8)
    assert _flag(argv, "--actor-num-nodes") == "1" and _flag(argv, "--rollout-num-gpus") == "8"
    assert _flag(argv, "--num-gpus-per-node") == "8"


def test_rejects_invalid_configs(launcher):
    with pytest.raises(AssertionError):
        launcher.ScriptArgs(mode="lora")
    with pytest.raises(AssertionError):
        launcher.ScriptArgs(pause_generation_mode="freeze")
    with pytest.raises(AssertionError):
        launcher.ScriptArgs(routing_policy="loose")
    with pytest.raises(AssertionError):
        launcher.ScriptArgs(rollout_num_gpus=12)
    with pytest.raises(AssertionError):
        launcher.ScriptArgs(max_tokens_per_gpu=4096)


# --- LAUNCHER_ARGS strings quoted in the deploy docs/scripts must be options of `serve` -----------
# (a serve_tinker.py flag such as --tinker-r3-check is rejected by typer only after the whole Ray
# cluster is up: deploy/abci/tinker_gateway.pbs execs the launcher after the ray-status wait)

_DOCUMENTED_LAUNCHER_ARGS_FILES = [
    REPO_ROOT / "deploy" / "abci" / "clients" / "smoke_q36.sh",
    REPO_ROOT / "deploy" / "abci" / "README.md",
    REPO_ROOT / "deploy" / "abci" / "submit_toolathlon.sh",
]


def _documented_launcher_args() -> list[str]:
    import re

    found = []
    for path in _DOCUMENTED_LAUNCHER_ARGS_FILES:
        for match in re.finditer(
            r"""LAUNCHER_ARGS=(?:\$\{LAUNCHER_ARGS:-)?(["']?)(--[^"'\n,]*?)\1(?=[\s,)}`]|$)""", path.read_text()
        ):
            found.append(match.group(2).strip())
    assert found, "no LAUNCHER_ARGS strings found in the deploy docs"
    return sorted(set(found))


def _invoke_serve(launcher, monkeypatch, argv: list[str]):
    from typer.testing import CliRunner

    captured = {}
    monkeypatch.setattr(U, "execute_train", lambda **kwargs: captured.update(kwargs))
    result = CliRunner().invoke(
        launcher.app, ["serve", "--hf-checkpoint", "/x/hf", "--checkpoint-root", "/x/ckpt", *argv]
    )
    return result, captured


@pytest.mark.parametrize("launcher_args", _documented_launcher_args())
def test_documented_launcher_args_are_accepted_by_the_typer_cli(launcher, monkeypatch, launcher_args):
    argv = shlex.split(launcher_args)
    if any(flag in launcher_args for flag in ("--n-adapters", "--max-tokens-per-gpu")):
        pytest.skip("README example for the generic LoRA launcher, not this typer script")
    assert not any(token.startswith("--tinker-") for token in argv), launcher_args
    result, captured = _invoke_serve(launcher, monkeypatch, argv)
    assert result.exit_code == 0, f"{launcher_args!r}: {result.output}"
    assert captured["train_script"] == "serve_tinker.py"


def test_r3_check_relaunch_reaches_the_trainer_actors(launcher, monkeypatch):
    """smoke_q36.sh step 3: --r3-check emits --tinker-r3-check and --extra-env-vars carries the
    threshold to the Ray actors (replay_base.py reads MILES_TEST_R3_THRESHOLD from os.environ)."""
    result, captured = _invoke_serve(
        launcher,
        monkeypatch,
        shlex.split("--check-weight-update-equal --r3-check --extra-env-vars MILES_TEST_R3_THRESHOLD=0.02"),
    )
    assert result.exit_code == 0, result.output
    argv = shlex.split(captured["train_args"])
    assert "--tinker-r3-check" in argv and "--check-weight-update-equal" in argv
    assert captured["config"].extra_env_vars == "MILES_TEST_R3_THRESHOLD=0.02"
    assert U.resolve_extra_env_vars({}, captured["config"]) == {"MILES_TEST_R3_THRESHOLD": "0.02"}


def test_serve_side_flags_are_rejected_by_the_typer_cli(launcher, monkeypatch):
    result, captured = _invoke_serve(launcher, monkeypatch, ["--check-weight-update-equal", "--tinker-r3-check"])
    assert result.exit_code == 2 and "No such option: --tinker-r3-check" in result.output
    assert not captured, "execute_train must not run"
