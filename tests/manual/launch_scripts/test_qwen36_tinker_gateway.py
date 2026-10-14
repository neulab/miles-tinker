from pathlib import Path

import pytest

from tests.fast.launch_scripts.py_harness import (
    call_entrypoint,
    format_recording,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)
from tests.fast.launch_scripts.sh_harness import REPO_ROOT, assert_matches_snapshot

_LAUNCHER = "examples/tinker/run_qwen3_6_35b_a3b_gateway.py"


@pytest.mark.parametrize("variant", ["35b-a3b", "27b", "4b"])
@pytest.mark.parametrize("entrypoint", ["prepare", "serve"])
def test_qwen36_gateway_recipe(monkeypatch, tmp_path: Path, variant: str, entrypoint: str):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / _LAUNCHER)
    call_entrypoint(module, entrypoint, {"model_variant": variant}, sandbox=tmp_path)
    snapshot = REPO_ROOT / "tests/snapshots/launch_scripts/py" / _LAUNCHER / f"{entrypoint}-{variant}.txt"
    assert_matches_snapshot(snapshot, format_recording(recording, sandbox=tmp_path), f"{variant}::{entrypoint}")


@pytest.mark.parametrize("overrides", [{"routing_replay": True}, {"ep": 8}, {"sglang_ep_size": 8}, {"tp": 8}])
def test_dense_recipe_rejects_moe_settings(overrides):
    module = import_launch_script(REPO_ROOT / _LAUNCHER)
    with pytest.raises(AssertionError):
        module.ScriptArgs(model_variant="27b", **overrides)
