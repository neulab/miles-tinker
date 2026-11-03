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


def _format(recording, sandbox):
    # Grouped argv strings contain trailing separators; snapshots need no EOL spaces.
    return "\n".join(line.rstrip() for line in format_recording(recording, sandbox=sandbox).splitlines()) + "\n"


@pytest.mark.parametrize("variant", ["35b-a3b", "27b"])
@pytest.mark.parametrize("entrypoint", ["prepare", "serve"])
def test_qwen36_gateway_recipe(monkeypatch, tmp_path: Path, variant: str, entrypoint: str):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / _LAUNCHER)
    call_entrypoint(module, entrypoint, {"model_variant": variant}, sandbox=tmp_path)
    snapshot = REPO_ROOT / "tests/snapshots/launch_scripts/py" / _LAUNCHER / f"{entrypoint}-{variant}.txt"
    assert_matches_snapshot(snapshot, _format(recording, tmp_path), f"{variant}::{entrypoint}")


@pytest.mark.parametrize("train_mtp", [False, True], ids=["frozen-draft", "trained-draft"])
def test_qwen36_gateway_mtp_recipe(monkeypatch, tmp_path: Path, train_mtp: bool):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / _LAUNCHER)
    call_entrypoint(
        module,
        "serve",
        {"model_variant": "27b", "enable_mtp_training": train_mtp, "mtp_speculative_decoding": True},
        sandbox=tmp_path,
    )
    suffix = "trained-draft" if train_mtp else "frozen-draft"
    snapshot = REPO_ROOT / "tests/snapshots/launch_scripts/py" / _LAUNCHER / f"serve-27b-{suffix}.txt"
    assert_matches_snapshot(snapshot, _format(recording, tmp_path), f"27b::{suffix}")
