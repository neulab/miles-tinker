"""Vanillux harness text: templates, observation truncation and the submit marker.

The templates live in ``vanillux_prompts.yaml``, vendored from tmax (which
vendored them from mini-swe-agent v2.2.x). Constants and the truncation rule
follow tmax's ``swerl_sandbox`` / ``swerl_vanillux_sandbox``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
LAST_STEP_WARNING = (
    f"You are running out of steps. You must end your next action with `echo {SUBMIT_MARKER}` to submit your work."
)

_PROMPTS: dict[str, Any] = yaml.safe_load((Path(__file__).parent / "vanillux_prompts.yaml").read_text())

SYSTEM_TEMPLATE: str = _PROMPTS["system_template"]
INSTANCE_TEMPLATE: str = _PROMPTS["instance_template"]
FORMAT_ERROR_TEMPLATE: str = _PROMPTS["format_error_template"]

_OBSERVATION = _PROMPTS["observation"]
_MAX_CHARS: int = int(_OBSERVATION.get("max_chars", 10000))
_HEAD_CHARS: int = int(_OBSERVATION.get("head_chars", 5000))
_TAIL_CHARS: int = int(_OBSERVATION.get("tail_chars", 5000))
_TOO_LONG_HINT: str = _OBSERVATION.get("too_long_hint", "Output truncated.")


def render_instance(task: str) -> str:
    """Render the instance template. ``{{task}}`` is its only placeholder."""
    return INSTANCE_TEMPLATE.replace("{{task}}", task)


def format_error_message(error: str) -> str:
    return FORMAT_ERROR_TEMPLATE.replace("{{error}}", error)


def truncate_observation(output: str) -> str:
    """Apply mini-swe-agent's head/tail truncation to long command output."""
    if len(output) <= _MAX_CHARS:
        return output
    elided = len(output) - _HEAD_CHARS - _TAIL_CHARS
    return (
        f"{_TOO_LONG_HINT}\n\n"
        f"---- HEAD ({_HEAD_CHARS} chars) ----\n"
        f"{output[:_HEAD_CHARS]}\n"
        f"---- {elided} chars elided ----\n"
        f"---- TAIL ({_TAIL_CHARS} chars) ----\n"
        f"{output[-_TAIL_CHARS:]}"
    )
