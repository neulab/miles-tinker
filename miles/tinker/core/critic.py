"""Validate role-specific PPO inputs without changing the request."""

import math

from miles.tinker.core.types import ModelRole, UserInputError


def validate_model_pass(role: ModelRole, kind: str, payload: dict) -> None:
    loss_fn = payload["loss_fn"]
    if loss_fn in {"ppo_critic", "ppo_value"}:
        raise UserInputError(f"unsupported loss_fn {loss_fn!r}; use 'ppo' for critic models")
    if role != "critic":
        if any({"old_values", "values", "returns"} & row.keys() for row in payload["rows"]):
            raise UserInputError("value regression inputs require a critic model")
        if any("loss_mask" in row for row in payload["rows"]):
            raise UserInputError("loss_mask is a critic input; mask policy loss using zero advantages or weights")
        return
    if loss_fn != "ppo":
        raise UserInputError("critic models require loss_fn='ppo'")
    config = payload.get("loss_fn_config") or {}
    if set(config) - {"value_clip"}:
        raise UserInputError("critic ppo only accepts value_clip in loss_fn_config")
    clip = config.get("value_clip", 0.2)
    if not isinstance(clip, (int, float)) or not math.isfinite(clip) or clip < 0:
        raise UserInputError("value_clip must be finite and nonnegative")
    for index, row in enumerate(payload["rows"]):
        if "old_values" in row or "loss_mask" in row:
            raise UserInputError("critic ppo uses values and weights instead of old_values/loss_mask")
        if kind == "forward_backward":
            missing = {"values", "returns", "weights"} - row.keys()
            if missing:
                raise UserInputError(f"critic datum {index} missing {sorted(missing)}")
        for key in ("values", "returns", "weights"):
            if key not in row:
                continue
            values = row[key]
            if len(values) != row["target_len"] or not all(math.isfinite(x) for x in values):
                raise UserInputError(f"critic datum {index}: {key} must be finite and match target length")
            if key == "weights" and any(x < 0 for x in values):
                raise UserInputError("critic weights must be nonnegative")
