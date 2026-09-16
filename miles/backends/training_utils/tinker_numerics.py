"""Inspect CPU per-datum results before returning them through the Tinker API."""

import torch


def finite_per_datum_reports(outputs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Make reporting tensors serializable and describe every invalid datum.

    These are detached reporting copies, never the loss or gradient tensors.
    The caller MUST veto the entire optimizer window if diagnostics is nonempty.
    """
    reports, diagnostics = [], []
    for output in outputs:
        loss = output["loss"]
        logprobs = output["logprobs"]
        bad_loss = int(not torch.isfinite(loss).item())
        bad_positions = (~torch.isfinite(logprobs)).nonzero().flatten()
        bad_logprobs = bad_positions.numel()
        if bad_loss or bad_logprobs:
            diagnostics.append(
                {
                    "sample_index": output["sample_index"],
                    "nonfinite_loss": bad_loss,
                    "nonfinite_logprobs": bad_logprobs,
                    "first_bad_positions": bad_positions[:8].tolist(),
                }
            )
            reports.append(
                {
                    **output,
                    "loss": torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0),
                    "logprobs": torch.nan_to_num(logprobs, nan=0.0, posinf=0.0, neginf=0.0),
                    "nonfinite_loss": bad_loss,
                    "nonfinite_logprobs": bad_logprobs,
                }
            )
        else:
            reports.append(output)
    return reports, diagnostics
