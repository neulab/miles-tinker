"""Reporting placeholders are explicit and never modify training tensors."""

import torch

from miles.backends.training_utils.tinker_numerics import finite_per_datum_reports
from miles.tinker.server.encoding import forward_backward_metrics


def test_nonfinite_reports_preserve_original_values_and_expose_counts():
    original = {
        "sample_index": 7,
        "loss": torch.tensor(float("nan")),
        "logprobs": torch.tensor([-2.0, float("inf"), float("nan"), float("-inf")]),
    }
    reports, diagnostics = finite_per_datum_reports([original])
    assert diagnostics == [
        {
            "sample_index": 7,
            "nonfinite_loss": 1,
            "nonfinite_logprobs": 3,
            "first_bad_positions": [1, 2, 3],
        }
    ]
    assert torch.isnan(original["loss"]) and torch.isnan(original["logprobs"][2])
    torch.testing.assert_close(reports[0]["logprobs"], torch.tensor([-2.0, 0.0, 0.0, 0.0]))
    assert reports[0]["loss"].item() == 0.0
    metrics = forward_backward_metrics(reports)
    assert metrics["nonfinite_loss:sum"] == 1 and metrics["nonfinite_logprobs:sum"] == 3


def test_finite_reports_are_unchanged():
    output = {"sample_index": 0, "loss": torch.tensor(1.5), "logprobs": torch.tensor([-1.0])}
    reports, diagnostics = finite_per_datum_reports([output])
    assert reports[0] is output and not diagnostics
    assert set(forward_backward_metrics(reports)) == {"loss:sum", "num_tokens:sum", "num_datums:sum"}
