"""Tinker losses are exact sums over datums; the runtime's DP padding rows
(all-zero loss mask) contribute nothing, even under losses whose terms do not
go through weights/advantages. The logprob scorer is stubbed."""

import importlib
import sys
import types
from argparse import Namespace

import pytest
import torch

LOSSES = "miles.backends.training_utils.loss_hub.tinker_losses"


@pytest.fixture
def losses(monkeypatch):
    """Import the losses against a scorer that returns per-token logprobs from the batch itself."""

    def get_log_probs_and_entropy(logits, *, args, unconcat_tokens, total_lengths, response_lengths, **kwargs):
        return {"log_probs": list(logits.split(response_lengths))}

    stub = types.ModuleType("miles.backends.training_utils.loss_hub.logit_processors")
    stub.get_log_probs_and_entropy = get_log_probs_and_entropy
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    monkeypatch.delitem(sys.modules, LOSSES, raising=False)
    module = importlib.import_module(LOSSES)
    yield module
    sys.modules.pop(LOSSES, None)


def _batch(logprobs: list[list[float]], masks: list[list[int]], **inputs) -> tuple[dict, torch.Tensor]:
    """``logits`` doubles as the per-token logprobs the stubbed scorer returns."""
    logits = torch.tensor([lp for row in logprobs for lp in row], requires_grad=True)
    batch = {
        "unconcat_tokens": None,
        "total_lengths": [len(row) + 1 for row in logprobs],
        "response_lengths": [len(row) for row in logprobs],
        "loss_masks": [torch.tensor(mask) for mask in masks],
        "sample_indices": list(range(len(logprobs))),
        **inputs,
    }
    return batch, logits


def test_cross_entropy_is_the_weighted_sum_of_datums(losses):
    batch, logits = _batch([[-1.0, -2.0], [-3.0]], [[1, 1], [1]], loss_weights=[[1.0, 0.5], [2.0]])
    loss, _ = losses.cross_entropy_loss_function(Namespace(), batch, logits, None)
    assert loss.item() == pytest.approx(1.0 + 1.0 + 6.0)


def test_masked_out_datums_contribute_nothing_under_dro(losses):
    """DRO's divergence penalty does not vanish with advantage 0; the mask must remove the pad row."""
    real = _batch([[-1.0]], [[1]], rollout_log_probs=[[-1.5]], advantages=[[0.7]])
    padded = _batch(
        [[-1.0], [-9.0]], [[1], [0]], rollout_log_probs=[[-1.5], [0.0]], advantages=[[0.7], [0.0]]
    )
    losses.start_per_datum_outputs()
    loss_real, _ = losses.dro_loss_function(Namespace(), *real, None)
    loss_padded, _ = losses.dro_loss_function(Namespace(), *padded, None)
    outputs = losses.drain_per_datum_outputs()

    assert loss_padded.item() == pytest.approx(loss_real.item())
    loss_padded.backward()
    assert padded[1].grad[1].item() == 0.0, "no gradient flows into the pad row"
    assert [o["sample_index"] for o in outputs] == [0, 0, 1] and outputs[2]["loss"].item() == 0.0


def test_pad_rows_are_inert_under_every_loss(losses):
    for name, fn in losses.TINKER_LOSS_FUNCTIONS.items():
        batch, logits = _batch(
            [[-1.0, -2.0], [-9.0]],
            [[1, 1], [0]],
            loss_weights=[[1.0, 1.0], [0.0]],
            rollout_log_probs=[[-1.2, -2.2], [0.0]],
            advantages=[[0.5, 0.5], [0.0]],
        )
        loss, _ = fn(Namespace(), batch, logits, None)
        loss.backward()
        assert logits.grad[2].item() == 0.0, f"{name}: the pad row must not reach the weights"
