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
    stub.get_values = lambda logits, **kwargs: {"values": list(logits.split(kwargs["response_lengths"]))}
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
    padded = _batch([[-1.0], [-9.0]], [[1], [0]], rollout_log_probs=[[-1.5], [0.0]], advantages=[[0.7], [0.0]])
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


@pytest.mark.parametrize("loss_fn", ["ppo", "importance_sampling"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("advantage", [0.0, 1.0, -1.0])
@pytest.mark.parametrize("current,old", [(-1.0, -100.0), (-100.0, -1.0)])
def test_actor_extreme_log_ratios_have_finite_loss_and_gradients(losses, loss_fn, dtype, advantage, current, old):
    batch, logits = _batch([[current]], [[1]], rollout_log_probs=[[old]], advantages=[[advantage]])
    logits = logits.detach().to(dtype).requires_grad_()
    loss, metrics = losses.TINKER_LOSS_FUNCTIONS[loss_fn](Namespace(), batch, logits, None)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["loss"])
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.item() == 0.0, "ratios beyond the numerical guard are saturated"
    if advantage == 0.0:
        assert loss.item() == 0.0


@pytest.mark.parametrize("loss_fn", ["ppo", "importance_sampling"])
def test_actor_preserves_normal_loss_and_gradients(losses, loss_fn):
    batch, logits = _batch(
        [[-1.1, -1.0, -0.9, -1.1, -1.0, -0.9]],
        [[1] * 6],
        rollout_log_probs=[[-1.0] * 6],
        advantages=[[1.0, 2.0, 3.0, -1.0, -2.0, -3.0]],
        loss_fn_config={"clip_low_threshold": 0.95, "clip_high_threshold": 1.05},
    )
    loss, _ = losses.TINKER_LOSS_FUNCTIONS[loss_fn](Namespace(), batch, logits, None)
    reference = logits.detach().double().requires_grad_()
    ratio = (reference + 1.0).exp()
    advantages = torch.tensor(batch["advantages"][0], dtype=torch.float64)
    objective = ratio * advantages
    if loss_fn == "ppo":
        objective = torch.minimum(objective, ratio.clamp(0.95, 1.05) * advantages)
    expected = -objective.sum()
    torch.testing.assert_close(loss, expected.float())
    loss.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, reference.grad.float())


def test_critic_ppo_clipping_weights_padding_and_fixed_targets(losses):
    old = torch.zeros(4, requires_grad=True)
    returns = torch.ones(4, requires_grad=True)
    weights = torch.tensor([1.0, 2.0, 0.5, 0.0], requires_grad=True)
    batch, logits = _batch(
        [[0.8, -0.8, 0.1, 3.0], [9.0]],
        [[1, 1, 1, 1], [0]],
        model_role="critic",
        values=[old, [0.0]],
        returns=[returns, [1.0]],
        loss_weights=[weights, [1.0]],
        loss_fn_config={"value_clip": 0.2},
    )
    losses.start_per_datum_outputs()
    loss, metrics = losses.TINKER_LOSS_FUNCTIONS["ppo"](Namespace(), batch, logits, None)
    outputs = losses.drain_per_datum_outputs()
    # Positive overshoot uses the clipped branch; negative overshoot uses the
    # unclipped branch. Client weights mask and normalize without server averaging.
    assert loss.item() == pytest.approx(0.64 + 2 * 3.24 + 0.5 * 0.81)
    assert metrics["loss"].item() == loss.item()
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([0.0, -7.2, -0.9, 0.0, 0.0]))
    assert old.grad is returns.grad is weights.grad is None
    assert [output["sample_index"] for output in outputs] == [0, 1]
    assert outputs[0]["loss"].item() == loss.item() and outputs[1]["loss"].item() == 0
    torch.testing.assert_close(outputs[0]["values"], logits[:4].detach())
    assert all("logprobs" not in output and not output["values"].requires_grad for output in outputs)


@pytest.mark.parametrize("clip", [0.0, 0.2, 1.0])
def test_sdk_critic_matches_native_value_loss_with_the_same_reduction(losses, monkeypatch, clip):
    from miles.backends.training_utils.loss_hub import losses as native

    monkeypatch.setattr(native, "get_values", losses.get_values)
    batch, logits = _batch(
        [[0.8, -0.8], [0.1]],
        [[1, 1], [1]],
        model_role="critic",
        values=[torch.tensor([0.0, 0.0]), torch.tensor([0.0])],
        returns=[torch.tensor([1.0, 1.0]), torch.tensor([1.0])],
        loss_weights=[[0.5, 2.0], [0.0]],
        loss_fn_config={"value_clip": clip},
    )
    args = Namespace(value_clip=clip)
    sdk_loss, _ = losses.ppo_loss_function(args, batch, logits, None)
    native_loss, _ = native.value_loss_function(args, batch, logits, lambda x: (x * torch.tensor([0.5, 2, 0])).sum())
    torch.testing.assert_close(sdk_loss, native_loss)
    sdk_grad = torch.autograd.grad(sdk_loss, logits, retain_graph=True)[0]
    native_grad = torch.autograd.grad(native_loss, logits)[0]
    torch.testing.assert_close(sdk_grad, native_grad)
