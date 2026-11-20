"""--tinker-custom-loss NAME=module:function: a per-datum loss imported once per process, reduced like the
built-in losses (raw sum over tokens, DP padding rows inert), its per-datum metrics summed into the
forward_backward result as ``<name>:sum``. The logprob scorer is stubbed: ``logits`` doubles as the
per-token trainer logprobs."""

import importlib
import sys
import types
from argparse import Namespace

import pytest
import torch

LOSSES = "miles.backends.training_utils.loss_hub.tinker_losses"
HERE = __name__


# ---- custom losses under test (importable as tests.fast.tinker.test_tinker_custom_loss:<name>) ----


def is_like(*, logprobs, sampling_logprobs, advantages, loss_mask, config):
    """The built-in importance_sampling loss, as a custom loss, with a scale from the config."""
    per_token = -torch.exp(logprobs - sampling_logprobs) * advantages * config.get("scale", 1.0)
    return per_token, {"tokens": loss_mask.sum(), "positive": float((advantages > 0).sum())}


def with_penalty(*, logprobs, sampling_logprobs, advantages, loss_mask, config):
    """A term that does not vanish with advantage 0 (the pad row must still be inert)."""
    return (logprobs - sampling_logprobs) ** 2, {}


def wrong_shape(**kwargs):
    return kwargs["logprobs"].sum(), {}


def not_a_tuple(**kwargs):
    return kwargs["logprobs"]


def bad_metrics(**kwargs):
    return kwargs["logprobs"] * 0, {"vector": torch.ones(3)}


NOT_CALLABLE = 3


@pytest.fixture
def losses(monkeypatch):
    def get_log_probs_and_entropy(logits, *, args, unconcat_tokens, total_lengths, response_lengths, **kwargs):
        return {"log_probs": list(logits.split(response_lengths))}

    stub = types.ModuleType("miles.backends.training_utils.loss_hub.logit_processors")
    stub.get_log_probs_and_entropy = get_log_probs_and_entropy
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    monkeypatch.delitem(sys.modules, LOSSES, raising=False)
    module = importlib.import_module(LOSSES)
    yield module
    sys.modules.pop(LOSSES, None)


def _batch(logprobs, sampling, advantages, masks=None, config=None):
    logits = torch.tensor([lp for row in logprobs for lp in row], dtype=torch.float64, requires_grad=True)
    batch = {
        "unconcat_tokens": None,
        "total_lengths": [len(row) + 1 for row in logprobs],
        "response_lengths": [len(row) for row in logprobs],
        "loss_masks": [torch.tensor(m) for m in (masks or [[1] * len(row) for row in logprobs])],
        "sample_indices": list(range(len(logprobs))),
        "rollout_log_probs": sampling,
        "advantages": advantages,
        "loss_weights": [[0.0] * len(row) for row in logprobs],
    }
    if config is not None:
        batch["loss_fn_config"] = config
    return batch, logits


def _args(*specs):
    return Namespace(tinker_custom_loss=list(specs))


def test_custom_loss_matches_the_builtin_it_reimplements(losses):
    rows = ([[-1.0, -2.0], [-0.5]], [[-1.2, -1.9], [-0.7]], [[0.5, 0.5], [-1.0]])
    fn = losses.get_tinker_loss_function(_args(f"mine={HERE}:is_like"), "mine")
    batch, logits = _batch(*rows, config={"scale": 1.0})
    losses.start_per_datum_outputs()
    loss, _ = fn(Namespace(), batch, logits, None)
    outputs = losses.drain_per_datum_outputs()
    loss.backward()

    ref_batch, ref_logits = _batch(*rows)
    ref_loss, _ = losses.importance_sampling_loss_function(Namespace(), ref_batch, ref_logits, None)
    ref_loss.backward()
    assert loss.item() == pytest.approx(ref_loss.item())
    torch.testing.assert_close(logits.grad, ref_logits.grad)
    assert [o["metrics"] for o in outputs] == [{"positive": 2.0, "tokens": 2.0}, {"positive": 0.0, "tokens": 1.0}]
    assert outputs[0]["loss"].item() == pytest.approx(
        -(torch.exp(torch.tensor([0.2, -0.1], dtype=torch.float64)) * 0.5).sum().item()
    )


def test_config_reaches_the_function(losses):
    fn = losses.get_tinker_loss_function(_args(f"mine={HERE}:is_like"), "mine")
    one, l1 = _batch([[-1.0]], [[-1.0]], [[1.0]], config={"scale": 1.0})
    three, l3 = _batch([[-1.0]], [[-1.0]], [[1.0]], config={"scale": 3.0})
    assert fn(Namespace(), three, l3, None)[0].item() == pytest.approx(3 * fn(Namespace(), one, l1, None)[0].item())


def test_pad_rows_are_inert(losses):
    fn = losses.get_tinker_loss_function(_args(f"pen={HERE}:with_penalty"), "pen")
    batch, logits = _batch([[-1.0], [-9.0]], [[-1.5], [0.0]], [[0.7], [0.0]], masks=[[1], [0]])
    loss, _ = fn(Namespace(), batch, logits, None)
    loss.backward()
    assert loss.item() == pytest.approx(0.25) and logits.grad[1].item() == 0.0


def test_metrics_are_summed_into_the_result():
    from miles.tinker.server.encoding import forward_backward_metrics

    plain = [{"loss": 1.0, "logprobs": [-1.0, -2.0]}]
    assert set(forward_backward_metrics(plain)) == {"loss:sum", "num_tokens:sum", "num_datums:sum"}
    outputs = [
        {"loss": 1.0, "logprobs": [-1.0, -2.0], "metrics": {"masked": 1.0, "loss": 99.0}},
        {"loss": 0.5, "logprobs": [-1.0], "metrics": {"masked": 2.0}},
    ]
    metrics = forward_backward_metrics(outputs)
    assert metrics["masked:sum"] == 3.0
    assert metrics["loss:sum"] == 1.5, "a custom metric never shadows the request's own"


def test_bad_specs_fail_fast(losses):
    with pytest.raises(ValueError, match="expected NAME=module.path:function"):
        losses.parse_custom_loss_specs(["mine"])
    with pytest.raises(ValueError, match="built-in"):
        losses.parse_custom_loss_specs([f"ppo={HERE}:is_like"])
    with pytest.raises(ValueError, match="given twice"):
        losses.parse_custom_loss_specs([f"a={HERE}:is_like", f"a={HERE}:with_penalty"])
    with pytest.raises(ValueError, match="cannot import"):
        losses.load_custom_loss("no_such_module_xyz:fn")
    with pytest.raises(ValueError, match="no callable"):
        losses.load_custom_loss(f"{HERE}:missing")
    with pytest.raises(ValueError, match="no callable"):
        losses.load_custom_loss(f"{HERE}:NOT_CALLABLE")
    with pytest.raises(ValueError, match="unknown Tinker loss_fn 'mine'"):
        losses.get_tinker_loss_function(_args(), "mine")
    assert losses.get_tinker_loss_function(_args(), "ppo") is losses.ppo_loss_function


@pytest.mark.parametrize(
    ("name", "match"),
    [("wrong_shape", "shape"), ("not_a_tuple", r"\(per_token_loss, metrics\)"), ("bad_metrics", "metrics must be")],
)
def test_bad_returns_raise(losses, name, match):
    fn = losses.get_tinker_loss_function(_args(f"x={HERE}:{name}"), "x")
    batch, logits = _batch([[-1.0, -2.0]], [[-1.0, -2.0]], [[1.0, 1.0]])
    with pytest.raises(TypeError, match=match):
        fn(Namespace(), batch, logits, None)


def test_imported_once(losses, monkeypatch):
    calls = []
    real = importlib.import_module
    monkeypatch.setattr(importlib, "import_module", lambda name, *a: calls.append(name) or real(name, *a))
    for _ in range(3):
        losses.load_custom_loss(f"{HERE}:is_like")
    assert calls.count(HERE) == 1
