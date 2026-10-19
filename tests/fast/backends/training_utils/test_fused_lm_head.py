"""CPU tests for the fused chunked lm head (miles/backends/training_utils/loss_hub/fused_lm_head.py).

On CPU the chunk logits are ``F.linear(h.float(), W.float())`` (fp32 GEMM on the
bf16 operands), the same arithmetic as the CUDA ``torch.mm(..., out_dtype=fp32)``
path, so the fp32 reference is exact for the forward. The backward casts the
logit gradient to bf16 before the two gradient GEMMs (like the standard head),
so gradients are checked against a rounding-aware reference (fp32 autograd
gradient rounded to bf16 once, then fp32 GEMMs) at fp32 tolerances and against
the pure fp32 autograd gradient at bf16 tolerances.
"""

from __future__ import annotations

import threading
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args, make_parallel_state

from miles.backends.training_utils.loss_hub import fused_lm_head, math_utils
from miles.backends.training_utils.loss_hub.fused_lm_head import (
    HeadState,
    activate,
    chunk_size,
    install,
    is_active,
    maybe_activate,
    target_logprobs,
    vocab_parallel_fused_linear_logprobs,
)
from miles.backends.training_utils.loss_hub.logit_processors import get_log_probs_and_entropy

T, H, V = 37, 64, 131
CHUNK_SIZES = [1, 5, 64]
BF16_RTOL = 2**-7  # one bf16 ulp of slack for summation-order differences


# ---------------------------------------------------------------------------
# Inputs and references
# ---------------------------------------------------------------------------


def _inputs(seed: int = 0, tokens: int = T, vocab: int = V, hidden: int = H):
    g = torch.Generator().manual_seed(seed)
    h = (torch.randn(tokens, hidden, generator=g) * 2).to(torch.bfloat16)
    w = (torch.randn(vocab, hidden, generator=g) * 0.2).to(torch.bfloat16)
    labels = torch.randint(0, vocab, (tokens,), generator=g)
    upstream = torch.randn(tokens, generator=g)
    return h, w, labels, upstream


def _reference_logprobs(h, w, labels, temperature=1.0):
    logits = F.linear(h.float(), w.float()) / temperature
    return torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def _reference_grads(h, w, labels, upstream, temperature=1.0):
    """fp32 autograd through the fp32 reference, plus the rounding-aware bf16-GEMM expectations."""
    h32 = h.float().requires_grad_(True)
    w32 = w.float().requires_grad_(True)
    logits = F.linear(h32, w32) / temperature
    logits.retain_grad()
    logprobs = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    (logprobs * upstream).sum().backward()
    g = logits.grad / temperature  # dL/d(raw logits)
    g_bf16 = g.to(torch.bfloat16).float()
    return {
        "hidden_fp32": h32.grad,
        "weight_fp32": w32.grad,
        "hidden_rounded": (g_bf16 @ w.float()).to(torch.bfloat16),
        "weight_rounded": g_bf16.t() @ h.float(),
        "logits_grad_bf16": g_bf16,
    }


def _shard_hidden_partials(ref, w, shards):
    """Per-TP-rank dHidden partials as the fused backward emits them (bf16 GEMM output per shard)."""
    return [(ref["logits_grad_bf16"][:, s] @ w[s].float()).to(torch.bfloat16).float() for s in shards]


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", CHUNK_SIZES)
def test_forward_matches_fp32_reference(chunk):
    h, w, labels, _ = _inputs()
    out = vocab_parallel_fused_linear_logprobs(h, w, labels, chunk_size=chunk)
    assert out.dtype == torch.float32 and out.shape == (T,)
    torch.testing.assert_close(out, _reference_logprobs(h, w, labels), rtol=0, atol=1e-6)


def test_forward_temperature_scales_in_fp32():
    h, w, labels, _ = _inputs(seed=1)
    out = vocab_parallel_fused_linear_logprobs(h, w, labels, temperature=0.7, chunk_size=5)
    # scaled logits are larger, so allow two fp32 ulps of summation-order slack
    torch.testing.assert_close(out, _reference_logprobs(h, w, labels, temperature=0.7), rtol=0, atol=4e-6)


def test_forward_keeps_label_shape_and_handles_empty_rows():
    h, w, labels, _ = _inputs()
    out = vocab_parallel_fused_linear_logprobs(h[:6].reshape(2, 3, H), w, labels[:6].reshape(2, 3), chunk_size=4)
    assert out.shape == (2, 3)
    empty = vocab_parallel_fused_linear_logprobs(h[:0], w, labels[:0], chunk_size=4)
    assert empty.shape == (0,) and empty.dtype == torch.float32


def test_argument_validation():
    h, w, labels, _ = _inputs()
    with pytest.raises(ValueError, match="leading dimensions"):
        vocab_parallel_fused_linear_logprobs(h, w, labels[:-1])
    with pytest.raises(ValueError, match="hidden size"):
        vocab_parallel_fused_linear_logprobs(h[:, :-1], w, labels)
    with pytest.raises(ValueError, match="chunk_size"):
        vocab_parallel_fused_linear_logprobs(h, w, labels, chunk_size=0)
    with pytest.raises(ValueError, match="temperature"):
        vocab_parallel_fused_linear_logprobs(h, w, labels, temperature=0.0)
    with pytest.raises(TypeError, match="torch.long"):
        vocab_parallel_fused_linear_logprobs(h, w, labels.int())


# ---------------------------------------------------------------------------
# Backward (single rank)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", CHUNK_SIZES)
def test_backward_matches_reference(chunk):
    h, w, labels, upstream = _inputs()
    ref = _reference_grads(h, w, labels, upstream)
    h_fused = h.clone().requires_grad_(True)
    w_fused = w.clone().requires_grad_(True)

    out = vocab_parallel_fused_linear_logprobs(h_fused, w_fused, labels, chunk_size=chunk)
    (out * upstream).sum().backward()

    assert h_fused.grad.dtype == torch.bfloat16 and w_fused.grad.dtype == torch.bfloat16
    # exact up to summation order against the rounding-aware reference ...
    torch.testing.assert_close(h_fused.grad, ref["hidden_rounded"], rtol=BF16_RTOL, atol=1e-6)
    torch.testing.assert_close(w_fused.grad.float(), ref["weight_rounded"], rtol=BF16_RTOL, atol=1e-6)
    # ... and within bf16 precision of the pure fp32 autograd gradient
    torch.testing.assert_close(h_fused.grad.float(), ref["hidden_fp32"], rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(w_fused.grad.float(), ref["weight_fp32"], rtol=2e-2, atol=2e-2)


def test_backward_with_temperature():
    h, w, labels, upstream = _inputs(seed=3)
    ref = _reference_grads(h, w, labels, upstream, temperature=0.7)
    h_fused = h.clone().requires_grad_(True)
    w_fused = w.clone().requires_grad_(True)
    out = vocab_parallel_fused_linear_logprobs(h_fused, w_fused, labels, temperature=0.7, chunk_size=5)
    (out * upstream).sum().backward()
    torch.testing.assert_close(h_fused.grad, ref["hidden_rounded"], rtol=BF16_RTOL, atol=1e-6)
    torch.testing.assert_close(w_fused.grad.float(), ref["weight_rounded"], rtol=BF16_RTOL, atol=1e-6)


def test_weight_grad_added_to_main_grad_when_ddp_managed():
    h, w, labels, upstream = _inputs()
    ref = _reference_grads(h, w, labels, upstream)
    weight = torch.nn.Parameter(w.clone())
    weight.main_grad = torch.full_like(w, 0.5, dtype=torch.float32)  # pre-accumulated by an earlier unit
    weight.grad_added_to_main_grad = False

    out = vocab_parallel_fused_linear_logprobs(h, weight, labels, chunk_size=5)
    (out * upstream).sum().backward()

    assert weight.grad is None
    assert weight.grad_added_to_main_grad is True
    torch.testing.assert_close(weight.main_grad - 0.5, ref["weight_rounded"], rtol=1e-5, atol=1e-6)


def test_weight_grad_returned_when_ddp_does_not_track_it():
    h, w, labels, upstream = _inputs()
    ref = _reference_grads(h, w, labels, upstream)
    weight = torch.nn.Parameter(w.clone())
    weight.main_grad = torch.zeros_like(w, dtype=torch.float32)  # no grad_added_to_main_grad attribute

    out = vocab_parallel_fused_linear_logprobs(h, weight, labels, chunk_size=5)
    (out * upstream).sum().backward()

    assert weight.grad is not None and weight.grad.dtype == torch.bfloat16
    assert torch.count_nonzero(weight.main_grad) == 0
    torch.testing.assert_close(weight.grad.float(), ref["weight_rounded"], rtol=BF16_RTOL, atol=1e-6)


def test_frozen_weight_gets_no_grad():
    h, w, labels, upstream = _inputs()
    h_fused = h.clone().requires_grad_(True)
    out = vocab_parallel_fused_linear_logprobs(h_fused, w, labels, chunk_size=5)
    (out * upstream).sum().backward()
    assert h_fused.grad is not None and w.grad is None


# ---------------------------------------------------------------------------
# Fake 2-shard tensor parallelism (threads; collectives over a barrier)
# ---------------------------------------------------------------------------


class _FakeTensorParallel:
    """Two 'ranks' on two threads; all_reduce / all_gather rendezvous on a barrier."""

    def __init__(self, world_size: int):
        self.world_size = world_size
        self.local = threading.local()
        self.barrier = threading.Barrier(world_size)
        self.slots: list[torch.Tensor | None] = [None] * world_size

    def rank(self) -> int:
        return self.local.rank

    def _exchange(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        self.slots[self.rank()] = tensor.detach().clone()
        self.barrier.wait()
        gathered = list(self.slots)
        self.barrier.wait()
        return gathered

    def all_reduce_(self, tensor: torch.Tensor, op) -> None:
        parts = torch.stack(self._exchange(tensor))
        reduced = parts.amax(dim=0) if op == torch.distributed.ReduceOp.MAX else parts.sum(dim=0)
        tensor.copy_(reduced)

    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat(self._exchange(tensor), dim=0)


class _FakeDist:
    ReduceOp = torch.distributed.ReduceOp

    @staticmethod
    def get_world_size(group):
        return group.world_size

    @staticmethod
    def get_rank(group):
        return group.rank()

    @staticmethod
    def all_reduce(tensor, op, group):
        group.all_reduce_(tensor, op)


class _FakeGatherFromSP(torch.autograd.Function):
    """forward: all-gather along the sequence dim; backward: reduce-scatter (Megatron's SP mapping)."""

    @staticmethod
    def forward(ctx, local, group):
        ctx.group = group
        ctx.rows = local.shape[0]
        return group.all_gather(local)

    @staticmethod
    def backward(ctx, grad):
        summed = grad.clone()
        ctx.group.all_reduce_(summed, torch.distributed.ReduceOp.SUM)
        r = ctx.group.rank()
        return summed[r * ctx.rows : (r + 1) * ctx.rows], None


def _fake_gather_from_sp_region(input_, group=None):
    return _FakeGatherFromSP.apply(input_, group)


def _fake_copy_to_tp_region(input_, group=None):
    return input_


def _run_ranks(group: _FakeTensorParallel, fn):
    results: list = [None] * group.world_size
    errors: list = []

    def run(rank):
        group.local.rank = rank
        try:
            results[rank] = fn(rank)
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)
            group.barrier.abort()

    threads = [threading.Thread(target=run, args=(r,)) for r in range(group.world_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return results


def _fake_output_layer(weight_shard, group, *, sequence_parallel):
    def original_forward(input_, weight=None, runtime_gather_output=None, **kwargs):
        return ("original", input_, weight, runtime_gather_output, kwargs), None

    layer = SimpleNamespace(
        weight=weight_shard,
        bias=None,
        gather_output=False,
        disable_grad_reduce=False,
        sequence_parallel=sequence_parallel,
        tp_group=group,
        config=SimpleNamespace(defer_embedding_wgrad_compute=False),
        forward=original_forward,
    )
    model = SimpleNamespace(
        post_process=True, output_layer=layer, config=SimpleNamespace(use_mup=False), mtp_process=False
    )
    return model, layer


def _install_fake(model):
    return install(
        [model], copy_to_tp_region=_fake_copy_to_tp_region, gather_from_sp_region=_fake_gather_from_sp_region
    )


@pytest.mark.parametrize("chunk", [1, 5, 64])
def test_two_shard_tensor_parallel(monkeypatch, chunk):
    monkeypatch.setattr(fused_lm_head, "dist", _FakeDist)
    vocab = 132
    h, w, labels, upstream = _inputs(seed=5, vocab=vocab)
    ref = _reference_grads(h, w, labels, upstream)
    expected = _reference_logprobs(h, w, labels)
    group = _FakeTensorParallel(2)
    shard = vocab // 2

    def rank_fn(rank):
        w_r = w[rank * shard : (rank + 1) * shard].clone().requires_grad_(True)
        h_r = h.clone().requires_grad_(True)
        out = vocab_parallel_fused_linear_logprobs(h_r, w_r, labels, tp_group=group, chunk_size=chunk)
        (out * upstream).sum().backward()
        return out, h_r.grad, w_r.grad

    results = _run_ranks(group, rank_fn)
    for out, _, _ in results:
        torch.testing.assert_close(out, expected, rtol=0, atol=1e-6)
    # dHidden: per-rank bf16 partials (the TP copy's backward all-reduces them outside); dW: per-rank shard
    shards = [slice(0, shard), slice(shard, vocab)]
    partials = _shard_hidden_partials(ref, w, shards)
    for rank, (_, h_grad, _) in enumerate(results):
        torch.testing.assert_close(
            h_grad.float(), partials[rank], rtol=BF16_RTOL, atol=BF16_RTOL * partials[rank].abs().max()
        )
    hidden_sum = results[0][1].float() + results[1][1].float()
    torch.testing.assert_close(hidden_sum, ref["hidden_fp32"], rtol=2e-2, atol=2e-2)
    for rank, (_, _, w_grad) in enumerate(results):
        torch.testing.assert_close(
            w_grad.float(), ref["weight_rounded"][rank * shard : (rank + 1) * shard], rtol=BF16_RTOL, atol=1e-6
        )


def test_two_shard_tensor_parallel_with_sequence_parallel(monkeypatch):
    """TP2 fallback: each rank owns half the sequence; the wrapper all-gathers, backward reduce-scatters."""
    monkeypatch.setattr(fused_lm_head, "dist", _FakeDist)
    vocab, tokens = 132, 36
    h, w, labels, upstream = _inputs(seed=6, tokens=tokens, vocab=vocab)
    ref = _reference_grads(h, w, labels, upstream)
    expected = _reference_logprobs(h, w, labels)
    group = _FakeTensorParallel(2)
    shard, half = vocab // 2, tokens // 2

    def rank_fn(rank):
        w_r = torch.nn.Parameter(w[rank * shard : (rank + 1) * shard].clone())
        model, layer = _fake_output_layer(w_r, group, sequence_parallel=True)
        state = _install_fake(model)
        h_half = h[rank * half : (rank + 1) * half].clone().unsqueeze(1).requires_grad_(True)  # [S/2, 1, H]
        with activate(state):
            mapped, bias = layer.forward(h_half)
            assert bias is None and mapped.shape == (tokens, 1, H)
            assert state.active_weight is w_r and state.fused_forward_calls == 1
            out = target_logprobs(mapped.squeeze(1), labels, group, temperature=1.0, chunk_size=5)
        (out * upstream).sum().backward()
        return out, h_half.grad.squeeze(1), w_r.grad

    results = _run_ranks(group, rank_fn)
    partials = _shard_hidden_partials(ref, w, [slice(0, shard), slice(shard, vocab)])
    reduced = (partials[0] + partials[1]).to(torch.bfloat16)  # the reduce-scatter sums bf16 partials
    slack = BF16_RTOL * max(p.abs().max() for p in partials)
    for rank, (out, h_grad, w_grad) in enumerate(results):
        torch.testing.assert_close(out, expected, rtol=0, atol=1e-6)
        torch.testing.assert_close(h_grad, reduced[rank * half : (rank + 1) * half], rtol=BF16_RTOL, atol=slack)
        torch.testing.assert_close(
            w_grad.float(), ref["weight_rounded"][rank * shard : (rank + 1) * shard], rtol=BF16_RTOL, atol=1e-6
        )


# ---------------------------------------------------------------------------
# install / wrapper / activation
# ---------------------------------------------------------------------------


def test_install_returns_none_without_post_process_head():
    assert _install_fake(SimpleNamespace(post_process=False)) is None
    assert (
        install([], copy_to_tp_region=_fake_copy_to_tp_region, gather_from_sp_region=_fake_gather_from_sp_region)
        is None
    )


def test_install_unwraps_ddp_and_float16_wrappers_and_is_idempotent():
    model, layer = _fake_output_layer(
        torch.nn.Parameter(torch.zeros(4, H, dtype=torch.bfloat16)), None, sequence_parallel=False
    )
    wrapped = SimpleNamespace(module=SimpleNamespace(module=model))
    state = install(
        [wrapped], copy_to_tp_region=_fake_copy_to_tp_region, gather_from_sp_region=_fake_gather_from_sp_region
    )
    assert isinstance(state, HeadState) and state.output_layer is layer
    assert _install_fake(model) is state


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda m, layer: setattr(layer, "bias", torch.zeros(4)), "bias"),
        (lambda m, layer: setattr(layer, "gather_output", True), "vocab-parallel"),
        (lambda m, layer: setattr(layer, "disable_grad_reduce", True), "disable_grad_reduce"),
        (lambda m, layer: setattr(layer.config, "defer_embedding_wgrad_compute", True), "deferred embedding wgrad"),
        (lambda m, layer: setattr(m.config, "use_mup", True), "MuP"),
        (lambda m, layer: setattr(m, "mtp_process", True), "MTP"),
    ],
)
def test_install_guards(mutate, message):
    model, layer = _fake_output_layer(
        torch.nn.Parameter(torch.zeros(4, H, dtype=torch.bfloat16)), None, sequence_parallel=False
    )
    mutate(model, layer)
    with pytest.raises(RuntimeError, match=message):
        _install_fake(model)


def test_install_rejects_missing_head_and_multiple_heads():
    with pytest.raises(RuntimeError, match="no output_layer"):
        _install_fake(SimpleNamespace(post_process=True))
    m1, _ = _fake_output_layer(
        torch.nn.Parameter(torch.zeros(4, H, dtype=torch.bfloat16)), None, sequence_parallel=False
    )
    m2, _ = _fake_output_layer(
        torch.nn.Parameter(torch.zeros(4, H, dtype=torch.bfloat16)), None, sequence_parallel=False
    )
    with pytest.raises(RuntimeError, match="exactly one lm head"):
        install([m1, m2], copy_to_tp_region=_fake_copy_to_tp_region, gather_from_sp_region=_fake_gather_from_sp_region)


def test_conditional_forward_passes_through_when_inactive():
    weight = torch.nn.Parameter(torch.zeros(4, H, dtype=torch.bfloat16))
    model, layer = _fake_output_layer(weight, None, sequence_parallel=False)
    _install_fake(model)
    x = torch.zeros(3, 1, H)
    out, bias = layer.forward(x, weight=weight, runtime_gather_output=True, output_cross_entropy_loss=True, labels=x)
    assert out[0] == "original" and out[1] is x and out[2] is weight and out[3] is True
    assert out[4] == {"output_cross_entropy_loss": True, "labels": x} and bias is None


def test_conditional_forward_when_active():
    weight = torch.nn.Parameter(torch.zeros(4, H, dtype=torch.bfloat16))
    other = torch.nn.Parameter(torch.ones(4, H, dtype=torch.bfloat16))
    model, layer = _fake_output_layer(weight, "group", sequence_parallel=False)
    state = _install_fake(model)
    x = torch.zeros(3, 1, H)
    with activate(state):
        assert is_active()
        mapped, bias = layer.forward(x)
        assert mapped is x and bias is None and state.active_weight is weight and state.fused_forward_calls == 1
        layer.forward(x, weight=other)
        assert state.active_weight is other and state.fused_forward_calls == 2
        with pytest.raises(RuntimeError, match="runtime_gather_output"):
            layer.forward(x, runtime_gather_output=True)
        with pytest.raises(RuntimeError, match="output_cross_entropy_loss"):
            layer.forward(x, output_cross_entropy_loss=True, labels=x)
        with pytest.raises(RuntimeError, match="unsupported output_layer kwargs"):
            layer.forward(x, labels=x)
    assert not is_active()
    assert layer.forward(x)[0][0] == "original"


def test_activate_none_is_inactive_and_target_logprobs_needs_state():
    with activate(None):
        assert not is_active()
        assert chunk_size() == fused_lm_head.DEFAULT_CHUNK_SIZE
    with pytest.raises(RuntimeError, match="active fused lm head"):
        target_logprobs(
            torch.zeros(2, H, dtype=torch.bfloat16),
            torch.zeros(2, dtype=torch.long),
            None,
            temperature=1.0,
            chunk_size=4,
        )
    model, layer = _fake_output_layer(
        torch.nn.Parameter(torch.zeros(4, H, dtype=torch.bfloat16)), None, sequence_parallel=False
    )
    state = _install_fake(model)
    with activate(state), pytest.raises(RuntimeError, match="has not run yet"):
        target_logprobs(
            torch.zeros(2, H, dtype=torch.bfloat16),
            torch.zeros(2, dtype=torch.long),
            None,
            temperature=1.0,
            chunk_size=4,
        )


def test_maybe_activate_is_noop_without_flag(monkeypatch):
    model, layer = _fake_output_layer(
        torch.nn.Parameter(torch.zeros(4, H, dtype=torch.bfloat16)), None, sequence_parallel=False
    )
    original = layer.forward
    monkeypatch.setattr(fused_lm_head, "install", lambda *a, **k: pytest.fail("install must not run"))
    with maybe_activate(Namespace(), [model]) as state:
        assert state is None and not is_active()
    assert layer.forward is original


def test_maybe_activate_installs_once_and_sets_chunk_size(monkeypatch):
    model, layer = _fake_output_layer(
        torch.nn.Parameter(torch.zeros(4, H, dtype=torch.bfloat16)), None, sequence_parallel=False
    )
    calls = []

    def fake_install(chunks):
        calls.append(chunks)
        return _install_fake(model)

    monkeypatch.setattr(fused_lm_head, "install", fake_install)
    args = Namespace(fused_lm_head_logprobs=True, fused_lm_head_chunk_size=7)
    with maybe_activate(args, [model]) as state:
        assert is_active() and chunk_size() == 7 and state.chunk_size == 7
    with maybe_activate(Namespace(fused_lm_head_logprobs=True), [model]):
        assert chunk_size() == fused_lm_head.DEFAULT_CHUNK_SIZE
    assert not is_active() and len(calls) == 2 and getattr(layer, fused_lm_head._STATE_ATTRIBUTE) is state


def test_maybe_activate_on_a_rank_without_head():
    with maybe_activate(Namespace(fused_lm_head_logprobs=True), [SimpleNamespace(post_process=False)]) as state:
        assert state is None and not is_active()


# ---------------------------------------------------------------------------
# Dispatch through math_utils / logit_processors
# ---------------------------------------------------------------------------


def _active_state_with(weight):
    model, layer = _fake_output_layer(weight, None, sequence_parallel=False)
    state = _install_fake(model)
    state.active_weight = weight
    state.chunk_size = 5
    return state


def test_dispatch_through_calculate_log_probs_and_entropy():
    h, w, labels, _ = _inputs()
    weight = torch.nn.Parameter(w.clone())
    expected = _reference_logprobs(h, w, labels)
    with activate(_active_state_with(weight)):
        log_prob, entropy = math_utils.calculate_log_probs_and_entropy(h, labels, None, chunk_size=-1)
        assert entropy is None and log_prob.shape == (T, 1)
        torch.testing.assert_close(log_prob.squeeze(-1), expected, rtol=0, atol=1e-6)
        with pytest.raises(AssertionError, match="no entropy"):
            math_utils.calculate_log_probs_and_entropy(h, labels, None, with_entropy=True)
        with pytest.raises(AssertionError, match="true-on-policy"):
            math_utils.calculate_log_probs_and_entropy(h, labels, None, true_on_policy=True)
        with pytest.raises(AssertionError, match="sampling masks"):
            math_utils.calculate_log_probs_and_entropy(
                h, labels, None, sampling_mask=torch.ones(T, V, dtype=torch.bool)
            )


def test_dispatch_through_get_log_probs_and_entropy_hidden_rows():
    """The loss sees [1, T, H] hidden states; only the response rows are projected."""
    make_parallel_state()
    args = make_args(true_on_policy_mode=False, rollout_temperature=1.0, log_probs_chunk_size=-1)
    h, w, labels, _ = _inputs(seed=8)
    weight = torch.nn.Parameter(w.clone())
    total_lengths, response_lengths = [17, 20], [9, 12]
    tokens = [labels[:17], labels[17:37]]
    # thd layout: hidden row i predicts concatenated token i+1; response rows are the last R-1..-1 of each sample
    expected = [_reference_logprobs(h[7:16], w, labels[8:17]), _reference_logprobs(h[24:36], w, labels[25:37])]

    with activate(_active_state_with(weight)):
        out = get_log_probs_and_entropy(
            h.unsqueeze(0),
            args=args,
            unconcat_tokens=tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            with_entropy=False,
        )
    assert [lp.shape for lp in out["log_probs"]] == [(9,), (12,)]
    torch.testing.assert_close(out["log_probs"][0], expected[0], rtol=0, atol=1e-6)
    torch.testing.assert_close(out["log_probs"][1], expected[1], rtol=0, atol=1e-6)


def test_dispatch_inactive_uses_the_standard_path(monkeypatch):
    seen = []

    def fake_compute_log_probs(logits, tokens, _tp_group, *, sampling_mask=None):
        seen.append(logits.shape)
        return torch.zeros((tokens.size(0), 1))

    monkeypatch.setattr(math_utils, "compute_log_probs", fake_compute_log_probs)
    math_utils.calculate_log_probs_and_entropy(
        torch.zeros(3, 8, dtype=torch.bfloat16), torch.zeros(3, dtype=torch.long), None
    )
    assert seen == [torch.Size([3, 8])]


# ---------------------------------------------------------------------------
# The head's output transform: logits * multiplier, then softcap * tanh(. / softcap) (Muse Glimmer)
# ---------------------------------------------------------------------------

GLIMMER = fused_lm_head.LogitTransform(multiplier=0.19611613513818404, softcap=20.0)
# The transformed log-probs reach -100 and below on these inputs: fp32 summation order alone moves them by a few 1e-6
FP32_LOGPROB_TOL = dict(rtol=4e-6, atol=4e-6)
TRANSFORMS = [
    fused_lm_head.LogitTransform(multiplier=3.0, softcap=5.0),  # the cap bites on most logits of these inputs
    GLIMMER,
    fused_lm_head.LogitTransform(multiplier=0.5),  # a multiplier alone
]


def _transform_reference(z: torch.Tensor, transform) -> torch.Tensor:
    y = z * transform.multiplier
    return y if transform.softcap is None else transform.softcap * torch.tanh(y / transform.softcap)


def _glimmer_scale_inputs(seed: int = 0):
    """Hidden states large enough that Glimmer's 0.196 x projection reaches its softcap of 20 on some logits."""
    h, w, labels, upstream = _inputs(seed=seed)
    return (h.float() * 12).to(torch.bfloat16), w, labels, upstream


def _reference_logprobs_transformed(h, w, labels, transform, temperature=1.0):
    logits = _transform_reference(F.linear(h.float(), w.float()), transform) / temperature
    return torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def _reference_grads_transformed(h, w, labels, upstream, transform, temperature=1.0):
    """fp32 autograd through the transformed reference, and the bf16-rounded projection gradient GEMMs."""
    h32 = h.float().requires_grad_(True)
    w32 = w.float().requires_grad_(True)
    z = F.linear(h32, w32)
    z.retain_grad()
    logits = _transform_reference(z, transform) / temperature
    logprobs = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    (logprobs * upstream).sum().backward()
    g_bf16 = z.grad.to(torch.bfloat16).float()  # dL/d(projection), rounded once as the fused backward does
    return {
        "hidden_fp32": h32.grad,
        "weight_fp32": w32.grad,
        "hidden_rounded": (g_bf16 @ w.float()).to(torch.bfloat16),
        "weight_rounded": g_bf16.t() @ h.float(),
        "logits_grad_bf16": g_bf16,
    }


@pytest.mark.parametrize("transform", TRANSFORMS, ids=lambda t: f"m{t.multiplier:g}-cap{t.softcap}")
@pytest.mark.parametrize("chunk", CHUNK_SIZES)
def test_forward_applies_the_output_transform(chunk, transform):
    h, w, labels, _ = _glimmer_scale_inputs()
    out = vocab_parallel_fused_linear_logprobs(h, w, labels, chunk_size=chunk, transform=transform)
    torch.testing.assert_close(out, _reference_logprobs_transformed(h, w, labels, transform), **FP32_LOGPROB_TOL)
    plain = _reference_logprobs(h, w, labels)
    assert (out - plain).abs().max() > 0.1, "the transform must change the scored distribution"


def test_glimmer_inputs_reach_the_softcap():
    h, w, _, _ = _glimmer_scale_inputs()
    z = F.linear(h.float(), w.float()) * GLIMMER.multiplier
    assert z.abs().max() > GLIMMER.softcap, "the test inputs would not exercise the cap"


def test_forward_transform_comes_before_the_temperature():
    h, w, labels, _ = _glimmer_scale_inputs(seed=2)
    out = vocab_parallel_fused_linear_logprobs(h, w, labels, temperature=0.7, chunk_size=5, transform=GLIMMER)
    expected = _reference_logprobs_transformed(h, w, labels, GLIMMER, temperature=0.7)
    torch.testing.assert_close(out, expected, **FP32_LOGPROB_TOL)


@pytest.mark.parametrize("transform", TRANSFORMS, ids=lambda t: f"m{t.multiplier:g}-cap{t.softcap}")
@pytest.mark.parametrize("chunk", CHUNK_SIZES)
def test_backward_through_the_output_transform(chunk, transform):
    h, w, labels, upstream = _glimmer_scale_inputs(seed=4)
    ref = _reference_grads_transformed(h, w, labels, upstream, transform)
    h_fused = h.clone().requires_grad_(True)
    w_fused = w.clone().requires_grad_(True)
    out = vocab_parallel_fused_linear_logprobs(h_fused, w_fused, labels, chunk_size=chunk, transform=transform)
    (out * upstream).sum().backward()
    slack = BF16_RTOL * ref["hidden_rounded"].float().abs().max()
    torch.testing.assert_close(h_fused.grad, ref["hidden_rounded"], rtol=BF16_RTOL, atol=slack)
    torch.testing.assert_close(
        w_fused.grad.float(), ref["weight_rounded"], rtol=BF16_RTOL, atol=BF16_RTOL * ref["weight_rounded"].abs().max()
    )
    torch.testing.assert_close(
        h_fused.grad.float(), ref["hidden_fp32"], rtol=2e-2, atol=2e-2 * ref["hidden_fp32"].abs().max()
    )
    torch.testing.assert_close(
        w_fused.grad.float(), ref["weight_fp32"], rtol=2e-2, atol=2e-2 * ref["weight_fp32"].abs().max()
    )


def test_identity_transform_is_the_plain_head():
    h, w, labels, upstream = _inputs()
    plain = vocab_parallel_fused_linear_logprobs(h, w, labels, chunk_size=5)
    identity = vocab_parallel_fused_linear_logprobs(h, w, labels, chunk_size=5, transform=fused_lm_head.IDENTITY)
    assert torch.equal(plain, identity) and fused_lm_head.IDENTITY.is_identity


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"multiplier": 0.0}, "multiplier"),
        ({"multiplier": float("inf")}, "multiplier"),
        ({"softcap": 0.0}, "softcap"),
        ({"softcap": -20.0}, "softcap"),
    ],
)
def test_transform_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        fused_lm_head.LogitTransform(**kwargs)


def test_two_shard_tensor_parallel_with_the_glimmer_transform(monkeypatch):
    monkeypatch.setattr(fused_lm_head, "dist", _FakeDist)
    vocab = 132
    h, w, labels, upstream = _inputs(seed=7, vocab=vocab)
    h = (h.float() * 12).to(torch.bfloat16)
    ref = _reference_grads_transformed(h, w, labels, upstream, GLIMMER)
    expected = _reference_logprobs_transformed(h, w, labels, GLIMMER)
    group = _FakeTensorParallel(2)
    shard = vocab // 2

    def rank_fn(rank):
        w_r = w[rank * shard : (rank + 1) * shard].clone().requires_grad_(True)
        h_r = h.clone().requires_grad_(True)
        out = vocab_parallel_fused_linear_logprobs(h_r, w_r, labels, tp_group=group, chunk_size=5, transform=GLIMMER)
        (out * upstream).sum().backward()
        return out, h_r.grad, w_r.grad

    results = _run_ranks(group, rank_fn)
    for out, _, _ in results:
        torch.testing.assert_close(out, expected, **FP32_LOGPROB_TOL)
    hidden_sum = results[0][1].float() + results[1][1].float()
    torch.testing.assert_close(hidden_sum, ref["hidden_fp32"], rtol=2e-2, atol=2e-2 * ref["hidden_fp32"].abs().max())
    for rank, (_, _, w_grad) in enumerate(results):
        expected_w = ref["weight_rounded"][rank * shard : (rank + 1) * shard]
        torch.testing.assert_close(w_grad.float(), expected_w, rtol=BF16_RTOL, atol=BF16_RTOL * expected_w.abs().max())


# --- which head classes the fused path knows (by the class that defines the forward the head runs)


def _head_class(name: str, module: str, bases: tuple = ()) -> type:
    def forward(self, input_, weight=None, runtime_gather_output=None, **kwargs):
        return input_, None

    return type(name, bases, {"forward": forward, "__module__": module})


_ColumnParallelLinear = _head_class("ColumnParallelLinear", "megatron.core.tensor_parallel.layers")
_MuseMixin = _head_class("MuseGlimmerOutputLayerMixin", "megatron.bridge.models.muse_glimmer.modeling_muse_glimmer")
_GemmaMixin = _head_class("Gemma4OutputLayer", "megatron.bridge.models.gemma.gemma4_provider")


def _extended(mixin: type) -> type:
    """megatron-bridge's extend_instance: a new class named like the head, the mixin first."""
    return type("ColumnParallelLinear", (mixin, _ColumnParallelLinear), {})


def _head(cls: type, **config):
    head = cls()
    head.weight, head.bias, head.gather_output, head.disable_grad_reduce = None, None, False, False
    head.sequence_parallel, head.tp_group = False, None
    head.config = SimpleNamespace(defer_embedding_wgrad_compute=False, **config)
    return head


def test_output_transform_of_known_heads():
    assert fused_lm_head.output_transform(_head(_ColumnParallelLinear)) is fused_lm_head.IDENTITY
    assert fused_lm_head.output_transform(SimpleNamespace(forward=lambda x: x)) is fused_lm_head.IDENTITY
    muse = _head(_extended(_MuseMixin), output_multiplier=0.196, final_logit_softcapping=20.0)
    assert fused_lm_head.output_transform(muse) == fused_lm_head.LogitTransform(multiplier=0.196, softcap=20.0)
    uncapped = _head(_extended(_MuseMixin), output_multiplier=0.5, final_logit_softcapping=0.0)
    assert fused_lm_head.output_transform(uncapped) == fused_lm_head.LogitTransform(multiplier=0.5)


def test_install_refuses_a_head_whose_forward_it_does_not_know():
    head = _head(_extended(_GemmaMixin))
    model = SimpleNamespace(post_process=True, output_layer=head, config=SimpleNamespace(use_mup=False))
    with pytest.raises(RuntimeError, match="does not know what .*Gemma4OutputLayer.forward does"):
        _install_fake(model)


def test_install_carries_the_muse_transform_to_target_logprobs():
    h, w, labels, _ = _glimmer_scale_inputs(seed=9)
    head = _head(_extended(_MuseMixin), output_multiplier=GLIMMER.multiplier, final_logit_softcapping=GLIMMER.softcap)
    head.weight = torch.nn.Parameter(w.clone())
    model = SimpleNamespace(post_process=True, output_layer=head, config=SimpleNamespace(use_mup=False))
    state = _install_fake(model)
    assert state.transform == GLIMMER
    with activate(state):
        mapped, _ = head.forward(h.unsqueeze(1))
        out = target_logprobs(mapped.squeeze(1), labels, None, temperature=1.0, chunk_size=5)
    torch.testing.assert_close(out, _reference_logprobs_transformed(h, w, labels, GLIMMER), **FP32_LOGPROB_TOL)
    # inactive, the head's own forward (here the mixin's) runs unchanged
    assert head.forward(h)[0] is h
