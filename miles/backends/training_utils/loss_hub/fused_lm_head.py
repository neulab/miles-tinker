"""Fused chunked lm head: target log-probs without a ``[T, V]`` logits tensor.

Ported from platoon's ``platoon/train/areal/fused_linear_logprobs.py`` for the
Tinker full fine-tuning path (``--fused-lm-head-logprobs``).

* When active, the Megatron ``output_layer`` (a ``ColumnParallelLinear``)
  returns its TP/SP-mapped *hidden states* instead of logits, so the tensor
  miles hands to the tinker losses is ``[1, T, H]``. ``math_utils
  .calculate_log_probs_and_entropy`` dispatches to :func:`target_logprobs`,
  which projects the response rows in chunks of ``chunk_size`` tokens inside
  one autograd function and recomputes the projection in backward. Peak
  transient memory is one ``[chunk, V/TP]`` fp32 tensor (plus its gradient in
  backward) instead of ``[T, V/TP]`` logits and their gradient.
* Numerics match SGLang's ``--enable-fp32-lm-head``: each chunk's logits are
  ``torch.mm(h_bf16, W_bf16.t(), out_dtype=torch.float32)`` (bf16 GEMM, fp32
  accumulate, fp32 output) on CUDA and ``torch.mm(h.float(), W.float().t())``
  elsewhere (CPU tests: ``aten::mm.dtype`` has no CPU kernel), in the forward
  and in the backward recompute; softmax statistics are fp32. The logit
  gradient is cast to the weight dtype before the two gradient GEMMs (as in
  the standard bf16 head); the weight gradient is accumulated in fp32 across
  chunks.
* Weight gradient. When the weight is DDP-managed (``weight.main_grad`` exists
  and DDP tracks ``weight.grad_added_to_main_grad``) the fp32 gradient is added
  into ``main_grad`` directly and ``None`` is returned for the weight (Megatron
  sets the flag the same way, ``tensor_parallel/layers.py``); otherwise the
  gradient is returned in the weight dtype and DDP's post-hook folds it into
  ``main_grad`` (``distributed_data_parallel.py`` ``_make_backward_post_hook``).
* Output transforms. A head whose class adds work after the projection would
  be bypassed with its ``forward``, so :func:`install` reads what the head
  class does and refuses one it does not know. Megatron's
  ``ColumnParallelLinear`` has none, nor does GPTModel's
  ``LinearCrossEntropyModule`` outside its fused cross-entropy; megatron-bridge's
  ``MuseGlimmerOutputLayerMixin`` (Muse Glimmer) multiplies the logits by
  ``config.output_multiplier`` and then soft-caps them,
  ``cap * tanh(logits / cap)`` with ``cap = config.final_logit_softcapping``.
  The fused path applies the same two steps in fp32 per chunk, in SGLang's
  order (``logit_scale``, then the softcap), and multiplies the logit gradient
  by their derivative, ``multiplier * (1 - tanh^2)``, in backward.
* Entropy, true-on-policy scoring and sampling masks are unsupported (the
  tinker losses use none of them).

Only torch is imported at module level; Megatron's TP mappings are imported
inside :func:`install` so the module (and its CPU tests) load without Megatron.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import math
from argparse import Namespace
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

FLAG_ATTR = "fused_lm_head_logprobs"
CHUNK_SIZE_ATTR = "fused_lm_head_chunk_size"
DEFAULT_CHUNK_SIZE = 1024

_LOW_PRECISION = (torch.bfloat16, torch.float16)


@dataclass(frozen=True)
class LogitTransform:
    """What an lm head does to its projection: ``logits * multiplier``, then ``softcap * tanh(logits / softcap)``
    when ``softcap`` is set."""

    multiplier: float = 1.0
    softcap: float | None = None

    def __post_init__(self):
        if not math.isfinite(self.multiplier) or self.multiplier == 0.0:
            raise ValueError(f"the logit multiplier must be finite and nonzero, got {self.multiplier}")
        if self.softcap is not None and not (math.isfinite(self.softcap) and self.softcap > 0.0):
            raise ValueError(f"the logit softcap must be positive (or None), got {self.softcap}")

    @property
    def is_identity(self) -> bool:
        return self.multiplier == 1.0 and self.softcap is None


IDENTITY = LogitTransform()


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


def _tp_world_size_and_rank(tp_group: Any | None) -> tuple[int, int]:
    if tp_group is None:
        return 1, 0
    return dist.get_world_size(tp_group), dist.get_rank(tp_group)


def _all_reduce_(tensor: torch.Tensor, op: dist.ReduceOp, tp_group: Any | None) -> None:
    world_size, _ = _tp_world_size_and_rank(tp_group)
    if world_size > 1:
        dist.all_reduce(tensor, op=op, group=tp_group)


def _mm_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` with fp32 accumulation and an fp32 result.

    bf16/fp16 operands on CUDA use the bf16 GEMM with ``out_dtype=float32``
    (SGLang's ``enable_fp32_lm_head`` expression); everything else (CPU, fp32
    models) upcasts the operands, which is the same arithmetic without the
    tensor-core path.
    """
    if a.is_cuda and a.dtype == b.dtype and a.dtype in _LOW_PRECISION:
        return torch.mm(a, b, out_dtype=torch.float32)
    return torch.mm(a.float(), b.float())


def _lm_head_compute_input(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Feed the GEMM the weight dtype (what ``ColumnParallelLinear`` computes in).

    ``autograd.Function.forward`` does not inherit the ambient autocast policy;
    miles' Float16Module output is already bf16, so this is a no-op there.
    """
    return hidden.to(dtype=weight.dtype)


def _transform_logits_(
    logits: torch.Tensor, transform: LogitTransform, *, with_slope: bool
) -> torch.Tensor | float | None:
    """Apply ``transform`` to fp32 ``logits`` in place, in SGLang's order (multiplier, then the softcap).

    With ``with_slope`` return d(transformed)/d(projection) for the backward: the multiplier without a softcap,
    else the tensor ``multiplier * (1 - tanh^2)``.
    """
    if transform.multiplier != 1.0:
        logits.mul_(transform.multiplier)
    if transform.softcap is None:
        return transform.multiplier if with_slope else None
    tanh = logits.div_(transform.softcap).tanh_()
    slope = torch.mul(tanh, tanh).neg_().add_(1.0).mul_(transform.multiplier) if with_slope else None
    tanh.mul_(transform.softcap)
    return slope


def _chunk_logprob_statistics(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    tp_group: Any | None,
    vocab_start: int,
    need_target: bool,
    transform: LogitTransform = IDENTITY,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | float | None]:
    """Return local-vocab probabilities and/or the global target log-probs of one chunk, and (for the backward,
    without ``need_target``) the slope of the head's output transform."""
    scaled_logits = _mm_fp32(hidden, weight.t())
    slope = None if transform.is_identity else _transform_logits_(scaled_logits, transform, with_slope=not need_target)
    if temperature != 1.0:
        scaled_logits.div_(temperature)

    global_max = scaled_logits.amax(dim=-1)
    _all_reduce_(global_max, dist.ReduceOp.MAX, tp_group)
    shifted_logits = scaled_logits.sub_(global_max.unsqueeze(-1))

    selected = None
    if need_target:
        local_vocab = weight.shape[0]
        local_labels = labels - vocab_start
        target_is_local = (local_labels >= 0) & (local_labels < local_vocab)
        safe_labels = local_labels.masked_fill(~target_is_local, 0)
        row_ids = torch.arange(hidden.shape[0], device=hidden.device)
        selected = shifted_logits[row_ids, safe_labels].clone().masked_fill_(~target_is_local, 0.0)

    probabilities = shifted_logits.exp_()
    sum_exp = probabilities.sum(dim=-1)

    if need_target:
        # One SUM collective carries both the denominator and the selected numerator.
        global_stats = torch.stack((sum_exp, selected), dim=-1)
        _all_reduce_(global_stats, dist.ReduceOp.SUM, tp_group)
        sum_exp, selected = global_stats.unbind(dim=-1)
        return None, selected - sum_exp.log(), None

    _all_reduce_(sum_exp, dist.ReduceOp.SUM, tp_group)
    probabilities.div_(sum_exp.unsqueeze(-1))
    return probabilities, None, slope


def _ddp_main_grad(weight: torch.Tensor) -> torch.Tensor | None:
    """The fp32 DDP grad buffer view when Megatron DDP manages this weight, else None."""
    if not hasattr(weight, "grad_added_to_main_grad"):
        return None
    return getattr(weight, "main_grad", None)


class _VocabParallelFusedLinearLogprobs(torch.autograd.Function):
    """Recompute a TP-sharded lm head in backward instead of saving logits."""

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        temperature: float,
        tp_group: Any | None,
        chunk_size: int,
        transform: LogitTransform,
    ) -> torch.Tensor:
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_labels = labels.reshape(-1)
        _, tp_rank = _tp_world_size_and_rank(tp_group)
        vocab_start = tp_rank * weight.shape[0]

        output = torch.empty(flat_labels.shape, dtype=torch.float32, device=flat_hidden.device)
        for start in range(0, flat_hidden.shape[0], chunk_size):
            stop = min(start + chunk_size, flat_hidden.shape[0])
            _, chunk_logprobs, _ = _chunk_logprob_statistics(
                _lm_head_compute_input(flat_hidden[start:stop], weight),
                weight,
                flat_labels[start:stop],
                temperature=temperature,
                tp_group=tp_group,
                vocab_start=vocab_start,
                need_target=True,
                transform=transform,
            )
            output[start:stop] = chunk_logprobs

        ctx.save_for_backward(hidden_states, weight, labels)
        ctx.weight_param = weight
        ctx.main_grad = _ddp_main_grad(weight)
        ctx.temperature = temperature
        ctx.tp_group = tp_group
        ctx.chunk_size = chunk_size
        ctx.vocab_start = vocab_start
        ctx.transform = transform
        return output.reshape(labels.shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[Any, ...]:
        hidden_states, weight, labels = ctx.saved_tensors
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_labels = labels.reshape(-1)
        flat_grad_output = grad_output.reshape(-1).float()

        hidden_grad = torch.empty_like(flat_hidden) if ctx.needs_input_grad[0] else None
        # fp32 accumulation across chunks; one cast at the autograd boundary (hook path only).
        weight_grad = torch.zeros_like(weight, dtype=torch.float32) if ctx.needs_input_grad[1] else None
        local_vocab = weight.shape[0]

        for start in range(0, flat_hidden.shape[0], ctx.chunk_size):
            stop = min(start + ctx.chunk_size, flat_hidden.shape[0])
            hidden_chunk = flat_hidden[start:stop]
            compute_hidden_chunk = _lm_head_compute_input(hidden_chunk, weight)
            labels_chunk = flat_labels[start:stop]
            probabilities, _, slope = _chunk_logprob_statistics(
                compute_hidden_chunk,
                weight,
                labels_chunk,
                temperature=ctx.temperature,
                tp_group=ctx.tp_group,
                vocab_start=ctx.vocab_start,
                need_target=False,
                transform=ctx.transform,
            )

            upstream = flat_grad_output[start:stop]
            probabilities.mul_(-upstream.unsqueeze(-1))
            local_labels = labels_chunk - ctx.vocab_start
            target_is_local = (local_labels >= 0) & (local_labels < local_vocab)
            safe_labels = local_labels.masked_fill(~target_is_local, 0)
            row_ids = torch.arange(hidden_chunk.shape[0], device=hidden_chunk.device)
            probabilities[row_ids, safe_labels] += upstream * target_is_local
            if ctx.temperature != 1.0:
                probabilities.div_(ctx.temperature)
            if slope is not None:  # through the head's output transform, back to the projection
                probabilities.mul_(slope)
                del slope

            # The gradient GEMMs run in the weight dtype like the standard head;
            # accumulation and outputs stay fp32 until the cast below.
            logits_grad = probabilities.to(weight.dtype)
            if hidden_grad is not None:
                hidden_grad[start:stop] = _mm_fp32(logits_grad, weight).to(hidden_chunk.dtype)
            if weight_grad is not None:
                weight_grad.add_(_mm_fp32(logits_grad.t(), compute_hidden_chunk))

        if hidden_grad is not None:
            hidden_grad = hidden_grad.reshape_as(hidden_states)
        return hidden_grad, _finish_weight_grad(ctx, weight_grad), None, None, None, None, None


def _finish_weight_grad(ctx, weight_grad: torch.Tensor | None) -> torch.Tensor | None:
    """DDP-managed weight: add fp32 into main_grad and return None; else return grad in weight dtype."""
    if weight_grad is None:
        return None
    weight = ctx.weight_param
    main_grad = ctx.main_grad
    if main_grad is not None:
        main_grad.view_as(weight).add_(weight_grad)
        weight.grad_added_to_main_grad = True
        return None
    return weight_grad.to(weight.dtype)


def vocab_parallel_fused_linear_logprobs(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 1.0,
    tp_group: Any | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    transform: LogitTransform = IDENTITY,
) -> torch.Tensor:
    """Project and select target log-probs without full-sequence logits.

    ``weight`` is the local contiguous vocabulary shard ``[V/TP, H]``. The
    hidden states must already carry Megatron's output-layer TP mapping
    (sequence-parallel states all-gathered, replicated states passed through
    ``copy_to_tensor_model_parallel_region`` whose backward all-reduces dHidden).
    ``transform`` is what the head does after the projection (a multiplier and a
    softcap), applied before the temperature. Returns fp32 log-probs shaped like
    ``labels``.
    """
    if hidden_states.shape[:-1] != labels.shape:
        raise ValueError(
            f"hidden-state leading dimensions must match labels: got {tuple(hidden_states.shape)} and {tuple(labels.shape)}"
        )
    if hidden_states.shape[-1] != weight.shape[-1]:
        raise ValueError(f"hidden size {hidden_states.shape[-1]} does not match weight {weight.shape[-1]}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if labels.dtype != torch.long:
        raise TypeError(f"labels must have dtype torch.long, got {labels.dtype}")
    return _VocabParallelFusedLinearLogprobs.apply(
        hidden_states, weight, labels, float(temperature), tp_group, int(chunk_size), transform
    )


# ---------------------------------------------------------------------------
# Head installation and activation
# ---------------------------------------------------------------------------


@dataclass
class HeadState:
    """State shared between the conditional output-layer wrapper and the loss dispatch."""

    output_layer: Any
    original_forward: Callable[..., Any]
    tp_group: Any
    active_weight: torch.Tensor | None = None
    fused_forward_calls: int = 0
    chunk_size: int = DEFAULT_CHUNK_SIZE
    transform: LogitTransform = IDENTITY


_ACTIVE: contextvars.ContextVar[HeadState | None] = contextvars.ContextVar("miles_active_fused_lm_head", default=None)
_STATE_ATTRIBUTE = "_miles_fused_lm_head_state"


def _unwrap_model_chunk(model: Any) -> Any:
    """DDP -> Float16Module -> GPTModel (or a VLM's language_model)."""
    seen: set[int] = set()
    current = model
    while id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "module"):
            current = current.module
            continue
        if hasattr(current, "language_model"):
            current = current.language_model
            continue
        break
    return current


def _find_output_layer(model_chunks: Iterable[Any]) -> tuple[Any, Any] | None:
    candidates: list[tuple[Any, Any]] = []
    missing_post_process_head = False
    for wrapped_model in model_chunks:
        model = _unwrap_model_chunk(wrapped_model)
        output_layer = getattr(model, "output_layer", None)
        if output_layer is None:
            missing_post_process_head |= bool(getattr(model, "post_process", False))
            continue
        candidates.append((model, output_layer))
    if missing_post_process_head:
        raise RuntimeError("a post-process Megatron model chunk has no output_layer")
    if not candidates:
        return None
    if len(candidates) != 1:
        raise RuntimeError("the fused lm head requires exactly one lm head per pipeline rank (VPP=1)")
    return candidates[0]


def _check_head_supported(model: Any, output_layer: Any) -> None:
    if getattr(output_layer, "bias", None) is not None:
        raise RuntimeError("the fused lm head does not support an lm-head bias")
    if bool(getattr(output_layer, "gather_output", False)):
        raise RuntimeError("the fused lm head requires vocab-parallel (ungathered) lm-head output")
    if bool(getattr(output_layer, "disable_grad_reduce", False)):
        raise RuntimeError("the fused lm head does not support disable_grad_reduce on the lm head")
    if bool(getattr(getattr(output_layer, "config", None), "defer_embedding_wgrad_compute", False)):
        raise RuntimeError("the fused lm head does not support deferred embedding wgrad")
    if bool(getattr(getattr(model, "config", None), "use_mup", False)):
        raise RuntimeError("the fused lm head does not implement MuP output scaling")
    if bool(getattr(model, "mtp_process", False)):
        raise RuntimeError("the fused lm head does not support MTP output processing (--mtp-num-layers 0)")


# The lm head classes whose forward the fused path reproduces, by the class that defines the forward the head runs.
# GPTModel's head (every Qwen and other GPT preset) is a LinearCrossEntropyModule: its forward is ColumnParallelLinear's
# unless asked for Megatron's fused cross-entropy loss, which the wrapper refuses while the fused path is active.
_PLAIN_HEAD_FORWARDS = (
    "megatron.core.tensor_parallel.layers.ColumnParallelLinear",
    "megatron.core.transformer.linear_cross_entropy.LinearCrossEntropyModule",
)
_MUSE_HEAD_FORWARDS = ("megatron.bridge.models.muse_glimmer.modeling_muse_glimmer.MuseGlimmerOutputLayerMixin",)


def _forward_owner(output_layer: Any) -> type | None:
    """The class whose ``forward`` the head runs (None when no class in its MRO defines one)."""
    return next((cls for cls in type(output_layer).__mro__ if "forward" in vars(cls)), None)


def output_transform(output_layer: Any) -> LogitTransform:
    """What the head's ``forward`` does after the projection, which the fused path must reproduce.

    Raises for a head class it does not know: bypassing such a forward would score a different distribution
    without any error (a Gemma or Muse softcap, a MuP scale).
    """
    owner = _forward_owner(output_layer)
    name = None if owner is None else f"{owner.__module__}.{owner.__qualname__}"
    if owner is None or name in _PLAIN_HEAD_FORWARDS:
        return IDENTITY
    if name in _MUSE_HEAD_FORWARDS:
        config = output_layer.config
        softcap = getattr(config, "final_logit_softcapping", None)
        return LogitTransform(multiplier=float(config.output_multiplier), softcap=float(softcap) if softcap else None)
    raise RuntimeError(
        f"the fused lm head does not know what {name}.forward does after the projection, so it cannot reproduce it; "
        "train with --no-fused-lm-head (the standard head), or teach fused_lm_head.output_transform this head"
    )


def install(
    model_chunks: Iterable[Any],
    *,
    copy_to_tp_region: Callable[..., torch.Tensor] | None = None,
    gather_from_sp_region: Callable[..., torch.Tensor] | None = None,
) -> HeadState | None:
    """Wrap the one local Megatron lm head so it can be bypassed under :func:`activate`.

    Returns ``None`` on pipeline ranks without a post-process chunk. Idempotent:
    a second call returns the installed state. Outside an activation the
    wrapper calls the original ``ColumnParallelLinear.forward`` unchanged.
    """
    found = _find_output_layer(model_chunks)
    if found is None:
        return None
    model, output_layer = found
    existing = getattr(output_layer, _STATE_ATTRIBUTE, None)
    if existing is not None:
        return existing
    _check_head_supported(model, output_layer)
    transform = output_transform(output_layer)

    if copy_to_tp_region is None or gather_from_sp_region is None:
        # Megatron is a runtime dependency of the trainer only; CPU tests inject fakes.
        from megatron.core.tensor_parallel.mappings import (
            copy_to_tensor_model_parallel_region,
            gather_from_sequence_parallel_region,
        )

        copy_to_tp_region = copy_to_tp_region or copy_to_tensor_model_parallel_region
        gather_from_sp_region = gather_from_sp_region or gather_from_sequence_parallel_region

    original_forward = output_layer.forward
    state = HeadState(
        output_layer=output_layer,
        original_forward=original_forward,
        tp_group=output_layer.tp_group,
        transform=transform,
    )

    def _conditional_forward(
        layer_self,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
        **kwargs,
    ):
        if _ACTIVE.get() is not state:
            return original_forward(input_, weight=weight, runtime_gather_output=runtime_gather_output, **kwargs)
        if runtime_gather_output:
            raise RuntimeError("the fused lm head cannot gather full-vocabulary output (runtime_gather_output)")
        if kwargs.get("output_cross_entropy_loss"):
            raise RuntimeError("the fused lm head cannot fuse the cross-entropy loss (output_cross_entropy_loss)")
        if kwargs:
            raise RuntimeError(f"unsupported output_layer kwargs while the fused lm head is active: {sorted(kwargs)}")
        active_weight = weight if weight is not None else layer_self.weight
        if active_weight is None:
            raise RuntimeError("the Megatron lm head did not supply its (shared) output weight")
        state.active_weight = active_weight
        state.fused_forward_calls += 1
        if bool(getattr(layer_self, "sequence_parallel", False)):
            mapped_hidden = gather_from_sp_region(input_, group=layer_self.tp_group)
        else:
            mapped_hidden = copy_to_tp_region(input_, group=layer_self.tp_group)
        return mapped_hidden, None

    output_layer.forward = MethodType(_conditional_forward, output_layer)
    setattr(output_layer, _STATE_ATTRIBUTE, state)
    logger.info(
        "fused lm head installed on %s (sequence_parallel=%s, output transform %s)",
        type(output_layer).__name__,
        bool(getattr(output_layer, "sequence_parallel", False)),
        transform,
    )
    return state


@contextlib.contextmanager
def activate(state: HeadState | None) -> Iterator[HeadState | None]:
    """Route the installed head through the fused path for the enclosed forward/backward."""
    token = _ACTIVE.set(state)
    try:
        yield state
    finally:
        _ACTIVE.reset(token)


@contextlib.contextmanager
def maybe_activate(args: Namespace, model_chunks: Iterable[Any]) -> Iterator[HeadState | None]:
    """``activate`` when ``args.fused_lm_head_logprobs`` is set (installing on first use), else a no-op."""
    if not getattr(args, FLAG_ATTR, False):
        yield None
        return
    state = install(model_chunks)
    if state is not None:
        state.chunk_size = int(getattr(args, CHUNK_SIZE_ATTR, DEFAULT_CHUNK_SIZE))
    with activate(state):
        yield state


def is_active() -> bool:
    return _ACTIVE.get() is not None


def chunk_size() -> int:
    state = _ACTIVE.get()
    return state.chunk_size if state is not None else DEFAULT_CHUNK_SIZE


def target_logprobs(
    hidden: torch.Tensor,
    tokens: torch.Tensor,
    tp_group: Any | None,
    *,
    temperature: float,
    chunk_size: int,
) -> torch.Tensor:
    """fp32 ``[R]`` log-probs of ``tokens`` given the TP-mapped hidden rows ``[R, H]`` of the active head."""
    state = _ACTIVE.get()
    if state is None:
        raise RuntimeError("target_logprobs needs an active fused lm head (activate/maybe_activate)")
    if state.active_weight is None:
        raise RuntimeError("the fused lm head is active but the output layer has not run yet")
    if tokens.dtype != torch.long:
        tokens = tokens.long()
    return vocab_parallel_fused_linear_logprobs(
        hidden,
        state.active_weight,
        tokens,
        temperature=temperature,
        tp_group=tp_group,
        chunk_size=chunk_size,
        transform=state.transform,
    )
