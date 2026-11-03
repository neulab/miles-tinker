"""Draft-only MTP with a token mean over a Tinker optimizer window.

The policy keeps Tinker's exact-sum semantics. Draft gradients accumulate as
sums, independently of Megatron's per-microbatch auxiliary-loss scaler, and
only MTP parameter gradients are normalized before the deferred DP reduction.
Megatron owns token rolling and packed-sequence boundaries.
"""

from __future__ import annotations

import contextvars
import copy
import inspect
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn.functional as F

from miles.backends.training_utils.loss_hub import fused_lm_head


class MTPWindow:
    """Detached sufficient statistics for the worker's open optimizer window."""

    def __init__(self):
        self.totals = None

    def record(self, loss_sum: torch.Tensor, token_count: torch.Tensor) -> None:
        values = torch.stack((loss_sum.detach(), token_count.detach())).to(torch.float64)
        if self.totals is None:
            self.totals = values
        else:
            self.totals.add_(values)

    def reset(self) -> None:
        self.totals = None

    def normalize_gradients(self, model, *, dp_group=None) -> dict[str, float]:
        if self.totals is None:
            raise RuntimeError("MTP was enabled but no draft loss was recorded in the optimizer window")
        totals = self.totals.clone()
        dp_size = 1
        if dist.is_initialized():
            dist.all_reduce(totals, group=dp_group)
            dp_size = dist.get_world_size(dp_group)
        # Deferred DDP reduction averages across DP; cancel that average here.
        scale = dp_size / totals[1].clamp_min(1)
        seen = set()
        for chunk in model:
            for name, param in chunk.named_parameters():
                if "mtp" not in name.split(".") or id(param) in seen:
                    continue
                seen.add(id(param))
                grad = getattr(param, "main_grad", None)
                if grad is None:
                    grad = param.grad
                if grad is not None:
                    grad.mul_(scale.to(grad.dtype))
        if not seen:
            raise RuntimeError("MTP loss was recorded but the model has no MTP parameters")
        loss_sum, tokens = totals.tolist()
        return {"mtp_loss": loss_sum / max(tokens, 1), "mtp_tokens": tokens}


class _AttachDraftLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, loss):
        ctx.save_for_backward(loss)
        return hidden

    @staticmethod
    def backward(ctx, grad_hidden):
        (loss,) = ctx.saved_tensors
        # The surrounding policy loss has client-defined scaling. The draft
        # has its own token-mean objective and must not inherit that scaling.
        return grad_hidden, torch.ones_like(loss)


def attach_loss(hidden: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
    return _AttachDraftLoss.apply(hidden, loss)


_WINDOW = contextvars.ContextVar("miles_tinker_mtp_window", default=None)
_PATCHED = contextvars.ContextVar("miles_mtp_loss_patched", default=False)


@contextmanager
def accumulate(window: MTPWindow):
    token = _WINDOW.set(window)
    try:
        yield
    finally:
        _WINDOW.reset(token)


def target_mask(batch, *, qkv_format: str) -> torch.Tensor:
    """Select client-supervised targets, aligned to next-token prediction rows.

    Tinker may supply an entire sequence with zero weights on prompt/tool
    tokens. Those are context, not targets for the draft. Zero-advantage tokens
    are excluded too, since the protocol carries no separate generation mask.
    """
    values = batch.get("loss_weights")
    if values is None:
        values = batch.get("advantages")
    if values is None:
        raise ValueError("MTP requires Tinker weights or advantages to identify supervised targets")
    masks = []
    for value, mask, total, response in zip(
        values, batch["loss_masks"], batch["total_lengths"], batch["response_lengths"], strict=True
    ):
        selected = (torch.as_tensor(value, device=mask.device) != 0) * mask
        masks.append(F.pad(selected, (total - response - 1, 1)))
    shape = batch["full_loss_masks"].shape
    if qkv_format == "bshd":
        packed = torch.stack([F.pad(mask, (0, shape[1] - mask.numel())) for mask in masks])
    elif qkv_format == "thd":
        packed = torch.cat(masks)
        packed = F.pad(packed, (0, shape[1] - packed.numel())).unsqueeze(0)
    else:
        raise ValueError(f"unsupported MTP qkv format: {qkv_format}")
    return packed


def _roll(values, *, arguments, roll_tensor, fields, depth):
    return roll_tensor(
        values,
        shifts=-1,
        dims=-1,
        cp_group=arguments.get("cp_group"),
        packed_seq_params=arguments.get("packed_seq_params"),
        roll_context=arguments.get("sequence_roll_context"),
        sequence_fields=fields,
        roll_depth=depth,
    )


def _process_window_loss(arguments, *, window: MTPWindow, roll_tensor):
    config = arguments["config"]
    if not config.mtp_detach_heads:
        raise RuntimeError("Tinker MTP requires detached policy hidden states, embedding and output head")
    states = torch.chunk(arguments["hidden_states"], 1 + config.mtp_num_layers, dim=0)
    policy_hidden = states[0]
    labels, mask = arguments.get("labels"), arguments.get("loss_mask")
    if labels is None:
        if arguments.get("input_ids") is None:
            raise RuntimeError("Tinker MTP needs input_ids to construct future-token labels")
        # Miles full_loss_masks already describes label[i] = input_ids[i+1].
        # Derive labels without shifting that already-aligned mask a second time.
        (labels,) = _roll(
            [arguments["input_ids"]],
            arguments=arguments,
            roll_tensor=roll_tensor,
            fields=["input_ids"],
            depth=0,
        )
    if mask is None:
        raise RuntimeError("Tinker MTP requires an explicit target mask")
    head = arguments["output_layer"]
    weight = arguments.get("output_weight")
    weight = (head.weight if weight is None else weight).detach()
    for depth, draft_hidden in enumerate(states[1:], start=1):
        labels, mask = _roll(
            [labels, mask],
            arguments=arguments,
            roll_tensor=roll_tensor,
            fields=["labels", "loss_mask"],
            depth=depth,
        )
        kwargs = {"weight": weight, "runtime_gather_output": arguments.get("runtime_gather_output")}
        if fused_lm_head.is_active():
            loss = head(draft_hidden, **kwargs, output_cross_entropy_loss=True, labels=labels)
        else:
            logits, _ = head(draft_hidden, **kwargs)
            if arguments.get("scale_logits_fn") is not None:
                logits = arguments["scale_logits_fn"](logits)
            loss = arguments["compute_language_model_loss"](labels, logits)
        loss_sum = (loss * mask).sum()
        window.record(loss_sum, mask.sum())
        draft_loss = config.mtp_loss_scaling_factor * loss_sum
        # FP16 optimizers unscale all parameters together. Preserve their loss
        # scale, but not the schedule's microbatch averaging or client weights.
        if getattr(config, "grad_scale_func", None) is not None:
            draft_loss = config.grad_scale_func(draft_loss)
        policy_hidden = attach_loss(policy_hidden, draft_loss)
    return policy_hidden


@contextmanager
def loss_context(enabled: bool):
    """Adapt GPT's MTP loss only for the enclosed forward/backward call."""
    if not enabled or _PATCHED.get():
        yield
        return
    # Optional runtime dependency: fast CPU tests do not import Megatron.
    from megatron.core.models.gpt import gpt_model
    from megatron.core.transformer.multi_token_prediction import roll_tensor

    original = gpt_model.process_mtp_loss
    signature = inspect.signature(original)
    if "input_ids" not in signature.parameters or "sequence_fields" not in inspect.signature(roll_tensor).parameters:
        raise RuntimeError("MTP needs a Megatron version with input_ids labels and packed MTP sequence rolling")

    def process(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        window = _WINDOW.get()
        if window is not None:
            return _process_window_loss(bound.arguments, window=window, roll_tensor=roll_tensor)
        if fused_lm_head.is_active():
            config = copy.copy(bound.arguments["config"])
            config.cross_entropy_loss_fusion = True
            config.cross_entropy_fusion_impl = "linear"
            bound.arguments["config"] = config
        return original(*bound.args, **bound.kwargs)

    token = _PATCHED.set(True)
    gpt_model.process_mtp_loss = process
    try:
        yield
    finally:
        gpt_model.process_mtp_loss = original
        _PATCHED.reset(token)
