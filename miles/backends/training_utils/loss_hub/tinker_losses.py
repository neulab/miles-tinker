"""Tinker protocol losses.

Raw sums over response tokens: normalization is the client's business via
weights/advantages. Under multi-LoRA the loss scaling path (global batch size
1) turns megatron's microbatch/DP averaging into an exact sum, matching the
protocol's "gradient = sum over datums" semantics.

The SDK's forward_backward_custom sends client-side gradients as
``weights = -dL/dlogprob`` with the cross_entropy loss, so cross_entropy is
also the custom-loss backend.

Config keys and defaults follow the Tinker docs.
"""

from argparse import Namespace
from collections.abc import Callable

import torch

from miles.backends.training_utils.loss_hub.logit_processors import get_log_probs_and_entropy, get_values
from miles.backends.training_utils.loss_hub.math_utils import compute_clipped_value_loss
from miles.utils.types import RolloutBatch

PPO_DEFAULTS = {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2, "value_clip": 0.2}
CISPO_DEFAULTS = {"clip_low_threshold": 0.0, "clip_high_threshold": 4.0}
DRO_DEFAULTS = {"beta": 0.05}

# Per-datum outputs collected across microbatches for protocol reassembly;
# active only between start/drain (the slot executor's forward_backward).
_per_datum_outputs: list[dict] | None = None


def start_per_datum_outputs() -> None:
    global _per_datum_outputs
    _per_datum_outputs = []


def drain_per_datum_outputs() -> list[dict]:
    global _per_datum_outputs
    outputs, _per_datum_outputs = _per_datum_outputs, None
    return outputs or []


def _record_per_datum(
    batch: RolloutBatch,
    predictions: list[torch.Tensor],
    per_sample_loss: list[torch.Tensor],
    output_key: str = "logprobs",
) -> None:
    if _per_datum_outputs is None:
        return
    sample_indices = batch.get("sample_indices")
    assert sample_indices is not None, "per-datum outputs need sample_indices in the batch"
    if not per_sample_loss:
        return
    # one device sync per microbatch instead of two per datum
    flat_predictions = torch.cat([prediction.detach() for prediction in predictions]).cpu()
    losses = torch.stack([loss.detach() for loss in per_sample_loss]).cpu()
    per_datum_predictions = torch.split(flat_predictions, [prediction.numel() for prediction in predictions])
    for index, prediction, loss in zip(sample_indices, per_datum_predictions, losses, strict=True):
        _per_datum_outputs.append({"sample_index": index, output_key: prediction, "loss": loss})


def _target_logprobs(args: Namespace, batch: RolloutBatch, logits: torch.Tensor) -> list[torch.Tensor]:
    outputs = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        with_entropy=False,
        max_seq_lens=batch.get("max_seq_lens", None),
    )
    return outputs["log_probs"]


def _like(values, reference: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(values, dtype=reference.dtype, device=reference.device)


def _drop_masked_out(batch: RolloutBatch, per_sample_loss: list[torch.Tensor]) -> list[torch.Tensor]:
    """A datum whose loss mask is all zero (the runtime's DP padding rows) contributes
    nothing under any loss, including terms that do not go through weights/advantages
    (the DRO divergence penalty)."""
    masks = batch.get("loss_masks")
    if masks is None or not per_sample_loss:
        return per_sample_loss
    return [loss * mask.any() for loss, mask in zip(per_sample_loss, masks, strict=True)]


def _finish(
    batch: RolloutBatch,
    logits: torch.Tensor,
    predictions: list[torch.Tensor],
    per_sample_loss: list[torch.Tensor],
    *,
    output_key: str = "logprobs",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    per_sample_loss = _drop_masked_out(batch, per_sample_loss)
    if per_sample_loss:
        loss = torch.stack(per_sample_loss).sum()
    else:
        # a microbatch with no supervised tokens still needs the graph alive; fp32 sum avoids fp16 inf -> nan
        loss = logits.sum(dtype=torch.float32) * 0
    _record_per_datum(batch, predictions, per_sample_loss, output_key=output_key)
    return loss, {"loss": loss.clone().detach()}


def cross_entropy_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    log_probs = _target_logprobs(args, batch, logits)
    per_sample = [
        -(_like(weights, lp) * lp).sum() for lp, weights in zip(log_probs, batch["loss_weights"], strict=True)
    ]
    return _finish(batch, logits, log_probs, per_sample)


def importance_sampling_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    log_probs = _target_logprobs(args, batch, logits)
    per_sample = []
    for lp, sampling, adv in zip(log_probs, batch["rollout_log_probs"], batch["advantages"], strict=True):
        ratio = torch.exp(lp - _like(sampling, lp))
        per_sample.append(-(ratio * _like(adv, lp)).sum())
    return _finish(batch, logits, log_probs, per_sample)


def ppo_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if batch.get("model_role", "policy") == "critic":
        return ppo_value_loss_function(args, batch, logits, sum_of_sample_mean)
    config = batch.get("loss_fn_config") or {}
    clip_low = config.get("clip_low_threshold", PPO_DEFAULTS["clip_low_threshold"])
    clip_high = config.get("clip_high_threshold", PPO_DEFAULTS["clip_high_threshold"])
    log_probs = _target_logprobs(args, batch, logits)
    per_sample = []
    for lp, sampling, adv in zip(log_probs, batch["rollout_log_probs"], batch["advantages"], strict=True):
        ratio = torch.exp(lp - _like(sampling, lp))
        advantages = _like(adv, lp)
        objective = torch.minimum(ratio * advantages, torch.clamp(ratio, clip_low, clip_high) * advantages)
        per_sample.append(-objective.sum())
    return _finish(batch, logits, log_probs, per_sample)


def ppo_value_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Native PPO value clipping with the SDK's client-weighted sum reduction."""
    values = get_values(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens"),
    )["values"]
    clip = (batch.get("loss_fn_config") or {}).get("value_clip", PPO_DEFAULTS["value_clip"])
    per_sample = []
    for value, old, target, weights in zip(
        values, batch["values"], batch["returns"], batch["loss_weights"], strict=True
    ):
        error = compute_clipped_value_loss(value, _like(old, value).detach(), _like(target, value).detach(), clip)
        per_sample.append((error * _like(weights, value).detach()).sum())
    return _finish(batch, logits, values, per_sample, output_key="values")


def cispo_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    config = batch.get("loss_fn_config") or {}
    clip_low = config.get("clip_low_threshold", CISPO_DEFAULTS["clip_low_threshold"])
    clip_high = config.get("clip_high_threshold", CISPO_DEFAULTS["clip_high_threshold"])
    log_probs = _target_logprobs(args, batch, logits)
    per_sample = []
    for lp, sampling, adv in zip(log_probs, batch["rollout_log_probs"], batch["advantages"], strict=True):
        ratio = torch.exp(lp - _like(sampling, lp))
        coefficient = torch.clamp(ratio, clip_low, clip_high).detach()
        per_sample.append(-(coefficient * lp * _like(adv, lp)).sum())
    return _finish(batch, logits, log_probs, per_sample)


def dro_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    config = batch.get("loss_fn_config") or {}
    beta = config.get("beta", DRO_DEFAULTS["beta"])
    log_probs = _target_logprobs(args, batch, logits)
    per_sample = []
    for lp, sampling, adv in zip(log_probs, batch["rollout_log_probs"], batch["advantages"], strict=True):
        divergence = lp - _like(sampling, lp)
        objective = lp * _like(adv, lp) - 0.5 * beta * divergence**2
        per_sample.append(-objective.sum())
    return _finish(batch, logits, log_probs, per_sample)


TINKER_LOSS_FUNCTIONS = {
    "cross_entropy": cross_entropy_loss_function,
    "importance_sampling": importance_sampling_loss_function,
    "ppo": ppo_loss_function,
    "cispo": cispo_loss_function,
    "dro": dro_loss_function,
}
