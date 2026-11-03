"""Tinker protocol losses.

Raw sums over response tokens: normalization is the client's business via
weights/advantages. Under multi-LoRA the loss scaling path (global batch size
1) turns megatron's microbatch/DP averaging into an exact sum, matching the
protocol's "gradient = sum over datums" semantics.

The SDK's forward_backward_custom sends client-side gradients as
``weights = -dL/dlogprob`` with the cross_entropy loss, so cross_entropy is
also the custom-loss backend.

Config keys and defaults follow the Tinker docs.

Context parallelism (ported from radixark/miles 3d8b5284, #3320): under CP>1
each rank scores its zigzag shard of every datum, so the full-length client
vectors (weights, advantages, masks) are sliced to that shard per call (never
in place: a recomputed loss must see the full batch again); rollout_log_probs
arrive pre-sliced from get_rollout_data. The loss stays a sum over the local
shard (the DP*CP gradient reduction sums the shards) and the per-datum
outputs are reassembled across the CP group, so every rank reports the
full-length logprobs and whole-datum losses.
"""

from argparse import Namespace
from collections.abc import Callable

import torch

from miles.backends.training_utils.cp_utils import (
    gather_cp_response_values,
    get_local_response_loss_masks,
    slice_log_prob_with_cp,
)
from miles.backends.training_utils.loss_hub.logit_processors import get_log_probs_and_entropy
from miles.utils.types import RolloutBatch

PPO_DEFAULTS = {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2}
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


# Set on a microbatch's batch dict once its outputs are recorded: a recomputed loss
# (--recompute-loss-function) re-runs the function in backward and must neither
# record twice nor issue an extra CP collective that only some ranks reach.
_RECORDED_KEY = "_tinker_per_datum_recorded"


def _gather_per_datum_outputs(
    args: Namespace,
    batch: RolloutBatch,
    predictions: list[torch.Tensor],
    losses: list[torch.Tensor],
    output_key: str = "logprobs",
) -> list[dict]:
    """Per-datum outputs for the client: full-response ``output_key`` vectors and
    whole-datum losses, identical on every CP rank. One batched CP all-reduce per
    microbatch carries both (and one device sync); at CP 1 it is a plain copy."""
    if not losses:
        return []
    full_predictions, full_losses = gather_cp_response_values(
        predictions,
        batch["total_lengths"],
        batch["response_lengths"],
        _qkv_format(args),
        batch.get("max_seq_lens", None),
        scalars=losses,
    )
    return [
        {"sample_index": index, output_key: prediction, "loss": loss}
        for index, prediction, loss in zip(batch["sample_indices"], full_predictions, full_losses, strict=True)
    ]


def _record_per_datum(
    args: Namespace,
    batch: RolloutBatch,
    log_probs: list[torch.Tensor],
    per_sample_loss: list[torch.Tensor],
    output_key: str = "logprobs",
) -> None:
    if _per_datum_outputs is None or batch.get(_RECORDED_KEY):
        return
    assert batch.get("sample_indices") is not None, "per-datum outputs need sample_indices in the batch"
    _per_datum_outputs.extend(_gather_per_datum_outputs(args, batch, log_probs, per_sample_loss, output_key))
    batch[_RECORDED_KEY] = True


def _qkv_format(args: Namespace) -> str:
    return getattr(args, "qkv_format", "thd")


def _local_response_values(args: Namespace, batch: RolloutBatch, values: list) -> list:
    """This rank's zigzag CP shard of each full-length client vector (the vectors themselves at CP 1)."""
    max_seq_lens = batch.get("max_seq_lens", None)
    return [
        slice_log_prob_with_cp(
            value,
            total_length,
            response_length,
            _qkv_format(args),
            max_seq_lens[i] if max_seq_lens is not None else None,
        )
        for i, (value, total_length, response_length) in enumerate(
            zip(values, batch["total_lengths"], batch["response_lengths"], strict=True)
        )
    ]


def _local_response_masks(args: Namespace, batch: RolloutBatch, log_probs: list[torch.Tensor]) -> list[torch.Tensor]:
    """This rank's shard of each loss mask, in the logprobs' dtype (all ones for a real datum)."""
    local_masks = get_local_response_loss_masks(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        _qkv_format(args),
        batch.get("max_seq_lens", None),
    )
    return [_like(mask, log_prob) for mask, log_prob in zip(local_masks, log_probs, strict=True)]


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
    (the DRO divergence penalty). The gate reads the full-length mask: under CP a real
    datum's shard can be empty on a rank, which is not padding."""
    masks = batch.get("loss_masks")
    if masks is None or not per_sample_loss:
        return per_sample_loss
    return [loss * mask.any() for loss, mask in zip(per_sample_loss, masks, strict=True)]


def _finish(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    log_probs: list[torch.Tensor],
    per_sample_loss: list[torch.Tensor],
    *,
    output_key: str = "logprobs",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    per_sample_loss = _drop_masked_out(batch, per_sample_loss)
    if any(log_prob.numel() for log_prob in log_probs):
        loss = torch.stack(per_sample_loss).sum()
    else:
        # No response token on this rank (a CP shard of short datums, or no datum at all): the
        # unfused head's empty logprobs are detached, so take a zero that is still connected to the
        # logits (or the fused head's hidden states) and run the same backward, with its TP/CP
        # collectives, as the other ranks. The fp32 sum of an empty slice cannot overflow.
        loss = logits[..., :0].sum(dtype=torch.float32)
    _record_per_datum(args, batch, log_probs, per_sample_loss, output_key=output_key)
    return loss, {"loss": loss.clone().detach()}


def cross_entropy_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    log_probs = _target_logprobs(args, batch, logits)
    per_sample = [
        -(_like(weights, lp) * lp * mask).sum()
        for lp, weights, mask in zip(
            log_probs,
            _local_response_values(args, batch, batch["loss_weights"]),
            _local_response_masks(args, batch, log_probs),
            strict=True,
        )
    ]
    return _finish(args, batch, logits, log_probs, per_sample)


def importance_sampling_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    log_probs = _target_logprobs(args, batch, logits)
    per_sample = []
    for lp, sampling, adv, mask in zip(
        log_probs,
        batch["rollout_log_probs"],
        _local_response_values(args, batch, batch["advantages"]),
        _local_response_masks(args, batch, log_probs),
        strict=True,
    ):
        ratio = torch.exp(lp - _like(sampling, lp))
        per_sample.append(-(ratio * _like(adv, lp) * mask).sum())
    return _finish(args, batch, logits, log_probs, per_sample)


def ppo_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    config = batch.get("loss_fn_config") or {}
    clip_low = config.get("clip_low_threshold", PPO_DEFAULTS["clip_low_threshold"])
    clip_high = config.get("clip_high_threshold", PPO_DEFAULTS["clip_high_threshold"])
    log_probs = _target_logprobs(args, batch, logits)
    per_sample = []
    for lp, sampling, adv, mask in zip(
        log_probs,
        batch["rollout_log_probs"],
        _local_response_values(args, batch, batch["advantages"]),
        _local_response_masks(args, batch, log_probs),
        strict=True,
    ):
        ratio = torch.exp(lp - _like(sampling, lp))
        advantages = _like(adv, lp)
        objective = torch.minimum(ratio * advantages, torch.clamp(ratio, clip_low, clip_high) * advantages)
        per_sample.append(-(objective * mask).sum())
    return _finish(args, batch, logits, log_probs, per_sample)


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
    for lp, sampling, adv, mask in zip(
        log_probs,
        batch["rollout_log_probs"],
        _local_response_values(args, batch, batch["advantages"]),
        _local_response_masks(args, batch, log_probs),
        strict=True,
    ):
        ratio = torch.exp(lp - _like(sampling, lp))
        coefficient = torch.clamp(ratio, clip_low, clip_high).detach()
        per_sample.append(-(coefficient * lp * _like(adv, lp) * mask).sum())
    return _finish(args, batch, logits, log_probs, per_sample)


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
    for lp, sampling, adv, mask in zip(
        log_probs,
        batch["rollout_log_probs"],
        _local_response_values(args, batch, batch["advantages"]),
        _local_response_masks(args, batch, log_probs),
        strict=True,
    ):
        divergence = lp - _like(sampling, lp)
        objective = lp * _like(adv, lp) - 0.5 * beta * divergence**2
        per_sample.append(-(objective * mask).sum())
    return _finish(args, batch, logits, log_probs, per_sample)


TINKER_LOSS_FUNCTIONS = {
    "cross_entropy": cross_entropy_loss_function,
    "importance_sampling": importance_sampling_loss_function,
    "ppo": ppo_loss_function,
    "cispo": cispo_loss_function,
    "dro": dro_loss_function,
}
