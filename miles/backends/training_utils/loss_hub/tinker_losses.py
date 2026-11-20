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

import functools
import importlib
import numbers
from argparse import Namespace
from collections.abc import Callable

import torch

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


def _metric_tensor(value, reference: torch.Tensor) -> torch.Tensor:
    value = value.detach() if isinstance(value, torch.Tensor) else torch.tensor(float(value))
    return value.reshape(()).to(device=reference.device, dtype=torch.float64)


def _record_per_datum(
    batch: RolloutBatch,
    log_probs: list[torch.Tensor],
    per_sample_loss: list[torch.Tensor],
    per_sample_metrics: list[dict[str, torch.Tensor]] | None = None,
) -> None:
    if _per_datum_outputs is None:
        return
    sample_indices = batch.get("sample_indices")
    assert sample_indices is not None, "per-datum outputs need sample_indices in the batch"
    if not per_sample_loss:
        return
    # one device sync per microbatch instead of two per datum
    flat_log_probs = torch.cat([lp.detach() for lp in log_probs]).cpu()
    losses = torch.stack([loss.detach() for loss in per_sample_loss]).cpu()
    per_datum_log_probs = torch.split(flat_log_probs, [lp.numel() for lp in log_probs])
    metric_rows = None
    if per_sample_metrics:
        # one device sync for all per-datum metrics of the microbatch
        names = sorted({name for metrics in per_sample_metrics for name in metrics})
        table = torch.stack(
            [
                torch.stack([_metric_tensor(metrics.get(name, 0.0), per_sample_loss[0]) for name in names])
                for metrics in per_sample_metrics
            ]
        ).cpu()
        metric_rows = [dict(zip(names, row.tolist(), strict=True)) for row in table]
    for i, (index, log_prob, loss) in enumerate(zip(sample_indices, per_datum_log_probs, losses, strict=True)):
        output = {"sample_index": index, "logprobs": log_prob, "loss": loss}
        if metric_rows is not None:
            output["metrics"] = metric_rows[i]
        _per_datum_outputs.append(output)


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
    log_probs: list[torch.Tensor],
    per_sample_loss: list[torch.Tensor],
    per_sample_metrics: list[dict[str, torch.Tensor]] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    per_sample_loss = _drop_masked_out(batch, per_sample_loss)
    if per_sample_loss:
        loss = torch.stack(per_sample_loss).sum()
    else:
        # a microbatch with no supervised tokens still needs the graph alive; fp32 sum avoids fp16 inf -> nan
        loss = logits.sum(dtype=torch.float32) * 0
    _record_per_datum(batch, log_probs, per_sample_loss, per_sample_metrics)
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


# -------- custom losses (--tinker-custom-loss NAME=module:function) --------
#
# A custom loss is a per-datum function, imported once per process:
#
#     fn(*, logprobs, sampling_logprobs, advantages, loss_mask, config) -> (per_token_loss, metrics)
#
#   logprobs           trainer logprobs of the target tokens, shape [T], carrying the gradient
#   sampling_logprobs  loss_fn_inputs["logprobs"] (the sampler's), shape [T], no gradient
#   advantages         loss_fn_inputs["advantages"], shape [T]
#   loss_mask          1 on the datum's tokens, 0 on the runtime's DP padding rows, shape [T]
#   config             the request's loss_fn_config (a dict)
#   per_token_loss     tensor of shape [T]; the datum's loss is its raw sum, as for the built-in losses
#                      (normalization stays the client's business, via advantages/weights)
#   metrics            {name: number or 0-dim tensor} per datum; summed over the request's datums and
#                      returned in the forward_backward result metrics as ``<name>:sum``


def parse_custom_loss_specs(specs: list[str] | None) -> dict[str, str]:
    """``["name=module:function", ...]`` -> {name: "module:function"}; built-in names are refused."""
    out: dict[str, str] = {}
    for spec in specs or []:
        name, sep, path = spec.partition("=")
        name, path = name.strip(), path.strip()
        if not sep or not name or ":" not in path:
            raise ValueError(f"--tinker-custom-loss {spec!r}: expected NAME=module.path:function")
        if name in TINKER_LOSS_FUNCTIONS:
            raise ValueError(f"--tinker-custom-loss {spec!r}: {name!r} is a built-in Tinker loss")
        if name in out:
            raise ValueError(f"--tinker-custom-loss: {name!r} given twice")
        out[name] = path
    return out


@functools.cache
def load_custom_loss(path: str) -> Callable:
    """Import ``module.path:function`` (once per process)."""
    module_name, _, attr = path.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ValueError(f"custom loss {path!r}: cannot import {module_name!r}: {e}") from e
    fn = getattr(module, attr, None)
    if not callable(fn):
        raise ValueError(f"custom loss {path!r}: {module_name!r} has no callable {attr!r}")
    return fn


def _check_custom_output(path: str, out, logprobs: torch.Tensor) -> tuple[torch.Tensor, dict]:
    if not (isinstance(out, tuple) and len(out) == 2):
        raise TypeError(f"custom loss {path!r} must return (per_token_loss, metrics), got {type(out).__name__}")
    per_token, metrics = out
    if not isinstance(per_token, torch.Tensor) or per_token.shape != logprobs.shape:
        got = tuple(per_token.shape) if isinstance(per_token, torch.Tensor) else type(per_token).__name__
        raise TypeError(f"custom loss {path!r}: per_token_loss must be a tensor of shape {tuple(logprobs.shape)}, got {got}")
    if not isinstance(metrics, dict) or not all(
        isinstance(k, str)
        and (isinstance(v, numbers.Real) or (isinstance(v, torch.Tensor) and v.numel() == 1))
        for k, v in metrics.items()
    ):
        raise TypeError(f"custom loss {path!r}: metrics must be a dict of str -> number or 1-element tensor")
    return per_token, metrics


def custom_loss_function(
    path: str,
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    fn = load_custom_loss(path)
    config = dict(batch.get("loss_fn_config") or {})
    log_probs = _target_logprobs(args, batch, logits)
    per_sample, per_sample_metrics = [], []
    masks = batch.get("loss_masks") or [torch.ones_like(lp) for lp in log_probs]
    for lp, sampling, adv, mask in zip(log_probs, batch["rollout_log_probs"], batch["advantages"], masks, strict=True):
        per_token, metrics = _check_custom_output(
            path,
            fn(
                logprobs=lp,
                sampling_logprobs=_like(sampling, lp),
                advantages=_like(adv, lp),
                loss_mask=_like(mask, lp),
                config=config,
            ),
            lp,
        )
        per_sample.append(per_token.sum())
        per_sample_metrics.append(metrics)
    return _finish(batch, logits, log_probs, per_sample, per_sample_metrics)


def get_tinker_loss_function(args: Namespace, loss_fn: str) -> Callable:
    """A built-in Tinker loss, or one registered with --tinker-custom-loss."""
    if loss_fn in TINKER_LOSS_FUNCTIONS:
        return TINKER_LOSS_FUNCTIONS[loss_fn]
    custom = parse_custom_loss_specs(getattr(args, "tinker_custom_loss", None))
    if loss_fn in custom:
        return functools.partial(custom_loss_function, custom[loss_fn])
    raise ValueError(
        f"unknown Tinker loss_fn {loss_fn!r} (built-in: {sorted(TINKER_LOSS_FUNCTIONS)}, custom: {sorted(custom)})"
    )


TINKER_LOSS_FUNCTIONS = {
    "cross_entropy": cross_entropy_loss_function,
    "importance_sampling": importance_sampling_loss_function,
    "ppo": ppo_loss_function,
    "cispo": cispo_loss_function,
    "dro": dro_loss_function,
}
