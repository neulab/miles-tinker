"""The frozen reference policy behind the Tinker KL penalty (--tinker-kl-coef).

Full fine-tuning serves one set of weights: the rollout engines hold the latest
push, so a Tinker sampling client "pinned to the base model" scores with the
current policy, not the one training started from. The reference therefore has
to live in the trainer. At startup the starting weights are copied to pinned CPU
memory under the "ref" tag; for every forward_backward unit the current weights
are parked under "actor", the reference is swapped in for one forward-only pass
that scores the unit's tokens, and the current weights come back before the
unit's own forward/backward runs. Gradients are untouched: they live in the
grad buffers, not the parameters, and the swap happens between passes.

Nothing here talks to the gateway or creates clients, so it cannot repeat the
client-side reference's leak (one ServiceClient per rollout, job 155424).

The "actor" copy is refreshed only when the weights changed since it was taken
(an optimizer step or a checkpoint load), not once per unit: within an optim
window every unit sees the same weights.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

REF_TAG = "ref"
ACTOR_TAG = "actor"


class ReferencePolicy:
    """Swap the frozen starting weights in for one scoring pass at a time."""

    def __init__(self, backuper: Any, set_active_tag: Callable[[str], None] | None = None) -> None:
        self._backuper = backuper
        self._set_active_tag = set_active_tag or (lambda _tag: None)
        backuper.backup(REF_TAG)
        self._actor_backup_current = False

    def weights_changed(self) -> None:
        """The live weights moved (optimizer step, checkpoint load): re-park them before the next swap."""
        self._actor_backup_current = False

    def score(self, compute: Callable[[], Any]) -> Any:
        """Run ``compute`` with the reference weights loaded, then restore the live weights."""
        if not self._actor_backup_current:
            self._backuper.backup(ACTOR_TAG)
            self._actor_backup_current = True
        self._backuper.restore(REF_TAG)
        self._set_active_tag(REF_TAG)
        try:
            return compute()
        finally:
            self._backuper.restore(ACTOR_TAG)
            self._set_active_tag(ACTOR_TAG)


def _cpu_vector(values: Any) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.detach().float().cpu().reshape(-1)
    return torch.as_tensor(values, dtype=torch.float32).reshape(-1)


def sampled_kl_to_reference(
    sampled_log_probs: Sequence[Sequence[float]],
    reference_log_probs: Sequence[torch.Tensor],
    weights: Sequence[Sequence[float]] | None,
) -> dict[str, float]:
    """KL(sampler || reference) estimated on the unit's own trained tokens.

    Uses the sampler's logprobs, so it measures how far the policy that produced
    the data has drifted from the reference -- the quantity the penalty bounds --
    without needing the current trainer logprobs. Trained tokens are the ones
    with a positive weight (prompt and tool-output tokens carry 0).
    """
    k1_sum = k3_sum = 0.0
    tokens = 0
    for i, (sampled, reference) in enumerate(zip(sampled_log_probs, reference_log_probs, strict=True)):
        # The trainer's batch holds these on the GPU and the reference scores on
        # the CPU (or either may be a plain list): compare everything on the CPU.
        sampled_t = _cpu_vector(sampled)
        reference_t = _cpu_vector(reference)
        n = min(len(sampled_t), len(reference_t))
        sampled_t, reference_t = sampled_t[:n], reference_t[:n]
        if weights is not None:
            mask = _cpu_vector(weights[i])[:n] > 0
        else:
            mask = torch.ones(n, dtype=torch.bool)
        if not mask.any():
            continue
        log_ratio = (sampled_t - reference_t)[mask]
        k1_sum += float(log_ratio.sum())
        k3_sum += float(((-log_ratio).exp() - 1 + log_ratio).sum())
        tokens += int(mask.sum())
    return {
        "kl_ref_k1": k1_sum / tokens if tokens else 0.0,
        "kl_ref_k3": k3_sum / tokens if tokens else 0.0,
        "kl_ref_tokens": float(tokens),
    }
