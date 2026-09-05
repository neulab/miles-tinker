"""Per-step AdamParams application shared by the Tinker executors.

Tinker ships the optimizer hyperparameters with every optim_step and there
is no scheduler, so the values are written straight into the Megatron param
groups right before the step. Kept torch-only so it is testable without
Megatron.
"""

import math
from collections.abc import Iterable

import torch


def apply_adam_params(param_groups: Iterable[dict], adam_params: dict) -> None:
    """Write lr/betas/eps/weight_decay into ``param_groups``.

    Honors Megatron's per-group multipliers the way its scheduler would:
    ``wd_mult`` (0.0 for norms and biases) and ``lr_mult``. FusedAdam may
    hold ``lr`` as a tensor; it is updated in place so the kernel sees it.
    AdamParams is materialized at the boundary; a missing key is an encoder bug.
    """
    for group in param_groups:
        lr = adam_params["learning_rate"] * group.get("lr_mult", 1.0)
        if torch.is_tensor(group.get("lr")):
            group["lr"].fill_(lr)
        else:
            group["lr"] = lr
        group["betas"] = (adam_params["beta1"], adam_params["beta2"])
        group["eps"] = adam_params["eps"]
        group["weight_decay"] = adam_params["weight_decay"] * group.get("wd_mult", 1.0)


def set_clip_grad(optimizer, grad_clip_norm: float) -> None:
    """Point this step's clipping at Tinker's ``grad_clip_norm`` (0.0 = no clipping).
    ChainedOptimizer and its children share one OptimizerConfig; set it on all
    of them so the value holds whichever object a step reads."""
    for target in (optimizer, *getattr(optimizer, "chained_optimizers", ())):
        config = getattr(target, "config", None)
        if config is not None:
            config.clip_grad = float(grad_clip_norm)


def should_veto_step(found_inf: bool, grad_norm: float) -> bool:
    """A non-finite gradient must not reach the weights (or the sampler)."""
    return bool(found_inf) or not math.isfinite(float(grad_norm))
