"""Model-config side of context parallelism: bridge provider fixups and post-build invariants.

Megatron's DDP (``gradient_scaling_factor``) and pipeline schedule (division by the
number of microbatches) read ``calculate_per_token_loss`` from the model config,
while miles' loss scaling (``loss.py``: ``loss * num_microbatches * |DP*CP|``) reads
args. The Qwen3-VL / Qwen3.5-VL megatron-bridge providers force the config flag on
at CP>1 (``finalize()``) and assert it in ``Qwen3VLModel.__init__``, so a bridge
model trained at CP>1 silently gets every gradient ``num_microbatches * |DP*CP|``
times too large. :func:`restore_per_token_loss` resets the flag on the built model
(after the bridge's own assert, before the DDP wrap) and
:func:`check_loss_scaling_invariants` checks the outcome on the DDP-wrapped chunks.

No Megatron import at module level: the CPU fast tests drive these with fakes.
"""

import argparse
import logging
import math
from collections.abc import Sequence

logger = logging.getLogger(__name__)

# where megatron-bridge defines Qwen3VLModel (Qwen3-VL and Qwen3.5/3.6-VL providers build it)
_QWEN_VL_MODEL_MODULE_SUFFIX = "modelling_qwen3_vl.model"


def apply_context_parallel_modes(provider, args: argparse.Namespace) -> None:
    """Copy the CP execution modes onto a bridge provider: ``cp_comm_type`` for the attention
    layers (Megatron's ``--cp-comm-type``; a one-element list collapses to its string, as in
    ``core_transformer_config_from_args``) and ``linear_cp_mode`` for linear-attention
    (GatedDeltaNet) layers (``--linear-cp-mode``)."""
    cp_comm_type = getattr(args, "cp_comm_type", None)
    if cp_comm_type is not None:
        if isinstance(cp_comm_type, (list, tuple)) and len(cp_comm_type) == 1:
            cp_comm_type = cp_comm_type[0]
        provider.cp_comm_type = cp_comm_type
    linear_cp_mode = getattr(args, "linear_cp_mode", None)
    if linear_cp_mode is not None:
        if not hasattr(provider, "linear_cp_mode"):
            raise ValueError(
                f"--linear-cp-mode {linear_cp_mode}: {type(provider).__name__} has no linear_cp_mode "
                "(this Megatron/bridge build has no linear-attention CP modes)"
            )
        provider.linear_cp_mode = linear_cp_mode


def restore_per_token_loss(model, args: argparse.Namespace) -> int:
    """Set ``calculate_per_token_loss`` back to the args value on every config of a built model,
    and on the copy MoE routers cache at construction (``router.calculate_per_token_loss``).
    Returns the number of values changed."""
    wanted = bool(args.calculate_per_token_loss)
    changed = 0
    seen: set[int] = set()
    for module in model.modules() if hasattr(model, "modules") else [model]:
        config = getattr(module, "config", None)
        if config is not None and id(config) not in seen and hasattr(config, "calculate_per_token_loss"):
            seen.add(id(config))
            if config.calculate_per_token_loss != wanted:
                config.calculate_per_token_loss = wanted
                changed += 1
        if "calculate_per_token_loss" in vars(module) and module.calculate_per_token_loss != wanted:
            module.calculate_per_token_loss = wanted
            changed += 1
    return changed


def check_qwen_vl_mrope_patch(model, args: argparse.Namespace) -> None:
    """At CP>1 a bridge Qwen3-VL/3.5-VL model receives miles' pre-sharded zigzag THD rows, and only
    miles' patch (``miles_plugins/models/qwen3_vl.py``) gives its MRoPE rank-local positions. The
    patch installs best-effort at import (a warning on failure); make its absence fatal here."""
    if args.context_parallel_size <= 1:
        return
    bridge_classes = [cls for cls in type(model).__mro__ if cls.__module__.endswith(_QWEN_VL_MODEL_MODULE_SUFFIX)]
    if not bridge_classes:
        return
    from miles_plugins.models.qwen3_vl import is_packed_mrope_patch_installed

    if not is_packed_mrope_patch_installed(bridge_classes[0]):
        raise RuntimeError(
            f"{type(model).__name__} at --context-parallel-size {args.context_parallel_size} needs miles' packed "
            "MRoPE patch (miles_plugins.models.qwen3_vl.install_qwen3_vl_packed_mrope_patch), which is not "
            "installed: look for 'THD packed mRoPE patch not applied' in the trainer log"
        )


def check_loss_scaling_invariants(model_chunks: Sequence, args: argparse.Namespace) -> None:
    """After the DDP wrap: the config Megatron reads agrees with args, and without per-token loss
    every DDP grad buffer (dense and expert) pre-scales by ``1 / |DP*CP|`` (the factor the loss
    scaling's ``* |DP*CP|`` cancels). With rollout routing replay, the router must take the unfused
    top-k: the fused kernel returns before the replay hook."""
    for chunk in model_chunks:
        config = _chunk_config(chunk)
        assert bool(config.calculate_per_token_loss) == bool(args.calculate_per_token_loss), (
            f"model config calculate_per_token_loss={config.calculate_per_token_loss} disagrees with "
            f"--calculate-per-token-loss={args.calculate_per_token_loss}: Megatron's DDP and schedule would scale "
            "the gradients differently from miles' loss"
        )
        if getattr(args, "use_rollout_routing_replay", False):
            assert not getattr(config, "moe_router_fusion", False), (
                "--use-rollout-routing-replay needs the unfused MoE router: with moe_router_fusion the fused top-k "
                "returns before the replay hook and the rollout routing is silently ignored"
            )
        if config.calculate_per_token_loss:
            continue
        _check_ddp_scaling(chunk)


def _chunk_config(chunk):
    """The TransformerConfig of a (possibly DDP / Float16Module wrapped) model chunk."""
    current = chunk
    for _ in range(8):
        config = getattr(current, "config", None)
        if config is not None:
            return config
        current = getattr(current, "module", None)
        if current is None:
            break
    raise AttributeError(f"no config on model chunk {type(chunk).__name__}")


def _check_ddp_scaling(chunk) -> None:
    ddp_config = getattr(chunk, "ddp_config", None)
    buffers = [*getattr(chunk, "buffers", []), *getattr(chunk, "expert_parallel_buffers", [])]
    if ddp_config is None or not buffers or getattr(ddp_config, "average_in_collective", False):
        return  # not Megatron's DDP, or the collective averages (factors 1 and edp/dp by design)
    dp_cp_size = chunk.dp_cp_group.size()
    for buffer in buffers:
        assert math.isclose(buffer.gradient_scaling_factor, 1.0 / dp_cp_size, rel_tol=1e-9), (
            f"DDP grad buffer gradient_scaling_factor={buffer.gradient_scaling_factor}, expected 1/{dp_cp_size}: "
            "gradients would not be the exact DP*CP sum the Tinker losses rely on"
        )
