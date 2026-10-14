"""Model-config side of context parallelism: bridge provider fixups and post-build invariants.

Megatron's DDP (``gradient_scaling_factor``), pipeline schedule (division by the
number of microbatches) and ``finalize_model_grads`` (division by the summed loss
normalizer) read ``calculate_per_token_loss`` from the model config, while miles'
loss scaling (``loss.py``: ``loss * num_microbatches * |DP*CP|``) reads args. The
Qwen3-VL / Qwen3.5-VL megatron-bridge providers force the config flag on at CP>1
(``finalize()``) and assert it in ``Qwen3VLModel.__init__``:

* Through the Tinker gateway, whose deferred finalize passes no token count
  (``full/executor.py``), every gradient came out ``num_microbatches * |DP*CP|``
  times too large.
* Elsewhere the schedule hands ``finalize_model_grads`` the summed normalizer
  (``num_microbatches`` ones, all-reduced over DP*CP), which cancels that for the
  main loss; the MoE aux/z-loss and the MTP loss, however, were scaled per token
  (``router.py``, ``schedules.py``), roughly tokens/CP times their per-microbatch
  scale.

:func:`restore_per_token_loss` resets the flag on the built model (after the
bridge's own assert, before the DDP wrap), so every entry point trains with the
scaling args ask for, and :func:`check_loss_scaling_invariants` checks the outcome
on the DDP-wrapped chunks.

No Megatron import at module level: the CPU fast tests drive these with fakes.
"""

import argparse
import logging
from collections.abc import Sequence

from miles.backends.training_utils.parallel import get_parallel_state

logger = logging.getLogger(__name__)

# where megatron-bridge defines Qwen3VLModel (Qwen3-VL and Qwen3.5/3.6-VL providers build it)
_QWEN_VL_MODEL_MODULE_SUFFIX = "modelling_qwen3_vl.model"

# Megatron's --cp-comm-type default. It fills a provider without a type of its own but does not
# override one the provider chose (megatron-bridge's Mistral provider picks a2a for sliding windows).
_MEGATRON_DEFAULT_CP_COMM_TYPE = ("p2p",)


def apply_context_parallel_modes(provider, args: argparse.Namespace) -> None:
    """At ``--context-parallel-size`` > 1, copy the CP execution modes onto a bridge provider.

    * ``cp_comm_type`` (attention layers, Megatron's ``--cp-comm-type``; a one-element list
      collapses to its string, as in ``core_transformer_config_from_args``): Megatron's default
      ``p2p`` only fills a provider whose own value is None, any other value overrides it.
    * ``linear_cp_mode`` (linear-attention / GatedDeltaNet layers, ``--linear-cp-mode``): only when
      given; unset, the provider keeps its own (Megatron's TransformerConfig default, chunkwise).

    At CP=1 neither mode applies and the provider is left as built."""
    cp_size = getattr(args, "context_parallel_size", 1)
    if cp_size <= 1:
        return
    cp_comm_type = getattr(args, "cp_comm_type", None)
    if cp_comm_type is not None:
        requested = tuple(cp_comm_type) if isinstance(cp_comm_type, (list, tuple)) else (cp_comm_type,)
        if getattr(provider, "cp_comm_type", None) is None or requested != _MEGATRON_DEFAULT_CP_COMM_TYPE:
            provider.cp_comm_type = requested[0] if len(requested) == 1 else list(requested)
    linear_cp_mode = getattr(args, "linear_cp_mode", None)
    if linear_cp_mode is not None:
        if not hasattr(provider, "linear_cp_mode"):
            raise ValueError(
                f"--linear-cp-mode {linear_cp_mode}: {type(provider).__name__} has no linear_cp_mode "
                "(this Megatron/bridge build has no linear-attention CP modes)"
            )
        provider.linear_cp_mode = linear_cp_mode
    comm_type, linear_mode = getattr(provider, "cp_comm_type", None), getattr(provider, "linear_cp_mode", None)
    logger.info(
        f"bridge CP modes at --context-parallel-size {cp_size}: cp_comm_type={comm_type}, linear_cp_mode={linear_mode}"
    )


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
    patch installs best-effort at import (a warning on failure); make its absence fatal here. Each
    forward on a pre-sharded row then checks that the patch built and handed over the positions."""
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


def check_per_token_loss_flag(config, args: argparse.Namespace) -> None:
    """The model config Megatron's DDP, schedule and finalize read agrees with the args miles'
    loss scaling reads. Cheap: the schedule re-reads the config on every step, so callers that
    run a step re-check it (``full/executor.forward_backward``)."""
    assert bool(config.calculate_per_token_loss) == bool(args.calculate_per_token_loss), (
        f"model config calculate_per_token_loss={config.calculate_per_token_loss} disagrees with "
        f"--calculate-per-token-loss={args.calculate_per_token_loss}: Megatron's DDP and schedule would scale "
        "the gradients differently from miles' loss"
    )


def check_loss_scaling_invariants(model_chunks: Sequence, args: argparse.Namespace) -> None:
    """After the DDP wrap: the config flag agrees with args (:func:`check_per_token_loss_flag`), and
    without per-token loss each DDP chunk reduces over the same DP*CP group whose size ``loss.py``
    multiplies the loss by (``intra_dp_cp``): DDP pre-scales by ``1 / dp_cp_group.size()`` (its own
    construction asserts that factor), so the two cancel to the exact DP*CP sum the Tinker losses
    rely on. With rollout routing replay, the router must take the unfused top-k: the fused kernel
    returns before the replay hook."""
    for chunk in model_chunks:
        config = _chunk_config(chunk)
        check_per_token_loss_flag(config, args)
        if getattr(args, "use_rollout_routing_replay", False):
            assert not getattr(config, "moe_router_fusion", False), (
                "--use-rollout-routing-replay needs the unfused MoE router: with moe_router_fusion the fused top-k "
                "returns before the replay hook and the rollout routing is silently ignored"
            )
        if not config.calculate_per_token_loss:
            _check_reduction_group(chunk)


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


def _check_reduction_group(chunk) -> None:
    dp_cp_group = getattr(chunk, "dp_cp_group", None)
    if dp_cp_group is None:
        return  # not Megatron's DDP
    loss_scale_size = get_parallel_state().intra_dp_cp.size
    assert dp_cp_group.size() == loss_scale_size, (
        f"DDP reduces over a DP*CP group of {dp_cp_group.size()} ranks (and pre-scales by its inverse) while "
        f"loss.py multiplies the loss by intra_dp_cp={loss_scale_size}: gradients would not be the exact DP*CP sum "
        "the Tinker losses rely on"
    )
