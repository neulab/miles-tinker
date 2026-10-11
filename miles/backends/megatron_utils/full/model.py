"""Command-grained full fine-tuning over the plain Megatron model + distributed optimizer.

The Tinker gateway's ``--tinker-train-mode full``: one model (slot 0) trains the base weights.

* Deferred finalize. ``forward_backward`` runs Megatron's schedule with
  ``config.finalize_model_grads_func`` replaced by a no-op that only records
  "pending". With overlap_grad_reduce off, that hook is the *only* place the
  DP reduce-scatter and the sequence-parallel / tied-embedding SUM all-reduces
  happen, so the fp32 ``main_grad`` buffers keep accumulating locally and
  linearly across calls (DDP's ``zero_grad_buffer`` is never called between
  batches; only ``reset_grad_metadata_keep_grads`` resets bookkeeping). The
  real ``finalize_model_grads`` runs exactly once, at ``optim_step``. Per-call
  finalize would be wrong: the distributed optimizer's reduce-scatter and the
  SP-layernorm sums are not idempotent. ``overlap_grad_reduce`` and
  ``overlap_param_gather`` must stay off (asserted at serve validation).
* Loss scaling. A batch carrying a Tinker ``loss_fn`` is an exact datum SUM
  (loss/objective.py); the DDP ``1/dp`` pre-scale at the single deferred
  reduce-scatter is cancelled by the loss's ``* dp_cp_size``. Padding datums
  (zero loss mask) add nothing.
* AdamParams arrive per optim_step (no scheduler): lr/betas/eps/weight_decay
  (times Megatron's ``wd_mult``/``lr_mult``) are written into the param groups
  and ``grad_clip_norm`` becomes ``OptimizerConfig.clip_grad`` (0 = no clip).
  A non-finite gradient norm vetoes the step: grads are zeroed, ``skipped=1``.
* Latest-only sampler. The gateway publishes sampler weights by pushing the
  base weights to the engines (``actor.update_weights``); engines hold one copy,
  so every sampler version of the model samples the newest push.
* Checkpoints. ``save_state`` writes a Megatron torch_dist checkpoint
  (model + optimizer, synchronous, fixed iteration 1) plus the gateway's
  ``META.json`` under the tinker checkpoint path; ``load_state`` reads it back
  with a temporary ``args.load`` (optimizer optional; a weights-only load resets
  Adam state so the semantics equal "fresh optimizer"). Resume needs the same
  parallel topology. ``reset_to_base`` reloads the weights the server started
  from (``args.load``) with fresh optimizer state, and is a no-op while the
  weights are still those.
"""

import logging
from argparse import Namespace
from collections.abc import Sequence
from contextlib import nullcontext

import torch.distributed as dist
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.optimizer import MegatronOptimizer

from miles.backends.megatron_utils import tinker_perf
from miles.backends.megatron_utils.adam_params import apply_adam_params, set_clip_grad, should_veto_step
from miles.backends.megatron_utils.checkpoint import load_checkpoint
from miles.backends.megatron_utils.lora.optimizer import reset_grad_metadata_keep_grads
from miles.backends.megatron_utils.model import (
    _has_loadable_ckpt,
    _zero_grads,
    finalize_model_grads_with_empty_cache,
    run_forward_backward_pass,
    save,
    setup_train_iteration_config,
)
from miles.backends.megatron_utils.optimizer_utils import reset_optimizer_state
from miles.backends.training_utils.checkpoint.io import write_checkpoint_dir
from miles.backends.training_utils.data.rollout import get_data_iterator
from miles.backends.training_utils.metrics.log_utils import aggregate_train_losses
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.argparse_utils import inplace_modify_args
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.dumper_utils import DumperMegatronUtil, DumperPhase
from miles.utils.timer import Timer, timer
from miles.utils.types import RolloutBatch

logger = logging.getLogger(__name__)

# One directory per client-chosen name; the iteration inside it is fixed.
SAVE_ITERATION = 1

# Single tenant: one accumulation window per process. ``pristine`` means the
# weights and optimizer are still the ones the server started from.
_state = {"pending_finalize": False, "dirty": False, "num_updates": 0, "pristine": True}


def _deferred_finalize(*_args, **_kwargs) -> None:
    """Installed as config.finalize_model_grads_func: the collectives run once, in optim_step."""
    _state["pending_finalize"] = True


def _check_accumulation_config(args: Namespace) -> None:
    assert not getattr(args, "overlap_grad_reduce", False) and not getattr(args, "overlap_param_gather", False), (
        "full fine-tuning accumulates grads across forward_backward calls by deferring finalize_model_grads; "
        "--overlap-grad-reduce / --overlap-param-gather would reduce per call"
    )


def forward_backward(
    args: Namespace,
    batch_id: int,
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    rollout_data: RolloutBatch,
    *,
    forward_only: bool = False,
) -> dict:
    _check_accumulation_config(args)
    data_iterator, num_microbatches = get_data_iterator(args, model, rollout_data)
    assert len(num_microbatches) == 1, "a work unit is a single forward/backward pass"

    for iterator in data_iterator:
        iterator.reset()
    for model_chunk in model:
        model_chunk.train()
    if forward_only:
        # scoring touches no gradients: no optimizer interaction, no accumulation bookkeeping
        setup_train_iteration_config(args, model, None, disable_optimizer=True)
    else:
        config = setup_train_iteration_config(args, model, optimizer, disable_optimizer=False)
        config.finalize_model_grads_func = _deferred_finalize
        reset_grad_metadata_keep_grads(model)

    dumper_phase_util = DumperMegatronUtil(args, model, DumperPhase.FWD_BWD, rollout_id=batch_id)
    try:
        with timer(tinker_perf.ACTOR_TRAIN_TIMER) if not forward_only else nullcontext():
            losses_reduced = run_forward_backward_pass(
                args,
                dumper_phase_util,
                data_iterator,
                model,
                num_microbatches[0],
                num_rollouts=None,
                forward_only=forward_only,
            )
    except Exception:
        if not forward_only:
            # partially accumulated grads: the next optim_step discards the window
            _state["dirty"] = True
        raise
    per_datum_outputs = [output for microbatch in losses_reduced for output in microbatch["per_datum"]]
    dumper_phase_util.finalize(model)
    if not forward_only:
        tinker_perf.note_unit(Timer().seq_lens)

    if get_parallel_state().is_pp_last_stage:
        return {"metrics": aggregate_train_losses(losses_reduced, None), "per_datum": per_datum_outputs}
    return {"metrics": {}, "per_datum": per_datum_outputs}


def optim_step(args: Namespace, model: Sequence[DDP], optimizer: MegatronOptimizer, adam_params: dict) -> dict:
    with timer(tinker_perf.ACTOR_TRAIN_TIMER):
        grad_norm, skipped = _step_accumulated_grads(model, optimizer, adam_params)
    if not skipped:
        _state["num_updates"] += 1
        _state["pristine"] = False
    metrics = {
        "grad_norm": grad_norm,
        "skipped": skipped,
        "lr": adam_params["learning_rate"],
        "num_updates": _state["num_updates"],
    }
    return {**metrics, **tinker_perf.pop_optim_window_metrics(args)}


def _step_accumulated_grads(model, optimizer, adam_params: dict) -> tuple[float, int]:
    """Finalize once, apply this step's AdamParams, veto non-finite grads, step. Returns (grad_norm, skipped)."""
    if _state["dirty"]:
        logger.warning("optim_step: discarding a window with a failed forward_backward")
        discard_grads(model, optimizer)
        return float("nan"), 1
    if _state["pending_finalize"]:
        # DP reduce-scatter + SP-layernorm / embedding all-reduces, exactly once per window
        finalize_model_grads_with_empty_cache(model, None)
        _state["pending_finalize"] = False

    apply_adam_params(optimizer.param_groups, adam_params)
    set_clip_grad(optimizer, adam_params["grad_clip_norm"])
    found_inf = optimizer.prepare_grads()
    grad_norm = float(optimizer.get_grad_norm())
    if should_veto_step(found_inf, grad_norm):
        logger.warning(f"optim_step: non-finite gradients (found_inf={found_inf}, norm={grad_norm}); step vetoed")
        discard_grads(model, optimizer)
        return grad_norm, 1

    # step() re-runs prepare_grads (an idempotent copy) then clips by config.clip_grad and all-gathers params
    update_successful, _, _ = optimizer.step()
    assert update_successful, "optimizer.step() reported failure after a finite grad norm"
    discard_grads(model, optimizer)
    return grad_norm, 0


def discard_grads(model: Sequence[DDP], optimizer: MegatronOptimizer) -> None:
    """Drop the accumulated window: zero DDP grad buffers and optimizer grads."""
    _zero_grads(model, optimizer, False)
    _state["pending_finalize"] = False
    _state["dirty"] = False


def save_state(
    args: Namespace, model: Sequence[DDP], optimizer: MegatronOptimizer, path: str, metadata: dict | None = None
) -> None:
    """Megatron torch_dist checkpoint (model + optimizer) plus META.json under ``path``; complete on return."""

    def write(checkpoint_dir) -> None:
        with inplace_modify_args(args, {"save": str(checkpoint_dir), "async_save": False}):
            save(SAVE_ITERATION, model, optimizer, None)

    write_checkpoint_dir(path, write, metadata=metadata)


def load_state(
    args: Namespace, model: Sequence[DDP], optimizer: MegatronOptimizer, path: str, load_optimizer: bool
) -> None:
    overrides = {
        "load": path,
        "finetune": False,
        "no_load_optim": not load_optimizer,
        "no_load_rng": True,
        "ckpt_step": None,
    }
    with inplace_modify_args(args, overrides):
        load_checkpoint(
            model,
            optimizer if load_optimizer else None,
            None,
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
        )
    if not load_optimizer:
        optimizer.reload_model_params()
        _reset_optimizer_state(args, optimizer)
    discard_grads(model, optimizer)
    _state["pristine"] = False
    _barrier()


def reset_to_base(args: Namespace, model: Sequence[DDP], optimizer: MegatronOptimizer) -> None:
    """Pristine base weights + fresh optimizer: reload what the server started from."""
    discard_grads(model, optimizer)
    if _state["pristine"]:
        return  # nothing trained or loaded since startup: already the base weights and a fresh optimizer
    assert _has_loadable_ckpt(args.load), f"cannot reset to base: --load {args.load!r} holds no checkpoint"
    # weights only; the same args.load/ckpt_step the init load used (HF via bridge, or a Megatron dir)
    with inplace_modify_args(args, {"finetune": True, "no_load_optim": True, "no_load_rng": True}):
        load_checkpoint(model, None, None, checkpointing_context={}, skip_load_to_model_and_opt=False)
    optimizer.reload_model_params()
    _reset_optimizer_state(args, optimizer)
    _state["num_updates"] = 0
    _state["pristine"] = True
    _barrier()


def _reset_optimizer_state(args: Namespace, optimizer: MegatronOptimizer) -> None:
    # refuses offloaded or streamed optimizer state, whose moments live outside what it clears
    reset_optimizer_state(
        optimizer,
        stream_optimizer_state_to_disk=args.stream_optimizer_state_to_disk,
        chunked_optimizer_state_offload=args.chunked_optimizer_state_offload,
    )


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier(group=get_gloo_group())
