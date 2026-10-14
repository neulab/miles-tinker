"""Command-grained full fine-tuning over the plain Megatron model + distributed optimizer.

Design (see scratchpad note v2_executor_full_ft_design.md, section 4):

* Deferred finalize. ``forward_backward`` runs Megatron's schedule with
  ``config.finalize_model_grads_func`` replaced by a no-op that only records
  "pending". With overlap_grad_reduce off, that hook is the *only* place the
  DP reduce-scatter and the sequence-parallel / tied-embedding SUM all-reduces
  happen, so the fp32 ``main_grad`` buffers keep accumulating locally and
  linearly across calls (DDP's ``zero_grad_buffer`` is never called between
  units; only ``reset_grad_metadata_keep_grads`` resets bookkeeping). The real
  ``finalize_model_grads`` runs exactly once, at ``optim_step``. Per-call
  finalize would be wrong: the distributed optimizer's reduce-scatter and the
  SP-layernorm sums are not idempotent. ``overlap_grad_reduce`` and
  ``overlap_param_gather`` must stay off (asserted at serve validation).
* Loss scaling. loss.py uses ``global_batch_size = 1`` for tinker units, so
  each call contributes the exact datum SUM; the DDP ``1/dp`` pre-scale at the
  single deferred reduce-scatter is cancelled by the ``* dp_cp_size`` in
  loss.py. Padding rows (weight 0) add nothing.
* AdamParams arrive per optim_step (no scheduler): lr/betas/eps/weight_decay
  (times Megatron's ``wd_mult``/``lr_mult``) are written into the param groups
  and ``grad_clip_norm`` becomes ``OptimizerConfig.clip_grad`` (0 = no clip).
  A non-finite gradient norm vetoes the step: grads are zeroed, ``skipped=1``.
* Latest-only sampler. ``push_slot`` is the ordinary base weight sync
  (``actor.update_weights``): engines hold one copy of the weights, so every
  sampler version resolves to the newest push (P1 documents the deviation).
* Checkpoints. ``save_state`` writes a Megatron torch_dist checkpoint
  (model + optimizer, synchronous, fixed iteration 1) under the tinker
  checkpoint path; ``load_state`` reads it back with a temporary ``args.load``
  (optimizer optional; a weights-only load resets Adam state so the semantics
  equal "fresh optimizer"). Resume needs the same parallel topology.
  ``reset_to_base`` reloads the weights the server started from (``args.load``:
  the HF checkpoint through the bridge, or the Megatron dir) and zeroes the
  optimizer moments and step clocks in place (same as --reset-optimizer-states;
  no optimizer re-creation).
* Memory. The fp32 grad buffer (DDP, replicated) stays live between calls; it
  is allocated permanently anyway. Qwen3-4B at TP=1/DP=8: ~30 GB static per
  rank plus activations; a checkpoint is ~56 GB (bf16 + fp32 master + Adam).
"""

import logging
from argparse import Namespace
from collections.abc import Sequence
from contextlib import ExitStack

import torch
import torch.distributed as dist
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.optimizer import MegatronOptimizer

from miles.backends.megatron_utils import tinker_perf
from miles.backends.megatron_utils.adam_params import apply_adam_params, set_clip_grad, should_veto_step
from miles.backends.megatron_utils.checkpoint import load_checkpoint
from miles.backends.megatron_utils.cp_model_config import check_per_token_loss_flag
from miles.backends.megatron_utils.lora.optimizer import reset_grad_metadata_keep_grads
from miles.backends.megatron_utils.model import (
    _has_loadable_ckpt,
    _zero_grads,
    finalize_model_grads_with_empty_cache,
    run_forward_backward_pass,
    save,
    setup_train_iteration_config,
)
from miles.backends.training_utils.data import get_data_iterator
from miles.backends.training_utils.log_utils import aggregate_train_losses
from miles.backends.training_utils.loss_hub import fused_lm_head
from miles.backends.training_utils.loss_hub.tinker_losses import drain_per_datum_outputs, start_per_datum_outputs
from miles.backends.training_utils.parallel import get_parallel_state
from miles.backends.training_utils.replay_data import rollout_replay_stage
from miles.backends.training_utils.tinker_numerics import finite_per_datum_reports
from miles.utils.argparse_utils import inplace_modify_args
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.dumper_utils import DumperMegatronUtil, DumperPhase
from miles.utils.timer import Timer, timer
from miles.utils.types import RolloutBatch

logger = logging.getLogger(__name__)

# One directory per client-chosen name; the iteration inside it is fixed.
SAVE_ITERATION = 1

# Single tenant: one accumulation window per process.
_state = {"pending_finalize": False, "dirty": False, "nonfinite_output": False, "num_updates": 0}


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
    unit_id: int,
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    rollout_data: RolloutBatch,
) -> dict:
    _check_accumulation_config(args)
    with tinker_perf.stage(tinker_perf.STAGE_DATA_ITER):
        data_iterator, num_microbatches = get_data_iterator(args, model, rollout_data)
    assert len(num_microbatches) == 1, "a work unit is a single forward/backward pass"

    for iterator in data_iterator:
        iterator.reset()
    for model_chunk in model:
        model_chunk.train()
    config = setup_train_iteration_config(args, model, optimizer, disable_optimizer=False)
    # the schedule reads the flag from the config on every call; a reset that did not hold (or a later
    # rebuild that turned it back on) would scale this unit's gradients by num_microbatches * |DP*CP|
    check_per_token_loss_flag(config, args)
    config.finalize_model_grads_func = _deferred_finalize
    reset_grad_metadata_keep_grads(model)

    dumper_phase_util = DumperMegatronUtil(args, model, DumperPhase.FWD_BWD, rollout_id=unit_id)
    start_per_datum_outputs()
    try:
        # ExitStack, not a nested `with`, only so the R3 fill (the context's __enter__:
        # ~29 GB of int32 routing per step) is timed apart from the pass it wraps.
        with ExitStack() as stack:
            with tinker_perf.stage(tinker_perf.STAGE_REPLAY_FILL):
                stack.enter_context(
                    rollout_replay_stage(
                        args, model, data_iterator, num_microbatches, rollout_data, stage="replay_backward"
                    )
                )
            with timer(tinker_perf.ACTOR_TRAIN_TIMER), fused_lm_head.maybe_activate(args, model):
                losses_reduced = run_forward_backward_pass(
                    args, dumper_phase_util, data_iterator, model, num_microbatches[0], num_rollouts=None
                )
    except Exception:
        # partially accumulated grads: optim_step discards the window
        _state["dirty"] = True
        raise
    finally:
        per_datum_outputs = drain_per_datum_outputs()
    per_datum_outputs = _check_per_datum_outputs(unit_id, per_datum_outputs, rollout_data)
    dumper_phase_util.finalize(model)
    tinker_perf.note_unit(Timer().seq_lens, tinker_perf.unit_shape(args, rollout_data, data_iterator))

    if get_parallel_state().is_pp_last_stage:
        return {"metrics": aggregate_train_losses(losses_reduced, None), "per_datum": per_datum_outputs}
    return {"metrics": {}, "per_datum": per_datum_outputs}


def optim_step(args: Namespace, model: Sequence[DDP], optimizer: MegatronOptimizer, adam_params: dict) -> dict:
    nonfinite_output = _state["nonfinite_output"]
    with timer(tinker_perf.ACTOR_TRAIN_TIMER), tinker_perf.stage(tinker_perf.STAGE_OPTIM):
        grad_norm, skipped = _step_accumulated_grads(model, optimizer, adam_params)
    if not skipped:
        _state["num_updates"] += 1
    metrics = {
        "grad_norm": grad_norm,
        "skipped": skipped,
        "lr": adam_params["learning_rate"],
        "num_updates": _state["num_updates"],
    }
    if nonfinite_output:
        metrics["nonfinite_output"] = 1
    return {**metrics, **tinker_perf.pop_optim_window_metrics(args)}


def _check_per_datum_outputs(unit_id: int, outputs: list[dict], rollout_data: RolloutBatch) -> list[dict]:
    reports, diagnostics = finite_per_datum_reports(outputs)
    invalid = torch.tensor(bool(diagnostics), dtype=torch.int32)
    # Every rank must take the same branch at optim_step, including ranks with
    # no bad local rows. Otherwise some would enter finalize's collectives alone.
    if dist.is_initialized():
        dist.all_reduce(invalid, op=dist.ReduceOp.MAX, group=get_gloo_group())
    if invalid.item():
        _state["dirty"] = True
        _state["nonfinite_output"] = True
    if diagnostics:
        indices = {int(index): i for i, index in enumerate(rollout_data.get("sample_indices", []))}
        for diagnostic in diagnostics:
            local_index = indices.get(int(diagnostic["sample_index"]))
            if local_index is not None:
                diagnostic["tokens"] = rollout_data["total_lengths"][local_index]
                for field in ("rollout_log_probs", "advantages"):
                    if field in rollout_data:
                        values = torch.as_tensor(rollout_data[field][local_index])
                        diagnostic[f"nonfinite_{field}"] = int((~torch.isfinite(values)).sum().item())
        logger.error("tinker nonfinite output: unit=%s; optimizer window vetoed; datums=%s", unit_id, diagnostics)
    return reports


def _step_accumulated_grads(model, optimizer, adam_params: dict) -> tuple[float, int]:
    """Finalize once, apply this step's AdamParams, veto non-finite grads, step. Returns (grad_norm, skipped)."""
    if _state["dirty"]:
        logger.warning("optim_step: discarding a window with a failed or non-finite forward_backward")
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
    _state["nonfinite_output"] = False
    _state["dirty"] = False


def save_state(args: Namespace, model: Sequence[DDP], optimizer: MegatronOptimizer, path: str) -> None:
    """Megatron torch_dist checkpoint (model + optimizer) under ``path``; complete on return."""
    _initialize_optimizer_state(optimizer)
    with inplace_modify_args(args, {"save": path, "async_save": False}):
        save(SAVE_ITERATION, model, optimizer, None)
    _barrier()


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
        reset_optimizer_state(optimizer)
    else:
        _reset_unstepped_optimizer_state(optimizer)
    discard_grads(model, optimizer)
    _barrier()


def reset_to_base(args: Namespace, model: Sequence[DDP], optimizer: MegatronOptimizer) -> None:
    """Pristine base weights + fresh optimizer: reload what the server started from."""
    assert _has_loadable_ckpt(args.load), f"cannot reset to base: --load {args.load!r} holds no checkpoint"
    # weights only; the same args.load/ckpt_step the init load used (HF via bridge, or a Megatron dir)
    with inplace_modify_args(args, {"finetune": True, "no_load_optim": True, "no_load_rng": True}):
        load_checkpoint(model, None, None, checkpointing_context={}, skip_load_to_model_and_opt=False)
    optimizer.reload_model_params()
    reset_optimizer_state(optimizer)
    discard_grads(model, optimizer)
    _state["num_updates"] = 0
    _barrier()


def reset_optimizer_state(optimizer: MegatronOptimizer) -> None:
    """Zero Adam moments and step clocks in place (group-level FusedAdam or per-param torch Adam)."""
    _initialize_optimizer_state(optimizer)
    for child in getattr(optimizer, "chained_optimizers", [optimizer]):
        inner = child.optimizer
        if inner is None:
            continue
        for group in inner.param_groups:
            if "step" in group:
                group["step"] = _zeroed_step(group["step"])
        for state in inner.state.values():
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key].zero_()
            if "step" in state:
                state["step"] = _zeroed_step(state["step"])


def _initialize_optimizer_state(optimizer: MegatronOptimizer) -> None:
    # FusedAdam allocates moments lazily. A checkpoint saved before its first
    # step must contain real zero moments, not leave load-time empty buffers.
    for child in getattr(optimizer, "chained_optimizers", [optimizer]):
        initialize = getattr(child, "init_state_fn", None)
        if child.optimizer is not None and initialize is not None:
            initialize(child.optimizer, child.config)


def _reset_unstepped_optimizer_state(optimizer: MegatronOptimizer) -> None:
    # Older pre-step checkpoints can restore uninitialized Adam moments. No
    # step means there is no history to preserve. Trained optimizers retain it.
    for child in getattr(optimizer, "chained_optimizers", [optimizer]):
        inner = child.optimizer
        if inner is None:
            continue
        clocks = [group.get("step", 0) for group in inner.param_groups]
        clocks.extend(state.get("step", 0) for state in inner.state.values())
        if all(float(step) == 0 for step in clocks):
            logger.info("Resetting unstepped Adam state after checkpoint load")
            reset_optimizer_state(child)


def _zeroed_step(step):
    if torch.is_tensor(step):
        return step.zero_()
    return 0


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier(group=get_gloo_group())
