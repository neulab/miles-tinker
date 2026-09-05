"""Per-slot checkpoints: adapter weights plus the slot's optimizer state.

Weight shards are (tp, pp, ep)-addressed and slot-agnostic (saved under
expose_adapter_slot), so a resume needs the same TP/PP/EP shard topology.
Optimizer state is keyed by the same slot-agnostic parameter names, scoped to
the shard coordinate (a name repeats across TP/PP/EP ranks with different
contents): each rank writes the Adam state of the main params it owns, and a
load merges the files of its own coordinate and picks the states of the params
the target slot owns on this rank. LayerWise scatters whole params across the
DP replicas of a coordinate, and which rank owns which param differs per slot
and per DP layout, so this is what lets a checkpoint load into any slot under
any DP size.

Every save/load phase agrees on its outcome across ranks over the gloo group
(``_consistent``): either all ranks proceed or all raise the same
SlotCheckpointError, with no rank left waiting in a collective. A load
validates the checkpoint before it touches the slot; the caller re-initializes
the slot if the apply phase fails.
"""

import logging
import os
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

import torch
import torch.distributed as dist
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.optimizer import MegatronOptimizer

from miles.backends.megatron_utils.lora.optimizer import _slot_children
from miles.backends.megatron_utils.lora.slots import adapter_shard_topology, megatron_shard_name, shard_coord_tag
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.distributed_utils import get_gloo_group

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SlotCheckpointError(ValueError):
    """A save or load failed on some rank; raised with the same message on every rank."""


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _all_gather(value) -> list:
    if not dist.is_initialized():
        return [value]
    group = get_gloo_group()
    gathered: list = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(gathered, value, group=group)
    return gathered


def _consistent(phase: Callable[[], T], context: str) -> T:
    """Run one per-rank phase and agree on its outcome: if any rank raised, every rank raises the
    same SlotCheckpointError, so no rank walks alone into a later collective. The gather is also
    the barrier between phases."""
    try:
        result, error = phase(), None
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"{context}: failed on rank {_rank()}")
        result, error = None, f"rank {_rank()}: {type(exc).__name__}: {exc}"
    outcomes = _all_gather(error)
    failures = [failure for failure in outcomes if failure is not None]
    if failures:
        raise SlotCheckpointError(f"{context} failed on {len(failures)}/{len(outcomes)} rank(s); first: {failures[0]}")
    return result


def _shard_tag() -> str:
    parallel_state = get_parallel_state()
    return shard_coord_tag(
        parallel_state.tp.rank, parallel_state.pp.rank, parallel_state.ep.rank, parallel_state.ep.size
    )


def _weight_shard_name() -> str:
    parallel_state = get_parallel_state()
    return megatron_shard_name(
        parallel_state.tp.rank, parallel_state.pp.rank, parallel_state.ep.rank, parallel_state.ep.size
    )


def _optim_shard_name(tag: str) -> str:
    return f"optim_{tag}_rank{_rank()}.pt"


def _optim_shard_glob(tag: str) -> str:
    return f"optim_{tag}_rank*.pt"


def save_slot(model: Sequence[DDP], optimizer: MegatronOptimizer, slot: int, path: str) -> None:
    """Write ``slot`` under ``path``: one adapter shard per (tp, pp, ep) coordinate and one
    optimizer file per rank, published by rename. Raises SlotCheckpointError on every rank if
    any rank failed to write."""
    is_shard_writer, _ = adapter_shard_topology()
    final_dir = Path(path)
    tmp_dir = final_dir.parent / f"_tmp_{final_dir.name}"
    context = f"save_slot(slot={slot}, path={path})"
    tag = _shard_tag()
    replicas = _all_gather(tag).count(tag)

    def prepare() -> None:
        if _rank() == 0:
            tmp_dir.mkdir(parents=True, exist_ok=True)

    def write_shards() -> None:
        named = _exposed_adapter_params(model, slot)
        assert named, f"slot {slot} exposed no adapter tensors"
        if is_shard_writer:
            torch.save({name: param.data.cpu() for name, param in named.items()}, tmp_dir / _weight_shard_name())
        names = {id(param): name for name, param in named.items()}
        torch.save(_optimizer_shard(optimizer, slot, names, tag, replicas), tmp_dir / _optim_shard_name(tag))

    def publish() -> None:
        # write-then-rename so readers never see a partial checkpoint
        if _rank() == 0:
            if final_dir.exists():
                shutil.rmtree(final_dir)
            os.replace(tmp_dir, final_dir)

    try:
        _consistent(prepare, context)
        _consistent(write_shards, context)
    except SlotCheckpointError:
        if _rank() == 0:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    _consistent(publish, context)


def load_slot(model: Sequence[DDP], optimizer: MegatronOptimizer, slot: int, path: str, load_optimizer: bool) -> None:
    """Restore ``path`` into ``slot``, which the caller has just initialized for the checkpoint's
    rank/alpha. Raises SlotCheckpointError on every rank at once: before any mutation when the
    checkpoint is missing or does not fit this slot, or after the apply phase failed on any rank
    (the slot is then partially written and the caller re-initializes it)."""
    from megatron.bridge.peft.multi_lora_layers import load_adapter

    checkpoint_dir = Path(path)
    context = f"load_slot(slot={slot}, path={path})"

    def read() -> tuple[dict[int, str], dict, dict | None]:
        named = _exposed_adapter_params(model, slot)
        weights = _read_weight_shard(checkpoint_dir, named)
        optim_state = _read_optimizer_shards(checkpoint_dir, _shard_tag()) if load_optimizer else None
        return {id(param): name for name, param in named.items()}, weights, optim_state

    names, weights, optim_state = _consistent(read, context)

    def apply() -> None:
        loaded = load_adapter(model, slot, weights)
        assert loaded == len(names), f"loaded {loaded} of {len(names)} adapter tensors"
        optimizer.reload_model_params()
        if optim_state is None:
            return  # weights-only load keeps the fresh Adam state the slot init just zeroed
        missing = _load_owned_optimizer_state(optimizer, slot, names, optim_state)
        if missing:
            logger.warning(
                f"{context}: {len(missing)} owned param(s) had no saved optimizer state and stay fresh "
                f"(e.g. {missing[0]})"
            )

    _consistent(apply, context)


def _exposed_adapter_params(model: Sequence[DDP], slot: int) -> dict[str, torch.nn.Parameter]:
    """The slot's adapter params under the slot-agnostic names load_adapter looks up."""
    from megatron.bridge.peft.multi_lora_layers import expose_adapter_slot

    with expose_adapter_slot(model, slot):
        return {
            name: param
            for model_chunk in model
            for name, param in model_chunk.named_parameters()
            if ".adapter." in name
        }


def _read_weight_shard(checkpoint_dir: Path, named: dict[str, torch.nn.Parameter]) -> dict[str, torch.Tensor]:
    """This rank's adapter shard, checked to fit the slot exactly (names both ways, shapes)."""
    shard_path = checkpoint_dir / _weight_shard_name()
    if not shard_path.is_file():
        raise FileNotFoundError(f"no adapter shard {shard_path}")
    weights = torch.load(shard_path, map_location="cpu", weights_only=True)
    missing = sorted(set(named) - set(weights))
    unknown = sorted(set(weights) - set(named))
    if missing or unknown:
        raise ValueError(
            f"{shard_path.name} does not match the slot's adapter params: {len(missing)} missing "
            f"(e.g. {missing[:1]}), {len(unknown)} unknown (e.g. {unknown[:1]}); did target_modules change?"
        )
    for name, param in named.items():
        if tuple(weights[name].shape) != tuple(param.shape):
            raise ValueError(
                f"{shard_path.name}: {name} has shape {tuple(weights[name].shape)}, the slot expects "
                f"{tuple(param.shape)}; was the gateway's max LoRA rank changed?"
            )
    return weights


def _read_optimizer_shards(checkpoint_dir: Path, tag: str) -> dict:
    """Merge the optimizer files of this shard coordinate; together they cover every param."""
    files = sorted(checkpoint_dir.glob(_optim_shard_glob(tag)))
    if not files:
        raise FileNotFoundError(f"no optimizer state ({_optim_shard_glob(tag)}) in {checkpoint_dir}")
    return _merge_optimizer_shards([torch.load(file, map_location="cpu", weights_only=True) for file in files])


def _merge_optimizer_shards(shards: list[dict]) -> dict:
    """``{"params": {name: state}, "step": int}`` from the files one DP replica set wrote."""
    replicas = {shard["replicas"] for shard in shards}
    if replicas != {len(shards)}:
        raise ValueError(f"incomplete optimizer state: found {len(shards)} of {sorted(replicas)} replica file(s)")
    params: dict[str, dict] = {}
    for shard in shards:
        params.update(shard["params"])
    steps = [step for shard in shards for step in shard["group_steps"]]
    # a slot's groups step in lockstep (step_adapter_slots); max tolerates untouched empty groups
    return {"params": params, "step": max(steps, default=0)}


def _main_params_by_model_param(child) -> list[tuple[torch.nn.Parameter, torch.Tensor]]:
    """(model param, main param) pairs of one Float16 child: the bf16 params and their fp32
    masters are aligned group by group; fp32 params are their own master."""
    pairs: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    for model_group, main_group in zip(child.float16_groups, child.fp32_from_float16_groups, strict=True):
        pairs += zip(model_group, main_group, strict=True)
    for group in child.fp32_from_fp32_groups:
        pairs += [(param, param) for param in group]
    return pairs


def _optimizer_shard(optimizer, slot: int, names: dict[int, str], tag: str, replicas: int) -> dict:
    """One rank's optimizer file: the states it owns for ``slot`` plus what a load needs to
    check the replica set of shard coordinate ``tag`` is complete."""
    params, group_steps = _owned_optimizer_state(optimizer, slot, names)
    return {
        "world_size": _world_size(),
        "shard": tag,
        "replicas": replicas,
        "group_steps": group_steps,
        "params": params,
    }


def _owned_optimizer_state(optimizer, slot: int, names: dict[int, str]) -> tuple[dict[str, dict], list[int]]:
    """This rank's Adam state of ``slot`` keyed by parameter name, plus every group's step clock."""
    params: dict[str, dict] = {}
    group_steps: list[int] = []
    for child in _slot_children(optimizer, slot):
        inner = child.optimizer
        group_steps += [_step_value(group.get("step", 0)) for group in inner.param_groups]
        for model_param, main_param in _main_params_by_model_param(child):
            # a never-stepped slot has no per-param state yet
            state = inner.state[main_param] if main_param in inner.state else {}
            params[names[id(model_param)]] = {
                key: value.cpu() if torch.is_tensor(value) else value for key, value in state.items()
            }
    return params, group_steps


def _step_value(step) -> int:
    return int(step.item()) if torch.is_tensor(step) else int(step)


def _load_owned_optimizer_state(optimizer, slot: int, names: dict[int, str], saved: dict) -> list[str]:
    """Copy the saved states into the main params this rank owns for ``slot``; returns the names
    that had no saved state (their fresh zeros stand)."""
    missing: list[str] = []
    for child in _slot_children(optimizer, slot):
        inner = child.optimizer
        # FusedAdam clocks steps on the param group, not in per-param state
        for group in inner.param_groups:
            _set_group_step(group, saved["step"])
        for model_param, main_param in _main_params_by_model_param(child):
            name = names[id(model_param)]
            if name not in saved["params"]:
                missing.append(name)
                continue
            _copy_param_state(inner.state[main_param], saved["params"][name], device=main_param.device)
    return missing


def _set_group_step(group: dict, step: int) -> None:
    existing = group.get("step")
    if torch.is_tensor(existing):
        existing.copy_(torch.as_tensor(step))
    elif "step" in group or step:
        group["step"] = step


def _copy_param_state(state: dict, saved_state: dict, device) -> None:
    for key, value in saved_state.items():
        if not torch.is_tensor(value):
            state[key] = value
        elif torch.is_tensor(state.get(key)):
            state[key].copy_(value.to(state[key].device))
        else:
            state[key] = value.to(device)
