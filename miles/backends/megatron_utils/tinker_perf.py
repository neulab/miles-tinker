"""MFU bookkeeping and MFU diagnostics for the Tinker executors.

One optim window = every forward_backward unit since the previous optim_step
plus the step itself. Units time their forward/backward under the
``actor_train`` timer and record the unit's sequence lengths here; optim_step
drains both into the same numbers the native ``perf/actor_train_*`` metrics
report (3 x forward FLOPs over the timed window). No collectives: every rank
computes the same values and the driver reads the first worker's.

On top of that headline MFU the window also carries the numbers that say *why*
it is what it is (``docs/fork/mfu.md``, "Diagnosing a low MFU"):

* ``rank_*`` - how unevenly the DP ranks were loaded. The batch is balanced by
  the SUM of sequence lengths with equal row counts
  (``get_seqlen_balanced_partitions(..., equal_size=True)``), which equalises
  tokens but not work: attention is quadratic on the full-attention layers, so
  two ranks with the same token count can differ in FLOPs. Every rank sees the
  whole batch's per-rank split (``Timer().dp_seq_lens``, filled by
  ``process_rollout_data_shard``), so this costs no collective.
* ``datum_seq_len_*`` / ``fwd_gflops_per_token`` - how much work a token of this
  batch is actually worth, against the 11.20 GFLOP/token of a uniform 64k batch.
  The length percentiles come from the gateway (``miles/tinker/runtime.py``),
  which sees the unpadded rows.
* ``actor_*_time`` - where the wall clock between two optimizer steps went;
  ``actor_train_time`` covers only run_forward_backward_pass.
* ``mb_*`` - micro-batch shape, which separates "many small kernels" from
  "one rank does all the work".
* ``pad_*`` (gateway side) - so padding is never again mistaken for the problem.

Everything past the headline metrics degrades to *absent* rather than raising:
a field the caller could not supply simply drops its keys.
"""

import logging
from argparse import Namespace
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from time import time

import torch
import torch.distributed as dist

from miles.utils.device_flops import local_peak_bf16_tflops
from miles.utils.flops_utils import calculate_fwd_flops, ensure_hf_layer_pattern, fwd_tflops_per_gpu
from miles.utils.timer import Timer

logger = logging.getLogger(__name__)

ACTOR_TRAIN_TIMER = "actor_train"

# Wall-clock stages of one optim window. DATA_FETCH / DATA_ITER / REPLAY_FILL sit
# *outside* the actor_train timer (they are the gap it does not explain); OPTIM sits
# inside it, so actor_fwd_bwd_time = actor_train_time - actor_optim_time.
STAGE_DATA_FETCH = "data_fetch"  # get_rollout_data: object-store get + H2D copies
STAGE_DATA_ITER = "data_iter"  # get_data_iterator: micro-batch balancing + its DP MAX all-reduce
STAGE_REPLAY_FILL = "replay_fill"  # R3 fill_replay_data (~29 GB of int32 [T-1, L, K] per step)
STAGE_OPTIM = "optim"  # the optimizer step itself (finalize_model_grads + step)
_STAGE_METRIC_KEYS = {
    STAGE_DATA_FETCH: "actor_data_fetch_time",
    STAGE_DATA_ITER: "actor_data_iter_time",
    STAGE_REPLAY_FILL: "actor_replay_fill_time",
    STAGE_OPTIM: "actor_optim_time",
}
_OUTSIDE_TRAIN_TIMER = (STAGE_DATA_FETCH, STAGE_DATA_ITER, STAGE_REPLAY_FILL)

_window_seq_lens: list[int] = []
_window_num_units = 0
_window_shapes: list[dict] = []
_window_stage_seconds: dict[str, float] = {}


def note_unit(seq_lens: Iterable[int], shape: dict | None = None) -> None:
    """Record one trained unit's (padded, batch-global) sequence lengths.

    ``shape`` is the optional diagnostic payload from :func:`unit_shape`; a unit
    without one still counts towards every headline metric.
    """
    global _window_num_units
    _window_seq_lens.extend(int(length) for length in seq_lens)
    _window_num_units += 1
    if shape:
        _window_shapes.append(shape)


def pop_window() -> tuple[list[int], int]:
    global _window_seq_lens, _window_num_units
    seq_lens, _window_seq_lens = _window_seq_lens, []
    num_units, _window_num_units = _window_num_units, 0
    return seq_lens, num_units


def pop_diagnostics() -> tuple[list[dict], dict[str, float]]:
    """The window's per-unit shapes and stage seconds, cleared."""
    global _window_shapes, _window_stage_seconds
    shapes, _window_shapes = _window_shapes, []
    stages, _window_stage_seconds = _window_stage_seconds, {}
    return shapes, stages


def reset_window() -> None:
    """Drop everything the open window holds (tests, and a poisoned window)."""
    pop_window()
    pop_diagnostics()


def discard_window() -> None:
    """Forget a window whose gradients were thrown away.

    ``_discard_dirty`` drops the gradients of a poisoned optim window and fails
    its optim_step, so ``pop_optim_window_metrics`` never runs for it. Without
    this the units that *did* complete stay in the window and are charged to the
    NEXT one: its MFU would be computed over tokens it never forwarded, and
    ``pad_rows`` (gateway counts minus the trainer's padded counts, and the
    gateway does reset its own per-slot window here) would come out inflated.
    Also drops the accumulated ``actor_train`` time, which belongs to the same
    discarded window.
    """
    reset_window()
    Timer().reset(ACTOR_TRAIN_TIMER)


@contextmanager
def stage(name: str):
    """Accumulate wall clock into the window's ``name`` stage. Never raises, and
    charges the stage even when the body does."""
    started = time()
    try:
        yield
    finally:
        _window_stage_seconds[name] = _window_stage_seconds.get(name, 0.0) + (time() - started)


def unit_shape(args: Namespace, rollout_data: dict, data_iterator: Sequence | None) -> dict:
    """The diagnostic shape of one unit: the whole batch's per-DP-rank sequence
    lengths, this rank's micro-batch token counts, and what the micro-batch
    schedule was allowed to pack. Never raises; an unavailable field is omitted
    (and its metrics with it)."""
    shape: dict = {}
    try:
        dp_seq_lens = getattr(Timer(), "dp_seq_lens", None)
        if isinstance(dp_seq_lens, (list, tuple)) and dp_seq_lens:
            shape["dp_seq_lens"] = [[int(length) for length in lens] for lens in dp_seq_lens]
    except Exception:  # noqa: BLE001 - perf bookkeeping must never take the trainer down
        pass
    try:
        local_lens = [int(length) for length in (rollout_data.get("total_lengths") or [])]
        iterator = data_iterator[0] if data_iterator else None
        mb_tokens = _microbatch_tokens(iterator, local_lens)
        if mb_tokens:
            shape["mb_tokens"] = mb_tokens
    except Exception:  # noqa: BLE001
        pass
    try:
        if getattr(args, "use_dynamic_batch_size", False) and getattr(args, "max_tokens_per_gpu", None):
            shape["mb_token_budget"] = int(args.max_tokens_per_gpu) * _cp_size()
        elif getattr(args, "micro_batch_size", None):
            shape["micro_batch_size"] = int(args.micro_batch_size)
    except Exception:  # noqa: BLE001
        pass
    return shape


def _cp_size() -> int:
    try:
        from miles.backends.training_utils.parallel import get_parallel_state

        return max(1, int(get_parallel_state().cp.size))
    except Exception:  # noqa: BLE001
        return 1


def _microbatch_tokens(iterator, local_lens: list[int]) -> list[int]:
    """This rank's tokens per micro-batch, from whichever schedule the iterator holds."""
    if iterator is None or not local_lens:
        return []
    indices = getattr(iterator, "micro_batch_indices", None)
    if indices is not None:
        return [sum(local_lens[i] for i in group if i < len(local_lens)) for group in indices]
    size = getattr(iterator, "micro_batch_size", None)
    if not size:
        return []
    return [sum(local_lens[i : i + size]) for i in range(0, len(local_lens), size)]


def perf_metrics(
    *, actor_train_time: float, total_fwd_tflops: float, num_tokens: int, peak_tflops: float | None
) -> dict[str, float]:
    """The native actor_train perf formulas over one optim window."""
    if actor_train_time <= 0:
        return {}
    tflops = 3 * total_fwd_tflops / actor_train_time
    metrics = {
        "actor_train_time": actor_train_time,
        "actor_train_tflops": tflops,
        "actor_train_tok_per_s": num_tokens / actor_train_time,
    }
    if peak_tflops:
        metrics["actor_train_mfu"] = tflops / peak_tflops
    return metrics


def pop_optim_window_metrics(args: Namespace) -> dict[str, float]:
    """Drain the actor_train timer and the window's units into perf metrics.
    ``num_tokens``/``num_datums`` count padded rows; the driver overwrites
    them with the exact per-slot counts (and keeps ``padded_num_*`` as the
    padded truth, which is what the ``pad_*`` metrics are read against)."""
    timer_instance = Timer()
    actor_train_time = timer_instance.log_dict().get(ACTOR_TRAIN_TIMER, 0.0)
    timer_instance.reset(ACTOR_TRAIN_TIMER)
    seq_lens, num_units = pop_window()
    shapes, stages = pop_diagnostics()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    # Once per process: linear-attention layer pattern from the HF config (never raises).
    ensure_hf_layer_pattern(args)
    total_fwd_tflops = fwd_tflops_per_gpu(seq_lens, args, world_size) if seq_lens else 0.0
    peak_tflops = getattr(args, "mfu_peak_tflops", None) or local_peak_bf16_tflops()
    metrics = perf_metrics(
        actor_train_time=actor_train_time,
        total_fwd_tflops=total_fwd_tflops,
        num_tokens=sum(seq_lens),
        peak_tflops=peak_tflops,
    )
    return {
        **metrics,
        **_peak_memory_metrics(),
        **diagnostic_metrics(
            args,
            seq_lens=seq_lens,
            shapes=shapes,
            stages=stages,
            world_size=world_size,
            total_fwd_tflops=total_fwd_tflops,
            actor_train_time=actor_train_time,
        ),
        "num_tokens": sum(seq_lens),
        "num_datums": len(seq_lens),
        "num_units": num_units,
    }


def diagnostic_metrics(
    args: Namespace,
    *,
    seq_lens: list[int],
    shapes: list[dict],
    stages: dict[str, float],
    world_size: int,
    total_fwd_tflops: float,
    actor_train_time: float,
) -> dict[str, float]:
    """The MFU-diagnostic half of the window. Each block is guarded on its own, so
    a diagnostic that cannot be computed drops its keys and leaves the rest
    standing; none of them can fail an optim_step."""
    metrics: dict[str, float] = {}
    for name, block in (
        ("window totals", lambda: _window_total_metrics(seq_lens, world_size, total_fwd_tflops)),
        ("rank imbalance", lambda: _rank_metrics(args, world_size, shapes)),
        ("micro-batch shape", lambda: _microbatch_metrics(shapes)),
        ("stage wall clock", lambda: _stage_metrics(stages, actor_train_time)),
    ):
        try:
            metrics.update(block())
        except Exception as exc:  # noqa: BLE001 - diagnostics never take the trainer down
            logger.warning("MFU diagnostics: dropped the %s metrics this window (%s)", name, exc)
    return metrics


def _window_total_metrics(seq_lens: list[int], world_size: int, total_fwd_tflops: float) -> dict[str, float]:
    if not seq_lens:
        return {}
    return {
        "padded_num_tokens": float(sum(seq_lens)),
        "padded_num_datums": float(len(seq_lens)),
        "actor_train_fwd_tflops": total_fwd_tflops,
        # Global forward FLOPs per global token: the per-GPU TFLOPs times the world
        # size, over the tokens they were earned on. A uniform 64k batch of
        # Qwen3.6-35B-A3B earns 11.20 GFLOP/token (docs/fork/mfu.md), so this is the
        # part of a low MFU that is "less work per token" rather than lost throughput.
        "fwd_gflops_per_token": total_fwd_tflops * world_size * 1e3 / sum(seq_lens),
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ratio(numerator: float, denominator: float) -> float:
    """max/mean, and 1.0 (perfectly balanced) when there is nothing to divide."""
    return numerator / denominator if denominator > 0 else 1.0


def _rank_metrics(args: Namespace, world_size: int, shapes: list[dict]) -> dict[str, float]:
    """Spread of tokens and forward FLOPs across the DP ranks.

    Absolute numbers are summed over the window's units, so ``rank_*_max`` is
    what the straggler rank did and ``rank_*_mean`` is the average rank; the
    step waits for the max. ``rank_fwd_tflops_*`` is per GPU (the whole batch's
    FLOPs divided by the GPUs behind one DP rank), so ``rank_fwd_tflops_mean``
    equals ``actor_train_fwd_tflops`` by construction.

    Costs no collective: every rank already computes the whole batch's DP split
    (``split_train_data_by_dp_raw`` runs on each trainer under
    ``--delay-split-train-data-by-dp``) and keeps it on ``Timer().dp_seq_lens``.
    """
    units = [shape["dp_seq_lens"] for shape in shapes if shape.get("dp_seq_lens")]
    if not units:
        return {}
    dp_size = len(units[0])
    if dp_size < 1 or any(len(unit) != dp_size for unit in units):
        return {}
    gpus_per_dp_rank = (world_size / dp_size) if world_size and dp_size else 1.0
    if gpus_per_dp_rank <= 0:
        gpus_per_dp_rank = 1.0

    tokens = {"max": 0.0, "mean": 0.0, "min": 0.0}
    tflops = {"max": 0.0, "mean": 0.0, "min": 0.0}
    worst = {"tokens": 0.0, "flops": 0.0}
    weighted = {"tokens": 0.0, "flops": 0.0}
    total_weight = 0.0
    for unit in units:
        per_rank = {
            "tokens": [float(sum(lens)) for lens in unit],
            "flops": [calculate_fwd_flops(lens, args) / 1e12 / gpus_per_dp_rank for lens in unit],
        }
        for totals, key in ((tokens, "tokens"), (tflops, "flops")):
            values = per_rank[key]
            totals["max"] += max(values)
            totals["mean"] += _mean(values)
            totals["min"] += min(values)
        weight = sum(per_rank["tokens"])
        total_weight += weight
        for key in ("tokens", "flops"):
            unit_ratio = _ratio(max(per_rank[key]), _mean(per_rank[key]))
            worst[key] = max(worst[key], unit_ratio)
            weighted[key] += unit_ratio * weight

    return {
        "rank_dp_size": float(dp_size),
        "rank_tokens_max": tokens["max"],
        "rank_tokens_mean": tokens["mean"],
        "rank_tokens_min": tokens["min"],
        "rank_token_imbalance": _ratio(tokens["max"], tokens["mean"]),
        "rank_token_imbalance_worst_unit": worst["tokens"],
        "rank_token_imbalance_weighted_mean": _ratio(weighted["tokens"], total_weight),
        "rank_fwd_tflops_max": tflops["max"],
        "rank_fwd_tflops_mean": tflops["mean"],
        "rank_fwd_tflops_min": tflops["min"],
        "rank_flops_imbalance": _ratio(tflops["max"], tflops["mean"]),
        "rank_flops_imbalance_worst_unit": worst["flops"],
        "rank_flops_imbalance_weighted_mean": _ratio(weighted["flops"], total_weight),
    }


def _microbatch_metrics(shapes: list[dict]) -> dict[str, float]:
    """Micro-batch shape: how many kernels the window ran and how big each was.

    ``mb_*`` is this rank's realised schedule; ``rank_mb_count_*`` replays the
    scheduler's own first-fit rule over every DP rank's rows, which is the count
    each rank would need before ``get_data_iterator`` raises them all to the DP
    max (so ``rank_mb_imbalance`` is the wasted-pass factor)."""
    metrics: dict[str, float] = {}
    counts = [len(shape["mb_tokens"]) for shape in shapes if shape.get("mb_tokens")]
    tokens = [count for shape in shapes for count in (shape.get("mb_tokens") or [])]
    if counts:
        metrics["mb_per_unit_max"] = float(max(counts))
        metrics["mb_per_unit_mean"] = _mean(counts)
    if tokens:
        metrics["mb_count"] = float(len(tokens))
        metrics["mb_tokens_mean"] = _mean(tokens)
        metrics["mb_tokens_max"] = float(max(tokens))

    per_unit_max = per_unit_mean = 0.0
    measured = False
    for shape in shapes:
        unit = shape.get("dp_seq_lens")
        if not unit:
            continue
        budget, size = shape.get("mb_token_budget"), shape.get("micro_batch_size")
        if budget:
            rank_counts = [float(_min_micro_batches(lens, budget)) for lens in unit]
        elif size:
            rank_counts = [float(-(-len(lens) // size)) for lens in unit]
        else:
            continue
        per_unit_max += max(rank_counts)
        per_unit_mean += _mean(rank_counts)
        measured = True
    if measured:
        metrics["rank_mb_count_max"] = per_unit_max
        metrics["rank_mb_count_mean"] = per_unit_mean
        metrics["rank_mb_imbalance"] = _ratio(per_unit_max, per_unit_mean)
    return metrics


def _min_micro_batches(lengths: Sequence[int], budget: int) -> int:
    """First-fit bin count, the rule ``get_data_iterator`` applies per rank before
    the DP MAX all-reduce. Kept in step with (and pinned against)
    ``miles.utils.data.get_minimum_num_micro_batch_size``, which is not imported
    here because that module pulls the rollout stack in behind it."""
    bins: list[int] = []
    for length in lengths:
        for i, packed in enumerate(bins):
            if packed + length <= budget:
                bins[i] = packed + length
                break
        else:
            bins.append(length)
    return len(bins)


def _stage_metrics(stages: dict[str, float], actor_train_time: float) -> dict[str, float]:
    """Seconds per window for the parts of the gap between two optimizer steps.

    ``actor_optim_time`` is inside ``actor_train_time``; the rest is outside it,
    so ``actor_window_accounted_time`` is what the trainer can explain of the
    wall clock between consecutive optim steps."""
    metrics = {_STAGE_METRIC_KEYS[name]: seconds for name, seconds in stages.items() if name in _STAGE_METRIC_KEYS}
    if actor_train_time > 0:
        metrics["actor_fwd_bwd_time"] = max(0.0, actor_train_time - stages.get(STAGE_OPTIM, 0.0))
    outside = sum(stages.get(name, 0.0) for name in _OUTSIDE_TRAIN_TIMER)
    if actor_train_time > 0 or outside > 0:
        metrics["actor_window_accounted_time"] = actor_train_time + outside
    return metrics


def _peak_memory_metrics() -> dict[str, float]:
    """This rank's peak CUDA memory over the window, then reset the peak.

    The headroom decides whether a bigger ``--max-tokens-per-gpu`` (or one more
    row per DP rank) fits, and an OOM 20 h into a run is otherwise the first
    time anyone learns how close it was. Reset per window so the number tracks
    the current shape rather than a startup transient.
    """
    if not torch.cuda.is_available():
        return {}
    gib = 1024**3
    peak = {
        "actor_peak_mem_alloc_gib": torch.cuda.max_memory_allocated() / gib,
        "actor_peak_mem_reserved_gib": torch.cuda.max_memory_reserved() / gib,
    }
    torch.cuda.reset_peak_memory_stats()
    return peak
