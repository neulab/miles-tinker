"""MFU bookkeeping for the Tinker executors.

One optim window = every forward_backward unit since the previous optim_step
plus the step itself. Units time their forward/backward under the
``actor_train`` timer and record the unit's sequence lengths here; optim_step
drains both into the same numbers the native ``perf/actor_train_*`` metrics
report (3 x forward FLOPs over the timed window). No collectives: every rank
computes the same values and the driver reads the first worker's.
"""

from argparse import Namespace
from collections.abc import Iterable

import torch
import torch.distributed as dist

from miles.utils.device_flops import local_peak_bf16_tflops
from miles.utils.flops_utils import ensure_hf_layer_pattern, fwd_tflops_per_gpu
from miles.utils.timer import Timer

ACTOR_TRAIN_TIMER = "actor_train"

_window_seq_lens: list[int] = []
_window_num_units = 0


def note_unit(seq_lens: Iterable[int]) -> None:
    """Record one trained unit's (padded, batch-global) sequence lengths."""
    global _window_num_units
    _window_seq_lens.extend(int(length) for length in seq_lens)
    _window_num_units += 1


def pop_window() -> tuple[list[int], int]:
    global _window_seq_lens, _window_num_units
    seq_lens, _window_seq_lens = _window_seq_lens, []
    num_units, _window_num_units = _window_num_units, 0
    return seq_lens, num_units


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
    them with the exact per-slot counts."""
    timer_instance = Timer()
    actor_train_time = timer_instance.log_dict().get(ACTOR_TRAIN_TIMER, 0.0)
    timer_instance.reset(ACTOR_TRAIN_TIMER)
    seq_lens, num_units = pop_window()
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
        "num_tokens": sum(seq_lens),
        "num_datums": len(seq_lens),
        "num_units": num_units,
    }


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
