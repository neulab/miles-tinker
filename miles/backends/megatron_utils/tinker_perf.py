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

import torch.distributed as dist

from miles.utils.device_flops import local_peak_bf16_tflops
from miles.utils.flops_utils import fwd_tflops_per_gpu
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
    total_fwd_tflops = fwd_tflops_per_gpu(seq_lens, args, world_size) if seq_lens else 0.0
    peak_tflops = getattr(args, "mfu_peak_tflops", None) or local_peak_bf16_tflops()
    metrics = perf_metrics(
        actor_train_time=actor_train_time,
        total_fwd_tflops=total_fwd_tflops,
        num_tokens=sum(seq_lens),
        peak_tflops=peak_tflops,
    )
    return {**metrics, "num_tokens": sum(seq_lens), "num_datums": len(seq_lens), "num_units": num_units}
