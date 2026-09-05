"""One optim window's units drain into the native actor_train perf formulas."""

from argparse import Namespace

import pytest

from miles.backends.megatron_utils import tinker_perf
from miles.utils.timer import Timer


@pytest.fixture(autouse=True)
def clean_window():
    tinker_perf.pop_window()
    Timer().reset(tinker_perf.ACTOR_TRAIN_TIMER)
    yield
    tinker_perf.pop_window()
    Timer().reset(tinker_perf.ACTOR_TRAIN_TIMER)


def test_perf_metrics_follow_the_native_formulas():
    metrics = tinker_perf.perf_metrics(
        actor_train_time=2.0, total_fwd_tflops=100.0, num_tokens=4000, peak_tflops=1000.0
    )
    assert metrics == {
        "actor_train_time": 2.0,
        "actor_train_tflops": 150.0,
        "actor_train_tok_per_s": 2000.0,
        "actor_train_mfu": 0.15,
    }


def test_perf_metrics_are_empty_without_a_timed_window():
    assert tinker_perf.perf_metrics(actor_train_time=0.0, total_fwd_tflops=1.0, num_tokens=1, peak_tflops=1.0) == {}


def test_optim_window_drains_units_and_timer(monkeypatch):
    monkeypatch.setattr(tinker_perf, "fwd_tflops_per_gpu", lambda seq_lens, args, world_size: 10.0)
    tinker_perf.note_unit([3, 5])
    tinker_perf.note_unit([2])
    Timer().add(tinker_perf.ACTOR_TRAIN_TIMER, 0.5)

    metrics = tinker_perf.pop_optim_window_metrics(Namespace(mfu_peak_tflops=600.0))

    assert metrics["num_tokens"] == 10 and metrics["num_datums"] == 3 and metrics["num_units"] == 2
    assert metrics["actor_train_time"] == 0.5
    assert metrics["actor_train_tflops"] == pytest.approx(60.0)
    assert metrics["actor_train_mfu"] == pytest.approx(0.1)
    assert metrics["actor_train_tok_per_s"] == pytest.approx(20.0)
    # drained: the next window starts empty
    assert tinker_perf.pop_window() == ([], 0)
    assert tinker_perf.ACTOR_TRAIN_TIMER not in Timer().log_dict()


def test_unmeasured_window_reports_counts_only():
    metrics = tinker_perf.pop_optim_window_metrics(Namespace(mfu_peak_tflops=600.0))
    assert metrics == {"num_tokens": 0, "num_datums": 0, "num_units": 0}
