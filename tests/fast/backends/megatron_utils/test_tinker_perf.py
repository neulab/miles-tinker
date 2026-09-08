"""One optim window's units drain into the native actor_train perf formulas, plus the
MFU diagnostics that say why the number is what it is (docs/fork/mfu.md)."""

from argparse import Namespace
from types import SimpleNamespace

import pytest

from miles.backends.megatron_utils import tinker_perf
from miles.utils.flops_utils import FULL_ATTENTION, LINEAR_ATTENTION, calculate_fwd_flops
from miles.utils.timer import Timer


@pytest.fixture(autouse=True)
def clean_window():
    def _reset():
        tinker_perf.reset_window()
        Timer().reset(tinker_perf.ACTOR_TRAIN_TIMER)
        Timer().dp_seq_lens = None

    _reset()
    yield
    _reset()


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


def test_optim_window_resolves_the_hf_layer_pattern_before_the_flops_model(monkeypatch):
    order = []
    monkeypatch.setattr(tinker_perf, "ensure_hf_layer_pattern", lambda args: order.append("pattern"))
    monkeypatch.setattr(
        tinker_perf, "fwd_tflops_per_gpu", lambda seq_lens, args, world_size: order.append("flops") or 10.0
    )
    tinker_perf.note_unit([4])
    Timer().add(tinker_perf.ACTOR_TRAIN_TIMER, 1.0)

    metrics = tinker_perf.pop_optim_window_metrics(Namespace(mfu_peak_tflops=600.0))

    assert order == ["pattern", "flops"]
    assert metrics["actor_train_tflops"] == pytest.approx(30.0)


def test_peak_memory_metrics_are_absent_without_cuda_and_reset_the_peak(monkeypatch):
    """Headroom per window: the numbers say whether a bigger --max-tokens-per-gpu fits (CP is always 1 here)."""
    import miles.backends.megatron_utils.tinker_perf as tp

    class _Cuda:
        def __init__(self):
            self.reset_calls = 0

        def is_available(self):
            return True

        def max_memory_allocated(self):
            return 40 * 1024**3

        def max_memory_reserved(self):
            return 44 * 1024**3

        def reset_peak_memory_stats(self):
            self.reset_calls += 1

    cuda = _Cuda()
    monkeypatch.setattr(tp.torch, "cuda", cuda)
    metrics = tp._peak_memory_metrics()
    assert metrics == {"actor_peak_mem_alloc_gib": 40.0, "actor_peak_mem_reserved_gib": 44.0}
    assert cuda.reset_calls == 1, "the peak must reset per window or it reports a startup transient forever"

    monkeypatch.setattr(tp.torch, "cuda", type("_NoCuda", (), {"is_available": staticmethod(lambda: False)})())
    assert tp._peak_memory_metrics() == {}


# --------------------------------------------------------------------------------------
# MFU diagnostics: rank imbalance, sequence shape, wall-clock stages, micro-batch shape
# --------------------------------------------------------------------------------------


def _flops_args(**overrides):
    """A tiny hybrid model: one Gated DeltaNet layer (linear in S) plus one full-attention
    layer (quadratic in S), so a rank's FLOPs depend on the SHAPE of its rows, not only
    on how many tokens it was given."""
    base = dict(
        hidden_size=64,
        num_attention_heads=4,
        num_query_groups=2,
        vocab_size=1024,
        num_layers=2,
        ffn_hidden_size=128,
        kv_channels=16,
        num_experts=None,
        moe_ffn_hidden_size=None,
        moe_router_topk=1,
        moe_shared_expert_intermediate_size=None,
        multi_latent_attention=False,
        q_lora_rank=None,
        kv_lora_rank=None,
        qk_head_dim=0,
        qk_pos_emb_head_dim=0,
        v_head_dim=0,
        attention_output_gate=False,
        flops_layer_types=[LINEAR_ATTENTION, FULL_ATTENTION],
        flops_linear_attention=dict(num_k_heads=2, num_v_heads=2, k_dim=16, v_dim=16, conv_kernel=4),
        enable_mtp_training=False,
        mtp_num_layers=None,
        mfu_peak_tflops=1000.0,
    )
    return Namespace(**(base | overrides))


# Two DP ranks with the SAME token count and a different shape, then a plainly skewed unit.
SAME_TOKENS_DIFFERENT_SHAPE = [[100, 100], [190, 10]]
SKEWED_TOKENS = [[300], [100]]


def _flat(unit):
    return [length for lens in unit for length in lens]


class TestRankImbalance:
    """The leading hypothesis: the batch is balanced by the SUM of sequence lengths with
    equal row counts, which equalises tokens but not work."""

    def _window(self):
        for unit in (SAME_TOKENS_DIFFERENT_SHAPE, SKEWED_TOKENS):
            tinker_perf.note_unit(_flat(unit), {"dp_seq_lens": unit})
        Timer().add(tinker_perf.ACTOR_TRAIN_TIMER, 1.0)
        return tinker_perf.pop_optim_window_metrics(_flops_args())

    def test_token_spread_and_ratios_are_the_hand_numbers(self):
        metrics = self._window()

        assert metrics["rank_dp_size"] == 2
        # per unit: max 200/300, mean 200/200, min 200/100 -> summed over the window
        assert metrics["rank_tokens_max"] == 500
        assert metrics["rank_tokens_mean"] == 400
        assert metrics["rank_tokens_min"] == 300
        assert metrics["rank_token_imbalance"] == pytest.approx(1.25)
        # unit 1 is perfectly balanced by tokens, unit 2 is 300/200
        assert metrics["rank_token_imbalance_worst_unit"] == pytest.approx(1.5)
        # both units weigh 400 tokens: (1.0 * 400 + 1.5 * 400) / 800
        assert metrics["rank_token_imbalance_weighted_mean"] == pytest.approx(1.25)

    def test_equal_tokens_still_cost_unequal_flops(self):
        """The whole point of the metric: 100+100 and 190+10 are the same tokens and not
        the same work, because attention is quadratic on the full-attention layers."""
        args = _flops_args()
        balanced = calculate_fwd_flops([100, 100], args)
        skewed = calculate_fwd_flops([190, 10], args)
        assert skewed > balanced, "the same tokens in fewer, longer rows cost more attention"
        big, small = calculate_fwd_flops([300], args), calculate_fwd_flops([100], args)
        unit_ratios = [skewed / ((skewed + balanced) / 2), big / ((big + small) / 2)]

        metrics = self._window()

        assert metrics["rank_flops_imbalance_worst_unit"] == pytest.approx(max(unit_ratios))
        assert (
            metrics["rank_token_imbalance_worst_unit"] < metrics["rank_flops_imbalance_worst_unit"]
        ), "balancing by the sum of sequence lengths hides work imbalance; that is the metric's whole job"

    def test_mean_rank_flops_reproduce_the_window_total(self):
        """rank_fwd_tflops is per GPU, so the mean over DP ranks must be exactly the
        forward TFLOPs the headline actor_train_tflops was computed from."""
        metrics = self._window()
        assert metrics["rank_fwd_tflops_mean"] == pytest.approx(metrics["actor_train_fwd_tflops"])
        assert metrics["rank_fwd_tflops_min"] < metrics["rank_fwd_tflops_mean"] < metrics["rank_fwd_tflops_max"]

    def test_absent_when_the_units_disagree_on_the_dp_size(self):
        tinker_perf.note_unit([1, 2], {"dp_seq_lens": [[1], [2]]})
        tinker_perf.note_unit([3, 4, 5], {"dp_seq_lens": [[3], [4], [5]]})
        Timer().add(tinker_perf.ACTOR_TRAIN_TIMER, 1.0)

        metrics = tinker_perf.pop_optim_window_metrics(_flops_args())
        assert not any(key.startswith("rank_tokens") for key in metrics)

    def test_absent_without_a_dp_split(self):
        tinker_perf.note_unit([1, 2])
        Timer().add(tinker_perf.ACTOR_TRAIN_TIMER, 1.0)

        metrics = tinker_perf.pop_optim_window_metrics(_flops_args())
        assert not any(key.startswith("rank_") for key in metrics)
        assert metrics["padded_num_datums"] == 2, "the window totals survive a missing dp split"


class TestWindowTotals:
    def test_flops_per_token_is_the_global_forward_flops_over_the_global_tokens(self, monkeypatch):
        monkeypatch.setattr(tinker_perf, "fwd_tflops_per_gpu", lambda seq_lens, args, world_size: 2.0)
        tinker_perf.note_unit([1000, 3000])
        Timer().add(tinker_perf.ACTOR_TRAIN_TIMER, 1.0)

        metrics = tinker_perf.pop_optim_window_metrics(_flops_args())

        assert metrics["padded_num_tokens"] == 4000
        assert metrics["padded_num_datums"] == 2
        assert metrics["actor_train_fwd_tflops"] == 2.0
        # world_size is 1 here: 2.0 TFLOP over 4000 tokens = 0.5 GFLOP/token
        assert metrics["fwd_gflops_per_token"] == pytest.approx(0.5)

    def test_real_model_flops_per_token_grows_with_sequence_length(self):
        """The legitimate half of a low MFU: a short-sequence window earns fewer FLOPs
        per token than a long one, because only the quadratic term scales with length."""
        args = _flops_args()
        short = calculate_fwd_flops([1024], args) / 1024
        long = calculate_fwd_flops([8192], args) / 8192
        assert long > short

        tinker_perf.note_unit([8192])
        Timer().add(tinker_perf.ACTOR_TRAIN_TIMER, 1.0)
        metrics = tinker_perf.pop_optim_window_metrics(args)
        assert metrics["fwd_gflops_per_token"] == pytest.approx(long / 1e9)


class TestStageWallClock:
    def _fake_clock(self, monkeypatch, ticks):
        values = iter(ticks)
        monkeypatch.setattr(tinker_perf, "time", lambda: next(values))

    def test_stages_account_for_the_gap_around_the_train_timer(self, monkeypatch):
        self._fake_clock(monkeypatch, [0.0, 1.0, 10.0, 13.0, 20.0, 20.5, 30.0, 32.0])
        with tinker_perf.stage(tinker_perf.STAGE_DATA_FETCH):
            pass
        with tinker_perf.stage(tinker_perf.STAGE_REPLAY_FILL):
            pass
        with tinker_perf.stage(tinker_perf.STAGE_DATA_ITER):
            pass
        with tinker_perf.stage(tinker_perf.STAGE_OPTIM):
            pass
        tinker_perf.note_unit([4])
        Timer().add(tinker_perf.ACTOR_TRAIN_TIMER, 5.0)

        metrics = tinker_perf.pop_optim_window_metrics(_flops_args())

        assert metrics["actor_data_fetch_time"] == pytest.approx(1.0)
        assert metrics["actor_replay_fill_time"] == pytest.approx(3.0)
        assert metrics["actor_data_iter_time"] == pytest.approx(0.5)
        assert metrics["actor_optim_time"] == pytest.approx(2.0)
        # the optimizer step is inside the train timer, the other three are outside it
        assert metrics["actor_fwd_bwd_time"] == pytest.approx(3.0)
        assert metrics["actor_window_accounted_time"] == pytest.approx(5.0 + 1.0 + 3.0 + 0.5)

    def test_a_stage_that_raises_is_still_charged_and_drains(self, monkeypatch):
        self._fake_clock(monkeypatch, [0.0, 7.0])
        with pytest.raises(ValueError):
            with tinker_perf.stage(tinker_perf.STAGE_REPLAY_FILL):
                raise ValueError("R3 fill blew up")
        tinker_perf.note_unit([4])
        Timer().add(tinker_perf.ACTOR_TRAIN_TIMER, 1.0)

        assert tinker_perf.pop_optim_window_metrics(_flops_args())["actor_replay_fill_time"] == pytest.approx(7.0)
        assert tinker_perf.pop_diagnostics() == ([], {}), "the stage seconds drained with the window"

    def test_stages_are_absent_from_an_untimed_window(self):
        assert tinker_perf.pop_optim_window_metrics(_flops_args()) == {
            "num_tokens": 0,
            "num_datums": 0,
            "num_units": 0,
        }


class TestMicroBatchShape:
    def test_local_schedule_and_cross_rank_demand(self):
        """mb_* is what this rank ran; rank_mb_* is what each rank needed before
        get_data_iterator raised them all to the DP max."""
        # rank 0 packs 3 x 100 into 3 bins of 100; rank 1 packs 3 x 50 into 2
        unit = {
            "dp_seq_lens": [[100, 100, 100], [50, 50, 50]],
            "mb_tokens": [100, 90],
            "mb_token_budget": 100,
        }
        tinker_perf.note_unit([100, 100, 100, 50, 50, 50], unit)
        tinker_perf.note_unit([80], {"mb_tokens": [80]})
        Timer().add(tinker_perf.ACTOR_TRAIN_TIMER, 1.0)

        metrics = tinker_perf.pop_optim_window_metrics(_flops_args())

        assert metrics["mb_per_unit_max"] == 2
        assert metrics["mb_per_unit_mean"] == pytest.approx(1.5)
        assert metrics["mb_count"] == 3
        assert metrics["mb_tokens_mean"] == pytest.approx(90.0)
        assert metrics["mb_tokens_max"] == 100
        assert metrics["rank_mb_count_max"] == 3
        assert metrics["rank_mb_count_mean"] == pytest.approx(2.5)
        assert metrics["rank_mb_imbalance"] == pytest.approx(1.2)

    def test_static_micro_batches_count_rows_per_rank(self):
        tinker_perf.note_unit([1] * 7, {"dp_seq_lens": [[1] * 5, [1] * 2], "micro_batch_size": 2})
        Timer().add(tinker_perf.ACTOR_TRAIN_TIMER, 1.0)

        metrics = tinker_perf.pop_optim_window_metrics(_flops_args())
        # ceil(5/2) = 3 and ceil(2/2) = 1
        assert metrics["rank_mb_count_max"] == 3
        assert metrics["rank_mb_count_mean"] == pytest.approx(2.0)

    def test_first_fit_matches_the_scheduler_helper(self):
        data = pytest.importorskip("miles.utils.data")
        for lengths, budget in (([100, 100, 100], 100), ([50, 50, 50], 100), ([190, 10], 150), ([7, 3, 5, 1], 8)):
            assert tinker_perf._min_micro_batches(lengths, budget) == data.get_minimum_num_micro_batch_size(
                lengths, budget
            )


class TestUnitShape:
    def test_reads_the_dp_split_and_the_dynamic_micro_batch_schedule(self):
        Timer().dp_seq_lens = [[10, 30], [20]]
        iterator = SimpleNamespace(micro_batch_indices=[[0, 2], [1]], micro_batch_size=None)
        args = Namespace(use_dynamic_batch_size=True, max_tokens_per_gpu=64, micro_batch_size=None)

        shape = tinker_perf.unit_shape(args, {"total_lengths": [10, 20, 30]}, [iterator])

        assert shape["dp_seq_lens"] == [[10, 30], [20]]
        assert shape["mb_tokens"] == [40, 20]
        assert shape["mb_token_budget"] == 64, "cp_size is 1 outside a parallel run"

    def test_reads_the_static_micro_batch_schedule(self):
        iterator = SimpleNamespace(micro_batch_indices=None, micro_batch_size=2)
        args = Namespace(use_dynamic_batch_size=False, max_tokens_per_gpu=None, micro_batch_size=2)

        shape = tinker_perf.unit_shape(args, {"total_lengths": [10, 20, 30]}, [iterator])

        assert shape["mb_tokens"] == [30, 30]
        assert shape["micro_batch_size"] == 2
        assert "dp_seq_lens" not in shape

    def test_degrades_to_an_empty_shape_instead_of_raising(self):
        assert tinker_perf.unit_shape(Namespace(), None, None) == {}
        Timer().dp_seq_lens = "not a split"
        shape = tinker_perf.unit_shape(Namespace(), {"total_lengths": [1]}, [object()])
        assert "mb_tokens" not in shape and "dp_seq_lens" not in shape


def test_discard_window_forgets_a_poisoned_window(monkeypatch):
    """A window whose gradients were discarded must not be charged to the next one.

    _discard_dirty drops the grads and fails the optim_step, so
    pop_optim_window_metrics never drains that window; without discard_window its
    units and its actor_train seconds would land in the following window and
    overstate its MFU (and inflate the gateway's pad_rows, which is computed
    against padded_num_datums).
    """
    monkeypatch.setattr(tinker_perf, "fwd_tflops_per_gpu", lambda seq_lens, args, world_size: float(sum(seq_lens)))
    tinker_perf.note_unit([100, 200], {"dp_seq_lens": [[100], [200]]})
    with tinker_perf.stage(tinker_perf.STAGE_DATA_FETCH):
        pass
    Timer().add(tinker_perf.ACTOR_TRAIN_TIMER, 7.0)

    tinker_perf.discard_window()

    assert tinker_perf.pop_window() == ([], 0)
    assert tinker_perf.pop_diagnostics() == ([], {})
    assert tinker_perf.ACTOR_TRAIN_TIMER not in Timer().log_dict()

    # the next window sees only its own work
    tinker_perf.note_unit([5])
    Timer().add(tinker_perf.ACTOR_TRAIN_TIMER, 1.0)
    metrics = tinker_perf.pop_optim_window_metrics(Namespace(mfu_peak_tflops=100.0))
    assert metrics["num_tokens"] == 5 and metrics["num_units"] == 1
    assert metrics["actor_train_time"] == 1.0
    assert "actor_data_fetch_time" not in metrics
