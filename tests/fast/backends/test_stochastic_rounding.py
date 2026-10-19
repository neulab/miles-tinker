"""Stochastic rounding of FP32 master weights into BF16: unbiased, exact on the grid,
sign-correct, and wired into the distributed optimizer's master-to-model copy."""

import types

import pytest
import torch

from miles.backends.megatron_utils import stochastic_rounding as sr


def _gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_values_on_the_bf16_grid_are_unchanged():
    grid = torch.tensor([0.0, -0.0, 1.0, -2.5, 2.0**-9 * 1.5, 3.0e38, float("inf"), -float("inf")]).bfloat16().float()
    out = sr.stochastic_round_to_bf16(grid, _gen())
    assert torch.equal(out.float(), grid)
    assert torch.isnan(sr.stochastic_round_to_bf16(torch.tensor([float("nan")]), _gen())).all()


@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("fraction", [0.1, 0.25, 0.5, 0.9])
def test_rounds_up_with_probability_equal_to_the_covered_fraction(sign, fraction):
    low = torch.tensor(2.0**-9 * 1.5)  # on the grid; spacing there is 2^-9 * 2^-7
    ulp = 2.0**-9 * 2.0**-7
    x = (low + fraction * ulp) * sign
    samples = sr.stochastic_round_to_bf16(x.repeat(200_000), _gen(1)).float()
    assert set(samples.abs().unique().tolist()) <= {low.item(), low.item() + ulp}
    up = (samples.abs() > low).float().mean().item()
    assert up == pytest.approx(fraction, abs=0.01)
    assert samples.mean().item() == pytest.approx(x.item(), rel=1e-4), "unbiased in expectation"


def test_nearest_rounding_would_freeze_a_slow_drift_that_stochastic_rounding_follows():
    """The burst mechanism in miniature: 2^-20 per update on a weight in [2^-9, 2^-8),
    whose BF16 spacing is 2^-16. Update 8 lands exactly on half a spacing (a tie, which
    round-to-nearest-even keeps), so nearest rounding changes every weight at update 9."""
    start = torch.full((50_000,), 2.0**-9 * 1.5)
    master = start.clone()
    nearest_changed, stochastic_mean = [], []
    for _ in range(9):
        master += 2.0**-20
        nearest_changed.append((master.bfloat16().float() != start).float().mean().item())
        stochastic_mean.append(sr.stochastic_round_to_bf16(master, _gen(len(stochastic_mean))).float().mean().item())
    assert nearest_changed[:8] == [0.0] * 8 and nearest_changed[8] == 1.0, "nearest: nothing, then everything at once"
    for update, mean in enumerate(stochastic_mean, start=1):
        assert mean == pytest.approx(start[0].item() + update * 2.0**-20, rel=2e-4), "stochastic: tracks the master"


class _Bucket:
    def __init__(self, n: int) -> None:
        self.param_data = torch.zeros(n, dtype=torch.bfloat16)


def _fake_distributed_optimizer():
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    opt = object.__new__(DistributedOptimizer)
    model_param = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
    master = torch.full((4,), 2.0**-9 * 1.5 + 0.5 * 2.0**-16)  # exactly half a BF16 spacing above the grid
    bucket = _Bucket(4)
    opt.is_stub_optimizer = False
    opt.ddp_config = types.SimpleNamespace(use_megatron_fsdp=False)
    opt.shard_fp32_from_float16_groups = [[master]]
    opt.model_float16_groups = [[model_param]]
    opt.model_param_gbuf_map = {model_param: (0, torch.bfloat16, 0)}
    opt.buffers = [types.SimpleNamespace(buckets=[bucket])]
    opt._get_model_param_range_map = lambda param: {"gbuf_world_in_bucket": types.SimpleNamespace(start=0, end=4)}
    opt._is_distopt_quantized_param = lambda param: False
    calls = []
    opt._copy_main_params_to_model_params = lambda: (calls.append("nearest"), bucket.param_data.copy_(master))
    return opt, master, bucket, calls


def test_install_wraps_the_master_to_model_copy(monkeypatch):
    opt, master, bucket, calls = _fake_distributed_optimizer()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    assert sr.install(types.SimpleNamespace(chained_optimizers=[opt]), seed=1234) == 1
    assert sr.install(types.SimpleNamespace(chained_optimizers=[opt]), seed=1234) == 1, "idempotent"
    seen = set()
    for _ in range(64):
        opt._copy_main_params_to_model_params()
        seen.update(bucket.param_data.float().tolist())
    assert calls == ["nearest"] * 64, "Megatron's own copy still runs first"
    low = torch.tensor(2.0**-9 * 1.5).item()
    assert seen == {low, low + 2.0**-16}, "both neighbours appear: the shard is rounded stochastically"


def test_non_distributed_optimizers_are_left_alone():
    plain = types.SimpleNamespace(_copy_main_params_to_model_params=lambda: None)
    assert sr.install(types.SimpleNamespace(chained_optimizers=[plain]), seed=0) == 0


def _installed(noise: str, monkeypatch):
    opt, master, bucket, calls = _fake_distributed_optimizer()
    master.copy_(torch.linspace(0.05, 0.95, 4) * 2.0**-16 + 2.0**-9 * 1.5)  # between grid points
    big = 4096
    opt.shard_fp32_from_float16_groups = [[torch.full((big,), master[1].item())]]
    bucket.param_data = torch.zeros(big, dtype=torch.bfloat16)
    opt._get_model_param_range_map = lambda param: {"gbuf_world_in_bucket": types.SimpleNamespace(start=0, end=big)}
    opt._copy_main_params_to_model_params = lambda: None
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    sr.install(types.SimpleNamespace(chained_optimizers=[opt]), seed=7, noise=noise)
    return opt, opt.shard_fp32_from_float16_groups[0][0], bucket


def test_fresh_noise_jitters_weights_whose_master_does_not_move(monkeypatch):
    """The measured problem: a constant off-grid master re-rolls its BF16 value every copy."""
    opt, master, bucket = _installed("fresh", monkeypatch)
    opt._copy_main_params_to_model_params()
    first = bucket.param_data.clone()
    opt._copy_main_params_to_model_params()
    assert (bucket.param_data != first).float().mean().item() > 0.2


def test_fixed_noise_leaves_weights_alone_while_their_master_stands_still(monkeypatch):
    opt, master, bucket = _installed("fixed", monkeypatch)
    opt._copy_main_params_to_model_params()
    first = bucket.param_data.clone()
    for _ in range(5):
        opt._copy_main_params_to_model_params()
        assert torch.equal(bucket.param_data, first)
    low = torch.tensor(2.0**-9 * 1.5).item()
    up = (first.float() > low).float().mean().item()
    assert up == pytest.approx((master[0].item() - low) / 2.0**-16, abs=0.05), "unbiased over weights"


def test_fixed_noise_changes_a_drifting_weight_once_per_spacing_spread_over_updates(monkeypatch):
    """2^-20 per update on spacing 2^-16: over 16 updates each weight changes exactly once
    (monotone, no back-and-forth), and about 1/16 of them change at each update -- no burst."""
    opt, master, bucket = _installed("fixed", monkeypatch)
    low = 2.0**-9 * 1.5
    master.fill_(low)
    opt._copy_main_params_to_model_params()
    prev = bucket.param_data.clone()
    changes = torch.zeros(master.numel())
    per_update = []
    for _ in range(16):
        master.add_(2.0**-20)
        opt._copy_main_params_to_model_params()
        changed = bucket.param_data != prev
        assert (bucket.param_data.float()[changed] > prev.float()[changed]).all(), "only moves with the drift"
        changes += changed.float()
        per_update.append(changed.float().mean().item())
        prev = bucket.param_data.clone()
    assert changes.max().item() == 1.0
    assert max(per_update) < 0.12, f"no burst: {per_update}"
