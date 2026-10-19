"""Stochastic rounding of the FP32 master weights into the BF16 model weights.

Megatron's distributed optimizer keeps FP32 master shards and, after each Adam
step, copies them into the BF16 model weights with round-to-nearest. At lr 1e-6
a master weight drifts by ~2^-20 per update, while a BF16 weight in
[2^e, 2^(e+1)) only changes once that drift crosses half its spacing, after
2^(e+12) updates. Every weight starts exactly on the BF16 grid, so a whole
magnitude bucket crosses together: e=-10 at update 4, e=-9 (13% of Qwen3.5-4B's
weights) at update 8, e=-8 at 16. Those updates move the policy several steps'
worth at once; the KL spikes one step later (logged steps 5, 9, 17) and batch
32's collapse began with the rollouts after update 8.

Stochastic rounding replaces round-to-nearest with rounding up with probability
equal to the fraction of the BF16 spacing the FP32 value has covered. The BF16
weight then equals the master weight in expectation and moves a little at every
update instead of in synchronized bursts.

Fresh noise at every copy (``noise="fresh"``) re-rolls every weight whose master
sits between two BF16 values, so once the masters have drifted off the grid
those weights jitter between their neighbours every step whether or not the
master moved: measured on Qwen3.5-4B at update 2, 0.243/0.133/0.069 of the
e-10/e-9/e-8 buckets changed against 0.108/0.054/0.027 of real drift, and the
sampler-vs-trainer KL rose to 0.0011. ``noise="fixed"`` draws the same noise for
each element at every copy, i.e. gives every weight its own fixed rounding
threshold: a weight changes only when its master crosses that threshold, so a
steady drift changes it once per spacing covered, at a point spread uniformly
across weights. No jitter and no synchronized bursts; still unbiased over
weights (each threshold is uniform), though not per weight over time.

Only the BF16 copy changes: the FP32 masters, Adam state and checkpoints are
untouched, so turning this on or off never alters what the optimizer stores.
"""

import logging
import types

import torch

logger = logging.getLogger(__name__)

# BF16 keeps the top 16 bits of an FP32 value.
_LOW_BITS = 1 << 16
_HIGH_MASK = -_LOW_BITS  # 0xFFFF0000 as a signed int32
# Shards are rounded in slices so the int32 noise stays small for a large shard.
_SLICE = 1 << 24


def stochastic_round_to_bf16(values: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Round an FP32 tensor to BF16, up with probability equal to the discarded fraction.

    Adding uniform noise below the kept bits and then truncating rounds the
    magnitude up exactly when the noise carries into the kept bits; FP32 is
    sign-magnitude, so the same holds for negative values and the result is
    unbiased either way. Values already on the BF16 grid, zeros, infinities and
    NaNs come through unchanged.
    """
    assert values.dtype == torch.float32, f"expected FP32 master weights, got {values.dtype}"
    bits = values.contiguous().view(torch.int32)
    noise = torch.randint(0, _LOW_BITS, bits.shape, dtype=torch.int32, device=bits.device, generator=generator)
    rounded = ((bits + noise) & _HIGH_MASK).view(torch.float32)
    # A finite value next to the largest BF16 magnitude could round up into inf; keep
    # non-finite inputs (and that edge) on round-to-nearest.
    rounded = torch.where(torch.isfinite(values) & torch.isfinite(rounded), rounded, values)
    return rounded.to(torch.bfloat16)  # exact: the low 16 bits are already zero


NOISE_MODES = ("fresh", "fixed")


def install(optimizer, seed: int, noise: str = "fresh") -> int:
    """Make each distributed optimizer under ``optimizer`` round its BF16 copy stochastically.

    Wraps the instance's ``_copy_main_params_to_model_params``, which Megatron
    calls after the inner Adam step and before the parameter all-gather, so the
    stochastically rounded shards are what every data-parallel rank gathers,
    what the forward pass uses and what the weight sync exports. Megatron's own
    copy still runs first (it also handles FP8/FP32 groups); the BF16 shards are
    then rewritten from the FP32 masters. ``noise="fixed"`` reseeds the stream
    before every copy, so each element sees the same noise every time (the shard
    layout and copy order are fixed). Returns how many optimizers were wrapped.
    """
    assert noise in NOISE_MODES, f"stochastic rounding noise must be one of {NOISE_MODES}, got {noise!r}"
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    wrapped = 0
    for child in getattr(optimizer, "chained_optimizers", None) or [optimizer]:
        if not isinstance(child, DistributedOptimizer):
            logger.warning(f"stochastic rounding: {type(child).__name__} is not a DistributedOptimizer; left as is")
            continue
        if getattr(child, "_stochastic_rounding_installed", False):
            wrapped += 1
            continue
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        # Own stream, made on the shards' device at first use: dropout and other RNG stay as they were.
        child._stochastic_rounding_seed = seed * 1_000_003 + rank
        child._stochastic_rounding_generator = None
        child._stochastic_rounding_fixed = noise == "fixed"
        child._copy_main_params_to_model_params = types.MethodType(
            _rounded_copy(child._copy_main_params_to_model_params), child
        )
        child._stochastic_rounding_installed = True
        wrapped += 1
    logger.info(f"stochastic rounding of BF16 weights ({noise} noise) installed on {wrapped} optimizer(s)")
    return wrapped


def _rounded_copy(nearest_copy):
    def copy(self) -> None:
        nearest_copy()
        if self.is_stub_optimizer or self.ddp_config.use_megatron_fsdp:
            return
        reseed = self._stochastic_rounding_fixed
        for shard_main_group, model_group in zip(self.shard_fp32_from_float16_groups, self.model_float16_groups):
            for shard_main_param, model_param in zip(shard_main_group, model_group):
                if self._is_distopt_quantized_param(model_param):
                    continue
                world_range = self._get_model_param_range_map(model_param)["gbuf_world_in_bucket"]
                gbuf_index, _, bucket_id = self.model_param_gbuf_map[model_param]
                shard_model_param = self.buffers[gbuf_index].buckets[bucket_id].param_data.view(-1)[
                    world_range.start : world_range.end
                ]
                if shard_model_param.dtype != torch.bfloat16:
                    continue
                if self._stochastic_rounding_generator is None:
                    self._stochastic_rounding_generator = torch.Generator(device=shard_main_param.device)
                    self._stochastic_rounding_generator.manual_seed(self._stochastic_rounding_seed)
                elif reseed:
                    self._stochastic_rounding_generator.manual_seed(self._stochastic_rounding_seed)
                reseed = False
                for source, target in zip(
                    shard_main_param.detach().view(-1).split(_SLICE), shard_model_param.split(_SLICE)
                ):
                    target.copy_(stochastic_round_to_bf16(source, self._stochastic_rounding_generator))

    return copy
