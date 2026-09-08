# MFU for hybrid linear-attention MoE models (Qwen3.5 / Qwen3.6)

`actor_train_mfu` (Tinker optim_step metrics, `perf.jsonl`, platoon
`optim/actor_train_mfu`) and the native `perf/actor_train_mfu` are
`3 x forward FLOPs / actor_train_time / peak` with the forward FLOPs from
`miles/utils/flops_utils.calculate_fwd_flops(seqlens, args)`. Before this change
the model charged quadratic softmax attention on every one of `args.num_layers`
layers and ignored the attention output gate and the MTP block. For
Qwen3.6-35B-A3B (30 Gated DeltaNet + 10 full-attention MoE layers) that
over-credited the trainer by ~2.4x at 64k context, so the reported MFU was
meaningless at the length this fork trains.

## What changed

- `calculate_qkv_projection_flops`: `q_flops *= 2` when
  `args.attention_output_gate` (Qwen3.5/3.6 `attn_output_gate: true`; the model
  args pass `--attention-output-gate`). The q projection also emits a per-head
  gate of the same width.
- `calculate_layer_flops(..., attention_kind="full_attention")`:
  `full_attention` is the previous formula; `linear_attention` is the Gated
  DeltaNet layer below, with no `S^2` term.
- `calculate_fwd_flops`: when `args.flops_layer_types` is a list (one of
  `full_attention` / `linear_attention` per layer, length `num_layers`) the
  layers are iterated one by one, the FFN width of layer `i` following
  `moe_layer_freq[i]` (or the int rule / all-MoE default). When the attribute is
  `None` or absent the previous two-bucket dense/MoE arithmetic runs unchanged.
  When `args.enable_mtp_training` and `args.mtp_num_layers` are set, each MTP
  depth adds one full-attention layer (MoE width when the model is MoE), the
  `eh_proj` (`2H -> H`), the embedding and one more lm-head pass.
- `ensure_hf_layer_pattern(args)`: fills `args.flops_layer_types` and
  `args.flops_linear_attention` once from `args.hf_checkpoint`'s HF config
  (`get_text_config()`, then `text_config`, then the config itself). It reads
  `layer_types` or derives it from `full_attention_interval` exactly like
  `miles_plugins/models/qwen3_5.py`, and the GDN dims from
  `linear_num_key_heads`, `linear_num_value_heads`, `linear_key_head_dim`,
  `linear_value_head_dim`, `linear_conv_kernel_dim`. Dense and plain-MoE
  configs (no `linear_attention` layer; kinds such as `sliding_attention` count
  as full attention) leave `flops_layer_types=None`. It never raises: a missing
  or unreadable config, a layer-count mismatch against `args.num_layers`, or
  missing GDN dims log one warning and keep the old formula. The result
  (including a failure) is cached on `args`, so the config is read once per
  process.
- Callers: `tinker_perf.pop_optim_window_metrics` (every rank, before the FLOPs
  model) and the native `perf/actor_train_*` site in
  `miles/utils/train_metric_utils.log_perf_data_raw` (primary rank). The FSDP
  adapter `flops_args_from_hf_config` carries the same three attributes
  (`attention_output_gate`, `flops_layer_types`, `flops_linear_attention`), so
  the FSDP perf path gets the same numbers without the Megatron args.
- No flag, no change to the reported metric names (`actor_train_mfu`,
  `actor_train_tflops`, `actor_train_tok_per_s`, `actor_train_time` and the
  `perf/*` equivalents). `deploy/clients/perf_report.py` is unchanged.

## Formulas (per layer, sequence length `S`, hidden `H`)

Full attention (heads `A`, query groups `G`, head dim `d = kv_channels`, gate `g`
= 2 with the output gate, else 1):

```
q       = 2 S H A d g
k, v    = 2 x 2 S H G d
scores  = 2 A S^2 d / 2        (causal)
A*V     =   A S^2 d
out     = 2 S H H
ffn     = 2 S H F x 3          F = ffn_hidden_size, or moe_ffn x topk + shared for MoE layers
```

Gated DeltaNet (`linear_attention`; `Kh` key heads x `Kd`, `Vh` value heads x `Vd`,
short conv kernel `c`; `K = Kh Kd`, `V = Vh Vd`):

```
q, k    = 2 S H (2K)
v, z    = 2 S H (2V)           z is the gate
a, b    = 2 S H (2 Vh)         decay and beta
conv    = 2 S (2K + V) c
recur   = 4 S Vh Kd Vd         state update + readout
out     = 2 S V H
ffn     = as above
```

MTP depth: `full_attention layer + 2 S (2H) H + 2 S H + 2 S H Vocab`.

Model total: `sum over layers + 2 S H Vocab` (+ `mtp_num_layers x` the MTP depth),
summed over the padded sequence lengths of the window; `actor_train_tflops` is
3x that over `actor_train_time`, divided by the world size.

## Qwen3.6-35B-A3B at S = 65536 (per token)

`H 2048, A 16, G 2, d 256, gate on, 40 layers, MoE 512 x 8 + shared 512, V 248320,
GDN Kh 16 Kd 128 Vh 32 Vd 128 c 4`:

| Term | MFLOPs / token |
|---|---|
| full-attention layer, attention part (q 33.6 + kv 4.2 + scores/AV 536.9 + out 8.4) | 583.0 |
| Gated DeltaNet layer, attention part | 69.5 |
| MoE FFN (8 x 512 + 512) | 56.6 |
| lm head | 1017.1 |
| MTP depth (layer 639.6 + eh_proj 16.8 + embedding + lm head 1017.1) | 1673.5 |

Total: `30 x (69.5 + 56.6) + 10 x (583.0 + 56.6) + 1017.1 = 11 198 MFLOPs/token`
(11.20 GFLOPs; the design's 11.0 figure omits the output gate). The old formula
gave 26.60 GFLOPs/token, an old/new ratio of **2.38** at 64k (1.86 at 32k, 1.20
at 8k, 0.97 at 2k: at short lengths the GDN projections cost slightly more than
the softmax attention they replace). With `--enable-mtp-training --mtp-num-layers 1`
add 1.67 GFLOPs/token.

Per 64k row per rank the trainer is credited `3 x 11.20 G x 65536 = 2.20 PFLOP`.
At 989 TFLOPS bf16 peak an MFU of 0.15 needs <= 14.8 s per row, including full
recompute, the EP8 alltoall, the fp32 GDN chunk kernels and the 10 quadratic
flash layers.

## Sanity band

For the 64k smoke and the 15-node runs, `actor_train_mfu` should land in
**0.05-0.35**. Below 0.05 points at idle DP ranks (units with few real rows:
check `num_datums` vs DP size and `--tinker-coalesce-wait-s`), a stalled
alltoall or a timer that includes waiting; above 0.35 means the layer pattern
was not applied (look for the `MFU: could not read the layer pattern` warning
or the absence of the `MFU: 30 linear-attention + 10 full-attention layers`
line in the trainer log). `actor_train_tok_per_s` and seconds per 64k row are
the primary throughput numbers; MFU is the cross-check.

## Diagnosing a low MFU

`actor_train_mfu` says the trainer is slow; it never says why. The same
optim-window dict carries the numbers that decompose it, so every one of them is
`optim/<key>` in platoon's W&B and a column of `perf.jsonl`. All of them are
free: no new collective, and the per-rank arithmetic is a few thousand Python
multiplications per window. Any of them may be **absent** - a field the trainer
or the gateway could not supply drops its keys rather than failing the step.

### 1. Rank imbalance (the leading hypothesis)

The batch is split by `get_seqlen_balanced_partitions(..., equal_size=True)`
(`miles/ray/rollout/train_data_conversion.py`), which equalises the SUM of
sequence lengths with equal row counts. That equalises tokens, not work: on the
10 full-attention layers the cost is quadratic, so `[100, 100]` and `[190, 10]`
are the same tokens and not the same FLOPs. The step waits for the slowest rank,
so `rank_*_max` is what it actually paid for.

| Metric | Meaning |
|---|---|
| `rank_dp_size` | DP ranks the split covered |
| `rank_tokens_max` / `_mean` / `_min` | tokens the busiest / average / idlest rank was given, summed over the window's units |
| `rank_token_imbalance` | `rank_tokens_max / rank_tokens_mean` (1.0 = perfect) |
| `rank_token_imbalance_worst_unit` | the worst single unit's ratio |
| `rank_token_imbalance_weighted_mean` | the per-unit ratio, token-weighted over the window |
| `rank_fwd_tflops_max` / `_mean` / `_min` | the same three ranks in forward TFLOPs **per GPU**; `rank_fwd_tflops_mean == actor_train_fwd_tflops` by construction |
| `rank_flops_imbalance`, `rank_flops_imbalance_worst_unit`, `rank_flops_imbalance_weighted_mean` | the same ratios on FLOPs |

`rank_flops_imbalance > rank_token_imbalance` is the signature of balancing by
length instead of by work, and `rank_flops_imbalance` is directly the factor by
which a work-balanced split could raise MFU. `--balance-by-flops` already exists
on the scheduled path (`miles/utils/dp_schedule.py`); this metric says whether
it is worth wiring into the tinker path.

No collective: under `--delay-split-train-data-by-dp` every trainer rank runs
`split_train_data_by_dp_raw` over the whole batch, so it already knows every
rank's rows. `_package_shards` writes them into each shard as `dp_seq_lens`
(a few hundred ints) and `process_rollout_data_shard` moves them onto
`Timer().dp_seq_lens`, next to the `Timer().seq_lens` the FLOPs model uses.

### 2. Sequence shape (the legitimate part of a low MFU)

| Metric | Meaning |
|---|---|
| `datum_seq_len_count` / `_mean` / `_p50` / `_p90` / `_max` | the window's **unpadded** datum lengths (gateway side, nearest-rank percentiles) |
| `fwd_gflops_per_token` | global forward FLOPs / global tokens this window earned |
| `actor_train_fwd_tflops` | per-GPU forward TFLOPs the MFU was computed from |
| `padded_num_tokens` / `padded_num_datums` | what the trainer actually forwarded (`num_tokens` / `num_datums` are the gateway's unpadded truth) |

Read `fwd_gflops_per_token` against the **11.20 GFLOP/token** of a uniform 64k
batch above. A 29k-mean window earns ~8 GFLOP/token, so ~1.36x of a "low" MFU is
simply less work per token and is not a throughput loss at all; what is left
over is.

### 3. Where the wall clock goes

`actor_train_time` covers only `run_forward_backward_pass` and the optimizer
step. These account for the rest of the gap between two optimizer steps:

| Metric | Where |
|---|---|
| `actor_data_fetch_time` | `get_rollout_data` in `miles/backends/megatron_utils/actor.py` (object store + H2D) |
| `actor_data_iter_time` | `get_data_iterator` (micro-batch balancing and its DP MAX all-reduce) |
| `actor_replay_fill_time` | the R3 `fill_replay_data`, ~29 GB of int32 `[T-1, 40, 8]` per step |
| `actor_optim_time` | the optimizer step, **inside** `actor_train_time` |
| `actor_fwd_bwd_time` | `actor_train_time - actor_optim_time` |
| `actor_window_accounted_time` | `actor_train_time` + the three stages outside it |

`(client step time) - actor_window_accounted_time` is what the trainer cannot
see: rollouts, transfer and the sampler push.

### 4. Micro-batch shape

| Metric | Meaning |
|---|---|
| `mb_count`, `mb_per_unit_max`, `mb_per_unit_mean` | micro-batches this rank ran |
| `mb_tokens_mean`, `mb_tokens_max` | tokens per micro-batch on this rank |
| `rank_mb_count_max`, `rank_mb_count_mean`, `rank_mb_imbalance` | micro-batches each DP rank *needed* before `get_data_iterator` raised them all to the DP max |

Small `mb_tokens_mean` with a healthy `rank_*_imbalance` means "many small
kernels" (raise `--max-tokens-per-gpu`); a large `rank_mb_imbalance` means the
DP max forces every rank through the straggler's pass count.

### 5. Padding

| Metric | Meaning |
|---|---|
| `real_rows`, `pad_rows`, `pad_row_frac` | rows a datum asked for vs rows the DP padding added |
| `pad_tokens`, `pad_token_frac` | the same in tokens |

Measured, not assumed: pad rows are ~32% of the row SLOTS and carry 2 tokens
each, so `pad_token_frac` is ~1e-5 (744 of 22.9M tokens on the live run).
Padding is not the problem. The block is emitted only when one slot stepped,
because the trainer's `padded_num_*` window is process-wide.

Tests: `tests/fast/utils/test_flops_layer_types.py` (dense/MoE goldens
unchanged, hand formulas for the GDN layer and the 64k total, the 2.2-2.6
ratio band, the MTP term, `ensure_hf_layer_pattern` on fake configs),
`tests/fast/utils/test_flops_from_hf_config.py`,
`tests/fast/backends/megatron_utils/test_tinker_perf.py` (the diagnostics'
arithmetic on hand-computable fixtures),
`tests/fast/utils/test_train_metric_utils.py`,
`tests/fast/tinker/test_runtime.py` (shape percentiles and the padding block),
`tests/fast/tinker/test_full_executor.py` (the wall-clock stages),
`tests/fast/ray/rollout/test_train_data_conversion.py` (`dp_seq_lens`).
