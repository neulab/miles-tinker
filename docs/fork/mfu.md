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

Tests: `tests/fast/utils/test_flops_layer_types.py` (dense/MoE goldens
unchanged, hand formulas for the GDN layer and the 64k total, the 2.2-2.6
ratio band, the MTP term, `ensure_hf_layer_pattern` on fake configs),
`tests/fast/utils/test_flops_from_hf_config.py`,
`tests/fast/backends/megatron_utils/test_tinker_perf.py`,
`tests/fast/utils/test_train_metric_utils.py`.
