# MI300 versus MI350 logprob investigation

Investigation date: 2026-09-14. Model:
`Qwen/Qwen3-4B-Instruct-2507`, snapshot
`cdbee75f17c01a7cc42f958dc650907174af0554`.

## Finding and configuration fix

The dominant discrepancy reproduced on MI300 comes from AITER BF16
normalization rounding. Keep AITER attention and export these two variables in the MI300 launch
shell before running `tools/amd_tinker.py serve`:

```text
CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=0
AITER_JIT_DIR=<runtime-root>/aiter-jit-bf16-rne
```

There are no kernel source patches. The separate cache is necessary because
AITER reuses existing shared libraries regardless of changed compiler flags.
The launcher now forwards these optional host environment variables through
Apptainer's `--cleanenv`. There is no model-specific branch or new CLI flag.

The existing MI350 launch defaults are unchanged. All experiments and service
restarts were on the allocated MI300 host, `k002-002` (eight MI300 GPUs).
Access to `k007-004` was read-only: logs, source files, process configuration,
and existing checkpoint metadata. Its active training job was not modified.
The client checkout and virtual environment are on the host reached by
`babel-compute-node`, under `/tmp/platoon-mi300-match`.

## Historical run comparison

| Run | First batch v1 | First batch v2 | RMS logprob error |
| --- | ---: | ---: | ---: |
| [MI350, 8cc8lafh](https://wandb.ai/9998_team/number-search-platoon-tinker/runs/8cc8lafh) | 0.00205527 | 0.00103057 | 0.0453998 |
| [MI300, amugbaqy](https://wandb.ai/9998_team/number-search-platoon-tinker/runs/amugbaqy) | 0.02771414 | 0.06044021 | 0.347679 |

The client training configurations are equal, and both recorded Platoon commit
`cea67e438c222c2b642dcd1f04e348b354297763`. The v2 difference is approximately
58.6-fold on the first batch. This metric is
`0.5 * mean((sampling_logprob - training_logprob)**2)` over action tokens,
not an exact KL over the entire vocabulary. The first logged `train/step=1`
uses forward results computed before the first optimizer update. Learning
rate, optimizer state, or accumulated policy drift cannot explain that first
discrepancy.

Historical MI350 server arguments were recovered from its initial checkpoint
`model-cbb2ae46da89/weights/000000/iter_0000001/common.pt`, whose timestamp
matches that run. It used four training GPUs and four TP1 inference engines,
TP/PP/CP all 1, context and packing budgets 65000, and requested memory
fraction 0.9. The currently running MI350 job uses a different 6/2 allocation
and smaller budgets; copying that current launch would not reproduce the
historical run.

The matched client uses full fine-tuning, batch 32, group 8, 32 rollout
workers, **max_staleness 3**, 100 updates, one minibatch/microbatch, CISPO
bounds 0/5, LR 1e-6, Adam betas 0.9/0.95, epsilon 1e-8, weight decay 0.017,
and gradient clipping 1. Sampling is temperature 1, top-p 1, maximum 2048
tokens; the renderer is `qwen3_instruct_exact`. Group advantages are centered
without standard-deviation normalization. Checkpoints are requested every
25 steps. The generic MI300 guide's earlier staleness-1 recipe is a different
workload; this comparison uses the historical run's value of 3.

## Attention and other suspected knobs

Both pinned SGLang stacks select **AITER** for this Qwen model on AMD. The
Megatron trainer requests **flash** attention through Transformer Engine.
An isolated MI300 probe with Qwen's head dimensions verified that TE selects
FlashAttention, with outputs identical to the installed FlashAttention call.
Thus the training and inference engines use different attention implementations,
but this is also the intended MI350 arrangement, not evidence of an MI300-only
silent fallback. Changing only inference attention to Triton did not reduce
the reproduced mismatch.

The observed inference settings include BF16 KV cache (`auto`), PyTorch
sampling, chunked prefill 4096, radix caching enabled, and CUDA graphs enabled
in the full server. No separate prefill/decode attention override was set.
SGLang reduces requested memory fraction 0.9 to 0.765 for AITER at this context
length on both architectures. That shared memory adjustment does not explain
the numerical gap. There was no evidence requiring MI300-specific graph,
cache, scheduling, or attention tuning for this problem.

Initial weight-equality checks passed for all four inference engines after
Megatron-to-HF export. Direct attention and RoPE probes did not show a gross
precision failure. The Apex import warning about lower-precision AITER RoPE
also appears on MI350; the packed trainer path examined here uses TE RoPE.
These checks do not establish that every possible AMD code path is bug-free.

## Controlled frozen-model ablations

Six fixed continuations, 128 tokens each, used prompt lengths 37, 487, and
3637. All comparisons below reuse the same token IDs and initial weights.
The reference is the MI300 Megatron forward pass. No backward or optimizer
request was made during these tests.

| Inference change | Mean prefill/trainer v2 |
| --- | ---: |
| Original AITER attention and operators | 0.01059855 |
| Only attention changed to Triton | 0.01067967 |
| `SGLANG_USE_AITER=0`, explicit AITER attention retained | 0.00187212 |
| Only normalization replaced with the existing vLLM operators | 0.00187212 |
| Only GEMM replaced with the PyTorch path | 0.01059855 |
| AITER retained, BF16 rounding mode 0 with freshly built RMSNorm | **0.00156027** |

The normalization-only outputs were bit-identical to the broader operator
fallback; GEMM-only outputs were bit-identical to baseline. The rounding
configuration reduced mean v2 by 85.3% while retaining AITER. These frozen
probe values are not directly comparable to different sampled number-search
batches in W&B. They isolate the cause; the matched training run is the
workload-level validation.

The original sampled-decode/trainer mean v2 was 0.01155377, while
sampled-decode/prefill mean v2 was 0.00086752. Most of the original gap was
therefore present in inference prefill as well; it was not just decode versus
prefill arithmetic or a KV-cache effect.

## Why the same AITER source behaves differently

Both images contain AITER commit
`417de6df4392120b766537efd50c7725cfa0d5af` and use `SGLANG_USE_AITER=1`.
In this source, `module_rmsnorm_quant` defaults
`CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT` to 2 (truncation). CK's BF16 conversion
uses that software path on `gfx942`, whereas `gfx950` takes the builtin
BF16 conversion path. This affects both normalized output and the residual
written by fused add+RMSNorm; a one-sided rounding bias can accumulate across
layers before any training.

Direct MI300 tests at widths 128 and 2560 matched a truncating reference
almost exactly with the original binary. Rebuilding with mode 0 changed the
outputs to match the round-to-nearest reference almost exactly. The current
MI350 process has neither CK nor OPUS rounding environment overrides.

An independently reported [AITER issue #5151](https://github.com/ROCm/aiter/issues/5151)
describes this same residual-rounding failure mode in RL on gfx942. Its newer
version uses an OPUS setting, so its environment variable should not be copied
blindly into this older CK-based image.

AMD support in the [Miles announcement](https://www.lmsys.org/blog/2026-03-17-rocm-miles-rl-amd/)
and [installation guide](https://miles.radixark.com/docs/getting-started/installation)
does not establish numerical equivalence between independently built inference
and training kernels. The image must also match the architecture: this MI300
image has TE 2.10 development and FlashAttention 2.8.0.post2, while the MI350
image has TE 2.8 and FlashAttention 2.8.3. The software stacks are not byte-for-byte
identical even when training hyperparameters and SGLang/AITER source are matched.

## Reproduction files and validation

The matched workload completed as
[zsr75q6r](https://wandb.ai/9998_team/number-search-platoon-tinker/runs/zsr75q6r).
Its first-batch v2 is **0.0010036085**, versus MI350's 0.0010305726 and the
original MI300 run's 0.0604402050. All 100 optimizer updates completed with
zero skipped updates; W&B reports the run as finished. Mean v2 across the run
was **0.0038031470**, versus MI350's 0.0041416372 and the original MI300's
0.05289756. The final v2 was 0.0056152362; the maximum was 0.0191294868.
All W&B `train` configuration fields match the historical MI350
run after excluding the rollout artifact directory.

The MI300 server and its dedicated SSH tunnel were subsequently stopped at
the user's request. Checkpoints and evidence were retained. The
[training guide](mi300-fresh-machine.md) contains the complete commands
for an independent reproduction.

- MI300 setup: `/tmp/miles-mi300-runtime`, image digest
  `545241a0dbbf8543ef9a6c6d14313b45ba2e78b49ce1ff8370f2c7ab354180e3`.
- Client checkout: `/tmp/platoon-mi300-match` on Babel, branch
  `legacy_code_areal` at the historical commit, with `.venv` and pinned client
  dependency constraints.
- Matched client configuration: `mi350-matched.yaml` in that checkout.
- Client log: `/tmp/platoon-mi300-match-training.log` on Babel.
- Fixed server log: `/tmp/miles-mi300-runtime/logs/server-rne.log` on MI300.
- Environment and server arguments for reproduction:
  `/tmp/mi300-analysis/rne-server-reproduction.json` on MI300.
- Probe scripts, token IDs, raw results, and historical W&B metadata:
  `/tmp/mi300-analysis` on MI300 and `/tmp/mi300-mi350-comparison` on Babel.
- Checkpoint root for the matched run:
  `/work1/grahamneubig/adityabs/miles-tinker/mi300-rne-checkpoints`.

The launcher change passes 13 focused host-side tests, including optional environment
propagation through `--cleanenv`. The tests
run with `--confcutdir=tests/fast` because the repository-wide conftest imports
unrelated SGLang session APIs absent from this pinned runtime.
