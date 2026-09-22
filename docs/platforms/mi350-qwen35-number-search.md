# Qwen3.5-4B number-search on MI350

## Status

**Real full fine-tuning works on the tested MI350 runtime with private dependency patches.**
The normal-execution numerical probe completed three optimizer updates with
batch sizes 1, 2, and 16. The number-search run has completed 16 rollout batches and 11 actual
optimizer updates; all-success batches with zero advantages omit model
work. Its step-15 full-state checkpoint saved successfully. The 30-batch
experiment is still running as of 2026-09-15 23:13 UTC.

- **Task run:** [W&B eda9q4xy](https://wandb.ai/9998_team/number-search-platoon-tinker/runs/eda9q4xy).
- **Numerical validation:** [W&B g2ij0f5t](https://wandb.ai/9998_team/number-search-platoon-tinker/runs/g2ij0f5t).
- Model: `Qwen/Qwen3.5-4B`, built-in `qwen3_5_disable_thinking` renderer.
- Server: `k007-004.hpcfund`, port `10635`; client: `babel-w5-28`.
- TP1, PP1, DP1, CP1. **No CP2 experiment.**

No Miles or Hyperion repository Python source was edited. The additions are
this guide and one Hyperion YAML config. Runtime dependency copies, an
isolated Babel environment, and a small client wrapper are described below.
The existing four modified platform documents were left untouched.

## Numerical evidence

Sampler and trainer score identical token IDs at the same weight version,
with temperature 1 and top-p 1. This separates numerical disagreement from
policy staleness. The normal-execution probe had no diagnostic hooks.

| Metric | Before updates | After three updates |
| --- | ---: | ---: |
| Generated tokens scored | 1,142 | 1,274 |
| Mean absolute logprob difference | 0.011705 | 0.010030 |
| Sampled KL, k3 estimate | 0.000452 | 0.000382 |
| Mean likelihood ratio | 1.000147 | 1.001024 |
| Maximum absolute logprob difference | 0.293405 | 0.248319 |

All three optimizer steps completed. Fixed-token logprobs changed by mean
absolute **0.036491**, confirming real updates; inference weights were
refreshed before the post-update comparison. These results pass the probe's
mean-absolute-difference `<0.02` and sampled-KL `<0.005` thresholds. They do
not establish bitwise equality or universal agreement on arbitrary contexts.

Standalone Qwen3.5 gated-delta output/gradient checks also passed against a
float32 recurrent reference, including uneven packed boundaries. Maximum
relative RMS errors across tested cases were approximately 0.00415 for
outputs and 0.00449 for input gradients. Actual training uses int32 packed
boundaries; an exploratory int64 test failed and is not a supported result.

An earlier DP4 probe passed three 16-datum updates, but sparse batches
subsequently failed. **DP4 has not been revalidated with the final patches.**
The current evidence supports the DP1 configuration below.

## Task run

The run starts from the original base checkpoint, not the CE-updated probe.

| Completed step | Mean reward, 64 rollouts | Datums | Batch seconds |
| --- | ---: | ---: | ---: |
| 1 | 0.34375 | 591 | 50.2 |
| 2 | 0.484375 | 560 | 43.9 |
| 3 | 0.59375 | 556 | 39.0 |
| 4 | 0.8125 | 567 | 55.8 |
| 5 | 0.96875 | 517 | 40.2 |

No optimizer updates were skipped in these steps. Sample/train KL v2 was
approximately 0.0001–0.0005. At step 1, the mean importance weight was
0.9998 with standard deviation 0.0323. Inspected successful step-5
trajectories narrowed the range using feedback and called the final guess.
Training reward on different batches is not a controlled held-out evaluation.

Settings:

- Full fine-tuning (`lora_rank: 0`), 8 tasks × 8 samples per step.
- 30 training steps; staleness 0; one minibatch and one client microbatch.
- CISPO, clip thresholds 0 and 5; no leave-one-out baseline or depth weighting.
- Adam LR 1e-6, betas 0.9/0.95, epsilon 1e-8, weight decay 0.017, clip norm 1.
- Up to 10 agent turns; 512 output tokens per turn; 16K context.
- Temperature 1, top-p 1; eight rollout workflow workers.
- Evaluation trigger every 10 steps on **1,000 tasks**, one sample each;
  see the rolling-evaluation limitation below.
- Full-state checkpoints initially, every 15 steps, and at completion.
- W&B online, entity `9998_team`, project `number-search-platoon-tinker`.

Step-15 checkpoint: `tinker://model-590598e33d26/weights/000015`; sampling
checkpoint: `tinker://model-590598e33d26/sampler_weights/000015`.

Babel config:
`~/hyperion_modal/plugins/number-search/platoon/number_search/number_search_qwen35_mi350.yaml`.

## Exact runtime

| Component | Tested version |
| --- | --- |
| Hardware | AMD Instinct MI350X, gfx950, 8 × approximately 288 GiB |
| Model HF revision | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Miles checkout revision | `bbe40deb9ae63b3f0de143492f0a8e993d793d23` |
| Container | `rlsys/miles@sha256:e60a69faa831ae2a146819290e8469cf23c81b97febeaf5d11e8baec2bfca285` |
| PyTorch / HIP | `2.9.0a0+git7bcbafe` / `7.0.51831-a3e329ad8` |
| Transformers / FLA | `5.3.0` / `0.4.2` |
| SGLang | `0.5.11.dev36+ga929eb728` |
| Transformer Engine | `2.8.0+a365f2de` |
| Megatron-LM | `d9a5080f2bafea5246add1489247f25368b8a120` |
| mbridge / Megatron Bridge | `0.15.1` / `0.3.0rc0` |
| Babel Tinker / cookbook | `0.16.1` / `0.1.1.dev198+g0be951bd3` |
| Babel IPython / W&B | `9.6.0` / `0.29.0` |

The image's Megatron Bridge lacks Qwen3.5 support. Use the existing local
model implementation with **`--megatron-to-hf-mode raw`** and a converted
text-only torch_dist **`--ref-load`** checkpoint. The model loads
4,205,751,296 text parameters and preserves 24 `A_log` tensors in float32.
No vision input or vision training is needed for this task.

One GPU trains; four separate one-GPU SGLang engines generate. The trainer
uses `HIP_VISIBLE_DEVICES=0`. Inference retains its separate assignments.
There is no activation recomputation in the validated configuration.
SGLang selected Triton gated-delta kernels and BF16 KV cache; it adjusted
the requested 0.7 static-memory fraction to 0.595.

## Runtime changes and what the evidence establishes

All server changes are in private copies under `/tmp/miles-qwen35-mi350/deps`,
bind-mounted over container dependencies. The container image is unchanged.

| Patch | Change | Evidence / remaining uncertainty |
| --- | --- | --- |
| `megatron-rocm-checkpoint.patch` | Disable nonblocking GPU-to-CPU staging before forked checkpoint writers on HIP | Original conversion save segfaulted; patched conversion, loading, and initial full-state saves pass. Pinned-memory inheritance is a suspected cause, not proven. |
| `fla-rocm-gdn-backward.patch` | Pin `chunk_bwd_kernel_dv_local` to 4 warps, 1 stage on HIP | Original isolated gated-delta backward faulted; layer/reference checks pass after patch. |
| `fla-rocm-wy-backward.patch` | Pin `prepare_wy_repr_bwd_kernel` to 4 warps, 1 stage on HIP | Full-model tracing located an illegal access here; sparse full-model and task updates pass with the patch. |
| `fla-rocm-l2norm-backward.patch` | Pin l2norm backward to BT32, 4 warps, 1 stage on HIP | Included in the working bundle after kernel-loading stalls; necessity has not been isolated. |
| `triton-hip-loader-gil.patch` | Release Python GIL around HIP module/function loading, then check statuses | Compiles and passes probes; included in working bundle. A GIL deadlock has not been proven, and necessity remains unisolated. |

The **bundle** is validated. This is not a claim that all five changes are
minimal or appropriate for every ROCm release. The CUDA branches of the FLA
config changes retain their original configuration spaces.

The WY failure was reproducible in full-model execution, despite successful
standalone replay of the exact captured tensors. Thus isolated kernel tests
alone were insufficient. Temporary synchronized training hooks localized the
fault; those hooks were removed before the final parity test and task run.
`AMD_SERIALIZE_KERNEL=1` was tried, did not resolve earlier stalls, and is
not set in the final configuration.

## Client changes and why older runs could work

The new run encountered three distinct client issues. These are not
requirements of the Miles Qwen3.5 backend, and the older model has not been
replayed in an identical environment to establish a model-specific cause.

1. **IPython API compatibility.** The initially used environment had IPython
   9.17, which rejected Hyperion's `run_cell_async(code)` call. A `/tmp`
   overlay pins IPython 9.6.0 and decorator 5.3.1 without altering the parent.
2. **Synchronous generated loop.** A rollout blocked the shared asyncio loop
   inside generated Python calling `guess`. Async timeouts cannot interrupt
   synchronous code that never yields. Existing `detect_while_loops` and
   `detect_interactive_input` options were disabled in number-search.
3. **Rich traceback overhead.** Repeated stack samples showed prolonged CPU
   time in IPython traceback formatting via `inspect.getmodule`. Plain
   tracebacks allowed the client to progress. A `1/0` check returns the
   exception text in the observation in approximately 1.4 ms.

The private Babel wrapper `/tmp/qwen35-mi350-analysis/run_number_search.py`
enables the two existing executor checks and calls
`self.shell.InteractiveTB.set_mode(mode="Plain")`, then runs the unchanged
number-search training module. Smoke checks confirm loop/input rejection
and normal execution of `guess(42)`.

**These settings affect accepted generated programs and error observations.**
They are documented experimental conditions, not an invisible equivalent
of the original client. They do not change the renderer, reward function,
training algorithm, or prompts. They are also not a general execution-timeout
or process-isolation solution. A clean repository implementation would
expose these executor settings in number-search config or set them in
`NumberSearchEnv`; this experiment avoids that source edit.

## Built-in renderer and prefix merging

The built-in non-thinking renderer matched HF tokenization for tested
single-turn and alternating number-search histories. Parsing a code action
also passed. This is not a certification for SWE tool-call serialization.

**Non-thinking mode still breaks prefix merging.** The installed renderer
adds an empty thinking block for generation and removes it from that turn
when rendering subsequent history:

```text
generation prefix: ...assistant\n<think>\n\n</think>\n\n
next-turn history: ...assistant\n<python>guess(42)</python><|im_end|>
```

A direct probe shared only 20 of 35 prior tokens. This is deterministic
renderer behavior, not tokenizer nondeterminism. It explains at least one
source of the observed 517–591 datums per 64-rollout batch. Other agent
history rewriting may add mismatches. Numerically valid training therefore
has substantial repeated-prefix computation in this experiment.

The run deliberately keeps the requested built-in renderer. An exact-history
renderer or token-preserving conversation path would be a separate efficiency
change needing template/parity validation. Reasoning-enabled training was
not run, and disabling reasoning should not be advertised as solving merging.

## Persistence and reproduction

Persistent artifact root:
`/work1/grahamneubig/adityabs/miles-tinker/qwen35-mi350/`.

- `checkpoints/`: full-state and sampling checkpoints.
- `validation/`: patches, five complete patched runtime files with SHA256
  manifest, version manifest, doctor log, launchers, converter, client wrapper,
  YAML copy, and validation scripts/results.
- `validation/sparse-parity/`: final normal-execution parity results and log.
- `validation/sparse-parity-debug/`: earlier successful instrumented test.
- `validation/investigation-notes.md`: chronological investigation and failed
  attempts. Its older settings are historical; use this guide and the final
  `validation/start_server.py` for the working configuration.

Ephemeral locations:

- Server image/env/model/dependencies: `/tmp/miles-qwen35-mi350`.
- Babel overlay: `/tmp/qwen35-number-search-venv`.
- Babel launchers: `/tmp/qwen35-mi350-analysis`.
- Local investigation helpers: `/tmp/qwen35-number-search-analysis`.

Babel task output:
`~/hyperion_modal/outputs/qwen35-mi350-number-search/20260915T225910Z`.
Client launch PID was `1779000`; server launcher PID was `1590529`.
PIDs are historical identifiers, not a reliable future health check.

### Server

Provision with the existing launcher:

```bash
python3 tools/amd_tinker.py setup \
  --root /tmp/miles-qwen35-mi350 \
  --model-id Qwen/Qwen3.5-4B --model-type qwen3.5-4B
```

Copy the pinned image's Megatron-LM, FLA, and Triton AMD driver into the
private dependency locations and apply the saved patches (or restore the
saved five patched files into otherwise identical dependency copies).
Use the private Megatron bind while running the saved `convert_checkpoint.py`
from the Miles root with `PYTHONPATH=.`. The converted checkpoint's marker
must read `release` at
`models/Qwen3.5-4B_torch_dist/latest_checkpointed_iteration.txt`.

The final saved `validation/start_server.py` applies all three binds,
disables core dumps, and runs this command from the Miles checkout:

```bash
export APPTAINER_BINDPATH=/tmp/miles-qwen35-mi350/deps/Megatron-LM:/root/Megatron-LM,/tmp/miles-qwen35-mi350/deps/fla:/opt/venv/lib/python3.10/site-packages/fla,/tmp/miles-qwen35-mi350/deps/triton-amd-driver.c:/opt/venv/lib/python3.10/site-packages/triton/backends/amd/driver.c
ulimit -c 0
python3 tools/amd_tinker.py serve \
  --root /tmp/miles-qwen35-mi350 \
  --checkpoint-root /work1/grahamneubig/adityabs/miles-tinker/qwen35-mi350/checkpoints \
  --model-id Qwen/Qwen3.5-4B --model-type qwen3.5-4B \
  --train-gpus 1 --rollout-gpus 4 --rollout-gpus-per-engine 1 \
  --tp 1 --pp 1 --cp 1 --context-length 16384 \
  --max-tokens-per-gpu 8192 --mem-fraction 0.7 --port 10635 -- \
  --megatron-to-hf-mode raw \
  --ref-load /tmp/miles-qwen35-mi350/models/Qwen3.5-4B_torch_dist \
  --train-env-vars '{"HIP_VISIBLE_DEVICES":"0"}' \
  --seed 42 --bf16 --optimizer adam --lr 1e-6 --lr-decay-style constant \
  --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
  --weight-decay 0.017 --clip-grad 1.0
```

### Babel client and tunnel

The old SSH alias points to `babel-x5-28`, whose allocation expired. Current
commands use `-o HostName=babel-w5-28 babel-compute-node`, without editing SSH
configuration. The client uses the user's existing allocation `10450197`;
no additional Babel GPU was requested for it. A briefly queued replacement
CPU job was canceled before starting.

From the MI350 host, forward the service to Babel:

```bash
ssh -NT -o HostName=babel-w5-28 -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -R 127.0.0.1:10635:127.0.0.1:10635 babel-compute-node
```

The active background tunnel uses control socket
`/tmp/miles-qwen35-mi350/runtime/babel-w5-tunnel.sock`.
The service is `http://127.0.0.1:10635` on both hosts while the tunnel lives.

Recreate the isolated environment on Babel, assuming its parent exists:

```bash
/tmp/issue-resolution-legacy-tinker-venv/bin/python -m venv /tmp/qwen35-number-search-venv
/tmp/qwen35-number-search-venv/bin/python - <<'PY'
from pathlib import Path
p = Path('/tmp/qwen35-number-search-venv/lib/python3.12/site-packages/client-base.pth')
p.write_text('import site; site.addsitedir("/tmp/issue-resolution-legacy-tinker-venv/lib/python3.12/site-packages")\n')
PY
/tmp/qwen35-number-search-venv/bin/pip install ipython==9.6.0 decorator==5.3.1
```

Restore the saved client wrapper/launcher under `/tmp/qwen35-mi350-analysis`
and the saved YAML under the Hyperion number-search plugin. Run the saved
`start_client.py` on Babel after backend readiness and parity validation.
It sets the checkout/plugin `PYTHONPATH`, local Tinker URL, HF cache,
`WANDB_MODE=online`, and invokes the wrapper. It inherits existing W&B
authentication; no credentials are stored in this guide or config.

## Train/validation separation audit

The entrypoint selects the first 1,000 IDs from the 50,000-row training file
and 1,000 IDs from the validation file. An audit found zero task-ID overlap
and zero `(low, target, high)` overlap, both for the selected training subset
and across the entire two files. All 1,216 training rollouts recorded at the
time of the audit had `number_search.train.*` IDs; all 623 recorded evaluation
rollouts had `number_search.val.*` IDs. Evaluation queues feed a completion
monitor and metrics, not the optimizer queue.

The agent prompt renders `str(task)`, which contains the goal/bounds and
step budget, not `task.misc["target"]`. Quick training-reward gains are
consistent with selecting an already available binary-search behavior in a
pretrained coding model and reducing malformed actions. They are not, by
themselves, a held-out generalization measurement.

## Existing evaluation limitations

`train_tinker.py` calls `get_task_ids("val", 100)`, but the second positional
parameter is `num_samples_train`; `num_samples_val` remains 1,000. Thus the
actual validation set has **1,000 tasks**, not 100. The initial 100-task
interpretation in the chronological notes was incorrect.

The existing asynchronous evaluation workers share the updating sampler.
Evaluation started at training step 10, but its event files already span
steps 10–13 as training advances. The resulting aggregate is a rolling
measurement across model versions, not a frozen step-10 score. This behavior
is unchanged in this experiment. A controlled follow-up should pass
`num_samples_val=100` explicitly and evaluate a fixed published checkpoint.

## Remaining scope

- Finish the 30-batch run; record final checkpoint and metrics, then run
  the prepared `validation/frozen_eval.py` on exactly 100 validation tasks
  with a fixed final sampling checkpoint.
- Minimize the dependency patch bundle and revalidate before upstreaming it.
- Revalidate DP4 and activation recomputation separately before scaling.
- Address prefix preservation before long SWE trajectories if efficiency is
  important. This number-search run does not validate SWE rebench integration.
- Reasoning-enabled renderer/history behavior remains a separate experiment.
