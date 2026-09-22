# MI300X Tinker server: startup and training configuration

To launch from an existing `~/miles-tinker` checkout, follow the
[complete number-search training commands](mi300-fresh-machine.md).
That guide pins the model/client revisions and reproduces the completed
100-step MI350-matched run, including `max_staleness: 3`.

This recipe runs a Tinker-compatible endpoint on one node with eight AMD
MI300X GPUs using Apptainer: four GPUs for Megatron full fine-tuning and four
GPUs for SGLang inference. It is independent of the client plugin. The concrete
model below is `Qwen/Qwen3-4B-Instruct-2507`.

Run commands from the existing `~/miles-tinker` checkout.
Keep images, model downloads, environments, and server logs under `/tmp`.
Choose persistent checkpoint storage with `--checkpoint-root`; the command
below uses `/work1/grahamneubig/adityabs/miles-tinker/checkpoints`.
The launcher binds the checkout and external checkpoint directory into Apptainer.
Ray uses `/tmp/miles-ray-<runtime-hash>` as its temporary base, with a stable
hash of the resolved runtime root rather than a PID. Its session logs are at
`/tmp/miles-ray-<runtime-hash>/ray/session_latest/logs`. Ray still appends its
own timestamp/PID session name; the short base keeps Unix socket paths within
Linux's 107-byte limit.

## 1. Provision the architecture-matched image

The launcher's original default image targets MI350 (`gfx950`). On MI300
(`gfx942`), explicitly select the following image on **both setup and serve**:

```bash
cd ~/miles-tinker
export MILES_AMD_IMAGE=docker://rlsys/miles@sha256:545241a0dbbf8543ef9a6c6d14313b45ba2e78b49ce1ff8370f2c7ab354180e3
export CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=0
export AITER_JIT_DIR=/tmp/miles-mi300-runtime/aiter-jit-bf16-rne

python3 tools/amd_tinker.py setup \
  --root /tmp/miles-mi300-runtime \
  --image-uri "$MILES_AMD_IMAGE" --image-workspace /app \
  --model-id Qwen/Qwen3-4B-Instruct-2507 \
  --model-type qwen3-4B-Instruct-2507

python3 tools/amd_tinker.py doctor \
  --root /tmp/miles-mi300-runtime \
  --image-uri "$MILES_AMD_IMAGE" --image-workspace /app \
  --peak-tflops 1307.4
```

Setup downloads the image and model and builds the temporary environment.
The image is pinned from the tag
`rlsys/miles:rocm7-mi300-sglang0.5.9-te2.12.0-latest`; its dependency workspace
is `/app`. Custom-image filenames include a hash of the URI. Use a separate
runtime root when changing image stacks, since the venv and writable AITER
copy also depend on the image.

## 2. Start the server

Run in a persistent shell after setup. The command includes the agreed
optimizer initialization and recomputation settings explicitly. SGLang's
chunked prefill, request limit, radix caching, and attention backend selection
retain the launcher's defaults; no manual overrides are needed for these.
Set the two AITER environment variables below for training/inference logprob
consistency on MI300; they apply independently of the model.

```bash
cd ~/miles-tinker
export MILES_AMD_IMAGE=docker://rlsys/miles@sha256:545241a0dbbf8543ef9a6c6d14313b45ba2e78b49ce1ff8370f2c7ab354180e3
export CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=0
export AITER_JIT_DIR=/tmp/miles-mi300-runtime/aiter-jit-bf16-rne
mkdir -p /tmp/miles-mi300-runtime/logs

MODAL_PROFILE=neulab python3 tools/amd_tinker.py serve \
  --root /tmp/miles-mi300-runtime \
  --checkpoint-root /work1/grahamneubig/adityabs/miles-tinker/checkpoints \
  --image-uri "$MILES_AMD_IMAGE" --image-workspace /app \
  --model-id Qwen/Qwen3-4B-Instruct-2507 \
  --model-type qwen3-4B-Instruct-2507 \
  --train-gpus 4 \
  --rollout-gpus 4 --rollout-gpus-per-engine 1 \
  --tp 1 --pp 1 --cp 1 \
  --context-length 65000 \
  --max-tokens-per-gpu 65000 \
  --mem-fraction 0.9 \
  --peak-tflops 1307.4 \
  --port 10613 \
  --modal-relay \
  -- \
  --seed 42 \
  --bf16 \
  --optimizer adam \
  --lr 1e-6 --lr-decay-style constant \
  --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
  --weight-decay 0.017 --clip-grad 1.0 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  > /tmp/miles-mi300-runtime/logs/server.log 2>&1
```

Arguments before `--` configure the host launcher; arguments after it are
passed to `serve_tinker.py` and its Megatron/SGLang parser. To inspect the
expanded container command without starting anything, add `--dry-run` before
that separator. Omit `--modal-relay` for a local or SSH-forwarded endpoint.

**Client optimizer requests override the startup Adam and clipping values.**
The full Tinker executor does not advance a server LR scheduler. Constant LR
means the client sends `learning_rate: 1e-6` on every optimizer step. Setting
`--lr-decay-style constant` makes initialization explicit but does not enforce
the client's future learning rates. See the required client settings below.

## 3. GPU allocation and Megatron knobs

| Setting | Startup flag or launcher behavior | Effect in this recipe |
| --- | --- | --- |
| Training allocation | `--train-gpus 4` | One actor node with four training ranks. |
| Training parallelism | `--tp 1 --pp 1 --cp 1` | DP = `4 / (1 × 1 × 1)` = 4; no tensor, pipeline, or context splitting. |
| Optimizer sharding | Miles enables the distributed optimizer for full-mode Adam | Model replicas across DP ranks, with distributed optimizer state; this is Megatron, not FSDP. |
| Inference allocation | `--rollout-gpus 4 --rollout-gpus-per-engine 1` | Four independent TP1 SGLang engines, separate from the training allocation. |
| Full fine-tuning | Launcher supplies `--tinker-train-mode full` | All model weights train; no LoRA. Client must also request full mode. |
| Precision | `--bf16` | BF16 model training; no FP8 recipe. |
| Packing | Launcher supplies `--use-dynamic-batch-size`; `--max-tokens-per-gpu 65000` | Token-length-based training packing budget per GPU, not rollout batch size. |
| Work-unit batch size | Tinker entrypoint enables dynamic global batch size | The submitted work unit determines batch size; client minibatches/microbatches are separate controls. |
| Activation checkpointing | `--recompute-granularity full --recompute-method uniform --recompute-num-layers 1` | Recompute full transformer layers in uniform groups of one; trades compute for activation memory. |
| Optimizer initialization | `--optimizer adam --lr 1e-6 --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 --weight-decay 0.017 --clip-grad 1.0` | Initializes the requested values; each optimizer request supplies the values actually used. |
| LR initialization | `--lr-decay-style constant` | No server scheduler advances in the Tinker executor. |
| Dropout | Launcher supplies `--attention-dropout 0.0 --hidden-dropout 0.0` | Both disabled. |
| Gradient arithmetic | Launcher supplies `--accumulate-allreduce-grads-in-fp32` | FP32 gradient accumulation/reduction. |
| Training attention | Launcher supplies `--attention-backend flash --attention-softmax-in-fp32` | Megatron attention configuration; separate from SGLang's inference attention backend. |
| Communication overlap | Leave `--overlap-grad-reduce` and `--overlap-param-gather` unset | Full Tinker accumulation requires both off and asserts this. |
| Weight export | Launcher supplies `--megatron-to-hf-mode bridge` | Megatron Bridge exports weights for inference. |
| Weight synchronization | Launcher supplies `--pause-generation-mode retract --update-weight-buffer-size 1073741824` | Retract mode during updates; 1 GiB weight-transfer buffer. |
| Seed | `--seed 42` | Explicit server seed; client rollout randomness is separately configured. |
| MFU reporting | `--peak-tflops 1307.4` | Per-GPU MI300X dense BF16 peak for utilization reporting, not a training hyperparameter. |

The model definition is loaded from
[`scripts/models/qwen3-4B-Instruct-2507.py`](../../scripts/models/qwen3-4B-Instruct-2507.py).
For another model, provide its own `--model-id` and `--model-type`
consistently to setup and serve. Setup downloads the model's `main` revision
by default; optionally pass `--revision <branch-or-tag-or-commit>` to setup
when a specific version is needed. Serve uses the downloaded local model and
does not fetch a new revision at startup.

Keep CP at 1 for this deployment. DP spreads samples over GPUs; it does not
split one long sequence across ranks. A 65000-token packing budget and
activation recomputation do not prove that an individual 65000-token training
sequence fits in memory.

## 4. SGLang knobs and token limits

### BF16 rounding for MI300 RL

The pinned AITER RMSNorm build defaults to BF16 truncation on `gfx942`.
Its `gfx950` conversion takes a different hardware path. In the fused
add+RMSNorm operator, truncation also affects the residual passed to the next
layer, so its bias accumulates even before the first optimizer update.
This is an AITER numerical issue, not a reason to switch attention backends.
See [AITER issue #5151](https://github.com/ROCm/aiter/issues/5151) for a report
of the same behavior in a newer stack; that newer version uses an OPUS flag
name rather than the CK flag used by this pinned image.

Export these settings in the MI300 launch shell. The launcher forwards them
through Apptainer's `--cleanenv`:

```text
CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=0
AITER_JIT_DIR=<runtime-root>/aiter-jit-bf16-rne
```

Mode 0 requests round-to-nearest-even. AITER checks for an existing `.so`
without including compiler flags in its cache key: **the rounding variable
alone can silently reuse a truncating kernel**. Keep the RNE cache separate;
do not copy an old `module_rmsnorm_quant.so` into it. First use compiles the
required kernels. The launcher explicitly forwards these optional host environment variables;
older versions of this launcher discarded them with `--cleanenv`.

These settings do not change `SGLANG_USE_AITER`, attention, optimizer settings,
or the default MI350 launch. Existing processes need a restart to load the
new kernel; apply it only to the MI300 runtime being changed.

On 2026-09-14, a frozen Qwen3-4B probe using identical tokens and weights
reduced mean inference-prefill/trainer `0.5 * mean(delta_logprob**2)` from
0.01060 to 0.00156 with these settings. Changing only attention to Triton
gave 0.01068; replacing only AITER normalization gave 0.00187. These are
controlled diagnostic measurements, not the number-search run's metrics.
The subsequent [matched number-search run](https://wandb.ai/9998_team/number-search-platoon-tinker/runs/zsr75q6r)
measured first-batch v2 0.0010036, versus the MI350 reference's 0.0010306.
See the [investigation](mi300-logprob-investigation.md) for the run comparison,
backend checks, and validation scope.

### Other inference settings

| Setting | Flag / chosen behavior | Meaning |
| --- | --- | --- |
| Context window | Launcher `--context-length 65000` → `--sglang-context-length 65000` | Total inference sequence limit, including prompt and generated tokens. |
| Output length | Client sampling limit `2048` | This is a per-request limit, not `--max-tokens-per-gpu` or a server startup flag. Reserve room inside the 65000-token context. |
| Static memory fraction | Launcher `--mem-fraction 0.9` → `--sglang-mem-fraction-static 0.9` | Request 0.9 and allow the image's AITER adjustment. |
| Chunked prefill | Launcher supplies `--sglang-chunked-prefill-size 4096` | Enabled; no additional command-line override. |
| Radix cache | Leave `--sglang-disable-radix-cache` unset | Enabled by default; no AReaL-specific override. |
| Running requests | Launcher supplies `--sglang-max-running-requests 128` | Per-engine scheduler cap; no additional command-line override. Distinct from client rollout concurrency and training batch size. |
| Attention backend | Leave `--sglang-attention-backend` unset | Pinned MI300 stack selected `aiter` in the verified run. |
| Inference parallelism | `--rollout-gpus-per-engine 1` | TP1 per engine. Four replicas arise from the four-GPU allocation; no SGLang DP-attention flag is enabled. |
| Other engine knobs | Leave defaults from the pinned Miles/SGLang stack | No additional cache, CUDA-graph, scheduling, or attention overrides in this recipe. |

In the pinned SGLang source, AITER multiplies `mem_fraction_static` by `0.85`
when the model context exceeds 8192. The observed effective value here was
`0.9 × 0.85 = 0.765`. This is why the requested value stays at 0.9 instead of
manually lowering it to 0.7. Inspect the emitted `ServerArgs` in `server.log`
when changing images; this adjustment is implementation-dependent.

## 5. Client training settings required for the agreed recipe

These apply to any client plugin. A server launch cannot choose the client's
advantages, loss, rollout groups, checkpoint cadence, or sampling limits.
Make these settings explicit in the client YAML and its Tinker request builder.

| Client setting | Required value / behavior |
| --- | --- |
| Training mode | Full fine-tuning; Platoon `lora_rank: 0`. |
| Learning rate | `learning_rate: 1e-6` on every `optim_step`; constant, no warmup or decay. |
| Adam | `beta1: 0.9`, `beta2: 0.95`, `eps: 1e-8`, `weight_decay: 0.017`. |
| Gradient clipping | `grad_clip_norm: 1.0` in each optimizer request. Startup `--clip-grad` alone is insufficient. |
| Loss | CISPO with `clip_low_threshold: 0.0`, `clip_high_threshold: 5.0`. These are importance-ratio bounds, not gradient clipping. |
| Batch / group | Preserve the source workload's batch and group sizes; the agreed verified configuration was batch 32, group 8. |
| Client splitting | `num_minibatches: 1`, `num_microbatches: 1`; let the server perform token-length-based packing. |
| Staleness | Platoon `max_staleness: 1` for the agreed one-step off-policy configuration. |
| Advantages | Group mean centering, no standard-deviation normalization; leave-one-out baseline and depth-level weighting disabled for this recipe. |
| Zero reward variance | Preserve the client's filtering behavior. In the verified configuration, `filter_zero_variance_groups: false` retains groups for accounting, with zero-variance groups excluded before model work. This is not a SGLang or Megatron startup switch. |
| Sampling | Context limit 65000, output limit 2048; verified sampling temperature 1.0. |
| Checkpoints | Client step cadence 25; keep initial/final save policy explicit. |
| Logging / evaluation | Configure explicitly per workload; neither W&B nor an evaluation schedule is enabled by starting the server. |

The gateway fills omitted optimizer fields with its wire defaults, which are
**not** this recipe: LR `1e-4`, epsilon `1e-12`, weight decay `0`, gradient
clipping `0` (disabled), betas `0.9/0.95`. An SDK may also serialize its own
defaults. Send every requested Adam field explicitly.

Weight decay still honors Megatron's parameter-group multipliers: norms and
biases use zero decay. Setting the scalar to 0.017 does not reproduce AReaL
applying decay to every trainable parameter. Also, full-mode sampler versions
resolve to the latest pushed weights; `max_staleness: 1` controls client
scheduling, not retention of historical inference weights.

## 6. Expose the endpoint to a remote client

### Modal HTTPS relay

`--modal-relay` creates an ephemeral Modal sandbox, an outbound SSH reverse
forward, and a public HTTPS URL. No inbound connection to the GPU node is
needed. The helper installs Modal into `<root>/modal-venv` at first use.
The sandbox requests the minimum allocation of 0.125 physical CPU cores and
128 MiB RAM, with no GPU. These are resource requests, not hard usage caps;
Modal bills the greater of requested and actual usage. See
[Modal resource configuration](https://modal.com/docs/guide/resources).

This host is already authenticated with active profile/workspace `neulab`.
`MODAL_PROFILE=neulab` pins the profile for the launch. On a fresh host,
authenticate with `<root>/modal-venv/bin/modal setup` after that environment
exists. Modal credentials normally live in `~/.modal.toml` on the server host;
remote training clients need only the endpoint URL and Tinker key.

At relay startup, the server stdout prints `TINKER_BASE_URL=...` and
`TINKER_API_KEY=...` for copying into the client environment. Prefix each line
with `export` in the client shell. With the command above, stdout is redirected
to `server.log`, so that log contains the relay API key. The files below remain
available for automated client configuration.

- Read the public URL from `/tmp/miles-mi300-runtime/runtime/modal-relay.json`.
- Read the Tinker key from `/tmp/miles-mi300-runtime/runtime/public-api-key`;
  transfer it privately and set the client's `TINKER_API_KEY`.
- Set client `TINKER_BASE_URL` to the public URL. For Platoon, also set
  `tinker_base_url` in the trainer config/CLI: its early registration helper
  creates a `ServiceClient()` that uses the environment variable.

The relay requires `X-API-Key`, including for `/health`. The key survives
relay restarts; the URL is ephemeral. The sandbox lifetime is at most 24 hours
and the request-body limit is 64 MiB. The server command supervises the relay
and cleans up SSH and the sandbox on exit; a relay failure stops the server.
SIGTERM to the launcher runs its cleanup path. A remaining limitation is that
an already-crashed server process can leave GPU workers alive; explicitly run
the runtime's `stop` command before restarting in that case.

### SSH forwarding

Omit `--modal-relay`, then run this on the GPU host in another persistent shell:

```bash
ssh -NT -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -R 127.0.0.1:10613:127.0.0.1:10613 babel-compute-node
```

The client on Babel then uses `TINKER_BASE_URL=http://127.0.0.1:10613`.
Keep the forwarding connection alive for the run.

## 7. Checkpoints, reuse, and shutdown

Client-requested full checkpoints are written to:

```text
/work1/grahamneubig/adityabs/miles-tinker/checkpoints/<model-id>/weights/<checkpoint-name>/
```

Change `--checkpoint-root` before the `--` separator to select another
directory. The launcher creates it on startup and binds it into Apptainer.
Without this option, storage defaults to `<root>/checkpoints`. This option
does not move existing checkpoints. If using the launcher's `prune` command,
pass the same `--checkpoint-root` to select the correct storage directory.

Set the **client** checkpoint cadence to 25 steps. The server does not schedule
these saves autonomously. Each full trained 4B checkpoint uses roughly 53 GiB;
allow space for multiple checkpoints during saves. Preserve the client run
state/logs alongside server checkpoints.

Resume requires an explicit client load of the full checkpoint **including
optimizer state**, with the same parallel topology. A weights-only load resets
Adam state. Restarting `serve` alone does not automatically select the latest
client checkpoint or restore the client's step counter/data progress. Actual
restart-and-resume has not been validated in this deployment.

The server stays up across optimizer requests and independent training runs.
Full mode has one live model slot: release the old model through
`/api/v1/unload_model`, wait for the returned future, then create the next
training client. This reuse was verified without restarting the server.

```bash
curl --fail http://127.0.0.1:10613/health
tail -f /tmp/miles-mi300-runtime/logs/server.log
```

Optimizer performance records are in
`/work1/grahamneubig/adityabs/miles-tinker/checkpoints/perf.jsonl` with the
startup command above. To stop this runtime:

```bash
python3 tools/amd_tinker.py stop --root /tmp/miles-mi300-runtime
```

## Image diagnosis and validation scope

The original MI350 image's Torch and AITER cover multiple architectures, but
its `sgl_kernel/common_ops` ELF bundles contain only `gfx950`, and Transformer
Engine's build information specifies `GPU_TARGETS: gfx950`. An isolated SGLang
rotary-embedding invocation segfaulted on MI300 before any client training
request. A successful Torch matrix multiplication does not validate those
separately compiled extensions. The selected MI300 image passed isolated
rotary-embedding and Transformer Engine forward/backward checks.

Image tags are not exact package manifests. Inspection of this MI300 image
found Megatron Core `0.16.0rc0`, Transformer Engine `2.10.0.dev0+307b5e86`, and
an older SGLang. Setup supplements it under `/tmp` with:

- RadixArk Megatron Bridge at `3fd3768045422d0aa5c97e90a4e6c659aea9acb9`
  for the three-field weight-export API used by this branch.
- SGLang at `a929eb72882a8c47478de4d1eefd4a8ebb0716ff` and matching native
  kernels built against this image for the detected GPU architecture.
- `apache-tvm-ffi==0.1.13.post3` for SGLang JIT kernels.

A remote short-conversation workload completed 45 actual optimizer updates
through Modal on 2026-09-13, with no skipped updates or non-finite gradients.
Step-25 and final checkpoints were inspected for the requested Adam values,
BF16, full mode, and activation recomputation. The client stalled in IPython
exit-time garbage collection after its successful final save and was
terminated; the server stayed healthy and accepted a fresh full-mode client.
Evidence is under `/tmp/miles-mi300-runtime/verification/`.

That run does not validate individual 65000-token sequences, maximum-size
relay requests, every client workload, or restart-and-resume. The command
above additionally makes optimizer initialization explicit; the verified
run supplied the same effective optimizer values through client requests.
The MI350 image and dependency defaults were preserved, but no MI350 hardware
execution was tested on this MI300 node. Model downloads now default to `main`
instead of the fixed snapshot used during validation.
