# Reusable AMD Tinker backend with Apptainer

For the MI350X startup command with explicit training, SGLang, checkpoint,
and remote-access settings, use the [MI350X server configuration guide](mi350-tinker-server.md).
For MI300X, use the [MI300X server configuration guide](mi300-tinker-server.md).
For a fresh MI300 machine and the matched number-search client, use the
[complete reproduction guide](mi300-fresh-machine.md).
The MI350 image pinned on this page contains architecture-specific gfx950
kernels and must not be treated as a universal MI300/MI350 image.

This branch runs the experimental Miles Tinker API with **Megatron full
fine-tuning and SGLang inference** on AMD MI350X GPUs. Apptainer supplies the
ROCm dependencies; an editable install supplies this branch's Miles code.
The tested allocation uses four training GPUs and four inference GPUs, with
one SGLang engine per inference GPU. Ray manages the workers.

This document covers the backend and the changes needed to make it work. It
does not prescribe a client training loop or a task plugin.

## Reproduce

Prerequisites: an allocated MI350X node, access to its GPU devices, Apptainer,
Git, and Python 3.9+ on the host. Docker is unnecessary. Keep images, caches,
temporary files, and virtual environments on local `/tmp`. Checkpoints default
to local `/tmp`; use `--checkpoint-root` to choose persistent storage.
The checkout can remain elsewhere and is explicitly bound into Apptainer.
The host tool uses only the standard library unless the optional Modal relay is enabled; GPU dependencies run in
the container. `/tmp` is temporary storage and must be copied elsewhere if
persistence is needed.

```bash
git clone --branch miles-tinker-amd https://github.com/neulab/miles-tinker.git /tmp/miles-tinker-amd
cd /tmp/miles-tinker-amd
python3 tools/amd_tinker.py setup --root /tmp/miles-amd-runtime
python3 tools/amd_tinker.py doctor --root /tmp/miles-amd-runtime
```

Setup pulls the pinned SIF, copies AITER into writable local storage, creates
an overlay venv inheriting the image's packages, installs this checkout
editable, and downloads the pinned model. Repeated setup reuses existing
assets. It does not reset or replace the checkout. Use a separate runtime
directory for separate deployments; a venv's editable install points to one
checkout at a time.

Start the backend in a persistent shell or tmux session:

```bash
python3 tools/amd_tinker.py serve --root /tmp/miles-amd-runtime \
  --train-gpus 4 --rollout-gpus 4 --port 10613 \
  > /tmp/miles-amd-runtime/logs/server.log 2>&1
```

`serve --dry-run` prints the container command without loading a model or
starting Ray. The tool refuses to start if its port is already occupied.
It loads architecture flags from the existing Python model definitions in
`scripts/models/`; the default is `qwen3-4B-Instruct-2507`.

```bash
curl -fsS http://127.0.0.1:10613/health
```

The default model is `Qwen/Qwen3-4B-Instruct-2507`, snapshot
`cdbee75f17c01a7cc42f958dc650907174af0554`. The server selects
`--tinker-train-mode full`; a client's LoRA-shaped create request is ignored
in this mode. The API's model information reports `is_lora: false`.
The SDK supplies the loss and optimizer parameters on training requests;
the launcher's initial learning rate is only a startup default.

### Configuration

| Option | Default | Meaning |
| --- | --- | --- |
| `--root` | `/tmp/miles-tinker-work` | Local deployment assets and runtime state |
| `--port` | `10613` | Tinker HTTP endpoint |
| `--train-gpus`, `--rollout-gpus` | `4`, `4` | Separate training and inference allocations |
| `--rollout-gpus-per-engine` | `1` | Inference tensor-parallel allocation |
| `--tp`, `--pp`, `--cp` | `1`, `1`, `1` | Megatron parallelism |
| `--peak-tflops` | `2309.6` | Dense BF16 peak **per training GPU**, used for MFU |
| `--max-tokens-per-gpu` | `16384` | Dynamic training batch token budget |
| `--context-length` | `32768` | SGLang context limit |
| `--mem-fraction` | `0.70` | SGLang memory allocation setting |
| `--checkpoint-root` | `<root>/checkpoints` | Checkpoint and performance-log directory; external paths are bound into Apptainer |
| `--model-id`, `--revision`, `--model-type` | Qwen3-4B, revision `main` | HF model, optional revision for setup, and Miles Python architecture definition |

Ray's temporary base is `/tmp/miles-ray-<runtime-hash>`, derived from the resolved
`--root` independently of the PID. This keeps its generated Unix socket paths
short; Ray session logs live under `<ray-base>/ray/session_latest/logs`.

For another model, supply its model ID, architecture definition, and appropriate
parallelism. The revision is optional and defaults to `main` during setup.
Other models and multi-node configurations have not been validated by this
deployment. Additional Miles arguments can follow `--`, for example
`serve -- --sglang-max-running-requests 64`. MI355X or other hardware needs
its own peak-TFLOPS denominator; do not silently use the MI350X value.

### Remote clients

From the GPU node, create a reverse SSH tunnel to the client host:

```bash
python3 tools/amd_tinker.py tunnel --root /tmp/miles-amd-runtime \
  --remote-host CLIENT_SSH_HOST --port 10613
```

The client then reaches the endpoint at `http://127.0.0.1:10613`. The remote
forward binds only the client's loopback interface. Set the endpoint on every
SDK client, including sampling proxies that independently construct clients.
Recent SDKs may require a syntactically valid `tml-`-prefixed key even for a
local endpoint. This gateway's tenant key is not a substitute for network
access control.

### Checkpoints and lifecycle

Full model-and-optimizer checkpoints are saved at:

```text
<root>/checkpoints/<model-id>/weights/<checkpoint-name>/
```

To change this location, pass `--checkpoint-root /path/to/checkpoints` to
`serve` before the `--` separator. The directory is created and bound into
Apptainer. Pass the same option to `prune` when using custom storage. Existing
checkpoints are not moved automatically.

A trained 4B checkpoint is approximately 53 GiB. To retain only the newest
completed checkpoint, run one cleanup process in another persistent shell:

```bash
python3 tools/amd_tinker.py prune --root /tmp/miles-amd-runtime --keep 1
```

The pruner ignores incomplete saves and removes older checkpoints only after
a newer save has its completion sidecar. Allow disk space for the old and new
checkpoint to coexist during a save. No checkpoints are transferred to the
client host. `--once` performs one cleanup pass. Resume requires the same
parallel topology; this deployment validated saving, not a full resume test.

The full-FT gateway supports one live training model at a time. After an
interrupted or completed client session, explicitly unload its model through
`/api/v1/unload_model`, or restart the backend before creating another model.
To stop this deployment, including otherwise-orphaned SGLang processes:

```bash
python3 tools/amd_tinker.py stop --root /tmp/miles-amd-runtime
```

The stop operation checks process ownership and deployment paths. It does not
delete checkpoints. Stopping only the gateway driver can leave inference
subprocesses holding GPU memory with this image.

## Image selection and dependency reuse

Pinned image: `rlsys/miles:MI350-355-latest`, digest
`sha256:e60a69faa831ae2a146819290e8469cf23c81b97febeaf5d11e8baec2bfca285`.
The SIF is about 15 GiB. Apptainer build cache and temporary storage are both
under the runtime directory; setup cleans its scoped build cache after pulling.

The inspected image contains Miles `9d810ffaa360ba19e1f1e06a35aab90b6cee59a3`,
Megatron-LM `d9a5080f2bafea5246add1489247f25368b8a120`, and SGLang
`a929eb72882a8c47478de4d1eefd4a8ebb0716ff` (0.5.11.dev36). Python is 3.10.12,
PyTorch is 2.9.0a0 with HIP 7.0, Transformer Engine is 2.8, and Megatron Bridge
is 0.3.0rc0. This branch started from Miles Tinker commit
`25b6a76fedb89b24f84f2ccb6c374ba53dc118a6`.

The inspected `MI350-355-rocmpatch` image adds ModelOpt 0.44 and a DeepSeek/GLM
RoPE fix. We install the small ModelOpt package in the overlay; the model-specific
RoPE fix is unnecessary for Qwen3. `rocm7.2-mi35x-dsv4` is a different DSV4 stack,
and MI300 images target a different GPU architecture. This branch's compatibility
code is validated against the pinned image, not every future AMD image.

## Changes required

| Change | Reason and scope |
| --- | --- |
| Apptainer `--rocm` | Explicit device binds enumerated GPUs but failed during HIP initialization. `--rocm` passed GPU matrix multiplication. |
| Local `/tmp` storage and temporary HOME | Avoid NFS for code, SIF/build cache, model data, and runtime/JIT writes. |
| Editable Miles overlay | Use this experimental fork while reusing the image's ROCm PyTorch, Megatron, SGLang, and Transformer Engine. A `.pth` explicitly inherits `/opt/venv` because nested venvs do not inherit another venv automatically. |
| Writable AITER copy | AITER compiles and imports kernels alongside its source. The immutable SIF cannot hold those writes; bind a local copy at its original image path. |
| Small Python dependency overlay | Ray 2.56 plus missing gateway dependencies; ModelOpt 0.44 for bridge imports; Tinker 0.27.1's generated protobuf schema. Exact direct requirements are in `tools/amd_tinker.py`. |
| Megatron argument compatibility | Support the image's older vocabulary-padding import path and Gloo process-group flag name. |
| SGLang CLI name mapping | Translate newer dataclass names to the older parser's TP/PP/CP/DP/EP names. |
| Launch-gate detection | The image starts engines immediately and has no gated-launch option. Skip that handshake for this version, retaining health checks. |
| Lazy DeepSeek V4 template import | Qwen does not require a DSV4 module absent from the older SGLang image. |
| Python 3.10 gateway compatibility | Load only the generated protobuf module without importing the SDK's Python-3.11-only client package; catch `asyncio.TimeoutError` explicitly. |
| Legacy SGLang weight-update lifecycle | Under `MILES_SGLANG_LEGACY_WEIGHT_UPDATE=1`, use the image's original `post_process_weights` pre/post protocol around RCCL broadcasts. Pause/retract, cache flush, version update, and resume remain intact. Adapter-only updates are rejected. |
| Bridge process-group filtering | The old bridge tried to copy a live process-group wrapper while exporting QKV weights. Filter process groups from that copied configuration. |
| Synchronous ROCm checkpoint writer | Forked Megatron writer children crashed inside `torch.save`. Write the same staged tensors and torch_dist format in each owning rank process, retaining optimizer state and fsync. |
| MFU denominator | Pass the existing `--mfu-peak-tflops` option so AMD runs emit MFU despite the NVIDIA-only automatic lookup table. |
| Deployment lifecycle tool | Python setup/serve/doctor/tunnel/stop/prune commands with configurable paths, topology, model definition, and memory limits. No Docker daemon or baked custom SIF is required. |

**No changes were made to loss functions, gradient computation/scaling, or
optimizer-step logic.** Inside the full trainer executor, the only change is
checkpoint writing. Model loading and inference weight transfer required the
compatibility changes listed above.

Separately, the remote client used Python 3.12 with SDK 0.16.1 and required
IPython 8.37 because its code executor expected the old `run_cell_async` API.
That IPython pin is a client concern, not a backend dependency. No client
training recipe is included here.

## MFU: missing metric and estimate for the completed run

`tinker_perf.perf_metrics` always emits TFLOPS for a timed training window, but
emits MFU only when a peak-TFLOPS denominator exists. The automatic table in
`miles/utils/device_flops.py` has only NVIDIA devices. On this node it returned
`None` for `AMD Instinct MI350X`, and the original launcher omitted the override.
This was a missing denominator, not missing training timing.

The reusable launcher passes `--mfu-peak-tflops 2309.6`, using AMD's published
**dense BF16 matrix** peak of 2.3096 PFLOPS per MI350X GPU
([AMD product brief](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi350x-gpu-brochure.pdf)).
The reported TFLOPS are already normalized per training GPU: do not multiply
the denominator by four or eight. MFU is a fraction; multiply by 100 for percent.

For the completed 100-step run, reconstructed from logged TFLOPS and timings:

| Quantity | Estimate |
| --- | --- |
| Time-weighted training-window MFU | **19.08%** |
| Time-weighted TFLOPS per training GPU | **440.62** |
| Median step MFU | **26.12%** |
| Step MFU range | 0.93%–32.95% |
| Sum of timed training windows | 62.59 seconds |
| Full client training-loop wall time | 3,748.33 seconds |

For step `i`, `MFU_i = TFLOPS_i / 2309.6`. The aggregate above is
`sum(TFLOPS_i * train_seconds_i) / sum(train_seconds_i) / 2309.6`, including
the first step's warmup. It uses the framework's modeled FLOP counts and
rounded logged timings, not hardware counter measurements. These windows
exclude rollout waiting and weight pushes outside the training timer. Amortizing
training FLOPs over the whole loop gives about 0.32% of the four training GPUs'
peak; that is a different metric and excludes inference work on the other GPUs.

Future launches emit `actor_train_mfu` in optimizer metrics; a client may prefix
this as `train/optim/actor_train_mfu`. An already-running server needs a restart
with the new launcher to acquire the denominator. Historical W&B metrics were
not rewritten.

## Validation and limitations

The tested full-FT SDK workload completed 100 optimizer updates without skipped
steps. Latest completed asynchronous evaluation reward was 0.907 over 1,000
tasks. Checkpoint saving succeeded through the final save. Comparing a saved
layer-0 QKV tensor to initialization found 936,954 changed values, confirming
actual parameter updates. The server uses latest pushed weights for sampling,
including clients created at earlier sampler versions; evaluation is therefore
not an immutable model snapshot.

The gateway/HTTP suite passed 102 tests with two SDK-import skips under Python
3.10. The backend-tool and MFU bookkeeping suites passed 10 tests. GPU smoke checks validated
the editable import, ROCm matrix multiplication, and explicit MFU calculation.
Four existing SGLang launcher tests exposed pre-existing CLI/constructor default
differences (`kt_method`, `kt_threadpool_count`, `pre_warm_nccl`); the real launch
supplies those values and passes its argument round-trip assertion.

The broader existing launcher suites produced 452 passes, 12 failures, and
40 errors in this image. An isolated unmodified base checkout produced the
same counts and the same failing test IDs. These include attempts to create
read-only `/root/models` and existing command-utils/snapshot incompatibilities;
they are not new failures introduced by the AMD backend tool.

The packaging refactors the validated shell deployment into a stdlib Python
host tool. Intended changes are configurable paths/topology, use of existing
Python model definitions, a generic temporary HOME, and the explicit MFU
denominator. Training flags and runtime environment otherwise retain the tested
recipe. Full fine-tuning is validated; LoRA, arbitrary models, checkpoint resume,
and multi-node operation are not claimed as tested.
