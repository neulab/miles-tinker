# MI350: Qwen3.5-4B with the ROCm 7.2 image

For concise Qwen3 and Qwen3.5 setup and launch commands, use the
[ROCm 7.2 quick-start](mi350-rocm720-quickstart.md).

This recipe provisions and starts a text-only, full-fine-tuning Tinker server
from this checkout. It uses one MI350 GPU for training and four single-GPU
SGLang inference engines. No experiment scripts or second checkout are needed.
Run all server commands on the allocated MI350 node, in the same Bash shell.
The host needs Apptainer, Python 3.9+, network access for image/model downloads,
and sufficient local `/tmp` space for the image, model, and runtime. Store
training checkpoints on persistent storage; a model/optimizer checkpoint in
the original validation occupied approximately 55 GiB.

## 1. Set paths and provision the image

The pinned image is `rocm/sgl-dev:miles-rocm720-mi35x-20260914`. The digest below
selects the exact image used in validation. `--preserve-image-packages` keeps
its native Ray, ModelOpt, Bridge, and SGLang stack; omitting it applies the
older-image compatibility setup.

```bash
cd /home1/adityabs/miles-tinker
export MILES_AMD_ROOT=/tmp/miles-qwen35-rocm720
export MILES_AMD_IMAGE=docker://rocm/sgl-dev@sha256:f37c0e4d76f64185fe61c2a4362ae96f121d299f113319098c21f2e298afe185
export MILES_AMD_CHECKPOINTS=/work1/grahamneubig/adityabs/miles-tinker/mi350-qwen35-rocm720-checkpoints
export MILES_AMD_CONTEXT=16384
export MILES_AMD_PORT=10635

# Use the launcher's explicit binds rather than inherited Apptainer binds.
unset APPTAINER_BINDPATH APPTAINER_BIND
ulimit -c 0

python3 tools/amd_tinker.py setup \
  --root "$MILES_AMD_ROOT" \
  --checkpoint-root "$MILES_AMD_CHECKPOINTS" \
  --image-uri "$MILES_AMD_IMAGE" --image-workspace /sgl-workspace \
  --preserve-image-packages \
  --model-id Qwen/Qwen3.5-4B --model-type qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a

python3 tools/amd_tinker.py doctor \
  --root "$MILES_AMD_ROOT" \
  --checkpoint-root "$MILES_AMD_CHECKPOINTS" \
  --image-uri "$MILES_AMD_IMAGE" --image-workspace /sgl-workspace \
  --preserve-image-packages
```

Setup downloads the image and HF model, creates the runtime environment, and
installs this checkout editable. The image's AITER source is copied into the
runtime for writable JIT build artifacts. Its kernels are not patched.
Use a separate runtime root from any older-image deployment.

## 2. Convert the base checkpoint

The server uses the existing raw text-only Qwen3.5 model definition. Convert
the downloaded HF weights to a Megatron distributed checkpoint once before
startup. This inline command uses the same container configuration as the
launcher and the repository's `tools/convert_hf_to_torch_dist.py`; it does not
create a helper script. Conversion uses GPU 0, which must be free.

```bash
python3 - <<'PY'
import os
from pathlib import Path

from tools.amd_tinker import BackendConfig, _container_command, _run

config = BackendConfig(
    root=Path(os.environ["MILES_AMD_ROOT"]),
    checkpoint_root=Path(os.environ["MILES_AMD_CHECKPOINTS"]),
    image_uri=os.environ["MILES_AMD_IMAGE"],
    preserve_image_packages=True,
    model_id="Qwen/Qwen3.5-4B",
    model_type="qwen3.5-4B",
)
output = config.root / "models/Qwen3.5-4B_torch_dist"
marker = output / "latest_checkpointed_iteration.txt"
if marker.exists() and marker.read_text().strip() == "release":
    print(f"Reusing converted checkpoint: {output}")
else:
    if output.exists():
        raise SystemExit(f"Incomplete conversion at {output}; move it aside before retrying")
    code = '''import runpy, shlex, sys
from miles.utils.external_utils.model_args_utils import load_model_args
checkpoint, output, context = sys.argv[1:]
sys.argv = [
    "tools/convert_hf_to_torch_dist.py",
    *shlex.split(load_model_args("qwen3.5-4B")),
    "--hf-checkpoint", checkpoint, "--save", output,
    "--bf16", "--seq-length", context,
    "--max-position-embeddings", "262144", "--attention-backend", "flash",
]
runpy.run_path(sys.argv[0], run_name="__main__")
'''
    _run(config, _container_command(config, [
        "env", "HIP_VISIBLE_DEVICES=0", "WORLD_SIZE=1", "RANK=0", "LOCAL_RANK=0",
        config.python, "-c", code, config.model_path, output,
        os.environ["MILES_AMD_CONTEXT"],
    ]))
    assert marker.read_text().strip() == "release", "Conversion did not finish"
PY
```

## 3. Start the server

Use a persistent terminal session for this foreground command. Logs are also
written to the runtime directory. The port must be free.

```bash
set -o pipefail
python3 tools/amd_tinker.py serve \
  --root "$MILES_AMD_ROOT" \
  --checkpoint-root "$MILES_AMD_CHECKPOINTS" \
  --image-uri "$MILES_AMD_IMAGE" --image-workspace /sgl-workspace \
  --preserve-image-packages \
  --model-id Qwen/Qwen3.5-4B --model-type qwen3.5-4B \
  --train-gpus 1 --rollout-gpus 4 --rollout-gpus-per-engine 1 \
  --tp 1 --pp 1 --cp 1 \
  --context-length "$MILES_AMD_CONTEXT" \
  --max-tokens-per-gpu 8192 --mem-fraction 0.7 \
  --port "$MILES_AMD_PORT" -- \
  --megatron-to-hf-mode raw \
  --ref-load "$MILES_AMD_ROOT/models/Qwen3.5-4B_torch_dist" \
  --train-env-vars '{"MILES_SGLANG_LEGACY_WEIGHT_UPDATE":"0"}' \
  --sglang-device cuda \
  --seed 42 --bf16 --optimizer adam --lr 1e-6 --lr-decay-style constant \
  --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
  --weight-decay 0.017 --clip-grad 1.0 \
  2>&1 | tee "$MILES_AMD_ROOT/logs/server.log"
```

The newer image requires `--sglang-device cuda` (also the device name on ROCm)
and `MILES_SGLANG_LEGACY_WEIGHT_UPDATE=0` for the weight-session API. Do not add
a trainer-only `HIP_VISIBLE_DEVICES=0` override to `--train-env-vars`: training
and inference processes need common device visibility for RCCL weight transfer.
The single-GPU visibility setting in checkpoint conversion is limited to that
separate process.

`MILES_AMD_CONTEXT` controls both SGLang's inference context and the gateway's
maximum tokens per training datum. For example, set it to `45000` before
startup to raise both limits. Inference context includes prompt plus generated
tokens. `--max-tokens-per-gpu` is a separate training microbatch budget.
The historical validation used 16,384 context tokens; larger contexts require
their own memory and workload validation.

The checkpoint-writer compatibility guard in this checkout allows newer
Megatron to use its native threaded writer while retaining the workaround
needed by older images. No private FLA, Triton, or Megatron patches are needed
for the validated configuration.

## 4. Check readiness and connect a client

In another terminal on the server node:

```bash
curl --fail http://127.0.0.1:10635/health
tail -f /tmp/miles-qwen35-rocm720/logs/server.log
```

Use the configured port and runtime path if you changed them above. The health
endpoint becomes available after server initialization.

If the client runs on a different machine, run this command **on the client
machine**, replacing `MI350_SSH_HOST` with the SSH hostname or alias for your
allocated MI350 node:

```bash
ssh -NT -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 10635:127.0.0.1:10635 MI350_SSH_HOST
```

Keep the tunnel open. Configure the client's Tinker base URL as
`http://127.0.0.1:10635` and model as `Qwen/Qwen3.5-4B`. For clients that read
these environment variables:

```bash
export TINKER_BASE_URL=http://127.0.0.1:10635
export TINKER_API_KEY=local-qwen35
```

The direct gateway uses the API key as a tenant identifier, not authentication;
use a consistent value for a client's requests. This recipe uses an SSH tunnel
for remote access. Client installation and task-specific training settings
belong to the client project.

## 5. Stop the server

From this checkout on the server node:

```bash
python3 tools/amd_tinker.py stop --root /tmp/miles-qwen35-rocm720
```

Use your selected runtime root if different. Checkpoints remain at the
persistent `--checkpoint-root`; runtime logs and downloaded base models remain
under `--root`.

## Validation provenance and scope

The original September 2026 image experiment passed three optimizer updates
with batch sizes 1, 2, and 16, then saved and reloaded model/optimizer state with
identical checked logprobs. A five-batch non-reasoning number-search run also
completed all optimizer updates and saved step-5 and final checkpoints.
These are historical results, not a new GPU validation of this documentation.

Validation covers dense text-only BF16 with DP1/TP1/PP1/CP1 and short
non-thinking rollouts. It does not establish multimodal training, context
parallelism, or long-run stability. In the tested client, retaining reasoning
history in the renderer alone did not preserve sampled token prefixes through
the client's structured-content normalization; reasoning-history support
requires separate client work.

Historical evidence remains in
`/work1/grahamneubig/adityabs/miles-tinker-codex/artifacts/qwen35-rocm720/`.
That directory is not required for any setup or startup command above.
