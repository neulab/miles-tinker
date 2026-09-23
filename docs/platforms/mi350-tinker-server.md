# Tinker endpoint on MI350

Run on an eight-GPU MI350 node with Apptainer, Python 3.9+, SSH, and internet
access. The Qwen3 example uses four training GPUs and four inference GPUs;
the Qwen3.5 example uses one training GPU and two inference GPUs. Both use the
ROCm 7.2 image pinned by the launcher; see
[implementation notes](mi350-tinker-changes.md).

## Qwen3-4B-Instruct

### Set up

```bash
cd ~/miles-tinker
export MILES_AMD_ROOT=/tmp/miles-tinker-rocm720
export MILES_AMD_CHECKPOINTS="${WORK:?Set WORK to your persistent work directory}/miles-tinker/checkpoints"
unset APPTAINER_BINDPATH APPTAINER_BIND

python3 tools/amd_tinker.py setup \
  --root "$MILES_AMD_ROOT" --checkpoint-root "$MILES_AMD_CHECKPOINTS"

python3 tools/amd_tinker.py doctor --root "$MILES_AMD_ROOT"

python3 -m venv "$MILES_AMD_ROOT/modal-venv"
"$MILES_AMD_ROOT/modal-venv/bin/pip" install modal==1.2.6
"$MILES_AMD_ROOT/modal-venv/bin/modal" setup
```

Choose your Modal workspace during authentication. `WORK` should point to your
persistent work directory, for example `/work1/grahamneubig/adityabs`. Checkpoints
then go under `/work1/grahamneubig/adityabs/miles-tinker/checkpoints/`.

### Start

Run in the same shell, inside a persistent terminal session.

```bash
python3 tools/amd_tinker.py serve \
  --root "$MILES_AMD_ROOT" --checkpoint-root "$MILES_AMD_CHECKPOINTS" \
  --train-gpus 4 --rollout-gpus 4 \
  --context-length 32768 --max-tokens-per-gpu 16384 --mem-fraction 0.7 \
  --modal-relay -- \
  --seed 42 --bf16 --optimizer adam --lr 1e-6 --lr-decay-style constant \
  --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
  --weight-decay 0.017 --clip-grad 1.0 \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  > "$MILES_AMD_ROOT/logs/server.log" 2>&1
```

`--context-length` sets inference context and the maximum training datum length.
`--max-tokens-per-gpu` controls training batching separately.

## Qwen3.5-4B

### Set up and convert

For a fresh deployment, run setup and conversion once before starting the
server. Use the existing Modal account, or authenticate with `modal setup`
below. Model weights use the default revision; no revision pin is required.

```bash
cd ~/miles-tinker
export MILES_AMD_ROOT=/tmp/miles-tinker-rocm720
export MILES_AMD_CHECKPOINTS="${WORK:?Set WORK to your persistent work directory}/miles-tinker/checkpoints"
MODEL_ARGS=(--model-id Qwen/Qwen3.5-4B --model-type qwen3.5-4B)
unset APPTAINER_BINDPATH APPTAINER_BIND

python3 tools/amd_tinker.py setup \
  --root "$MILES_AMD_ROOT" --checkpoint-root "$MILES_AMD_CHECKPOINTS" \
  "${MODEL_ARGS[@]}"

python3 tools/amd_tinker.py convert \
  --root "$MILES_AMD_ROOT" --checkpoint-root "$MILES_AMD_CHECKPOINTS" \
  "${MODEL_ARGS[@]}" --context-length 16384

python3 tools/amd_tinker.py doctor --root "$MILES_AMD_ROOT"

python3 -m venv "$MILES_AMD_ROOT/modal-venv"
"$MILES_AMD_ROOT/modal-venv/bin/pip" install modal==1.2.6
"$MILES_AMD_ROOT/modal-venv/bin/modal" setup
```

Conversion prepares the training checkpoint from the downloaded HF weights
and needs GPU 0 free. Subsequent launches only need the serve command.

### Start

Run in the same shell, inside a persistent terminal session. This starts one
training GPU and two independent, one-GPU inference engines. The launcher
automatically selects the Qwen3.5 model definition and converted checkpoint.

```bash
python3 tools/amd_tinker.py serve \
  --root "$MILES_AMD_ROOT" --checkpoint-root "$MILES_AMD_CHECKPOINTS" \
  --model-id Qwen/Qwen3.5-4B --model-type qwen3.5-4B \
  --train-gpus 1 --rollout-gpus 2 --rollout-gpus-per-engine 1 \
  --context-length 16384 --max-tokens-per-gpu 8192 --mem-fraction 0.7 \
  --modal-relay -- \
  --seed 42 --bf16 --optimizer adam --lr 1e-6 --lr-decay-style constant \
  --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
  --weight-decay 0.017 --clip-grad 1.0 \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  > "$MILES_AMD_ROOT/logs/server.log" 2>&1
```

This serves text-only Qwen3.5. The commands below for readiness, client
settings, and stopping apply to either model.

## Check readiness and get client settings

In another terminal on the GPU node:

```bash
curl --fail http://127.0.0.1:10613/health
tail -n 30 /tmp/miles-tinker-rocm720/logs/server.log

python3 - <<'PY'
import json
from pathlib import Path
root = Path('/tmp/miles-tinker-rocm720/runtime')
print('export TINKER_BASE_URL=' + json.loads((root / 'modal-relay.json').read_text())['url'])
print('export TINKER_API_KEY=' + (root / 'public-api-key').read_text().strip())
PY
```

Copy the two printed exports into the client shell. Use the model you launched:
`Qwen/Qwen3-4B-Instruct-2507` or `Qwen/Qwen3.5-4B`.
The public URL changes on restart; the relay lasts
up to 24 hours per launch. Relay failure stops the server. A printed URL alone
does not mean model initialization has finished; wait for the health check.

## Stop

```bash
cd ~/miles-tinker
python3 tools/amd_tinker.py stop --root /tmp/miles-tinker-rocm720
```

Checkpoints remain on disk. Optional cleanup, retaining the newest completed
checkpoint across this checkpoint root:

```bash
python3 tools/amd_tinker.py prune \
  --checkpoint-root "${WORK:?Set WORK to your persistent work directory}/miles-tinker/checkpoints" \
  --keep 1 --once
```
