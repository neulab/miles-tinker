# MI350 Tinker server: ROCm 7.2 quick-start

Run these commands on an MI350 node with Apptainer and Python 3.9+.
Choose Qwen3 or Qwen3.5 below and run one server at a time. Both launch
commands enable a public, authenticated Modal HTTPS relay.

## 1. Set up the image and model

```bash
cd /home1/adityabs/miles-tinker
export MILES_AMD_IMAGE=docker://rocm/sgl-dev@sha256:f37c0e4d76f64185fe61c2a4362ae96f121d299f113319098c21f2e298afe185
export MILES_AMD_ROOT=/tmp/miles-rocm720
unset APPTAINER_BINDPATH APPTAINER_BIND
```

Choose **one** model:

```bash
# Qwen3
export MODEL_ID=Qwen/Qwen3-4B-Instruct-2507
export MODEL_TYPE=qwen3-4B-Instruct-2507
export MODEL_REVISION=main
```

```bash
# Qwen3.5
export MODEL_ID=Qwen/Qwen3.5-4B
export MODEL_TYPE=qwen3.5-4B
export MODEL_REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
```

Then provision the image and download the selected model:

```bash
python3 tools/amd_tinker.py setup \
  --root "$MILES_AMD_ROOT" \
  --image-uri "$MILES_AMD_IMAGE" --preserve-image-packages \
  --model-id "$MODEL_ID" --model-type "$MODEL_TYPE" \
  --revision "$MODEL_REVISION"

python3 tools/amd_tinker.py doctor \
  --root "$MILES_AMD_ROOT" --image-uri "$MILES_AMD_IMAGE"
```

For **Qwen3.5 only**, convert the base checkpoint once before starting the
server. GPU 0 must be free:

```bash
python3 tools/amd_tinker.py convert \
  --root "$MILES_AMD_ROOT" \
  --image-uri "$MILES_AMD_IMAGE" --preserve-image-packages \
  --model-id Qwen/Qwen3.5-4B --model-type qwen3.5-4B \
  --context-length 16384
```

## 2. Start the server

Run the matching command in the same shell, inside a persistent terminal
session. Checkpoints go to persistent storage; logs go under the runtime root.

Authenticate with Modal once on this host (reuse existing authentication if
already configured). Choose the `neulab` workspace:

```bash
python3 -m venv "$MILES_AMD_ROOT/modal-venv"
"$MILES_AMD_ROOT/modal-venv/bin/pip" install modal==1.2.6
"$MILES_AMD_ROOT/modal-venv/bin/modal" setup
```

### Qwen3-4B-Instruct-2507

```bash
MODAL_PROFILE=neulab python3 tools/amd_tinker.py serve \
  --root "$MILES_AMD_ROOT" \
  --image-uri "$MILES_AMD_IMAGE" --preserve-image-packages \
  --checkpoint-root /work1/grahamneubig/adityabs/miles-tinker/qwen3-rocm720-checkpoints \
  --model-id Qwen/Qwen3-4B-Instruct-2507 --model-type qwen3-4B-Instruct-2507 \
  --train-gpus 4 --rollout-gpus 4 --rollout-gpus-per-engine 1 \
  --tp 1 --pp 1 --cp 1 \
  --context-length 65000 --max-tokens-per-gpu 65000 --mem-fraction 0.9 \
  --port 10613 --modal-relay -- \
  --sglang-device cuda \
  --train-env-vars '{"MILES_SGLANG_LEGACY_WEIGHT_UPDATE":"0"}' \
  --seed 42 --bf16 --optimizer adam --lr 1e-6 --lr-decay-style constant \
  --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
  --weight-decay 0.017 --clip-grad 1.0 \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  > "$MILES_AMD_ROOT/logs/server.log" 2>&1
```

### Qwen3.5-4B (text only)

```bash
MODAL_PROFILE=neulab python3 tools/amd_tinker.py serve \
  --root "$MILES_AMD_ROOT" \
  --image-uri "$MILES_AMD_IMAGE" --preserve-image-packages \
  --checkpoint-root /work1/grahamneubig/adityabs/miles-tinker/qwen35-rocm720-checkpoints \
  --model-id Qwen/Qwen3.5-4B --model-type qwen3.5-4B \
  --train-gpus 1 --rollout-gpus 4 --rollout-gpus-per-engine 1 \
  --tp 1 --pp 1 --cp 1 \
  --context-length 16384 --max-tokens-per-gpu 8192 --mem-fraction 0.7 \
  --port 10613 --modal-relay -- \
  --megatron-to-hf-mode raw \
  --ref-load "$MILES_AMD_ROOT/models/Qwen3.5-4B_torch_dist" \
  --sglang-device cuda \
  --train-env-vars '{"MILES_SGLANG_LEGACY_WEIGHT_UPDATE":"0"}' \
  --seed 42 --bf16 --optimizer adam --lr 1e-6 --lr-decay-style constant \
  --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
  --weight-decay 0.017 --clip-grad 1.0 \
  > "$MILES_AMD_ROOT/logs/server.log" 2>&1
```

`--context-length` sets both the inference context and the gateway's per-datum
token limit. Change it to your desired window. Qwen3.5 was validated at 16K;
the Qwen3 command carries over the older-image settings and has not yet been
GPU-validated on this image.

In another terminal, check readiness:

```bash
curl --fail http://127.0.0.1:10613/health
tail -f /tmp/miles-rocm720/logs/server.log
```

The relay starts automatically. Read the public URL and client API key:

```bash
cat /tmp/miles-rocm720/runtime/modal-relay.json
cat /tmp/miles-rocm720/runtime/public-api-key
```

Set the client's `TINKER_BASE_URL` to the public URL and `TINKER_API_KEY` to
that key. No manual SSH tunnel is needed. The URL changes on relay restart.

To stop either server:

```bash
python3 tools/amd_tinker.py stop --root /tmp/miles-rocm720
```
