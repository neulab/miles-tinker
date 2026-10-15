# Single-node Tinker on ABCI: GRPO and PPO

If this is the first time you are running this codebase on ABCI, complete [one-time setup](ENVIRONMENT.md), then run GRPO or PPO in terminal 1 and Modal relay in terminal 2. Each block sources
the environment; `miles_python` runs Python in your NVIDIA container.

Set scratch in each host terminal before running the commands below. If
`PBS_LOCALDIR` is absent, set `ABCI_SCRATCH` to your current allocation's
directory under `/local`:

```bash
# If needed, replace JOB_ID with your current PBS job ID (including .pbs1).
# export ABCI_SCRATCH=/local/JOB_ID
export ABCI_SCRATCH="${ABCI_SCRATCH:-${PBS_LOCALDIR:-}}"
: "${ABCI_SCRATCH:?Set ABCI_SCRATCH to the local scratch directory for your current allocation}"
```

Both recipes use HF Bridge without raw Megatron checkpoint conversion, following the
[AMD GRPO](../../docs/platforms/amd-tinker.md) and
[PPO](../../docs/fork/tinker-ppo-client.md) examples. The client owns the data,
rewards, training schedule and saves.

Edit the tuning variables at the top of either example. `MODEL_TYPE` must
match a definition in `scripts/models/`; GPU counts should total eight.
`CONTEXT_LENGTH` covers prompt plus response; `MAX_TOKENS_PER_GPU` is the
training microbatch packing budget for sequence packing.

## GRPO — terminal 1

Two training GPUs and six inference GPUs.

```bash
(
set -euo pipefail
source "$HOME/miles-env/abci-env.sh"

# Tune these for your workload.
MODEL_ID=Qwen/Qwen3-4B-Instruct-2507
MODEL_TYPE=qwen3-4B-Instruct-2507
TRAIN_GPUS=2
ROLLOUT_GPUS=6
CONTEXT_LENGTH=66000
MAX_TOKENS_PER_GPU=50000
INFERENCE_MEM_FRACTION=0.9

# Paths and logging.
MODEL_DIR="$PROJECT_ROOT/models/$MODEL_ID"
CHECKPOINT_ROOT="$PROJECT_ROOT/checkpoints"
umask 077
mkdir -p "$CHECKPOINT_ROOT/logs"
LOG="$CHECKPOINT_ROOT/logs/server-$(date +%Y%m%d-%H%M%S).log"

miles_python -c 'import sys; from huggingface_hub import snapshot_download; snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])' \
  "$MODEL_ID" "$MODEL_DIR"
MODEL_FLAGS=$(miles_python -m miles.utils.external_utils.model_args_utils "$MODEL_TYPE")
read -r -a MODEL_ARGS <<< "$MODEL_FLAGS"

SERVER_ARGS=(
  # Model, endpoint and checkpoints.
  --hf-checkpoint "$MODEL_DIR" --megatron-to-hf-mode bridge
  --tinker-base-model "$MODEL_ID" --tinker-train-mode full
  --tinker-server-port 10613
  --tinker-checkpoint-root "$CHECKPOINT_ROOT" --tinker-checkpoints-to-keep 3

  # GPU allocation and parallelism.
  --num-gpus-per-node 8 --actor-num-nodes 1 --actor-num-gpus-per-node "$TRAIN_GPUS"
  --rollout-num-gpus "$ROLLOUT_GPUS" --rollout-num-gpus-per-engine 1
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1

  # Training batches and inference capacity.
  --use-dynamic-batch-size --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU"
  --sglang-context-length "$CONTEXT_LENGTH" --sglang-mem-fraction-static "$INFERENCE_MEM_FRACTION"
  --sglang-chunked-prefill-size 4096 --sglang-max-running-requests 128

  # Weight transfer.
  --pause-generation-mode retract --update-weight-buffer-size 1073741824

  # Optimizer initialization; the client supplies per-step Adam parameters.
  --bf16 --optimizer adam --lr 1e-6 --lr-decay-style constant
  --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8
  --weight-decay 0.017 --clip-grad 1.0

  # Precision, activation memory and workload balance.
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
  --balance-data
)

miles_python serve_tinker.py "${MODEL_ARGS[@]}" "${SERVER_ARGS[@]}" 2>&1 | tee "$LOG"
)
```

## PPO — terminal 1, instead of GRPO

Four training GPUs shared by resident actor/critic, plus four inference GPUs.

```bash
(
set -euo pipefail
source "$HOME/miles-env/abci-env.sh"

# Tune these for your workload.
MODEL_ID=Qwen/Qwen3-4B-Instruct-2507
MODEL_TYPE=qwen3-4B-Instruct-2507
TRAIN_GPUS=4
ROLLOUT_GPUS=4
CONTEXT_LENGTH=66000
MAX_TOKENS_PER_GPU=50000
INFERENCE_MEM_FRACTION=0.95

# Paths and logging.
MODEL_DIR="$PROJECT_ROOT/models/$MODEL_ID"
CHECKPOINT_ROOT="$PROJECT_ROOT/checkpoints"
umask 077
mkdir -p "$CHECKPOINT_ROOT/logs"
LOG="$CHECKPOINT_ROOT/logs/server-$(date +%Y%m%d-%H%M%S).log"

miles_python -c 'import sys; from huggingface_hub import snapshot_download; snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])' \
  "$MODEL_ID" "$MODEL_DIR"
MODEL_FLAGS=$(miles_python -m miles.utils.external_utils.model_args_utils "$MODEL_TYPE")
read -r -a MODEL_ARGS <<< "$MODEL_FLAGS"

SERVER_ARGS=(
  # Model, PPO residency, endpoint and checkpoints.
  --hf-checkpoint "$MODEL_DIR" --megatron-to-hf-mode bridge
  --tinker-base-model "$MODEL_ID" --tinker-train-mode full
  --tinker-enable-critic --offload-train
  --tinker-server-port 10613
  --tinker-checkpoint-root "$CHECKPOINT_ROOT" --tinker-checkpoints-to-keep 3

  # GPU allocation and parallelism.
  --num-gpus-per-node 8 --actor-num-nodes 1 --actor-num-gpus-per-node "$TRAIN_GPUS"
  --rollout-num-gpus "$ROLLOUT_GPUS" --rollout-num-gpus-per-engine 1
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1

  # Training batches and inference capacity.
  --use-dynamic-batch-size --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU"
  --sglang-context-length "$CONTEXT_LENGTH" --sglang-mem-fraction-static "$INFERENCE_MEM_FRACTION"
  --sglang-chunked-prefill-size 4096 --sglang-max-running-requests 128

  # Weight transfer.
  --pause-generation-mode retract --update-weight-buffer-size 1073741824

  # Optimizer initialization; the client supplies per-step Adam parameters.
  --bf16 --optimizer adam --lr 1e-6 --seed 1234

  # Precision, activation memory and workload balance.
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
  --balance-data
)

miles_python serve_tinker.py "${MODEL_ARGS[@]}" "${SERVER_ARGS[@]}" 2>&1 | tee "$LOG"
)
```

Actor and critic execute sequentially; `--offload-train` offloads each while the other trains. Use `--no-offload-train` to disable actor/critic offloading. Do not add `--colocate` to share GPUs between inference and training engines; this is not supported.

Server output is saved under `$PROJECT_ROOT/checkpoints/logs/`; relay output
stays in terminal 2. Ray starts locally.

## Modal relay — terminal 2

Run on the same node, outside the container. Reuse an existing relay on port
10613 if one is already running. `--ssh-over-tls` uses outbound port 443 for
ABCI's network restrictions and requires host `openssl`.

```bash
source "$HOME/miles-env/abci-env.sh"
module load python/3.12/3.12.9
curl --fail --noproxy '*' http://127.0.0.1:10613/api/v1/healthz && \
  "$PROJECT_ROOT/venvs/modal/bin/python" "$MILES_SRC/tools/amd_tinker_modal.py" \
    --root "$PROJECT_ROOT/relay" --port 10613 --ssh-over-tls
```

Retry after startup if the health check fails. The helper works on NVIDIA and
prints the authenticated HTTPS `TINKER_BASE_URL` and `TINKER_API_KEY` for your
remote client. Keep the key private; credentials live in
`$PROJECT_ROOT/relay/runtime/`.

Ctrl+C stops the relay and removes its sandbox; stop the server separately.
Client exit stops neither. End the allocation when finished.
