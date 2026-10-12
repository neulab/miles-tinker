# DAPO-Math-17K PPO through Tinker

Host Qwen3-4B-Instruct-2507 across four allocated nodes with one MI350 GPU
per node. Actor and critic stay resident on two shared training GPUs;
two dedicated GPUs serve inference. Images, environments, models, and data
are stored under `PPO_ROOT`.

To host a Tinker endpoint, complete server preparation and run the server
command below. The client section is optional: it runs DAPO-Math-17K PPO
against that endpoint.

In **both terminals**, set the same paths and a fresh run name:

```bash
cd ~/miles_final
PPO_ROOT="$WORK/miles_artifacts"
PPO_RUN=tinker-ppo-dapo
```

## First-time server preparation

Complete the one-time [AMD setup](../platforms/amd-tinker.md#setup) on an
allocated MI350 node. Setup installs the image and environment, runs the GPU
doctor, and prepares Modal access. Use the same shared repository, image,
venv, model, and checkpoint paths on all four nodes. Each node needs Python
3.9+, Apptainer, tmux, `ip`, and GNU `tee`.
Qwen3 loads HF weights through Bridge; no checkpoint conversion is needed.
Bridge initializes the new critic head with mean-zero Gaussian weights,
standard deviation `1 / (hidden_size + 1)`, and zero bias.

```bash
python3 tools/amd_tinker.py setup --root "$PPO_ROOT" --model-family qwen3
```

## Server — terminal 1

Run this on the first node in `NODES`, which becomes the Ray head. Replace
the hostnames with your current allocation; the head needs passwordless SSH
access to all four nodes, including itself. Set the CPU count no higher than
the CPUs allocated per node. See the [multinode guide](../platforms/amd-tinker-cluster.md)
for deployment details.

The launcher starts Ray and the server in tmux sessions, using the prepared
GPU Apptainer image. The two GPU counts must add up to the number of nodes.

```bash
NODES=(k007-005-v1 k007-005-v2 k007-005-v3 k007-005-v5)
TRAIN_GPUS=2
INFERENCE_GPUS=2
RAY_CPUS_PER_NODE=24
MODEL_ID=Qwen/Qwen3-4B-Instruct-2507
MODEL_TYPE=qwen3-4B-Instruct-2507

python3 tools/amd_tinker_cluster.py start \
  --root "$PPO_ROOT" --checkpoint-root "$PPO_ROOT/runs/$PPO_RUN/checkpoints" \
  --nodes "${NODES[@]}" \
  --train-gpus "$TRAIN_GPUS" --rollout-gpus "$INFERENCE_GPUS" \
  --cpus-per-node "$RAY_CPUS_PER_NODE" \
  --interface eth0 --ray-port 16379 --port 11613 -- \
  --model-id "$MODEL_ID" --model-type "$MODEL_TYPE" \
  --tp 1 --pp 1 --cp 1 \
  --context-length 32768 --max-tokens-per-gpu 50000 --mem-fraction 0.95 \
  --keep-checkpoints 3 --modal-relay -- \
  --tinker-enable-critic --no-offload-train \
  --bf16 --optimizer adam --lr 1e-6 --lr-decay-style constant \
  --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
  --weight-decay 0.017 --clip-grad 1.0 --seed 1234 \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --balance-data
```

| PPO setup | Purpose |
|---|---|
| `--tinker-enable-critic` | Adds the critic; the launcher already selects full-model training. |
| `--train-gpus 2 --rollout-gpus 2` | Shared actor/critic training pool and separate inference pool. Do not add `--colocate`. |
| `--tp 1 --pp 1 --cp 1` | Data parallel training across the two training GPUs. |
| `--no-offload-train` | Keep both models resident on the shared training GPUs; execution is sequential and both must fit in memory. |
| HF Bridge (launcher default for Qwen3) | Load HF weights directly; initialize the critic head with small Gaussian weights and zero bias. |
| `--rollout-temperature` omitted | Keep the server default of `1.0`; the PPO client's sampling temperature must match. |
| `--modal-relay` | Expose the endpoint through Modal and print the client URL and API key. |
| `--balance-data` | Balance training sequences across ranks. |
| `--keep-checkpoints 3` | Retain three checkpoints per model, including across resume. |

The first `--` forwards serve options; the second forwards training options.
`--mem-fraction` controls inference memory, while both models' training state
and active-model activations must fit on the training GPUs. Request capacity
uses the launcher's default of 128. Server logs are under
`$PPO_ROOT/runs/$PPO_RUN/checkpoints/logs/`.

Once ready, the launcher prints `export TINKER_BASE_URL=...` and
`export TINKER_API_KEY=...`. Run those exports in the client terminal before
continuing. The Modal relay has a 24-hour lifetime. To check readiness and
print the connection settings again:

```bash
python3 tools/amd_tinker_cluster.py status --root "$PPO_ROOT"
```

## Client — terminal 2

The CPU-only client uses a separate Python 3.12 image and unmodified
`tinker==0.16.1`; the prepared GPU image's Python 3.10 cannot run this SDK.

```bash
cd ~/miles_final
PPO_ROOT="$WORK/miles_artifacts"
PPO_RUN=tinker-ppo-dapo
: "${TINKER_BASE_URL:?Export the URL printed by the server launcher}"
: "${TINKER_API_KEY:?Export the API key printed by the server launcher}"

PPO_IMAGE="$PPO_ROOT/images/ppo-client-python312.sif"
PPO_ENV="$PPO_ROOT/venvs/ppo-client-container"
if [ -n "${WANDB_API_KEY:-}" ]; then
  export APPTAINERENV_WANDB_API_KEY="$WANDB_API_KEY"
fi
PPO_CLIENT=(
  apptainer exec --cleanenv --home "$HOME"
  --bind "$PWD:$PWD" --bind "$PPO_ROOT:$PPO_ROOT" --pwd "$PWD"
  --env "PYTHONPATH=$PWD" --env "TINKER_API_KEY=$TINKER_API_KEY"
  "$PPO_IMAGE"
)
```

If the client environment is not prepared, run once:

```bash
[ -f "$PPO_IMAGE" ] || apptainer pull "$PPO_IMAGE" docker://python:3.12-slim
[ -x "$PPO_ENV/bin/python" ] || "${PPO_CLIENT[@]}" python -m venv "$PPO_ENV"
"${PPO_CLIENT[@]}" "$PPO_ENV/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cpu torch
"${PPO_CLIENT[@]}" "$PPO_ENV/bin/python" -m pip install \
  -r deploy/clients/requirements-dapo-ppo.txt aiohttp
```

Download the dataset once if it is not already present:

```bash
"${PPO_CLIENT[@]}" "$PPO_ENV/bin/hf" download --repo-type dataset \
  zhuzilin/dapo-math-17k --local-dir "$PPO_ROOT/datasets/dapo-math-17k"
```

Miles' grader and sequence balancing require the checkout, CPU PyTorch, and
`aiohttp`. No Miles-specific SDK or custom transport is installed.

Start synchronous PPO once the server is healthy. Synchronous sampling is
the default; do not add `--async-rollouts` for this run.

```bash
curl --fail -H "X-API-Key: $TINKER_API_KEY" "${TINKER_BASE_URL%/}/health" &&
"${PPO_CLIENT[@]}" "$PPO_ENV/bin/python" deploy/clients/dapo_ppo.py \
  --base-url "$TINKER_BASE_URL" --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --model-path "$PPO_ROOT/models/Qwen3-4B-Instruct-2507" \
  --data "$PPO_ROOT/datasets/dapo-math-17k/dapo-math-17k.jsonl" \
  --output "$PPO_ROOT/runs/$PPO_RUN" \
  --rollouts 300 --save-interval 10 --temperature 1.0 \
  --wandb-project miles-amd-ppo --wandb-entity 9998_team
```

The client uses the URL and API key printed by the launcher, so it can run
from another host with access to its environment, tokenizer, and dataset.
W&B uses your home-directory credentials or forwarded `WANDB_API_KEY`.
Metrics also go to the run's `metrics.jsonl`.

The client command explicitly sets temperature `1.0` to match the server.
The DAPO client's own default remains `0.8`; to use it, change the client
flag to `--temperature 0.8` and add `--rollout-temperature 0.8` after the
server command's second `--`. Sampling requests do not change the server's
training-logit temperature.

Other client defaults are 8 prompts × 8 responses per rollout, 32 responses
per optimizer step, maximum response length 8192, actor/critic learning
rates 1e-6/1e-5, GAE γ=λ=1, and clipping 0.2.
Each rollout has two optimizer steps per model; the first trains only the critic.

## Async and resume options

- **Async:** append `--async-rollouts` to the client command. It samples one
  batch ahead using generating-policy logprobs; actor/critic training stays
  sequential. Use a fresh run name for a separate sync/async comparison.
- **Resume:** rerun the client command with
  `--resume "$PPO_ROOT/runs/$PPO_RUN/latest.json"`. `--rollouts` is the total
  target, not an additional count. A restarted server must use the same
  checkpoint root and training topology.
- **Verify resume:** optionally add `--verify-resume` to check saved actor
  logprobs and critic values before continuing.

The command saves both models every 10 rollout batches and at completion;
`latest.json` records the pair after both saves succeed. Retention keeps three
per role. For a short checkpoint test, use `--rollouts 25`, then resume with
a larger total. See [PPO semantics](tinker-ppo.md) for API and numerical limitations.

## Stop the deployment

Let the client finish saving an actor/critic pair if you need to resume, then
stop the client. Stop the server and Ray across all four nodes with:

```bash
python3 tools/amd_tinker_cluster.py stop --root "$PPO_ROOT"
```

Checkpoints and logs are retained. This stops the deployment; release the
scheduler allocation separately.
