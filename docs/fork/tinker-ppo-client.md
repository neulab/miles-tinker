# DAPO-Math-17K PPO through Tinker

Run Qwen3-4B on one 8-GPU AMD node: actor and critic share four training
GPUs with offloading; four dedicated GPUs serve inference. Use two Bash
terminals on the allocated node, with Python 3.9+ and Apptainer available.
Images, environments, models, and data are stored under `PPO_ROOT`.
This synchronous PPO setup has been validated on both MI300 and MI350 using HF Bridge.

In **both terminals**, set the same paths and a fresh run name:

```bash
cd ~/miles_final
PPO_ROOT="$WORK/miles_artifacts"
PPO_RUN=tinker-ppo-dapo
```

## First-time server preparation

Run setup once per GPU type, using the same root as the
[AMD guide](../platforms/amd-tinker.md#setup). Setup and serve automatically
detect the current node's MI300 or MI350 GPUs. Setup installs
the image and environment, runs the GPU doctor, and prepares Modal access.
Qwen3-4B loads HF weights through Bridge; no checkpoint conversion is needed.
Bridge initializes the new critic head with mean-zero Gaussian weights,
standard deviation `1 / (hidden_size + 1)`, and zero bias.

```bash
python3 tools/amd_tinker.py setup --root "$PPO_ROOT" --model-family qwen3
```

## Server — terminal 1

The launcher runs the server inside the prepared GPU Apptainer image.

```bash
python3 tools/amd_tinker.py serve \
  --root "$PPO_ROOT" --model-family qwen3 \
  --model-id Qwen/Qwen3-4B --model-type qwen3-4B \
  --train-gpus 4 --rollout-gpus 4 --tp 1 --pp 1 --cp 1 \
  --mem-fraction 0.9 --keep-checkpoints 3 \
  --checkpoint-root "$PPO_ROOT/runs/$PPO_RUN/checkpoints" -- \
  --bf16 --megatron-to-hf-mode bridge \
  --tinker-enable-critic --offload-train --balance-data \
  --rollout-temperature 0.8 --seed 1234 \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
```

| PPO setup | Purpose |
|---|---|
| `--tinker-enable-critic` | Adds the critic; the launcher already selects full-model training. |
| `--train-gpus 4 --rollout-gpus 4` | Shared actor/critic training pool and separate inference pool. Do not add `--colocate`. |
| `--tp 1 --pp 1 --cp 1` | Data parallel training across the four training GPUs. |
| `--offload-train` | Switch actor/critic residency automatically. |
| `--megatron-to-hf-mode bridge` | Load HF weights directly; initialize the critic head with small Gaussian weights and zero bias. |
| `--rollout-temperature 0.8` | Must match the client's sampling temperature. |
| `--balance-data` | Balance training sequences across ranks. |
| `--keep-checkpoints 3` | Retain three checkpoints per model, including across resume. |

BF16, activation recomputation, and inference memory fraction are retained from
the tested launch. Request capacity uses the launcher's default of 128. Server logs are under
`$PPO_ROOT/runs/$PPO_RUN/checkpoints/logs/`.

## Client — terminal 2

The CPU-only client uses a separate Python 3.12 image and unmodified
`tinker==0.16.1`; the prepared GPU image's Python 3.10 cannot run this SDK.

```bash
cd ~/miles_final
PPO_ROOT="$WORK/miles_artifacts"
PPO_RUN=tinker-ppo-dapo

PPO_IMAGE="$PPO_ROOT/images/ppo-client-python312.sif"
PPO_ENV="$PPO_ROOT/venvs/ppo-client-container"
if [ -n "${WANDB_API_KEY:-}" ]; then
  export APPTAINERENV_WANDB_API_KEY="$WANDB_API_KEY"
fi
PPO_CLIENT=(
  apptainer exec --cleanenv --home "$HOME"
  --bind "$PWD:$PWD" --bind "$PPO_ROOT:$PPO_ROOT" --pwd "$PWD"
  --env "PYTHONPATH=$PWD" --env "TINKER_API_KEY=tml-local-ppo"
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
curl --fail http://127.0.0.1:10613/health &&
"${PPO_CLIENT[@]}" "$PPO_ENV/bin/python" deploy/clients/dapo_ppo.py \
  --base-url http://127.0.0.1:10613 --base-model Qwen/Qwen3-4B \
  --model-path "$PPO_ROOT/models/Qwen3-4B" \
  --data "$PPO_ROOT/datasets/dapo-math-17k/dapo-math-17k.jsonl" \
  --output "$PPO_ROOT/runs/$PPO_RUN" \
  --rollouts 300 --save-interval 10 \
  --wandb-project miles-amd-ppo --wandb-entity 9998_team
```

The base URL is the server address; `127.0.0.1` works because both containers
run on the same node. `tml-local-ppo` is the local gateway's tenant identifier.
W&B uses your home-directory credentials or forwarded `WANDB_API_KEY`.
Metrics also go to the run's `metrics.jsonl`.

Defaults preserve the reference recipe: 8 prompts × 8 responses per rollout,
32 responses per optimizer step, maximum response length 8192, temperature
0.8, actor/critic learning rates 1e-6/1e-5, GAE γ=λ=1, and clipping 0.2.
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
