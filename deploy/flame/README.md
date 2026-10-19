# OpenHands RL: client on FLAME or Babel, Tinker server on AMD MI350

Terminal-agent RL (Platoon + OpenHands sandboxes on the tmax pool) splits into
two jobs that talk over the Tinker API:

| Side | Runs | Launcher | Hardware |
|---|---|---|---|
| Client | Platoon trainer loop, OpenHands agents, task sandboxes, eval | `deploy/flame/train_openhands.sbatch` | a FLAME or Babel node (CPU only when the server is remote) |
| Server | Megatron trainer + SGLang rollout engines behind the Tinker gateway | `deploy/amd/serve_openhands.sbatch` | AMD HPC Fund `mi3508x` (8x MI350X per node), one or two nodes |

The server publishes itself through a Modal relay and prints a URL and an API
key; the client gets them as `TINKER_BASE_URL` / `TINKER_API_KEY`. Without
`TINKER_BASE_URL` the client script instead runs trainer, engines and client on
one 8-GPU FLAME node.

The per-setting reference is in the header comments of the two launchers; this
page is the runbook.

## 1. One-time setup on AMD

Follow the [AMD setup](../../docs/platforms/amd-tinker.md#setup) (image, venv,
`$MILES_AMD_ROOT`, default `$WORK/miles_artifacts`), then convert the base model
once into `$MILES_AMD_ROOT/models` on `$WORK`, where every job and every node can
read it:

```bash
python3 tools/amd_tinker.py convert --root "$MILES_AMD_ROOT" \
    --model-family qwen3.5 --model-id Qwen/Qwen3.8-27B --model-type qwen3.8-27B
```

## 2. Start the server

One node (4 GPUs train, 4 serve), as a batch job or inside an allocation you hold:

```bash
sbatch deploy/amd/serve_openhands.sbatch                     # defaults: Qwen3.8-27B, mi3508x, 12 h
srun --jobid=<jobid> --overlap --pty bash deploy/amd/serve_openhands.sbatch
grep -E 'TINKER_(BASE_URL|API_KEY)' oh-tinker-<jobid>.log     # once it is up
```

Settings that worked for Qwen3.8-27B on one node:

```bash
CHECKPOINT_ROOT=$WORK/miles_artifacts/runs/<run>/checkpoints \
PYTORCH_HIP_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TP=4 SEQUENCE_PARALLEL=1 MAX_TOKENS_PER_GPU=8192 CONTEXT_WINDOW=131072 STOCHASTIC_ROUNDING=0 KL_COEF=0 \
bash deploy/amd/serve_openhands.sbatch
```

- Training GPUs peak at ~293-301 GB of 309 GB with these settings: stable, but
  near the limit. TP=2 (even with sequence parallelism) and recomputing only 48
  of 64 layers ran out of memory (`HSA_STATUS_ERROR_OUT_OF_RESOURCES`).
- Context parallelism is not available: the Tinker gateway requires CP=1.
- `KL_COEF=0` skips the reference-policy pass (~30% more training throughput).
- Measured: ~5k trained tokens/s, MFU ~9-11%.

### Two nodes

Hold a two-node allocation and run the same script on the first node with
`NODES` (or submit it with `sbatch -N 2`). Training takes all 8 GPUs of the first
node (two TP=4 replicas), the rollout engines the 8 of the second:

```bash
NODES="<node1> <node2>" TRAIN_GPUS=8 ROLLOUT_GPUS=8 \
CHECKPOINT_ROOT=$WORK/miles_artifacts/runs/<run>/checkpoints \
<the settings above> bash deploy/amd/serve_openhands.sbatch
```

- The model must be converted under `$MILES_AMD_ROOT/models` and
  `CHECKPOINT_ROOT` must be on shared storage; the script checks both.
- It goes through `tools/amd_tinker_cluster.py` with `--roce`, which sets the
  mi3508x RoCE NCCL settings on every node, including `NCCL_NET_GDR_LEVEL=0`:
  with GPU-direct RDMA, RCCL 2.27.7 fails the 1-to-8 cross-node weight push with
  "Message truncated". `NO_ROCE=1` leaves them out; `DRY_RUN=1` prints the plan.
- The server's own log, with the engines' output, is
  `$CHECKPOINT_ROOT/logs/server-*.log`; the launcher's terminal output is not.
- Stop it with Ctrl+C in the launcher, or from any shell:
  `python3 tools/amd_tinker_cluster.py stop --root "$MILES_AMD_ROOT"`.

## 3. Start the client

On FLAME, against the remote server (no GPUs needed):

```bash
TINKER_BASE_URL=https://... TINKER_API_KEY=... \
MODEL_NAME=Qwen/Qwen3.8-27B CONTEXT_WINDOW=131072 KL_COEF=0 \
STEP_TIMEOUT=<s> SAMPLE_TIMEOUT=<s> ROLLOUT_WORKERS=<n> \
RUN_NAME=<run> sbatch --gres=none deploy/flame/train_openhands.sbatch
```

On Babel the same script runs with Babel's partition/account and the pool on
shared storage: `DATA_DIR=<pool> WORK_DIR=<work> OVERLAY_DIR=<node-local dir>
STAGE=0`.

These must match the server, or the run fails or silently trains something else:

| Client | Server |
|---|---|
| `MODEL_NAME` | `MODEL_ID` |
| `CONTEXT_WINDOW` | `CONTEXT_WINDOW` (at least) |
| `KL_COEF` (0 or not) | `KL_COEF` |

Sizing the client for a 27B policy:

- `STEP_TIMEOUT` bounds an agent turn and `SAMPLE_TIMEOUT` the sampling call in
  it; the defaults (300 s / 600 s) are too short for a busy 27B server. Raise
  them together.
- `ROLLOUT_WORKERS` x `GROUP_SIZE` (8) rollouts run at once. Watch the engines'
  `full token usage` in the server log: above ~0.9 the KV cache evicts the
  agents' shared prefixes, every turn re-prefills its whole context, and calls
  queue. On two nodes, 72 workers (~64 rollouts per engine) reached 0.95 and a
  27-49% prefix-cache hit rate; use fewer.

## 4. The training recipe

The settings of the Qwen3.8-27B runs, as one
client command (on a Babel node inside an allocation, the pool on shared
storage; on FLAME the same variables go before `sbatch --gres=none`):

```bash
export TINKER_BASE_URL=https://... TINKER_API_KEY=...       # from the server log
export RUN_NAME=<run name> MODEL_NAME=Qwen/Qwen3.8-27B KL_COEF=0
P=<task pool directory>                                    # see "Task pools"

# What is trained on, and how the loss is formed
export TOOLS=terminal,file_editor,finish LOSS_NORM=trajectory ACTIVE_SAMPLING=1 MALFORMED_CREDIT=off
export TASK_SELECTION=adaptive ADAPTIVE_PRIOR=$P/<screening prior>.tsv ADAPTIVE_PRIOR_WEIGHT=8
export TERMINAL_RL_EXCLUDED_TASKS=$P/<excluded task list>.txt
# Batch, cadence, eval
export BATCH_SIZE=32 NUM_MICROBATCHES=8 MAX_STALENESS=2 ROLLOUT_WORKERS=<n>
export STEPS=200 CHECKPOINT_EVERY=1 EVAL_EVERY=3 EVAL_SIZE=50 EVAL_AT_START=0 WANDB_MODE=online
# Token budgets and timeouts
export REASONING_EFFORT=medium CONTEXT_WINDOW=131072 MAX_INPUT_TOKENS=73728 \
       MAX_COMPLETION_TOKENS=49152 MAX_DATUM_TOKENS=81920
export STEP_TIMEOUT=2400 SAMPLE_TIMEOUT=2460 ROLLOUT_TIMEOUT=3600

REPO=<this checkout> STAGE=0 DATA_DIR=$P ROOT=/data/user_data/$USER/<client dir> \
OVERLAY_DIR=/scratch/$USER-ovl bash deploy/flame/train_openhands.sbatch 2>&1 | tee client.log
```


| Setting | What it does | Why this value |
|---|---|---|
| `TOOLS=terminal,file_editor,finish` | the tools the agent is offered (train and eval) | drops `think`, whose takeover collapsed earlier runs, and the barely used `task_tracker` |
| `LOSS_NORM=trajectory` | average the loss over each trajectory's tokens, then over trajectories (`patches/trajectory_loss_norm.py`) | the default (all tokens of the step) lets long step-limit rollouts outweigh short solved ones |
| `ACTIVE_SAMPLING=1` | keep drawing groups until each microbatch has groups whose rewards differ (`patches/active_sampling.py`) | all-pass and all-fail groups carry no gradient; early on they were 40-49% of groups |
| `MALFORMED_CREDIT=off` | a malformed final response scores its task reward like any other ending | the default ("trajectory") scores a malformed unsolved ending the failure penalty over the whole trajectory; "targeted" puts it on the broken tokens instead, and reached \|advantage\| ~7,700 on a 3-token span, dominating clipped steps |
| `KL_COEF` | `KL_COEF` x KL(policy \|\| starting weights) per trained token, computed in the trainer (`--tinker-kl-coef`) | 0 here: skips the reference pass (~30% faster); must equal the server's |
| `TASK_SELECTION=adaptive` | draw tasks by learnability 4p(1-p) of each task's smoothed pass rate, with cooldowns after all-pass/all-fail groups and 5% uniform exploration (`terminal_rl/adaptive_tasks.py`) | spends rollouts on tasks that give a gradient; works with active sampling; its state is saved with every checkpoint and restored on resume |
| `ADAPTIVE_PRIOR`, `ADAPTIVE_PRIOR_WEIGHT=8` | seed the pass rates from a screening TSV (task, solved, rollouts), each task counted as at most 8 attempts | the sampler starts from measured rates instead of p = 0.5 everywhere, and online results soon dominate |
| `TERMINAL_RL_EXCLUDED_TASKS` | task ids left out of training and of the eval split | tasks that fail validation or that the run should not see; the eval's held-out tasks are picked by hash from the remaining ones, so changing the list changes the eval set: keep it fixed across runs you compare |
| `BATCH_SIZE=32 NUM_MICROBATCHES=8` | 32 tasks (x 8 rollouts) per optimizer step, sent as 8 forward_backward calls of 4 tasks as their groups finish | training overlaps rollout collection |
| `MAX_STALENESS=2` | rollouts may come from up to 2 policy versions back | keeps the engines busy while the trainer runs a step |
| `REASONING_EFFORT=medium` | the chat template's reasoning effort (Qwen3.8 defaults to xhigh) | shorter reasoning per turn, so long episodes fit the token budgets |
| `MAX_INPUT_TOKENS` + `MAX_COMPLETION_TOKENS` <= `CONTEXT_WINDOW` | the condenser compacts history past 73,728 prompt tokens; a completion may use 49,152 | the script refuses budgets that do not fit the window |
| `MAX_DATUM_TOKENS=81920` | leave a rollout whose training sequence is longer out of training (`patches/drop_long_datums.py`) | an OOM guard for the learner; at 81,920 it drops ~0.6% of rollouts (1.5% of tokens), at 65,536 ~9% |
| `STEP_TIMEOUT=2400 SAMPLE_TIMEOUT=2460` | limit per agent turn, and per sampling call inside it | a 27B with long completions behind a busy server needs far more than 300 s / 600 s |
| `CHECKPOINT_EVERY=1 EVAL_EVERY=3 EVAL_AT_START=0 EVAL_SIZE=50` | checkpoint every step, a 50-task eval every 3 | a 12 h server job fits few steps, so none is lost; the eval pauses training while it runs |

## Task pools

A pool is a directory the client reads through `DATA_DIR` (`tasks.py:_staged`):

```text
selected.json        one entry per task: task_id, image, sif_name (harbor: true for Harbor tasks)
tasks/<task_id>/     instruction.md, setup.sh or task.toml, tests/, solution/ (oracle checks only)
sifs-oh/<sif>        the task's Apptainer image with the OpenHands runtime added
```

Every task image needs the OpenHands agent server inside it (its terminal tool
runs in the task's container). The pool scripts add one prebuilt payload, the
`/openhands-venv` and its uv CPython.

The scripts that build and screen pools are kept with the experiment that
uses them, in neulab/science-agent (`experiments/qwen38_27b_tinker_amd/pool/`).
