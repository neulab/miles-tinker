#!/bin/bash
# OpenHands terminal RL on the tmax pool, on one FLAME node (8x H100).
#
# The FLAME port of deploy/babel/train_openhands.sbatch: same recipe and
# hyperparameters (see that file for why each setting is what it is), with
# everything the run reads staged onto node-local /tmp first by stage.sh.
#
#   sbatch deploy/flame/train_openhands.sbatch                  # fresh run
#   RESUME_RUN_ID=<run id> sbatch deploy/flame/train_openhands.sbatch
#   STEPS=2 WANDB_MODE=offline sbatch deploy/flame/train_openhands.sbatch   # smoke
#
# Batch size 64 by gradient accumulation: four forward_backward calls of 16 tasks
# each, submitted as their groups finish, then one optimizer step. 64 tasks is
# 4x the samples per step, so 50 steps match the default run's 200 in data, and
# checkpointing/eval every 3 steps keeps them a few hours apart:
#   BATCH_SIZE=64 NUM_MICROBATCHES=4 STEPS=50 CHECKPOINT_EVERY=3 EVAL_EVERY=3 \
#       RUN_NAME=oh-rl-flame-bs64 sbatch deploy/flame/train_openhands.sbatch
#
# FP32_LM_HEAD=1 computes the LM head in FP32 in both Megatron and SGLang
# (--fp32-lm-head) to shrink the training-inference logprob mismatch.
#
# WEIGHT_MOVEMENT=1 (default) logs, per optimizer step, the share of BF16 weights
# that changed by magnitude bucket and the relative update norm
# (--log-weight-movement; optim/weights_changed_frac*, optim/weight_update_rel_norm).
#
# STOCHASTIC_ROUNDING=1 copies the FP32 master weights into the BF16 weights with
# stochastic rounding (--stochastic-rounding), so they change a little every step
# instead of in bursts at updates 4, 8, 16 (the KL spikes at steps 5, 9, 17).
# STOCHASTIC_ROUNDING=1 draws fresh noise every step, so weights between two BF16
# values jitter every step (KL 0.0011 at step 2); STOCHASTIC_ROUNDING=fixed gives
# each weight a fixed threshold, so it changes only when its master crosses it
# (--stochastic-rounding-noise fixed).
#
# ACTIVE_SAMPLING=1 fills each microbatch with groups whose rewards differ
# instead of dropping all-equal groups (deploy/babel/patches/active_sampling.py).
#
# LOSS_NORM=trajectory averages the loss over each trajectory's tokens, then
# over trajectories, instead of over all tokens in the step, so long step-limit
# rollouts no longer outweigh short solved ones
# (deploy/babel/patches/trajectory_loss_norm.py). Default: token.
#
# TOOLS lists the tools the agent is offered in training and in-training eval
# rollouts alike, comma-separated, from terminal, file_editor, task_tracker,
# finish, think (TERMINAL_RL_TOOLS; see openhands_rollout.py). Default: all five,
# the evaluation's set. TOOLS=terminal,file_editor,finish drops think, whose
# takeover collapsed jobs 155637/155644, and the barely used task_tracker.
#
# Reward: REWARD_SCHEME=penalized (default) scores a raw tool call in the final
# message, or a run into the step limit that did not solve the task, -1.0, and
# otherwise the verifier's reward (a solved run keeps its 1.0 even if it ran out
# or its final message is malformed);
# REWARD_SCHEME=gated restores 0 / 0.2 + 0.8 * task. REWARD_MODE=passrate grades
# the task by the fraction of its unit tests that pass instead of the verifier's 0/1.
#
# STEP_LIMIT_PENALTY sets what an unsolved run into the step limit scores (default:
# the failure penalty, -1); STEP_LIMIT_PENALTY=0 scores it like a wrong finish.
# STEP_LIMIT_PENALTY=drop leaves those rollouts out of training (no gradient; a
# group left with < 2 rollouts is skipped, so pair it with ACTIVE_SAMPLING=1).
#
# MAX_TOOL_CALLS=K caps the tool calls in one assistant turn at K (train and
# eval): sampling stops at </tool_call>, and a turn with fewer than K calls is
# sampled on from there. SINGLE_TOOL_CALL=1 is MAX_TOOL_CALLS=1. Without a cap
# openhands runs every call in a response for the price of one step, and in
# 156019 the policy learned to pack dozens per turn until the 16k token cap
# (terminal_rl/single_tool_call.py).
#
# LENGTH_PENALTY_N=N (0 = off, default) penalises long completions instead of
# capping tool calls: a trajectory whose longest completion has n > N tokens
# (reasoning included) loses LENGTH_PENALTY_ALPHA (default 0.3) * ln(n / N) from
# its score, floored at -1, so a solved run scores 1 - alpha * ln(n / N)
# (terminal_rl/openhands_env.py). N=4096 leaves >99% of completions alone.
# LENGTH_PENALTY_CREDIT=targeted puts it on each long response's overflow tokens
# (those after the N-th) instead of the trajectory reward, so the turn that ran
# long -- or stacked calls -- carries it (patches/targeted_length_credit.py);
# "trajectory" (default) subtracts it from the trajectory reward.
#
# MAX_DATUM_TOKENS (default 45056 = 44k; 0 = off) leaves a rollout whose training
# sequence is longer out of the training group, so one long trajectory cannot
# CUDA-OOM the learner (155785 at ~52k, 156491 at step 58;
# patches/drop_long_datums.py). Affects < 1% of rollouts (p99 is 40-44k).
#
# SANDBOX_MEM_LIMIT_GB (default 32) / SANDBOX_PIDS_LIMIT (default 2000; 0 = off)
# kill a task sandbox whose processes exceed that resident memory or process
# count (a watchdog in terminal_rl/backend.py; cgroup limits are unavailable
# under --fakeroot). The rollout is dropped as infrastructure-killed. Sandbox
# processes also run at oom_score_adj 1000 (SANDBOX_OOM_SCORE_ADJ) so the kernel
# does not prefer the learner. Healthy sandboxes on 157001: peak RSS p99 ~1.5 GB,
# <= ~20 processes; 2000 leaves room for make -j$(nproc) on a 208-core node.
# 156937's node ran out of RAM with a load of ~2.6 million.
#
# DIST_TIMEOUT_MIN (default 60) is the learner's NCCL collective timeout in
# minutes (miles --distributed-timeout-minutes, whose own default is 10). A run's
# first forward_backward units tune Triton kernels for every new sequence shape
# and take ~460 s (156879, 156937); 156938's went past 600 s and the NCCL watchdog
# killed the learner.
#
# MALFORMED_CREDIT=targeted scores a malformed final response's trajectory on the
# task and puts a fixed -MALFORMED_PENALTY (default 1.0) on the tokens where the
# response broke (terminal_rl/malformed_break.py,
# patches/targeted_malformed_credit.py); "trajectory" (default) keeps the old
# whole-trajectory FAILURE_PENALTY; "off" scores a malformed
# ending its task reward like any other.
#
# KL_COEF (default 0.01; 0 = off) adds KL_COEF * KL(policy || starting weights)
# per trained token to the loss, averaged like the policy-gradient term;
# KL_TYPE picks the estimator (default k3). The trainer logs "[kl-ref]" lines
# with KL(sampler || reference) per unit in gateway.log.
#
# ANNOUNCE_BUDGET=1 (default) ends the task message with the agent's turn budget
# (max_steps minus the 6-step headroom: 94); 0 omits it.
#
# The in-training eval runs with the training rollout settings (steps, tokens,
# timeouts, temperature, tools, prompt). EVAL_SIZE (default 100) held-out tasks,
# chosen by EVAL_SPLIT=hashed (a fixed pseudo-random sample, excluded tasks
# removed; "tail" restores the old last-N split), each run EVAL_GROUP_SIZE times
# (default 4), plus once at step 0 (EVAL_AT_START=1, default). The eval pauses
# training while it runs (~1 h for 100 x 4).
#
# Remote trainer: TINKER_BASE_URL=<url> TINKER_API_KEY=<key> runs only the Platoon
# client and its sandboxes here, against a Tinker server started elsewhere (the
# AMD cluster: deploy/amd/serve_openhands.sbatch prints both values in its log).
# The node then needs no GPUs:
#   TINKER_BASE_URL=https://... TINKER_API_KEY=... sbatch --gres=none deploy/flame/train_openhands.sbatch
# Trainer-side settings (FP32_LM_HEAD, WEIGHT_MOVEMENT, KL_COEF, KL_TYPE,
# STOCHASTIC_ROUNDING, DIST_TIMEOUT_MIN) then belong to the server job and are
# ignored here, except that KL_COEF must match it: the client sends the loss
# weights the KL term needs only when KL_COEF is non-zero. MODEL_NAME (default
# Qwen/Qwen3.5-4B) must match the server's MODEL_ID, e.g. Qwen/Qwen3.8-27B.
#
# Inside an allocation you already hold, run it with bash instead of sbatch:
#   srun --jobid=<jobid> --overlap --pty bash deploy/flame/train_openhands.sbatch
#
# /tmp is private to each Slurm job (job_container/tmpfs) and deleted when the job
# ends, checkpoints included. PERSIST=1 (off by default) mirrors the run to
# $GCS/runs/<run id>/ every 10 minutes (deploy/flame/persist_run.sh: the newest
# two checkpoints, checkpoints.jsonl, rollouts, logs), and RESUME_RUN_ID pulls
# the run back from there when it is not on the node.
#SBATCH --partition=flame
#SBATCH --qos=flame-16gpu_qos
#SBATCH --account=gneubig

#SBATCH --job-name=oh-rl
#SBATCH --time=07-00:00:00
#SBATCH --signal=SIGTERM@128
#SBATCH --output=%x-%j.log

#SBATCH --ntasks-per-node=1
#SBATCH --mem=1860368M
#SBATCH --cpus-per-task=208
#SBATCH --gres=gpu:8

set -euo pipefail

REPO="${REPO:-$HOME/flame-openhands/repo}"
ROOT="${ROOT:-/tmp/$USER-oh-rl}"
# WORK_DIR/DATA_DIR override the layout under ROOT on another cluster (not WORK,
# which some clusters set for every job).
WORK="${WORK_DIR:-$ROOT/work}"
DATA="${DATA_DIR:-$ROOT/tmax15k}"
IMAGE=$ROOT/images/miles-sglang.sif
JOB="${SLURM_JOB_ID:-local-$$}"

# Apptainer is a user-space install on FLAME; rollouts find it on PATH too.
export PATH="$HOME/flame-openhands/tools/apptainer/bin:$HOME/.local/bin:$PATH"

echo "node=$(hostname)  job=$JOB  start=$(date -Is)"
REMOTE_ARGS=()
if [[ -n "${TINKER_BASE_URL:-}" ]]; then
    : "${TINKER_API_KEY:?set TINKER_API_KEY from the server log too}"
    export TINKER_API_KEY
    REMOTE_ARGS=(--tinker-url "$TINKER_BASE_URL")
    echo "Client only, against the Tinker server at $TINKER_BASE_URL"
else
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
fi

WANDB_MODE="${WANDB_MODE:-online}"
if [[ "$WANDB_MODE" == online && -z "${WANDB_API_KEY:-}" && ! -f "${NETRC:-$HOME/.netrc}" ]]; then
    echo "W&B online needs WANDB_API_KEY or ~/.netrc; export one or set WANDB_MODE=offline" >&2
    exit 1
fi

# EVAL_SET=tb2 evaluates on Terminal-Bench 2.0 instead of the 100 hashed held-out
# tmax tasks: stage.sh merges the tb2-v1 pool (<science-agent>/experiments/qwen38_27b_tinker_amd/pool/build_tb2_pool.py,
# <science-agent>/experiments/qwen38_27b_tinker_amd/pool/build_tb2_sifs.sh) and the eval runs the tasks in
# terminal_rl/tb2_eval_tasks.txt (those whose reference solution passes in our
# sandbox). TB2 tasks are eval-only; training then uses every kept tmax task.
# EVAL_SET=tb2-32 runs 32 of them (terminal_rl/tb2_eval_tasks_32.txt: the same
# easy/medium/hard mix, biased to tasks with short rollouts and to tasks a policy
# has solved), about 30% of the full eval's rollout time.
if [[ "${EVAL_SET:-}" == tb2* ]]; then
    export EXTRA_POOLS="${EXTRA_POOLS:-} tb2-v1"
fi
# STAGE=0 uses DATA_DIR as it is (a pool already on shared storage); it only
# checks that every task has its image. A client-only run stages no server image
# or model: the client runs on the host and fetches the tokenizer from the Hub.
if [[ -n "${TINKER_BASE_URL:-}" ]]; then export STAGE_SERVER=0; fi
if [[ "${STAGE:-1}" == 1 ]]; then
    bash "$REPO/deploy/flame/stage.sh"
else
    python3 - "$DATA" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
entries = json.loads((root / "selected.json").read_text())
missing = sorted({e["sif_name"] for e in entries if not (root / "sifs-oh" / e["sif_name"]).is_file()})
no_task = [e["task_id"] for e in entries if not (root / "tasks" / e["task_id"]).is_dir()]
print(f"pool {root}: {len(entries)} tasks, {len(missing)} missing images, {len(no_task)} missing task dirs")
if missing or no_task:
    sys.exit(f"incomplete pool, e.g. {(missing or no_task)[0]}")
PY
fi

# Leftover rollout sandboxes outlive a killed client; stop them on exit.
cleanup() {
    local instance
    jobs -p | xargs -r kill 2>/dev/null || true
    # Only orphans: a sandbox is named litecoder-<client pid>-<id>, and another
    # job's client on this node (or a dry run next to a live run) still holds its pid.
    for instance in $(apptainer instance list 2>/dev/null | awk 'NR > 1 && $1 ~ /^(litecoder|terminal)-/ {print $1}'); do
        owner=$(cut -d- -f2 <<< "$instance")
        if [[ "$owner" =~ ^[0-9]+$ ]] && kill -0 "$owner" 2>/dev/null; then continue; fi
        apptainer instance stop "$instance" >/dev/null 2>&1 || true
    done
    # Last mirror of the run (logs, index; a new checkpoint only if it fits the time left).
    if [[ "${PERSIST:-0}" == 1 && -n "${RUN_OUTPUT:-}" ]]; then
        local run
        run=$(ls -td "$RUN_OUTPUT"/*/ 2>/dev/null | head -1)
        [[ -n "$run" ]] && PERSIST_ONCE=1 timeout 110 bash "$REPO/deploy/flame/persist_run.sh" "$run" || true
    fi
}
trap cleanup EXIT

export TERMINAL_RL_RETAIN_REASONING_MODEL=openai/gpt-4o
export TERMINAL_RL_FINISH_REWARD_WEIGHT=0.1
export TERMINAL_RL_TOOL_JSON_REWARD_WEIGHT=0.1
export TERMINAL_RL_DUMP_PROMPTS=2
# MAX_INPUT_TOKENS: the condenser's budget, i.e. the longest prompt before history is
# compacted. MAX_INPUT_TOKENS + MAX_COMPLETION_TOKENS must fit CONTEXT_WINDOW.
export TERMINAL_RL_MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-30000}"
# REASONING_EFFORT (unset = the chat template's default; Qwen3.8: xhigh): medium or low.
[[ -n "${REASONING_EFFORT:-}" ]] && export TERMINAL_RL_REASONING_EFFORT="$REASONING_EFFORT"
echo "Reasoning effort: ${REASONING_EFFORT:-template default}; condenser budget ${TERMINAL_RL_MAX_INPUT_TOKENS} tokens"
export TERMINAL_RL_CLOSE_SCORE_TIMEOUT=600
export TERMINAL_RL_EVAL_SIZE="${EVAL_SIZE:-${TERMINAL_RL_EVAL_SIZE:-100}}"
export TERMINAL_RL_EVAL_SPLIT="${EVAL_SPLIT:-hashed}"
# TASK_ORDER=listed trains on the task list in its own order (no shuffle), for a
# curriculum such as medium_pool_curriculum.txt (easy -> hard); default shuffles.
# TASK_ORDER=reshuffle draws a new random order of the whole list on every pass
# (the default shuffles once and replays it, dropping the same partial batch).
if [[ "${TASK_ORDER:-shuffled}" == listed ]]; then
    export PLATOON_TRAIN_SHUFFLE=0
    echo "Training tasks in list order (curriculum)"
elif [[ "${TASK_ORDER:-shuffled}" == reshuffle ]]; then
    export PLATOON_TRAIN_SHUFFLE=epoch
    echo "Training tasks reshuffled every pass"
fi
# TASK_LIST=<file of task ids>: train and eval on exactly these tasks (the
# fixed-task learning test; e.g. deploy/babel/terminal_rl/platoon/terminal_rl/overfit_tasks_32.txt).
# Pair it with NUM_EPOCHS large enough to repeat them for the whole run.
# TASK_SELECTION=adaptive draws training tasks by learnability, 4p(1-p) of each
# task's smoothed pass rate, with cooldowns after all-pass / all-fail groups and
# 5% uniform exploration (terminal_rl/adaptive_tasks.py; ported from science-agent).
# Works with ACTIVE_SAMPLING=1. ADAPTIVE_PRIOR=<tsv task/solved/rollouts> seeds the
# pass rates (each task capped at ADAPTIVE_PRIOR_WEIGHT=4 attempts). Overrides TASK_ORDER.
if [[ "${TASK_SELECTION:-}" == adaptive ]]; then
    export PLATOON_TASK_SELECTION=adaptive
    [[ -n "${ADAPTIVE_PRIOR:-}" ]] && export PLATOON_ADAPTIVE_PRIOR="$(realpath "$ADAPTIVE_PRIOR")"
    [[ -n "${ADAPTIVE_PRIOR_WEIGHT:-}" ]] && export PLATOON_ADAPTIVE_PRIOR_WEIGHT="$ADAPTIVE_PRIOR_WEIGHT"
    echo "Training tasks drawn adaptively by learnability${PLATOON_ADAPTIVE_PRIOR:+ (prior: $PLATOON_ADAPTIVE_PRIOR)}"
fi
if [[ -n "${TASK_LIST:-}" ]]; then
    export TERMINAL_RL_TASK_LIST="$(realpath "$TASK_LIST")"
    if [[ -n "${EVAL_TASK_LIST:-}" ]]; then
        export TERMINAL_RL_EVAL_TASK_LIST="$(realpath "$EVAL_TASK_LIST")"
        echo "Fixed task list: $TERMINAL_RL_TASK_LIST ($(grep -c '^task_' "$TERMINAL_RL_TASK_LIST") train tasks); eval list: $TERMINAL_RL_EVAL_TASK_LIST ($(grep -c '^task_' "$TERMINAL_RL_EVAL_TASK_LIST") tasks)"
    else
        echo "Fixed task list: $TERMINAL_RL_TASK_LIST ($(grep -c '^task_' "$TERMINAL_RL_TASK_LIST") tasks, train = eval)"
    fi
fi
export PLATOON_FORWARD_BACKWARD_TIMEOUT=7200
export PLATOON_LOGPROB_DUMP_DIR=$WORK/logprob_gap
export PLATOON_LOGPROB_DUMP_TOPK=64
export PLATOON_LOGPROB_DUMP_GRADK=64
export PLATOON_LOGPROB_DUMP_SAMPLE=64

# Sandbox writable layers on node-local NVMe, never tmpfs (Babel used /scratch;
# FLAME has no /scratch, and /tmp is the NVMe array). OVERLAY_DIR overrides.
export TERMINAL_RL_OVERLAY_DIR="${OVERLAY_DIR:-$ROOT/ovl}"
mkdir -p "$TERMINAL_RL_OVERLAY_DIR"
rm -rf "${TERMINAL_RL_OVERLAY_DIR:?}"/* 2>/dev/null || true

RUN_OUTPUT=$WORK/output/tmax-15k-openhands
# GCS: a gs:// prefix you can write to; needed only for PERSIST=1 and to pull a run
# that is not on the node.
GCS="${GCS:-}"
export GCS
RESUME_ARGS=()
if [[ -n "${RESUME_RUN_ID:-}" ]]; then
    RESUME_ARGS=(--resume-run-id "$RESUME_RUN_ID")
    echo "resuming from run $RESUME_RUN_ID"
    # TRAIN_BATCH_START=N: resume the task loader at dataloader batch N. Active
    # sampling consumes more than one batch per step, so the default (the step
    # number) re-serves tasks; work/batch_progress.py finds N from the rollouts.
    if [[ -n "${TRAIN_BATCH_START:-}" ]]; then
        export PLATOON_TRAIN_BATCH_START="$TRAIN_BATCH_START"
        echo "task loader resumes at batch $PLATOON_TRAIN_BATCH_START"
    fi
    if [[ ! -d "$RUN_OUTPUT/$RESUME_RUN_ID" ]]; then
        : "${GCS:?run $RESUME_RUN_ID is not under $RUN_OUTPUT: set GCS to pull it}"
        echo "pulling run $RESUME_RUN_ID from $GCS/runs/"
        mkdir -p "$RUN_OUTPUT/$RESUME_RUN_ID"
        gcloud storage rsync -r "$GCS/runs/$RESUME_RUN_ID" "$RUN_OUTPUT/$RESUME_RUN_ID"
        rm -f "$RUN_OUTPUT/$RESUME_RUN_ID/.persisted"
    fi
fi
if [[ "${PERSIST:-0}" == 1 ]]; then
    : "${GCS:?PERSIST=1 needs GCS}"
    # The run id is made by run.py; mirror whichever run directory appears.
    (
        until run=$(ls -td "$RUN_OUTPUT"/*/ 2>/dev/null | head -1) && [[ -n "$run" ]]; do sleep 30; done
        exec nice bash "$REPO/deploy/flame/persist_run.sh" "$run" 600
    ) &
    echo "Persisting the run to $GCS/runs/ every 10 minutes"
fi

export TERMINAL_RL_REWARD_SCHEME="${REWARD_SCHEME:-penalized}"
export TERMINAL_RL_REWARD_MODE="${REWARD_MODE:-binary}"
echo "Reward: scheme=$TERMINAL_RL_REWARD_SCHEME task=$TERMINAL_RL_REWARD_MODE"

export PLATOON_ACTIVE_SAMPLING="${ACTIVE_SAMPLING:-0}"
if [[ "$PLATOON_ACTIVE_SAMPLING" == 1 ]]; then
    echo "Active sampling: microbatches wait for groups with a gradient"
fi

export PLATOON_LOSS_NORM="${LOSS_NORM:-token}"
echo "Loss normalization: $PLATOON_LOSS_NORM"

export TERMINAL_RL_TOOLS="${TOOLS:-terminal,file_editor,task_tracker,finish,think}"
echo "Agent tools (train and eval): $TERMINAL_RL_TOOLS"
# One JSON line per sampled completion: stop reason, last token, open tool call
# (proxy_bridge.py). Tells a call cut off by an end-of-turn token from the cap.
export TERMINAL_RL_STOP_LOG="$RUN_OUTPUT/stop_reasons-${SLURM_JOB_ID:-local}.jsonl"
mkdir -p "$RUN_OUTPUT"
echo "Stop-reason log: $TERMINAL_RL_STOP_LOG"
if [[ -n "${STEP_LIMIT_PENALTY:-}" ]]; then
    export TERMINAL_RL_STEP_LIMIT_PENALTY="$STEP_LIMIT_PENALTY"
    echo "Step limit (unsolved) scores: $TERMINAL_RL_STEP_LIMIT_PENALTY"
fi
export TERMINAL_RL_MAX_TOOL_CALLS="${MAX_TOOL_CALLS:-0}"
if [[ "${SINGLE_TOOL_CALL:-0}" == 1 ]]; then export TERMINAL_RL_MAX_TOOL_CALLS=1; fi
if (( TERMINAL_RL_MAX_TOOL_CALLS > 0 )); then
    echo "Tool calls per turn: at most $TERMINAL_RL_MAX_TOOL_CALLS (sampling stops at </tool_call>)"
fi
export TERMINAL_RL_SANDBOX_MEM_LIMIT_GB="${SANDBOX_MEM_LIMIT_GB:-32}"
export TERMINAL_RL_SANDBOX_PIDS_LIMIT="${SANDBOX_PIDS_LIMIT:-2000}"
export TERMINAL_RL_SANDBOX_OOM_SCORE_ADJ="${SANDBOX_OOM_SCORE_ADJ:-1000}"
echo "Sandbox guard: kill at ${TERMINAL_RL_SANDBOX_MEM_LIMIT_GB} GB resident or ${TERMINAL_RL_SANDBOX_PIDS_LIMIT} processes; oom_score_adj ${TERMINAL_RL_SANDBOX_OOM_SCORE_ADJ}"
export TERMINAL_RL_MAX_DATUM_TOKENS="${MAX_DATUM_TOKENS:-45056}"
if (( TERMINAL_RL_MAX_DATUM_TOKENS > 0 )); then
    echo "Training sequences over $TERMINAL_RL_MAX_DATUM_TOKENS tokens: rollout left out of the group (OOM guard)"
fi
export TERMINAL_RL_LENGTH_PENALTY_N="${LENGTH_PENALTY_N:-0}"
export TERMINAL_RL_LENGTH_PENALTY_ALPHA="${LENGTH_PENALTY_ALPHA:-0.3}"
export TERMINAL_RL_LENGTH_PENALTY_CREDIT="${LENGTH_PENALTY_CREDIT:-trajectory}"
if (( TERMINAL_RL_LENGTH_PENALTY_N > 0 )); then
    if [[ "$TERMINAL_RL_LENGTH_PENALTY_CREDIT" == targeted ]]; then
        echo "Output-length penalty (targeted): each response with n > $TERMINAL_RL_LENGTH_PENALTY_N tokens gets $TERMINAL_RL_LENGTH_PENALTY_ALPHA * ln(n/$TERMINAL_RL_LENGTH_PENALTY_N) on its overflow tokens"
    else
        echo "Output-length penalty: longest completion n > $TERMINAL_RL_LENGTH_PENALTY_N tokens costs $TERMINAL_RL_LENGTH_PENALTY_ALPHA * ln(n/$TERMINAL_RL_LENGTH_PENALTY_N), floored at -1"
    fi
fi
export TERMINAL_RL_MALFORMED_CREDIT="${MALFORMED_CREDIT:-trajectory}"
export TERMINAL_RL_MALFORMED_PENALTY="${MALFORMED_PENALTY:-1.0}"
echo "Malformed credit: $TERMINAL_RL_MALFORMED_CREDIT (penalty $TERMINAL_RL_MALFORMED_PENALTY)"
export TERMINAL_RL_ANNOUNCE_BUDGET="${ANNOUNCE_BUDGET:-1}"
echo "Turn budget announced to the agent: $TERMINAL_RL_ANNOUNCE_BUDGET"

# Rollout settings shared by training and the in-training eval, so the eval
# measures the policy under exactly the conditions it is trained in; only the
# tasks differ. The eval used to run 120 steps, 8192 completion tokens and a
# 2400 s rollout timeout, against training's 100 / 16384 / 3600.
GROUP_SIZE="${GROUP_SIZE:-8}"
# LR: the Adam learning rate (default 1e-6). LR=1e-12 freezes the policy in
# effect, so every rollout samples the starting weights: the difficulty-scoring
# pass (analysis/build_medium_pool.py) uses it that way.
LR="${LR:-1e-6}"
# Rollout groups run at once (train and eval). Concurrent rollouts = workers x
# group size; 16 x 8 = 128 is what the 4 SGLang engines have been serving.
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-16}"
echo "Learning rate: $LR"
MAX_STEPS=100
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-16384}"
# CONTEXT_WINDOW: prompt + completion limit the proxy enforces; the server must be started
# with at least this (serve_openhands.sbatch CONTEXT_WINDOW).
CONTEXT_WINDOW="${CONTEXT_WINDOW:-98304}"
if (( TERMINAL_RL_MAX_INPUT_TOKENS + MAX_COMPLETION_TOKENS > CONTEXT_WINDOW )); then
    echo "MAX_INPUT_TOKENS ($TERMINAL_RL_MAX_INPUT_TOKENS) + MAX_COMPLETION_TOKENS ($MAX_COMPLETION_TOKENS) exceeds CONTEXT_WINDOW ($CONTEXT_WINDOW)" >&2
    exit 1
fi
MAX_STALENESS="${MAX_STALENESS:-1}"
# ROLLOUT_TIMEOUT bounds a whole episode. A group waits for its slowest rollout, so this
# sets the tail of batch collection: on the 27B science run, 1800 s cut 5% of rollouts
# and shortened the mean group by ~19% against 3600 s.
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-3600}"
# STEP_TIMEOUT bounds each agent turn; SAMPLE_TIMEOUT bounds the sampling call
# inside it (deploy/babel/patches/sample_timeout.py), so raise them together.
# A larger policy behind a busy server (27B, ~190 concurrent rollouts) saw
# turns queue past 300s and calls past 600s.
STEP_TIMEOUT="${STEP_TIMEOUT:-300}"
export PLATOON_SAMPLE_TIMEOUT="${SAMPLE_TIMEOUT:-600}"
if (( STEP_TIMEOUT > PLATOON_SAMPLE_TIMEOUT )); then
    echo "STEP_TIMEOUT=${STEP_TIMEOUT}s exceeds SAMPLE_TIMEOUT=${PLATOON_SAMPLE_TIMEOUT}s: a slow call fails at the sample timeout first" >&2
fi
# Four rollouts per eval task, not training's eight: the eval's noise is set by
# the number of tasks, so doubling the group would double its cost for little.
EVAL_GROUP_SIZE="${EVAL_GROUP_SIZE:-4}"
# One eval before any training, so every run has its own step-0 baseline on
# its eval split (deploy/babel/patches/eval_at_step_zero.py).
export PLATOON_EVAL_AT_START="${EVAL_AT_START:-1}"
echo "Rollouts (train and eval): steps=$MAX_STEPS max_completion_tokens=$MAX_COMPLETION_TOKENS timeout=${ROLLOUT_TIMEOUT}s step_timeout=${STEP_TIMEOUT}s sample_timeout=${PLATOON_SAMPLE_TIMEOUT}s temperature=1.0"
if [[ "${EVAL_SET:-}" == tb2* ]]; then
    case "$EVAL_SET" in
        tb2) eval_list=tb2_eval_tasks.txt ;;
        tb2-32) eval_list=tb2_eval_tasks_32.txt ;;
        *) echo "unknown EVAL_SET=$EVAL_SET (tb2, tb2-32)" >&2; exit 1 ;;
    esac
    export TERMINAL_RL_EVAL_TASK_LIST="$REPO/deploy/babel/terminal_rl/platoon/terminal_rl/$eval_list"
    echo "Eval: Terminal-Bench 2.0, $(grep -c '^tb2_' "$TERMINAL_RL_EVAL_TASK_LIST") tasks x $EVAL_GROUP_SIZE rollouts, every ${EVAL_EVERY:-10} steps, at step 0: $PLATOON_EVAL_AT_START (test timeout capped at ${TERMINAL_RL_TEST_TIMEOUT_MAX:-1800}s)"
else
echo "Eval: $TERMINAL_RL_EVAL_SIZE held-out tasks ($TERMINAL_RL_EVAL_SPLIT split) x $EVAL_GROUP_SIZE rollouts, every ${EVAL_EVERY:-10} steps, at step 0: $PLATOON_EVAL_AT_START"
fi

EXTRA_ARGS=()
if [[ "${FP32_LM_HEAD:-0}" == 1 ]]; then
    EXTRA_ARGS+=(--extra-arg=--fp32-lm-head)
    echo "LM head in FP32 (trainer and rollout engines)"
fi
EXTRA_ARGS+=(--extra-arg=--distributed-timeout-minutes="${DIST_TIMEOUT_MIN:-60}")
echo "Learner NCCL timeout: ${DIST_TIMEOUT_MIN:-60} min"
if [[ "${WEIGHT_MOVEMENT:-1}" == 1 ]]; then
    EXTRA_ARGS+=(--extra-arg=--log-weight-movement)
    echo "Logging per-step BF16 weight movement"
fi
# KL penalty to the starting weights, in the trainer (miles --tinker-kl-coef; the
# reference is a pinned-CPU copy of the starting weights swapped in for one
# forward pass per unit, so no extra clients or sessions). The client sends its
# per-token loss normalization as `weights` so the penalty averages like the
# policy-gradient term (deploy/babel/patches/send_loss_weights.py).
KL_COEF="${KL_COEF:-0.01}"
if [[ "$KL_COEF" != 0 && "$KL_COEF" != 0.0 ]]; then
    EXTRA_ARGS+=(--extra-arg=--tinker-kl-coef="$KL_COEF" --extra-arg=--tinker-kl-type="${KL_TYPE:-k3}")
    export PLATOON_SEND_LOSS_WEIGHTS=1
    echo "KL penalty to the starting weights: coef=$KL_COEF type=${KL_TYPE:-k3}"
else
    echo "KL penalty: off"
fi
if [[ "${STOCHASTIC_ROUNDING:-0}" == 1 ]]; then
    EXTRA_ARGS+=(--extra-arg=--stochastic-rounding)
    echo "Stochastic rounding of BF16 weights (fresh noise every step)"
elif [[ "${STOCHASTIC_ROUNDING:-0}" == fixed ]]; then
    EXTRA_ARGS+=(--extra-arg=--stochastic-rounding --extra-arg=--stochastic-rounding-noise=fixed)
    echo "Stochastic rounding of BF16 weights (fixed threshold per weight)"
fi

# Default run name: every setting that distinguishes these runs, so W&B and the
# job log say what a run was without opening its sbatch line. RUN_NAME overrides.
tools_tag=$(echo "$TERMINAL_RL_TOOLS" | sed -e 's/terminal/term/; s/file_editor/fe/; s/task_tracker/tt/; s/finish/fin/; s/think/think/' -e 's/,/-/g')
sr_tag=$(case "${STOCHASTIC_ROUNDING:-0}" in 1) echo srfresh;; fixed) echo srfixed;; *) echo nosr;; esac)
# (if-statements, not `[[ ]] && echo`: a false test there fails the script under set -e)
if [[ "$KL_COEF" == 0 || "$KL_COEF" == 0.0 ]]; then kl_tag=nokl; else kl_tag="kl${KL_COEF}${KL_TYPE:+-$KL_TYPE}"; fi
DEFAULT_RUN_NAME="oh-rl-${TERMINAL_RL_REWARD_MODE}-${TERMINAL_RL_REWARD_SCHEME}-mc${TERMINAL_RL_MALFORMED_CREDIT}-${kl_tag}-${sr_tag}-tools${tools_tag}-${PLATOON_LOSS_NORM}norm"
if [[ "$PLATOON_ACTIVE_SAMPLING" == 1 ]]; then DEFAULT_RUN_NAME+="-as"; fi
if [[ "${FP32_LM_HEAD:-0}" == 1 ]]; then DEFAULT_RUN_NAME+="-fp32head"; fi
if [[ "$TERMINAL_RL_ANNOUNCE_BUDGET" == 1 ]]; then DEFAULT_RUN_NAME+="-budget"; fi
if [[ "${TERMINAL_RL_STRIP_ANSI:-1}" == 0 ]]; then DEFAULT_RUN_NAME+="-ansi"; fi
if [[ -n "${TERMINAL_RL_STEP_LIMIT_PENALTY:-}" ]]; then DEFAULT_RUN_NAME+="-sl${TERMINAL_RL_STEP_LIMIT_PENALTY}"; fi
if (( TERMINAL_RL_MAX_TOOL_CALLS == 1 )); then DEFAULT_RUN_NAME+="-1call"; elif (( TERMINAL_RL_MAX_TOOL_CALLS > 1 )); then DEFAULT_RUN_NAME+="-calls${TERMINAL_RL_MAX_TOOL_CALLS}"; fi
if [[ "$TERMINAL_RL_MAX_DATUM_TOKENS" != 45056 ]]; then DEFAULT_RUN_NAME+="-maxdatum${TERMINAL_RL_MAX_DATUM_TOKENS}"; fi
if (( TERMINAL_RL_LENGTH_PENALTY_N > 0 )); then DEFAULT_RUN_NAME+="-lenN${TERMINAL_RL_LENGTH_PENALTY_N}a${TERMINAL_RL_LENGTH_PENALTY_ALPHA}"; fi
if (( TERMINAL_RL_LENGTH_PENALTY_N > 0 )) && [[ "$TERMINAL_RL_LENGTH_PENALTY_CREDIT" == targeted ]]; then DEFAULT_RUN_NAME+="tgt"; fi
if [[ "$LR" != "1e-6" ]]; then DEFAULT_RUN_NAME+="-lr${LR}"; fi
if [[ "$GROUP_SIZE" != 8 ]]; then DEFAULT_RUN_NAME+="-g${GROUP_SIZE}"; fi
if [[ "${PLATOON_TRAIN_SHUFFLE:-1}" == 0 ]]; then DEFAULT_RUN_NAME+="-curriculum"; fi
if [[ "${PLATOON_TRAIN_SHUFFLE:-1}" == epoch ]]; then DEFAULT_RUN_NAME+="-reshuffle"; fi
if [[ "${PLATOON_TASK_SELECTION:-}" == adaptive ]]; then DEFAULT_RUN_NAME+="-adaptive"; fi
if [[ "${EVAL_SET:-}" == tb2* ]]; then DEFAULT_RUN_NAME+="-eval${EVAL_SET/-/_}"; fi
DEFAULT_RUN_NAME+="-bs${BATCH_SIZE:-16}"
if [[ -n "${TERMINAL_RL_TASK_LIST:-}" ]]; then DEFAULT_RUN_NAME+="-fixed$(grep -c '^task_' "$TERMINAL_RL_TASK_LIST")tasks"; fi
RUN_NAME="${RUN_NAME:-$DEFAULT_RUN_NAME}"
echo "Run name: $RUN_NAME-$JOB"

cd "$REPO"
if [[ -z "${TINKER_BASE_URL:-}" ]]; then
    bash deploy/babel/gpu_memory_log.sh "$WORK/gpu_memory-$JOB.csv" 20 &
fi
bash deploy/babel/cgroup_memory_log.sh "$WORK/cgroup_memory-$JOB.csv" 30 &

python3 deploy/babel/run.py \
    --task tmax-15k-openhands \
    --learning-rate "$LR" \
    "${RESUME_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    "${REMOTE_ARGS[@]}" \
    --image "$IMAGE" \
    --data-dir "$DATA" \
    --work-dir "$WORK" \
    --model-name "${MODEL_NAME:-Qwen/Qwen3.5-4B}" \
    --renderer qwen3_5 \
    --steps "${STEPS:-200}" \
    --wandb-mode "$WANDB_MODE" \
    --context-length "$CONTEXT_WINDOW" \
    --tp 4 \
    --recompute \
    --learner-gpus 4 \
    --rollout-gpus 4 \
    --max-tokens-per-gpu 4096 \
    --extra-arg=--tinker-max-tokens-per-datum="$CONTEXT_WINDOW" \
    --extra-arg=--log-probs-chunk-size=1024 \
    --extra-arg=--rollout-max-context-len="$CONTEXT_WINDOW" \
    --extra-arg=--tinker-lease-timeout-s=172800 \
    --wandb-project tmax-15k-rl \
    --run-name "$RUN_NAME-$JOB" \
    --client-arg=--watchdog.timeout_seconds=10800 \
    --client-arg=--train.batch_size="${BATCH_SIZE:-16}" \
    --client-arg=--train.num_microbatches="${NUM_MICROBATCHES:-1}" \
    --client-arg=--train.num_epochs="${NUM_EPOCHS:-3}" \
    --client-arg=--train.workflow_config.group_size="$GROUP_SIZE" \
    --client-arg=--train.max_staleness="$MAX_STALENESS" \
    --client-arg=--train.optimizer.grad_clip_norm=1.0 \
    --client-arg=--train.optimizer.beta2=0.999 \
    --client-arg=--train.context_window_length="$CONTEXT_WINDOW" \
    --client-arg=--train.workflow_config.rollout_config.max_steps="$MAX_STEPS" \
    --client-arg=--train.workflow_config.rollout_config.timeout="$ROLLOUT_TIMEOUT" \
    --client-arg=--train.workflow_config.rollout_config.step_timeout="$STEP_TIMEOUT" \
    --client-arg=--train.workflow_config.rollout_config.inference_params.temperature=1.0 \
    --client-arg=--train.workflow_config.rollout_config.inference_params.max_completion_tokens="$MAX_COMPLETION_TOKENS" \
    --client-arg=--train.num_concurrent_rollout_workflow_workers="$ROLLOUT_WORKERS" \
    --client-arg=--eval.workflow_config.group_size="$EVAL_GROUP_SIZE" \
    --client-arg=--eval.num_concurrent_rollout_workflow_workers="$ROLLOUT_WORKERS" \
    --client-arg=--eval.workflow_config.rollout_config.max_steps="$MAX_STEPS" \
    --client-arg=--eval.workflow_config.rollout_config.timeout="$ROLLOUT_TIMEOUT" \
    --client-arg=--eval.workflow_config.rollout_config.step_timeout="$STEP_TIMEOUT" \
    --client-arg=--eval.workflow_config.rollout_config.inference_params.temperature=1.0 \
    --client-arg=--eval.workflow_config.rollout_config.inference_params.max_completion_tokens="$MAX_COMPLETION_TOKENS" \
    --client-arg=--eval.strategy=step \
    --client-arg=--eval.every="${EVAL_EVERY:-10}" \
    --client-arg=--checkpoint.every="${CHECKPOINT_EVERY:-10}" \
    "$@"
