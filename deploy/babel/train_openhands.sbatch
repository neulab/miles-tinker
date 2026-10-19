#!/bin/bash
# Terminal-RL training with the openhands agent, on the tmax-15k task pool.
#
# Same resources as ~/train.sh, but running the training instead of holding the
# node idle, and with the partition's maximum wall time (aasai MaxTime is
# 20-01:00:00; aasai_qos caps a user at 8 GPUs, not at a shorter wall clock).
#
#   sbatch train_openhands.sbatch                 # fresh run
#   RESUME_RUN_ID=260928-024322-325 sbatch ...    # continue from that run's
#                                                 # last checkpoint
#SBATCH --job-name=oh-rl
#SBATCH --partition=aasai
#SBATCH --qos=aasai_qos
#SBATCH --gres=gpu:8
#SBATCH --time=20-01:00:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=960G
#SBATCH --output=/data/user_data/%u/miles-babel/slurm-%j.out
#SBATCH --open-mode=append

set -euo pipefail

echo "node=$(hostname)  job=$SLURM_JOB_ID  start=$(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

REPO=${REPO:-$HOME/miles-tinker}
WORK=/data/user_data/$USER/miles-babel
# DATA_DIR selects another staged pool, e.g. one written by prepare_harbor.py.
DATA=${DATA_DIR:-/data/user_data/$USER/tmax15k}
IMAGE=$WORK/apptainer-cache/cache/oci-tmp/3828109b1b56225bfba11f4564640b014ec652f240f62a4f572e2df5aeff2dd3

# Sandbox writable layers go to node-local NVMe, never tmpfs: at 64 concurrent
# rollouts the RAM-backed overlays reached the job's cgroup limit and the kernel
# killed a Megatron actor. /scratch is per-node, so create it on whichever node
# Slurm gave us, and clear any residue from an earlier job.
# Reward shaping. The verifier's 0/1 score is scaled to 0.8 and two behavioural
# terms make up the rest, so a perfect rollout still scores exactly 1.0 and the
# worst solved rollout (0.8) still outranks the best unsolved one (0.2).
#
#   finish   -- 1.0 when the agent ended by calling the finish tool. That is the
#               behaviour worth teaching: openhands ends a conversation as soon
#               as the model returns prose with no tool call, so an agent that
#               narrates instead of acting has its episode closed mid-task. The
#               policy does this in 9 of 4281 rollouts (0.21%), which is too
#               rare to vary within a group, so it starts near-inert and relies
#               on bootstrapping from the rare group where one member finishes.
#
#   tool_json -- 0.0 when the final message carries a raw <tool_call> block, ie
#               the model emitted a tool call as text. This fires on 14.2% of
#               rollouts and none of those 107 ever solved a task. Its real job
#               is variance: GRPO drops any group whose rewards are all equal,
#               and 70% of groups score uniformly zero. Measured over 546 real
#               groups, usable groups go 126 -> 400 with this term and only
#               126 -> 133 with finish alone, so it is what keeps the batch
#               producing gradient while finish is still rare.
#
# Set a weight to 0 to drop that term; the task weight derives from the other
# two, so the total stays normalised to 1.0.
# Keep prior-turn reasoning in the history the model is shown. openhands only
# echoes reasoning back for names it lists as thinking models, and the Qwen3.5
# template only renders a <think> block when it is given one, so both ends have
# to agree. With this on, turn N's generated tokens are a literal prefix of turn
# N+1's prompt (measured: 59% shared without, 100% with), which is what lets the
# datum builder merge a trajectory's steps into one sequence instead of
# re-encoding the whole history per step. Must match TERMINAL_RL_CONTAINER_MODEL.
export TERMINAL_RL_RETAIN_REASONING_MODEL=openai/gpt-4o

export TERMINAL_RL_FINISH_REWARD_WEIGHT=0.1
export TERMINAL_RL_TOOL_JSON_REWARD_WEIGHT=0.1

# Write the first few sampling calls of each rollout to events/prompts_*.txt:
# the decoded prompt and response, plus whether each prompt extends the last.
# Nothing else records what the model was actually shown, and the three open
# questions -- does history match the evaluation's format, do tool results
# arrive as role "tool", does the model ever close its think block -- are all
# answerable by eye from that file. Set to 0 to turn it off.
export TERMINAL_RL_DUMP_PROMPTS=2

export TERMINAL_RL_OVERLAY_DIR=/scratch/$USER/ovl
mkdir -p "$TERMINAL_RL_OVERLAY_DIR"
rm -rf "${TERMINAL_RL_OVERLAY_DIR:?}"/* 2>/dev/null || true

# Match the terminal-bench 2.0 evaluation: LLM_MAX_INPUT_TOKENS=40000 and
# LLM_MAX_OUTPUT_TOKENS=8192 in run_eval_tb2.sh. Temperature stays at the
# config's 1.0 rather than the evaluation's 0.0, because RL needs exploration.
# The condenser budget, which must leave room under the sampler's hard ceiling:
# context 65536 - max_completion_tokens 8192 = 57344 usable input tokens. At
# 40000 the rendered prompt still reached 57464 and 58721 tokens once retaining
# prior-turn reasoning added tokens this accounting does not see, and sglang
# answered 400. Three retries later the single "sample failed" tore down the
# client session, which set last_heartbeat=-inf and let the lease sweep free the
# model mid-run -- that is what killed jobs 10598023 and 10600529 at step ~10.
# The 15h run 260928-050043-457 predates reasoning retention, never overflowed,
# and never died. 30000 leaves ~27k of slack against the same overshoot.
#
# Training max_completion_tokens is now 16384 (TMax range), and the context goes
# 65536 -> 98304. The gateway clamps max_tokens to the room left
# (miles/tinker/runtime.py _clamp_max_new_tokens) and rejects only a prompt that
# alone fills the context. That rejection is fatal: the tinker SDK's sample_async
# reports it with SESSION_END, the gateway frees the model, and the in-flight
# forward_backward dies with "lease expired". Job 10687294 died this way at step
# 17 on a single 67,615-token prompt, despite the 30000 condenser budget; its
# successful calls peaked at 52,673 (p99.9 38,247). Clamping would also have
# silently shortened 2,871 of its 125,860 calls at 65536 with a 32768 budget.
#
# The guard is --train.context_window_length (below), set to the context. The
# platoon proxy then refuses prompt + max_tokens > context before reaching the
# SDK (proxy.py _check_context_window_length). The ValueError reaches OpenHands
# as a 502 from the bridge, so only that rollout fails and no SESSION_END is
# sent. The check is stricter than the gateway's clamp: a 16384 budget refuses
# prompts above 81,920 tokens. Keep it equal to --context-length.
export TERMINAL_RL_MAX_INPUT_TOKENS=30000
# close() scores a rollout the loop never scored. 180s was set when one close
# ran 1942s unbounded, but 128 rollouts now finish in a wave and queue for the
# verifier together: job 10615209 left 26 of 427 rollouts (6%) UNSCORED, which
# keeps their initial reward of 0.0 rather than the 0.1 a failure earns. Three
# groups were wiped that way -- all eight members at 0.0, so max==min and the
# group is dropped with no gradient. 600s still bounds a hung close but absorbs
# the queue.
export TERMINAL_RL_CLOSE_SCORE_TIMEOUT=600
# 20 held-out tasks instead of the default 2, so the eval metric has enough
# resolution to show a trend rather than only 0.0/0.5/1.0.
export TERMINAL_RL_EVAL_SIZE=20
# The trainer's forward_backward watchdog, hardcoded upstream at 900s and made
# configurable by patches/forward_backward_timeout.py. Batch size here follows
# how long the rollouts survive rather than any configured size, and 900s killed
# jobs 10595807, 10597116 and 10597592 at step 1 with nothing actually hung.
# Raised with batch_size 8 -> 16: 128 rollouts per step instead of 64, so datums
# roughly double (606 median before) and forward_backward already peaked at 1761s
# against the old 2700s bound. The learner GPUs run at 96% util, so this phase is
# linear in tokens and has no slack to absorb the increase.
# Raised 5400 -> 7200. With the act() stall fixed, rollouts run 61 steps on
# average instead of 22, so a batch now carries 31-48M tokens and the learner
# needs 3200-4900s at its fixed ~9700 tok/s. Two of the first six steps came
# within ~500s of 5400, and a batch at the densest observed tokens-per-turn
# would exceed it -- at which point forward_backward_timeout.py raises and the
# run dies, losing every step since the last checkpoint.
export PLATOON_FORWARD_BACKWARD_TIMEOUT=7200

# Per-token sample-vs-train logprob gaps, tagged with rollout staleness
# (patches/dump_token_logprob_gap.py). Left on for every step deliberately: the
# KL spike showed up at step 10 in both earlier Qwen3.5 runs and not at step 20,
# so there is no safe window to restrict it to.
#
# Each row is one token: its id, absolute position, the sampler's logprob, the
# trainer's logprob, their gap, the importance ratio, the raw advantage, and
# clip(iw)*adv -- the last of these separates "large ratio" from "actually moves
# the gradient", which importance_weight_max alone cannot.
#
# stale = train_step - checkpoint_version, computed per rollout. stale_rollouts
# in the metrics cannot substitute: rl.py:601 counts only rollouts that EXCEED
# max_staleness and get dropped, so the stale=1 rollouts that actually train are
# invisible there.
#
# Cost measured at 19k action tokens/datum: 10.3 ms per datum at the 128 threads
# this job runs with, ~18 s on a 29-minute step (about 1%), and 40 KB per datum
# -> roughly 70 MB per step. Lower the three K values to 16 for ~18 MB/step if
# disk becomes a concern.
export PLATOON_LOGPROB_DUMP_DIR=$WORK/logprob_gap
export PLATOON_LOGPROB_DUMP_TOPK=64      # rows by |gap|
export PLATOON_LOGPROB_DUMP_GRADK=64     # rows by |clip(iw)*adv|
export PLATOON_LOGPROB_DUMP_SAMPLE=64    # uniform rows, for the background distribution

RESUME_ARGS=()
if [[ -n "${RESUME_RUN_ID:-}" ]]; then
    RESUME_ARGS=(--resume-run-id "$RESUME_RUN_ID")
    echo "resuming from run $RESUME_RUN_ID"
fi

cd "$REPO"
# No wall-clock cap on a training rollout: rollout_config.timeout defaults to
# None, and max_steps (100 - CONVERSATION_HEADROOM 6 = 94; was 120) is the bound
# instead. Eval keeps 120 so the metric stays comparable with earlier runs.
# The old 700s cap was set when a straggler tail dominated the step, but that
# tail was the ConversationStateUpdateEvent bug in get_actions_for_last_obs,
# fixed by patches/state_updates_are_transparent.py. With it gone the cap was
# cutting off working agents: 24.5% of rollouts, median 75 steps, still acting
# when the clock expired.
#
# step_timeout carries the safety instead, at 300s. 120s was tried and was a
# unit error on my part: 0.235% of steps exceed it, which looks negligible until
# it compounds over a ~114-step episode -- 1-(1-0.00235)^114 = 24%, and job
# 10614643 measured 29.2% of rollouts truncated, against 0.5% at 300s in job
# 10609927. Backing out the per-step hazard from that 0.5% (median 49 steps)
# gives 0.0102%, i.e. ~1.2% of rollouts over 114 steps. Truncation is not free:
# those rollouts had already run 30 steps and 387s, and scored 0.208 mean
# against 0.486 for rollouts that finished, some of them cut after solving.
# --watchdog.timeout_seconds remains the backstop at 10800s.
#
# eval keeps timeout=2400 and step_timeout=300: tightening them would change the
# metric, and discarding there would drop exactly the hardest tasks and quietly
# inflate the eval reward.
# Memory sampling, started here rather than by hand so it runs on the allocated
# node and dies with the job. Timestamps are epoch seconds, so query with
# `date -d @<ts>`, not an ISO string comparison. These are what showed the
# 3.8GB-per-sandbox cost and the 476GB peak that sized --mem.
bash deploy/babel/gpu_memory_log.sh    "$WORK/gpu_memory-$SLURM_JOB_ID.csv"    20 &
bash deploy/babel/cgroup_memory_log.sh "$WORK/cgroup_memory-$SLURM_JOB_ID.csv" 30 &

exec python deploy/babel/run.py \
    --task tmax-15k-openhands \
    "${RESUME_ARGS[@]}" \
    --image "$IMAGE" \
    --data-dir "$DATA" \
    --work-dir "$WORK" \
    --model-name Qwen/Qwen3.5-4B \
    --renderer qwen3_5 \
    --steps 200 \
    --context-length 98304 \
    --mask-flash-cute \
    --tp 4 \
    --recompute \
    --learner-gpus 4 \
    --rollout-gpus 4 \
    --max-tokens-per-gpu 4096 \
    --extra-arg=--tinker-max-tokens-per-datum=98304 \
    --extra-arg=--log-probs-chunk-size=1024 \
    --extra-arg=--rollout-max-context-len=98304 \
    --extra-arg=--tinker-lease-timeout-s=172800 \
    --wandb-project tmax-15k-rl \
    --run-name "oh-rl-$SLURM_JOB_ID" \
    --client-arg=--watchdog.timeout_seconds=10800 \
    --client-arg=--train.batch_size=16 \
    --client-arg=--train.num_epochs=3 \
    --client-arg=--train.workflow_config.group_size=8 \
    --client-arg=--train.max_staleness=1 \
    --client-arg=--train.workflow_config.rollout_config.max_steps=100 \
    --client-arg=--train.workflow_config.rollout_config.step_timeout=300 \
    --client-arg=--train.workflow_config.rollout_config.inference_params.max_completion_tokens=16384 \
    --client-arg=--train.optimizer.grad_clip_norm=1.0 \
    --client-arg=--train.optimizer.beta2=0.999 \
    --client-arg=--train.context_window_length=98304 \
    --client-arg=--train.num_concurrent_rollout_workflow_workers=16 \
    --client-arg=--eval.workflow_config.group_size=4 \
    --client-arg=--eval.num_concurrent_rollout_workflow_workers=10 \
    --client-arg=--eval.workflow_config.rollout_config.max_steps=120 \
    --client-arg=--eval.workflow_config.rollout_config.timeout=2400 \
    --client-arg=--eval.workflow_config.rollout_config.step_timeout=300 \
    --client-arg=--eval.workflow_config.rollout_config.inference_params.max_completion_tokens=8192 \
    --client-arg=--eval.strategy=step \
    --client-arg=--eval.every=10 \
    --client-arg=--checkpoint.every=10
