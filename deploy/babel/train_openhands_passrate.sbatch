#!/bin/bash
# Terminal-RL training with the openhands agent, on the tmax-15k task pool,
# training Qwen3-4B-Instruct-2507 with a graded (pass-rate) task reward.
#
# Identical to train_openhands.sbatch apart from the model, its renderer, the
# reasoning switch and the prompt-dump limit.
#
# Reasoning is OFF and cannot be turned on: Instruct-2507's chat template has
# no thinking branch, no enable_thinking toggle and zero mentions of
# reasoning_content (the <think>/</think> ids 151667/151668 exist in the
# vocabulary but nothing renders them). Measured on job 10625680, ~55% of
# every generated response was reasoning -- 174 of 316 tokens -- and only
# generated tokens carry mask=1.0, so this moves that share of the gradient
# onto the visible answer and the tool call, which is what the reward scores.
#
# The weights are already prepared at <work-dir>/models/Qwen3-4B-Instruct-2507
# (7.5G, all 3 shards), so prepare has nothing to download.
#
# Shares WORK with train_openhands.sbatch, which is safe only because
# aasai_qos caps a user at 8 GPUs: the two cannot hold the partition at once,
# and run.py re-patches <work-dir>/platoon at every launch.
#
# Same resources as ~/train.sh, but running the training instead of holding the
# node idle, and with the partition's maximum wall time (aasai MaxTime is
# 20-01:00:00; aasai_qos caps a user at 8 GPUs, not at a shorter wall clock).
#
#   sbatch train_openhands.sbatch                 # fresh run
#   RESUME_RUN_ID=260928-024322-325 sbatch ...    # continue from that run's
#                                                 # last checkpoint
#SBATCH --job-name=oh-rl-pr
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

REPO=$HOME/miles-tinker
WORK=/data/user_data/$USER/miles-babel
DATA=/data/user_data/$USER/tmax15k-rl2
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
# echoes reasoning back for names it lists as thinking models, and the Qwen3
# template only renders a <think> block when it is given one, so both ends have
# to agree. With this on, turn N's generated tokens are a literal prefix of turn
# N+1's prompt (measured: 59% shared without, 100% with), which is what lets the
# datum builder merge a trajectory's steps into one sequence instead of
# re-encoding the whole history per step. Must match TERMINAL_RL_CONTAINER_MODEL.
# Deliberately unset. This registers a model name as a thinking model inside
# each sandbox (agent_server.py:140 _allow_reasoning_resend) so openhands
# resends reasoning between turns. Qwen3-4B-Instruct-2507 never produces any:
# its chat template has no <think> branch and no reasoning_content handling at
# all, so the edit would be an in-sandbox no-op run once per rollout.
# The three reasoning patches stay registered and go inert the same way --
# retain_thinking_in_history skips on its hasattr guard, and the other two
# find no thinking parts and nothing inbound to fold.
export TERMINAL_RL_RETAIN_REASONING_MODEL=

# Grade by the fraction of the task's unit tests that pass, instead of its own
# 1/0. 1486 tests over the 500-task pool, mean 2.97 per task, so most tasks gain
# 3-4 reward levels. The point is GRPO's group-relative advantage: a group whose
# members all score alike is dropped, and 54% of groups in job 10635972 were
# all-0.2 with nobody solving. Graded tests give those groups variance.
#
# Caveat worth watching: tests are not equally hard. "the executable exists" is
# worth the same third of a task as "it computes the right answer", and a
# group-relative advantage rewards beating your groupmates -- so cheap tests are
# a cheap way to do that. If reward climbs while the solve rate does not, that is
# the first thing to check.
export TERMINAL_RL_REWARD_MODE=passrate

# Evaluate once before training, at step 0 (eval_at_step_zero.py). The graded
# reward changes the scale of eval/reward itself, so the binary runs' numbers
# are not a baseline for this one -- 15.0/15.0/16.2/18.8% from job 10635972 is
# measured on a different y-axis. Without a step-0 point the first eval lands
# at step 10, and a rise that happened before it is indistinguishable from a
# flat series. Costs one eval cycle (20 tasks x 4) ahead of the first step.
export PLATOON_EVAL_AT_START=1

# Both are inert under TERMINAL_RL_REWARD_SCHEME=gated (the unset default):
# the gate tests `tool_reward <= 0.0` as a boolean and never multiplies by its
# weight, and finish_reward is computed and logged but never enters the total.
# They take effect only under scheme=linear. Kept so switching schemes needs
# one edit, not three.
export TERMINAL_RL_FINISH_REWARD_WEIGHT=0.1
export TERMINAL_RL_TOOL_JSON_REWARD_WEIGHT=0.1

# Write the first few sampling calls of each rollout to events/prompts_*.txt:
# the decoded prompt and response, plus whether each prompt extends the last.
# Nothing else records what the model was actually shown, and the three open
# questions -- does history match the evaluation's format, do tool results
# arrive as role "tool", does the model ever close its think block -- are all
# answerable by eye from that file. Set to 0 to turn it off.
# Raised from 2 for this run. Calls 1 and 2 are where the prefix chain is
# provably clean -- 0 genuine breaks in 2549 observations on job 10625680,
# every non-extension there being openhands' conversation-title call -- while
# the 17-38% of breaks measured per batch all happen deeper than a limit of 2
# can see. Diagnostic only: one file per rollout, nothing the trainer reads.
# Set back to 2 to match job 10625680 exactly.
export TERMINAL_RL_DUMP_PROMPTS=200

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
# Raised when the step went to 128 rollouts (then batch 16 x group 8; now
# batch 8 x group 16 -- same 128, same token cost), so datums
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

RESUME_ARGS=()
if [[ -n "${RESUME_RUN_ID:-}" ]]; then
    RESUME_ARGS=(--resume-run-id "$RESUME_RUN_ID")
    echo "resuming from run $RESUME_RUN_ID"
fi

cd "$REPO"
# No wall-clock cap on a training rollout: rollout_config.timeout defaults to
# None, and max_steps (120 - CONVERSATION_HEADROOM 6 = 114) is the bound instead.
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
# Group size 16 at batch size 8, rather than 8 at 16. Same 128 rollouts per
# step and the same token cost to the learner; what changes is how they are
# spent. GRPO centres the advantage on the group mean, so with 8 members that
# baseline is estimated from 8 samples and a group is dropped entirely when all
# 8 score alike. Doubling the group halves the baseline's standard error and
# makes an all-identical group much less likely -- on this pool 54.7% of
# binary-scored groups were degenerate, and the graded reward already cut that
# to ~23%.
#
# The cost is task coverage per step: 8 fresh tasks instead of 16, drawn from a
# pool of 500. Step-to-step reward will be noisier (job 10648148 showed a
# per-step sd of 0.056 at 16 tasks), so read the eval series rather than the
# training curve for trend. Epochs also halve in length -- 3 epochs at batch 8
# is half the optimizer steps of 3 at batch 16, which is a real reduction in
# training, not a free change.
#
# num_concurrent_rollout_workflow_workers drops 16 -> 8 to match. A worker runs
# its whole group concurrently (group_rollout_workflow.py:186 gathers
# group_size episodes), so in-flight rollouts are workers x group_size. Leaving
# it at 16 would put 256 rollouts in flight instead of 128 and double the load
# on the sampler, which in job 10648148 degraded from 4-5 concurrent requests
# down to 1 over 26 hours and took the median call from 10.8s to 54s.

bash deploy/babel/gpu_memory_log.sh    "$WORK/gpu_memory-$SLURM_JOB_ID.csv"    20 &
bash deploy/babel/cgroup_memory_log.sh "$WORK/cgroup_memory-$SLURM_JOB_ID.csv" 30 &

exec python deploy/babel/run.py \
    --task tmax-15k-openhands \
    "${RESUME_ARGS[@]}" \
    --image "$IMAGE" \
    --data-dir "$DATA" \
    --work-dir "$WORK" \
    --model-name Qwen/Qwen3-4B-Instruct-2507 \
    --renderer qwen3_instruct \
    --steps 200 \
    --context-length 65536 \
    --mask-flash-cute \
    --tp 4 \
    --recompute \
    --learner-gpus 4 \
    --rollout-gpus 4 \
    --max-tokens-per-gpu 4096 \
    --extra-arg=--tinker-max-tokens-per-datum=65536 \
    --extra-arg=--log-probs-chunk-size=1024 \
    --extra-arg=--rollout-max-context-len=65536 \
    --extra-arg=--tinker-lease-timeout-s=172800 \
    --wandb-project tmax-15k-rl \
    --run-name "oh-rl-pr-$SLURM_JOB_ID" \
    --client-arg=--watchdog.timeout_seconds=10800 \
    --client-arg=--train.batch_size=8 \
    --client-arg=--train.num_epochs=3 \
    --client-arg=--train.workflow_config.group_size=16 \
    --client-arg=--train.max_staleness=1 \
    --client-arg=--train.workflow_config.rollout_config.max_steps=120 \
    --client-arg=--train.workflow_config.rollout_config.step_timeout=300 \
    --client-arg=--train.workflow_config.rollout_config.inference_params.max_completion_tokens=8192 \
    --client-arg=--train.num_concurrent_rollout_workflow_workers=8 \
    --client-arg=--eval.workflow_config.group_size=4 \
    --client-arg=--eval.num_concurrent_rollout_workflow_workers=10 \
    --client-arg=--eval.workflow_config.rollout_config.max_steps=120 \
    --client-arg=--eval.workflow_config.rollout_config.timeout=2400 \
    --client-arg=--eval.workflow_config.rollout_config.step_timeout=300 \
    --client-arg=--eval.workflow_config.rollout_config.inference_params.max_completion_tokens=8192 \
    --client-arg=--eval.strategy=step \
    --client-arg=--eval.every=10 \
    --client-arg=--checkpoint.every=10
