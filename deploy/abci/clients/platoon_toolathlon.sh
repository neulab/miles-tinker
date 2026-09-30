#!/bin/bash
# CLIENT_SCRIPT for tinker_gateway.pbs: the Platoon OpenReward RL client on the job's head node.
# The historical filename is retained; SERVICE_ENVIRONMENT also supports SWE and Frontier-CS.
# The launcher exports TINKER_BASE_URL TINKER_API_KEY ENDPOINT_FILE JOBDIR RUN_NAME HF_HOME no_proxy MILES_SRC
# PLATOON_ROOT SERVICE_NODES OPENREWARD_SESSION_URLS_TOOLATHLON OPENREWARD_SESSION_URLS PBS_LOCALDIR WORKERS (C10).
# The client environment (C10/C11 exports + the secrets, from /home/aci18912it/rasp/config/abci.env, which chains
# secrets.env, mode 0600) lives in clients/platoon_env.sh and is sourced THERE ONLY - never in the batch shell, the
# service helper or any container. smoke_q36.sh step 9 sources the same file, so the smoke runs this environment.
# Knobs: CONFIG (absolute or plugin-relative yaml), WORKERS (3), MAX_RESTARTS (20), RESTART_DELAY (700 s >= lease 600 + sweep 30
# + margin), KEEP_CKPTS (2), WANDB_MODE (online), WANDB_TRIAL (RUN_NAME), EXTRA (verbatim platoon --a.b.c overrides).
# Restart/resume: platoon.train.tinker.restart_wrapper restarts the trainer on the watchdog exit code (2). The SDK
# sends SESSION_END (TINKER_TELEMETRY=1) only from a fatal-exception path - never on a clean exit, an os._exit(2) or
# the graceful SIGTERM exit of rl.py - so in practice the gateway frees the single full-FT slot on lease expiry
# (600 s + <= 30 s sweep), which RESTART_DELAY covers; the next attempt then re-acquires it and rl.py resumes from
# <log_path>/<experiment>/<trial>/checkpoints.jsonl.
set -euo pipefail
MILES_SRC=${MILES_SRC:-/home/aci18912it/source/miles}
source "$MILES_SRC/deploy/clients/hostguard.sh"
source /etc/profile.d/modules.sh 2>/dev/null || true; module load python/3.12/3.12.9 2>/dev/null || true
PLATOON_ROOT=${PLATOON_ROOT:-/home/aci18912it/source/platoon-tinker}
PLUGIN=$PLATOON_ROOT/plugins/openreward
PY=$PLUGIN/.venv/bin/python
CONFIG=${CONFIG:-platoon/openreward/configs/tinker/abci_toolathlon_15node_recursive_qwen36_miles_tinker_r3_v1.yaml}
[[ -x $PY ]] || { echo "platoon venv python missing: $PY" >&2; exit 2; }
[[ $CONFIG == /* ]] || CONFIG=$PLUGIN/$CONFIG
[[ -r $CONFIG ]] || { echo "config missing: $CONFIG" >&2; exit 2; }
# shellcheck source=deploy/abci/clients/platoon_env.sh
source "$MILES_SRC/deploy/abci/clients/platoon_env.sh"   # C10/C11 env + secrets; shared with smoke_q36.sh step 9
LOG_PATH=/groups/gai51740/aci18912it/miles-tinker/runs/$RUN_NAME/logs
CKPT_ROOT=/groups/gai51740/aci18912it/miles-tinker/checkpoints/$RUN_NAME
mkdir -p "$LOG_PATH"
echo "platoon ${SERVICE_ENVIRONMENT:-toolathlon} -> $TINKER_BASE_URL config=$CONFIG workers=${WORKERS:-3} services=$OPENREWARD_SESSION_URLS ($(date))"
echo "platoon: $(git -C "$PLATOON_ROOT" rev-parse --short HEAD 2>/dev/null) $(git -C "$PLATOON_ROOT" branch --show-current 2>/dev/null); log_path=$LOG_PATH ckpt_root=$CKPT_ROOT"
# checkpoint pruner: every 10 min delete the ~490 GB `recovery-*` checkpoints except the newest KEEP_CKPTS of them
# (never the last record's state_path, never a directory written in the last 30 min); permanent checkpoints
# (NNNNNN from checkpoint.every, *_interrupted, final) are kept forever
( while sleep 600; do
    python3 "$MILES_SRC/deploy/abci/clients/prune_tinker_checkpoints.py" --log-path "$LOG_PATH" --checkpoint-root "$CKPT_ROOT" --keep "${KEEP_CKPTS:-2}" || true
  done ) & PRUNE_PID=$!
trap 'kill $PRUNE_PID 2>/dev/null || true' EXIT
cd "$PLUGIN"
# shellcheck disable=SC2086
status=0
"$PY" -m platoon.train.tinker.restart_wrapper --max-restarts "${MAX_RESTARTS:-20}" --restart-delay "${RESTART_DELAY:-700}" -- \
  "$PY" -m platoon.openreward.train_scripts.tinker.train_tinker --config "$CONFIG" \
    --tinker_base_url "$TINKER_BASE_URL" --log_path "$LOG_PATH" \
    --stats.trial_name "${WANDB_TRIAL:-$RUN_NAME}" --stats.wandb.mode "$WANDB_MODE" \
    --train.num_concurrent_rollout_workflow_workers "${WORKERS:-3}" ${EXTRA:-} || status=$?
kill "$PRUNE_PID" 2>/dev/null || true
cd "$MILES_SRC"
python3 deploy/clients/perf_report.py --perf-jsonl "$CKPT_ROOT/perf.jsonl" --every 10 || true
echo "platoon ${SERVICE_ENVIRONMENT:-toolathlon} exited status=$status ($(date))"
exit "$status"
