#!/bin/bash
# platoon textcraft-synth Tinker training (SDK 0.16.1, cispo) against a gateway; acceptance run B.
#   deploy/clients/run_platoon_textcraft.sh <endpoint-file-or-url> <log_path> [extra --overrides ...]
# Small smoke defaults (override with env): STEPS=5 BATCH_SIZE=4 GROUP_SIZE=4 WORKERS=4 ROLLOUT_MAX_STEPS=30
#   CONTEXT_LEN=4096 EVAL_EVERY=100 (eval cannot be disabled; a large interval skips it) CKPT_EVERY=1
# Full acceptance settings: STEPS=200 BATCH_SIZE=16 GROUP_SIZE=8 WORKERS=16 ROLLOUT_MAX_STEPS=200 CONTEXT_LEN=8192 EVAL_EVERY=5
# Env: PLATOON_ROOT, PLATOON_PYTHON (plugin venv with the tinker extra), TINKER_API_KEY (must start with tml-),
#      HF_HOME (Qwen3-4B-Instruct-2507 tokenizer), WANDB_MODE (offline|disabled).
set -euo pipefail
ENDPOINT=${1:?endpoint file or http://host:port}; LOG_PATH=${2:?log_path}; shift 2
[[ -f $ENDPOINT ]] && ENDPOINT=$(tr -d '[:space:]' < "$ENDPOINT")
PLATOON_ROOT=${PLATOON_ROOT:-/home/aci18912it/source/platoon}
PLUGIN=$PLATOON_ROOT/plugins/textcraft
PLATOON_PYTHON=${PLATOON_PYTHON:-$PLUGIN/.venv/bin/python}
CONFIG=${CONFIG:-platoon/textcraft/configs/tinker/textcraft_synth_tinker.yaml}   # relative to $PLUGIN; the script's default path is wrong
[[ -f $PLUGIN/$CONFIG ]] || { echo "config not found: $PLUGIN/$CONFIG" >&2; exit 2; }
export TINKER_API_KEY=${TINKER_API_KEY:-tml-local}       # SDK 0.16.1 requires the tml- prefix
export TINKER_BASE_URL=$ENDPOINT                         # the litellm proxy's ServiceClient() takes no base_url
export TINKER_TELEMETRY=${TINKER_TELEMETRY:-0}
export HF_HOME=${HF_HOME:-/groups/gai51740/aci18912it/rasp/cache/huggingface}
export no_proxy="${no_proxy:-}${no_proxy:+,}${ENDPOINT#http://}"
echo "platoon textcraft synth -> $ENDPOINT (log_path=$LOG_PATH, steps=${STEPS:-5})"
cd "$PLUGIN"
exec "$PLATOON_PYTHON" -m platoon.textcraft.train_scripts.tinker.train_tinker_synth \
  --config "$CONFIG" \
  --tinker_base_url "$ENDPOINT" \
  --log_path "$LOG_PATH" \
  --stats.wandb.mode "${WANDB_MODE:-offline}" \
  --train.max_training_steps "${STEPS:-5}" \
  --train.batch_size "${BATCH_SIZE:-4}" \
  --train.workflow_config.group_size "${GROUP_SIZE:-4}" \
  --train.num_concurrent_rollout_workflow_workers "${WORKERS:-4}" \
  --train.workflow_config.rollout_config.max_steps "${ROLLOUT_MAX_STEPS:-30}" \
  --train.context_window_length "${CONTEXT_LEN:-4096}" \
  --eval.every "${EVAL_EVERY:-100}" \
  --checkpoint.every "${CKPT_EVERY:-1}" \
  "$@"
