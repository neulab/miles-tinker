#!/bin/bash
# CLIENT_SCRIPT for tinker_gateway.pbs: platoon number-search (SDK 0.16.1, cispo) with W&B online logging.
# W&B credentials: run `wandb login` once on a login node (stored in ~/.netrc); nothing is passed on the command line.
# Knobs: STEPS (100), LR (1e-6 full / 1e-5 lora), CKPT_EVERY (25), WANDB_MODE (online), EXTRA (verbatim platoon --a.b.c overrides).
set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true; module load python/3.12/3.12.9 2>/dev/null || true
PLUGIN=${PLATOON_ROOT:-/home/aci18912it/source/platoon}/plugins/number-search
PY=${PLATOON_PYTHON:-$PLUGIN/.venv/bin/python}
LOG_PATH=${LOG_PATH:-/groups/gai51740/aci18912it/miles-tinker/runs/$RUN_NAME-number-search/logs}
export TINKER_API_KEY=${TINKER_API_KEY:-tml-local} TINKER_TELEMETRY=${TINKER_TELEMETRY:-0} WANDB_MODE=${WANDB_MODE:-online}
echo "number-search -> $TINKER_BASE_URL, steps=${STEPS:-100}, lr=${LR:-1e-6} ($(date))"
cd "$PLUGIN"
"$PY" -m platoon.number_search.train_tinker --config platoon/number_search/number_search_tinker.yaml \
  --tinker_base_url "$TINKER_BASE_URL" \
  --train.model_name Qwen/Qwen3-4B-Instruct-2507 --train.renderer_name qwen3_instruct \
  --train.max_training_steps "${STEPS:-100}" --train.num_epochs 1000 --train.optimizer.learning_rate "${LR:-1e-6}" \
  --checkpoint.strategy step --checkpoint.every "${CKPT_EVERY:-25}" \
  --log_path "$LOG_PATH" --stats.wandb.mode "$WANDB_MODE" \
  --stats.experiment_name "${WANDB_EXPERIMENT:-number-search-miles-tinker}" --stats.trial_name "${WANDB_TRIAL:-$RUN_NAME}" \
  ${EXTRA:-}
cd - >/dev/null
python3 deploy/clients/perf_report.py --perf-jsonl "/groups/gai51740/aci18912it/miles-tinker/checkpoints/$RUN_NAME/perf.jsonl" --every 10 || true
