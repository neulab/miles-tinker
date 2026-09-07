#!/bin/bash
# CLIENT_SCRIPT for tinker_gateway.pbs: tinker-cookbook multiplayer_rl/guess_number, async RL, 25 steps.
# Extra key=value overrides via GN_ARGS (e.g. GN_ARGS="learning_rate=1e-5 max_steps=50").
set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true; module load python/3.12/3.12.9 2>/dev/null || true
PY=${COOKBOOK_PYTHON:-/home/aci18912it/miles-tinker/cookbook-venv/bin/python}
LOG_PATH=${LOG_PATH:-/groups/gai51740/aci18912it/miles-tinker/runs/$RUN_NAME-guess-number}
export TINKER_API_KEY=${TINKER_API_KEY:-tml-cookbook}
echo "guess_number -> $TINKER_BASE_URL, log_path=$LOG_PATH ($(date))"
"$PY" deploy/clients/cookbook_guess_number_async.py endpoint_file="$ENDPOINT_FILE" log_path="$LOG_PATH" ${GN_ARGS:-}
"$PY" deploy/clients/perf_report.py --perf-jsonl "/groups/gai51740/aci18912it/miles-tinker/checkpoints/$RUN_NAME/perf.jsonl" --every 5 || true
