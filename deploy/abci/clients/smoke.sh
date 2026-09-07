#!/bin/bash
# CLIENT_SCRIPT for tinker_gateway.pbs: real-SDK smoke + sampler/trainer probe + a short MFU benchmark.
# Runs on the head compute node with TINKER_BASE_URL / ENDPOINT_FILE / JOBDIR exported by the launcher.
set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true; module load python/3.12/3.12.9 2>/dev/null || true
PY=${COOKBOOK_PYTHON:-/home/aci18912it/miles-tinker/cookbook-venv/bin/python}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/groups/gai51740/aci18912it/miles-tinker/checkpoints/$RUN_NAME}
echo "smoke against $TINKER_BASE_URL on $(hostname -s) ($(date))"
"$PY" deploy/clients/tinker_sdk_smoke.py --endpoint-file "$ENDPOINT_FILE" --lora-rank 16 --steps 2 ${SMOKE_ARGS:-}
"$PY" deploy/clients/tinker_onpolicy_check.py --endpoint-file "$ENDPOINT_FILE" --train-steps 3 --learning-rate 1e-4
"$PY" deploy/clients/tinker_mfu_bench.py --endpoint-file "$ENDPOINT_FILE" --num-datums "${BENCH_DATUMS:-64}" --seq-len "${BENCH_SEQ_LEN:-8192}" --steps 2 --perf-jsonl "$CHECKPOINT_ROOT/perf.jsonl"
echo "smoke done ($(date))"
