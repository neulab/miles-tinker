#!/bin/bash
# CLIENT_SCRIPT for tinker_gateway.pbs: routed-experts probe, sampler/trainer probe, a short MFU benchmark, then the
# real-SDK smoke (last: its resume-with-optimizer step is the most memory-hungry call, see --resume-without-optimizer).
# Runs on the head compute node with TINKER_BASE_URL / ENDPOINT_FILE / JOBDIR exported by the launcher.
# MODE (lora|full) and MODEL_NAME (the --tinker-base-model the gateway advertises) come from the job environment
# (`qsub -v`) or, when unset, from the job's launcher.log / head.log.
set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true; module load python/3.12/3.12.9 2>/dev/null || true
PY=${COOKBOOK_PYTHON:-/home/aci18912it/miles-tinker/cookbook-venv/bin/python}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/groups/gai51740/aci18912it/miles-tinker/checkpoints/$RUN_NAME}
MODE=${MODE:-$(sed -n '1s/.* mode=\([a-z]*\) .*/\1/p' "$JOBDIR/launcher.log")}
MODEL_NAME=${MODEL_NAME:-$(grep -o -- '--tinker-base-model [^ ]*' "$JOBDIR/head.log" | sed -n 1p | cut -d' ' -f2)}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}
if [[ $MODE == full ]]; then
  # single-tenant gateway: free the first model before the resume check; FFT learning rate for the probe
  SMOKE_ARGS=${SMOKE_ARGS:-"--unload-before-resume"}
  ONPOLICY_LR=${ONPOLICY_LR:-1e-6}
else
  SMOKE_ARGS=${SMOKE_ARGS:-}
  ONPOLICY_LR=${ONPOLICY_LR:-1e-4}
fi
echo "smoke against $TINKER_BASE_URL on $(hostname -s): mode=$MODE model=$MODEL_NAME ($(date))"
# engine-side check (independent of the gateway): does /generate return routed experts for this model?
"$PY" deploy/clients/sglang_routed_experts_probe.py --head-log "$JOBDIR/head.log" ${ROUTED_EXPERTS_ARGS:-} \
  || echo "routed-experts probe exit status $? (2 = engines answered without routed_experts)"
"$PY" deploy/clients/tinker_onpolicy_check.py --endpoint-file "$ENDPOINT_FILE" --base-model "$MODEL_NAME" --train-steps 3 --learning-rate "$ONPOLICY_LR"
"$PY" deploy/clients/tinker_mfu_bench.py --endpoint-file "$ENDPOINT_FILE" --base-model "$MODEL_NAME" --num-datums "${BENCH_DATUMS:-64}" --seq-len "${BENCH_SEQ_LEN:-8192}" --steps 2 --perf-jsonl "$CHECKPOINT_ROOT/perf.jsonl"
"$PY" deploy/clients/tinker_sdk_smoke.py --endpoint-file "$ENDPOINT_FILE" --base-model "$MODEL_NAME" --lora-rank 16 --steps 2 $SMOKE_ARGS
echo "smoke done ($(date))"
