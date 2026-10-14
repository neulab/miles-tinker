#!/bin/bash
# CLIENT_SCRIPT for tinker_gateway.pbs (full fine-tuning): the context-parallel parity probe
# (deploy/clients/tinker_cp_parity.py run) on a fresh model, then the sampler/trainer on-policy probe and,
# unless SDK_SMOKE=0, the real-SDK smoke (save_state + resume with optimizer). Runs on the head compute node
# with TINKER_BASE_URL / ENDPOINT_FILE / JOBDIR exported by the launcher. The gateway's CP and TP sizes, pad
# multiplier and model id come from head.log; PARITY_OUT (default $JOBDIR/parity-cp<N>.json) receives the dump.
# Compare the dumps of a CP=1, a CP=N and a TP2 (noise reference) gateway with
#   python3 deploy/clients/tinker_cp_parity.py compare <cp1.json> <cpN.json> --noise <tp2.json>
set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true; module load python/3.12/3.12.9 2>/dev/null || true
PY=${COOKBOOK_PYTHON:-/home/aci18912it/miles-tinker/cookbook-venv/bin/python}
MODEL_NAME=${MODEL_NAME:-$(grep -o -- '--tinker-base-model [^ ]*' "$JOBDIR/head.log" | head -1 | cut -d' ' -f2)}
CP=${CP:-$(grep -o -- '--context-parallel-size [0-9]*' "$JOBDIR/head.log" | head -1 | cut -d' ' -f2)}
CP=${CP:-1}
TP=${TP:-$(grep -o -- '--tensor-model-parallel-size [0-9]*' "$JOBDIR/head.log" | head -1 | cut -d' ' -f2)}
TP=${TP:-1}
# set only through EXTRA_ARGS, so usually absent from head.log (|| true: no match is not an error)
PAD_MULTIPLIER=${PAD_MULTIPLIER:-$( (grep -o -- '--data-pad-size-multiplier [0-9]*' "$JOBDIR/head.log" || true) | tail -1 | cut -d' ' -f2)}
PAD_MULTIPLIER=${PAD_MULTIPLIER:-128}
PARITY_OUT=${PARITY_OUT:-$JOBDIR/parity-cp$CP.json}
echo "cp parity against $TINKER_BASE_URL on $(hostname -s): model=$MODEL_NAME cp=$CP tp=$TP out=$PARITY_OUT ($(date))"
"$PY" deploy/clients/tinker_cp_parity.py run --endpoint-file "$ENDPOINT_FILE" --base-model "$MODEL_NAME" \
  --cp "$CP" --tp "$TP" --pad-multiplier "$PAD_MULTIPLIER" --label "${RUN_NAME:-cp$CP}" --out "$PARITY_OUT" \
  ${PARITY_ARGS:-}
"$PY" deploy/clients/tinker_onpolicy_check.py --endpoint-file "$ENDPOINT_FILE" --base-model "$MODEL_NAME" \
  --train-steps 3 --learning-rate "${ONPOLICY_LR:-1e-6}" || echo "on-policy probe exit status $?"
if [[ ${SDK_SMOKE:-1} == 1 ]]; then
  "$PY" deploy/clients/tinker_sdk_smoke.py --endpoint-file "$ENDPOINT_FILE" --base-model "$MODEL_NAME" \
    --lora-rank 16 --steps 2 --unload-before-resume
fi
echo "cp parity done ($(date))"
