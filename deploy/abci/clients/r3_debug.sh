#!/bin/bash
# CLIENT_SCRIPT for tinker_gateway.pbs: the smallest run that exercises R3 end to end.
# One tinker_r3_probe.py run (3 prefix-extending sample turns + one forward_backward + one
# optim_step) on a 2-node gateway (1 train node, 1 engine node, SERVICES=0), then the R3
# diagnostics the trainer prints: the per-build router registration count, the per-fill row
# counts, and the gateway's prefix-parent hits. Used to reproduce/close job 2245536's
# "pop_forward out of range" and its missing prefix-parent hits.
#   qsub -v RTYPE=rt_HF,MODE=full,RUN_NAME=r3dbg-1,SERVICES=0,TRAIN_NODES=1,\
#     GATEWAY_LAUNCHER=examples/tinker/run_qwen3_6_35b_a3b_gateway.py,HF_SNAPSHOT=...,\
#     MILES_SRC=...,LAUNCHER_ARGS="--context-len 8191 --max-tokens-per-gpu 8192",\
#     CLIENT_SCRIPT=$MILES_SRC/deploy/abci/clients/r3_debug.sh -l walltime=01:30:00 deploy/abci/tinker_gateway.pbs
# CLIENT_SCRIPT must be absolute: tinker_gateway.pbs checks it from the batch shell's cwd ($HOME),
# not from MILES_SRC (job 2245772 came up healthy and then refused the relative path).
# The launcher asserts --max-tokens-per-gpu >= --context-len + 1, so both must shrink together.
# Knobs: MODEL_NAME, R3_POLICY (natural), R3_MAX_TOKENS (16 generated tokens per turn), SECOND_PROBE (1).
set -uo pipefail
MILES_SRC=${MILES_SRC:-/home/aci18912it/source/miles}
source "$MILES_SRC/deploy/clients/hostguard.sh"
PLATOON_PY=${PLATOON_PYTHON:-${PLATOON_ROOT:-/home/aci18912it/source/platoon-tinker}/plugins/openreward/.venv/bin/python}
MODEL_NAME=${MODEL_NAME:-$(grep -o -- '--tinker-base-model [^ ]*' "$JOBDIR/head.log" | head -1 | cut -d' ' -f2)}
MODEL_NAME=${MODEL_NAME:-apurvaga/Qwen3.6-35B-A3B-preserve-thinking}
HEAD_LOG=$JOBDIR/head.log
FAILED=0
step() { echo; echo "===== [$1] $2 ($(date +%T))"; }
run() { echo "+ $*"; "$@"; local st=$?; (( st == 0 )) || { echo "FAIL: exit $st: $*"; FAILED=$((FAILED + 1)); }; return $st; }
expect_log() { local what=$1 pattern=$2; if grep -qE -- "$pattern" "$HEAD_LOG"; then echo "PASS $what: $(grep -E -- "$pattern" "$HEAD_LOG" | head -1 | cut -c1-240)"; else echo "FAIL $what: /$pattern/ not in head.log"; FAILED=$((FAILED + 1)); fi; }
echo "r3_debug against $TINKER_BASE_URL on $(hostname -s): model=$MODEL_NAME ($(date))"
cd "$MILES_SRC"

step 1 "R3 registration: one line per model build, per trainer rank"
grep -E 'R3 registration' "$HEAD_LOG" | head -10
expect_log "routing manager enabled with routers registered" 'R3 registration.*routing replay manager enabled=True, [1-9][0-9]* routers'

step 2 "R3 probe (policy=${R3_POLICY:-natural})"
run "$PLATOON_PY" deploy/clients/tinker_r3_probe.py --endpoint-file "$ENDPOINT_FILE" \
  --base-model "$MODEL_NAME" --policy "${R3_POLICY:-natural}" --max-tokens "${R3_MAX_TOKENS:-16}"

step 3 "R3 fill: rows recorded per registered router"
grep -E 'R3 fill' "$HEAD_LOG" | head -10
expect_log "every registered router got its rows" 'R3 fill rollout_routed_experts: microbatches=[1-9][0-9]* replays=[1-9][0-9]*'

step 4 "prefix-parent hits (routing_cache cut ladder)"
grep -E 'prefix parent hit' "$HEAD_LOG" | head -5
expect_log "gateway saw a prefix-parent hit on an extended turn" 'prefix parent hit .*routed_experts_start_len=[1-9]'
expect_log "optim metrics count prefix-parent hits" '"routing_prefix_hits": [1-9]'

step 5 "no replay error, no opaque router IndexError"
if grep -qE 'ReplayError|pop_forward out of range' "$HEAD_LOG"; then
  echo "FAIL replay error in head.log:"; grep -E 'ReplayError|pop_forward out of range' "$HEAD_LOG" | head -5; FAILED=$((FAILED + 1))
else echo "PASS no replay error"; fi

if [[ ${SECOND_PROBE:-1} == 1 ]]; then
  step 6 "a second client: the first probe gave the single full-FT slot back (service._release_model)"
  run "$PLATOON_PY" deploy/clients/tinker_r3_probe.py --endpoint-file "$ENDPOINT_FILE" \
    --base-model "$MODEL_NAME" --policy "${R3_POLICY:-natural}" --max-tokens "${R3_MAX_TOKENS:-16}"
  grep -E 'R3 fill' "$HEAD_LOG" | tail -4
fi

echo; echo "r3_debug done: $FAILED failed check(s) ($(date))"
exit $(( FAILED > 0 ))
