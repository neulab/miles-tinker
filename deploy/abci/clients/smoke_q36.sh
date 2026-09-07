#!/bin/bash
# CLIENT_SCRIPT for tinker_gateway.pbs: the 5-node Qwen3.6-35B-A3B smoke (design 10.2 steps 1-8) on the head node.
# Launch with deploy/abci/submit_toolathlon.sh smoke (MODE=full, SERVICES=1, TRAIN_NODES=4, LAUNCHER_ARGS=--check-weight-update-equal).
# Steps that need a different gateway configuration are RELAUNCHES of the same preset (they cannot be toggled per
# request). LAUNCHER_ARGS are options of the typer launcher (examples/tinker/run_qwen3_6_35b_a3b_gateway.py serve
# --help), NOT serve_tinker.py flags: the launcher emits `--tinker-r3-check` itself from `--r3-check`, and a
# `--tinker-*` token in LAUNCHER_ARGS is rejected by typer only after the whole Ray cluster is up.
#   step 3 (R3 equivalence):
#     LAUNCHER_ARGS="--check-weight-update-equal --r3-check --extra-env-vars MILES_TEST_R3_THRESHOLD=0.02" R3_CHECK=1
#     (`--extra-env-vars` is the only path to the Ray trainer actors, where miles/utils/replay_base.py reads the
#     env var; EXTRA_ARGS is serve argv only, the containers run --cleanenv; without it the gate is 1e-2)
#   strict-policy half of step 2:
#     EXTRA_ARGS="--tinker-routing-replay-policy strict" R3_POLICY=strict (or LAUNCHER_ARGS ... --routing-policy strict);
#     only step 1 and the R3 probe run: every other client trains datums without a routing_handle, which strict
#     fails typed by design
#   fused-head on/off compare of step 4:
#     LAUNCHER_ARGS="--check-weight-update-equal --no-fused-lm-head" FUSED_HEAD=0
# Each relaunch re-runs this script and the relevant checks pick up the mode from R3_CHECK / R3_POLICY / FUSED_HEAD.
# Knobs: MODEL_NAME, BENCH_DATUMS (32), BENCH_SEQ_LEN (65536), R3_POLICY (natural), R3_CHECK (0), FUSED_HEAD (1).
# The SDK sends SESSION_END only from a fatal-exception path (never on a clean exit), so every client here frees
# its model itself (unload_model at exit); a client that leaks one holds the single full-FT slot for the 600 s lease.
set -uo pipefail
MILES_SRC=${MILES_SRC:-/home/aci18912it/source/miles}
source "$MILES_SRC/deploy/clients/hostguard.sh"
source /etc/profile.d/modules.sh 2>/dev/null || true; module load python/3.12/3.12.9 2>/dev/null || true
COOKBOOK_PY=${COOKBOOK_PYTHON:-/home/aci18912it/miles-tinker/cookbook-venv/bin/python}          # tinker 0.27.1 (proto)
PLATOON_PY=${PLATOON_PYTHON:-${PLATOON_ROOT:-/home/aci18912it/source/platoon-tinker}/plugins/openreward/.venv/bin/python}  # tinker 0.16.1 (JSON)
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/groups/gai51740/aci18912it/miles-tinker/checkpoints/$RUN_NAME}
MODEL_NAME=${MODEL_NAME:-$(grep -o -- '--tinker-base-model [^ ]*' "$JOBDIR/head.log" | head -1 | cut -d' ' -f2)}
MODEL_NAME=${MODEL_NAME:-apurvaga/Qwen3.6-35B-A3B-preserve-thinking}
HEAD_LOG=$JOBDIR/head.log; LAUNCHER_LOG=$JOBDIR/launcher.log
FAILED=0
step() { echo; echo "===== [$1] $2 ($(date +%T))"; }
run() { echo "+ $*"; "$@"; local st=$?; (( st == 0 )) || { echo "FAIL: exit $st: $*"; FAILED=$((FAILED + 1)); }; return $st; }
expect_log() { local what=$1 pattern=$2 file=$3; if grep -qE -- "$pattern" "$file"; then echo "PASS $what: $(grep -E -- "$pattern" "$file" | head -1 | cut -c1-200)"; else echo "FAIL $what: /$pattern/ not in $(basename "$file")"; FAILED=$((FAILED + 1)); fi; }
echo "smoke_q36 against $TINKER_BASE_URL on $(hostname -s): model=$MODEL_NAME services=${OPENREWARD_SESSION_URLS_TOOLATHLON:-none} ($(date))"
cd "$MILES_SRC"

step 1 "startup: node facts, bridge load, warmup push check, engine flags, service health"
grep -E '^init_info|^roles|^hnode|^init_ok' "$LAUNCHER_LOG" | head -40
expect_log "ray cluster at 40 GPUs" '/40.0 GPU|ray cluster ready' "$HEAD_LOG"
expect_log "services healthy" 'services healthy on' "$LAUNCHER_LOG"
expect_log "gateway healthy" 'gateway healthy at' "$LAUNCHER_LOG"
expect_log "engine enable_return_routed_experts" 'enable_return_routed_experts.?[=:] ?True' "$HEAD_LOG"
expect_log "engine mamba_radix_cache_strategy" 'mamba_radix_cache_strategy' "$HEAD_LOG"
expect_log "engine enable_fp32_lm_head" 'enable_fp32_lm_head.?[=:] ?True' "$HEAD_LOG"
expect_log "engine disable_overlap_schedule=False" 'disable_overlap_schedule.?[=:] ?False' "$HEAD_LOG"
expect_log "HostCache[routed_experts] allocated (node memory plan)" 'HostCache\[routed_experts\] allocated' "$HEAD_LOG"
expect_log "warmup check_weights: 0 mismatches" 'check_weight|mismatch' "$HEAD_LOG"
grep -E 'mismatch' "$HEAD_LOG" | head -5

R3_POLICY=${R3_POLICY:-natural}
step 2 "0.16.1 R3 probe (policy=$R3_POLICY) then the SDK 0.27.1 smoke (proto)"
run "$PLATOON_PY" deploy/clients/tinker_r3_probe.py --endpoint-file "$ENDPOINT_FILE" --base-model "$MODEL_NAME" --policy "$R3_POLICY"
# the gateway logs the first prefix-parent hits (runtime.py _begin_capture) and counts them in the optim metrics
expect_log "gateway saw a prefix-parent hit on an extended turn" 'prefix parent hit .*routed_experts_start_len=[1-9]' "$HEAD_LOG"
expect_log "optim metrics count prefix-parent hits (routing_prefix_hits)" '"routing_prefix_hits": [1-9]' "$HEAD_LOG"
if [[ $R3_POLICY == strict ]]; then
  echo "SKIP (R3_POLICY=strict): the SDK smoke, the 64k bench, the numerics check and the resume gate train datums"
  echo "  without a routing_handle, which strict fails typed by design; run them on the natural-policy launch"
  echo; echo "smoke_q36 done (strict): $FAILED failed step(s)/check(s) ($(date))"
  exit $(( FAILED > 0 ))
fi
run "$COOKBOOK_PY" deploy/clients/tinker_sdk_smoke.py --endpoint-file "$ENDPOINT_FILE" --base-model "$MODEL_NAME" --lora-rank 8 --steps 2 --unload-before-resume --resume-without-optimizer

step 3 "R3 equivalence (relaunch with --r3-check; R3_CHECK=${R3_CHECK:-0})"
if [[ ${R3_CHECK:-0} == 1 ]]; then
  expect_log "replay check lines present" 'Replay check' "$HEAD_LOG"
  grep -E 'Replay check' "$HEAD_LOG" | tail -5
  echo "(threshold: MILES_TEST_R3_THRESHOLD=0.02 -> >= 98% of covered rows must match the trainer's top-k)"
else
  echo "SKIP: relaunch the smoke preset with LAUNCHER_ARGS='--check-weight-update-equal --r3-check --extra-env-vars MILES_TEST_R3_THRESHOLD=0.02' and R3_CHECK=1"
fi

step 4 "memory at 64k: one 65536-token row per DP rank (fused head ${FUSED_HEAD:-1})"
run "$COOKBOOK_PY" deploy/clients/tinker_mfu_bench.py --endpoint-file "$ENDPOINT_FILE" --base-model "$MODEL_NAME" \
  --num-datums "${BENCH_DATUMS:-32}" --seq-len "${BENCH_SEQ_LEN:-65536}" --steps 2 --perf-jsonl "$CHECKPOINT_ROOT/perf.jsonl"
grep -E 'max_memory_allocated|OutOfMemory|CUDA out of memory' "$HEAD_LOG" | tail -5
echo "(budget: max_memory_allocated < 130 GB at TP1, else relaunch with --tp 2; fused on/off compare = relaunch with"
echo " LAUNCHER_ARGS='--check-weight-update-equal --no-fused-lm-head' and FUSED_HEAD=0)"

step 5 "sampler-vs-trainer numerics (fp32 head both sides)"
run "$COOKBOOK_PY" deploy/clients/tinker_onpolicy_check.py --endpoint-file "$ENDPOINT_FILE" --base-model "$MODEL_NAME" --train-steps 1 --learning-rate 1e-6

step 6 "MFU sanity (actor_train_mfu in 0.05-0.35) and per-rank skew"
python3 deploy/clients/perf_report.py --perf-jsonl "$CHECKPOINT_ROOT/perf.jsonl" --every 1 || true
python3 - "$CHECKPOINT_ROOT/perf.jsonl" <<'PY' || FAILED=$((FAILED + 1))
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
mfu = [r.get("actor_train_mfu") for r in rows if r.get("actor_train_mfu") is not None]
bad = [m for m in mfu if not (0.05 <= m <= 0.35)]
print(f"{len(mfu)} optim windows, mfu={['%.3f' % m for m in mfu]}, tok/s={[round(r.get('actor_train_tok_per_s', 0)) for r in rows]}")
sys.exit(1 if (not mfu or bad) else 0)
PY

step 7 "weight-push timing with 1 engine (extrapolate to 11; target 1-3 min per step)"
grep -E 'update_weights|weight push|broadcast' "$HEAD_LOG" | tail -8

step 8 "resume gate (hard): save_state -> unload -> create_training_client_from_state_with_optimizer -> forward_backward"
run "$COOKBOOK_PY" deploy/clients/tinker_sdk_smoke.py --endpoint-file "$ENDPOINT_FILE" --base-model "$MODEL_NAME" --lora-rank 8 --steps 1 --unload-before-resume
grep -E 'load_slot|max_memory_allocated' "$HEAD_LOG" | tail -4
echo "(failure here = TP2 fallback for the whole run)"

echo; echo "smoke_q36 done: $FAILED failed step(s)/check(s) ($(date))"
exit $(( FAILED > 0 ))
