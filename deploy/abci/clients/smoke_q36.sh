#!/bin/bash
# CLIENT_SCRIPT for tinker_gateway.pbs: the 5-node Qwen3.6-35B-A3B smoke (design 10.2 steps 1-9) on the head node.
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
# Step 9 (design 10.2 step 9) is the only end-to-end step: a 2-optim-step platoon + OpenHands + Toolathlon run of the
# production client argv (clients/platoon_toolathlon.sh) with the production environment (clients/platoon_env.sh,
# sourced by both scripts) and a tiny run size. It runs LAST, after the resume gate, so steps 1-8 always print their
# verdicts first, and it is bounded by `timeout -k 120 $PLATOON_TIMEOUT`.
# Knobs: MODEL_NAME, BENCH_DATUMS (32), BENCH_SEQ_LEN (65536), R3_POLICY (natural), R3_CHECK (0), FUSED_HEAD (1),
#   PLATOON_STEP (1; 0 skips step 9), PLATOON_TIMEOUT (2400 s), SMOKE_WANDB (offline; `online` writes to the
#   production W&B project), SMOKE_EXPERIMENT (toolathlon-smoke), SMOKE_TRIAL ($RUN_NAME-platoon-smoke),
#   CONFIG (the recursive yaml, plugin-relative), PLATOON_ROOT.
# Step-9 run size (defaults = the tiny run; raise them to reproduce production concurrency on 5 nodes):
#   PLATOON_WORKERS (2), PLATOON_GROUP_SIZE (2), PLATOON_BATCH_SIZE (2), PLATOON_QUORUM (2), PLATOON_MIN_SUCCESS (1),
#   PLATOON_TRAIN_STEPS (2), PLATOON_MAX_STEPS (12), PLATOON_ROLLOUT_TIMEOUT (600), PLATOON_STEP_TIMEOUT (240),
#   PLATOON_STRAGGLER_TIMEOUT (180), PLATOON_WATCHDOG (1200).  Production is WORKERS=3 GROUP_SIZE=8 (24 concurrent
#   root trajectories in one asyncio process), which is what step 9's mcp_failure_rate_pct check is calibrated for.
# The SDK sends SESSION_END only from a fatal-exception path (never on a clean exit), so every client here frees
# its model itself (unload_model at exit); a client that leaks one holds the single full-FT slot for the 600 s lease.
# Step 9's trainer does not (production does not either): it is the last step, and the lease frees the slot.
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
# like expect_log, but polls: the driver flushes its `tinker perf` line a few seconds after the client's call returns
expect_log_soon() { local what=$1 pattern=$2 file=$3 budget=${4:-90} t=0
  while (( t < budget )); do grep -qE -- "$pattern" "$file" && break; sleep 5; t=$((t + 5)); done
  expect_log "$what" "$pattern" "$file"; }
expect_no_log() { local what=$1 pattern=$2 file=$3; if grep -qE -- "$pattern" "$file" 2>/dev/null; then echo "FAIL $what: $(grep -E -- "$pattern" "$file" | head -1 | cut -c1-200)"; FAILED=$((FAILED + 1)); else echo "PASS $what"; fi; }
# PASS when any file named $3 under directory $4 matches the ERE $2 (step 9's rollout artifacts)
expect_files() { local what=$1 pattern=$2 include=$3 dir=$4 hit; hit=$(grep -rlE --include="$include" --binary-files=without-match -- "$pattern" "$dir" 2>/dev/null | head -1); if [[ -n $hit ]]; then echo "PASS $what: ${hit#"$dir"/}"; else echo "FAIL $what: /$pattern/ in no $include under $dir"; FAILED=$((FAILED + 1)); fi; }
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
expect_log_soon "optim metrics count prefix-parent hits (routing_prefix_hits)" 'routing_prefix_hits": [1-9]' "$HEAD_LOG"
if [[ $R3_POLICY == strict ]]; then
  echo "SKIP (R3_POLICY=strict): the SDK smoke, the 64k bench, the numerics check and the resume gate train datums"
  echo "  without a routing_handle, which strict fails typed by design; run them on the natural-policy launch."
  echo "  Step 9 (platoon) does send a routing_handle per datum, but it is skipped here too: run it on that launch."
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

# ---------------------------------------------------------------------------------------------- step 9
# Last, after the resume gate, so steps 1-8 always produce their verdicts first even if this hangs or times out.
step 9 "end to end: platoon + OpenHands + Toolathlon + gateway, 2 optim steps (PLATOON_STEP=${PLATOON_STEP:-1})"
if [[ ${PLATOON_STEP:-1} != 1 ]]; then
  echo "SKIP (PLATOON_STEP=${PLATOON_STEP:-1}): the end-to-end platoon run; steps 1-8 stand on their own"
else
  PLATOON_ROOT=${PLATOON_ROOT:-/home/aci18912it/source/platoon-tinker}
  PLUGIN=$PLATOON_ROOT/plugins/openreward
  PLATOON_CONFIG=${CONFIG:-platoon/openreward/configs/tinker/abci_toolathlon_15node_recursive_qwen36_miles_tinker_r3_v1.yaml}
  SMOKE_WANDB_MODE=${SMOKE_WANDB:-offline}                 # never the production W&B project unless SMOKE_WANDB=online
  SMOKE_EXPERIMENT=${SMOKE_EXPERIMENT:-toolathlon-smoke}   # distinct experiment/trial: never resumes a production run
  SMOKE_TRIAL=${SMOKE_TRIAL:-$RUN_NAME-platoon-smoke}
  SMOKE_LOG_PATH=$JOBDIR/platoon-smoke
  SMOKE_RUNDIR=$SMOKE_LOG_PATH/$SMOKE_EXPERIMENT/$SMOKE_TRIAL
  ROLLOUTS=$SMOKE_RUNDIR/rollouts/train                    # <log_path>/<exp>/<trial>/rollouts/train/<version>/...
  PLATOON_LOG=$SMOKE_LOG_PATH/platoon-step9.log
  FACTS=$SMOKE_LOG_PATH/step9-facts.txt
  PERF_JSONL=$CHECKPOINT_ROOT/perf.jsonl
  SESSION_URL=${OPENREWARD_SESSION_URLS_TOOLATHLON%%,*}
  PERF_BEFORE=0; [[ -f $PERF_JSONL ]] && PERF_BEFORE=$(wc -l < "$PERF_JSONL")
  STEP9_FAILED_BEFORE=$FAILED
  mkdir -p "$SMOKE_LOG_PATH"
  [[ -x $PLATOON_PY ]] || { echo "FAIL step 9: platoon venv python missing: $PLATOON_PY"; FAILED=$((FAILED + 1)); }
  [[ -r $PLUGIN/$PLATOON_CONFIG ]] || { echo "FAIL step 9: config missing: $PLUGIN/$PLATOON_CONFIG"; FAILED=$((FAILED + 1)); }
  # A tiny real run of the production entry point: same restart_wrapper -> train_tinker argv and the same environment
  # (clients/platoon_env.sh, sourced by clients/platoon_toolathlon.sh too), only the run size is overridden.
  # group_size 2 REQUIRES the straggler_quorum / min_successful_group_size overrides: the yaml's 6 / 4 fail
  # WorkflowConfig.__post_init__ ("must be in [1, group_size]"). platoon's _parse_value maps "1" to True, which is 1 in
  # every use (`steps % recovery_every`, `1 <= min_successful_group_size <= group_size`).
  # checkpoint.every 1000 = no periodic permanent save; recovery_every 1 = one ~490 GB recovery save per step (plus
  # one at launch). The trainer still writes a permanent `final` on a clean finish (rl.py, unconditional).
  # Every run-size value is a knob so the smoke can also be run at PRODUCTION concurrency (the client-side
  # failure modes of the 15-node runs only appear once rollouts overlap): PLATOON_WORKERS x PLATOON_GROUP_SIZE is
  # the number of concurrent root trajectories in the one asyncio client process (production: 3 x 8 = 24).
  # Keep PLATOON_QUORUM and PLATOON_MIN_SUCCESS in [1, PLATOON_GROUP_SIZE] (WorkflowConfig.__post_init__), and
  # PLATOON_BATCH_SIZE >= PLATOON_WORKERS so every worker has a group to run.
  PLATOON_ARGV=(
    "$PLATOON_PY" -m platoon.train.tinker.restart_wrapper --max-restarts 0 --restart-delay 10 --
    "$PLATOON_PY" -m platoon.openreward.train_scripts.tinker.train_tinker --config "$PLATOON_CONFIG"
    --tinker_base_url "$TINKER_BASE_URL" --log_path "$SMOKE_LOG_PATH"
    --stats.experiment_name "$SMOKE_EXPERIMENT" --stats.trial_name "$SMOKE_TRIAL"
    --stats.wandb.mode "$SMOKE_WANDB_MODE"
    --train.max_training_steps "${PLATOON_TRAIN_STEPS:-2}" --train.batch_size "${PLATOON_BATCH_SIZE:-2}"
    --train.num_concurrent_rollout_workflow_workers "${PLATOON_WORKERS:-2}"
    --train.workflow_config.group_size "${PLATOON_GROUP_SIZE:-2}"
    --train.workflow_config.straggler_quorum "${PLATOON_QUORUM:-2}"
    --train.workflow_config.min_successful_group_size "${PLATOON_MIN_SUCCESS:-1}"
    --train.workflow_config.straggler_timeout_seconds "${PLATOON_STRAGGLER_TIMEOUT:-180}"
    --train.workflow_config.rollout_config.max_steps "${PLATOON_MAX_STEPS:-12}"
    --train.workflow_config.rollout_config.timeout "${PLATOON_ROLLOUT_TIMEOUT:-600}"
    --train.workflow_config.rollout_config.step_timeout "${PLATOON_STEP_TIMEOUT:-240}"
    --checkpoint.every 1000 --checkpoint.recovery_every 1
    --watchdog.timeout_seconds "${PLATOON_WATCHDOG:-1200}"
  )
  run_platoon() {
    (
      cd "$PLUGIN" || exit 2
      WANDB_MODE=$SMOKE_WANDB_MODE
      # shellcheck source=deploy/abci/clients/platoon_env.sh
      source "$MILES_SRC/deploy/abci/clients/platoon_env.sh"   # exits 2 here, in this subshell, on a missing input
      exec timeout -k 120 "${PLATOON_TIMEOUT:-2400}" "${PLATOON_ARGV[@]}"
    ) 2>&1 | tee "$PLATOON_LOG"
  }
  echo "log_path=$SMOKE_LOG_PATH experiment=$SMOKE_EXPERIMENT trial=$SMOKE_TRIAL wandb=$SMOKE_WANDB_MODE service=$SESSION_URL"
  printf '+ timeout -k 120 %s' "${PLATOON_TIMEOUT:-2400}"; printf ' %q' "${PLATOON_ARGV[@]}"; echo
  run run_platoon
  # `timeout` signals only the restart_wrapper; on a timeout its trainer child survives and would hold the gateway's
  # single full-FT slot until the 600 s lease expires. Reap it (a no-op after a clean exit).
  pkill -u "$(id -u)" -f 'platoon\.openreward\.train_scripts\.tinker\.train_tinker' 2>/dev/null &&
    echo "(reaped a step-9 trainer that outlived the timeout)" || true
  python3 - "$PERF_JSONL" "$PERF_BEFORE" "$SMOKE_RUNDIR/checkpoints.jsonl" "$ROLLOUTS" > "$FACTS" <<'PY'
import glob, json, os, sys
perf, before, ckpt, rollouts = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
def rows(path):
    try:
        return [json.loads(line) for line in open(path) if line.strip()]
    except OSError:
        return []
def last(records, key):
    values = [r[key] for r in records if r.get(key) is not None]
    return values[-1] if values else 0
records = rows(perf)
old, new = records[:before], records[before:]
windows = [r for r in new if r.get("num_datums", 0) > 0]  # one record per optim_step window
print(f"perf_jsonl={perf} records_before={len(old)} records_total={len(records)}")
print(f"optim_windows_with_datums={len(windows)}")
print(f"routing_covered_datums={sum(r.get('routing_covered_datums', 0) for r in windows)}")
print(f"routing_uncovered_datums={sum(r.get('routing_uncovered_datums', 0) for r in windows)}")
print(f"routing_prefix_hits_delta={max(0, last(new, 'routing_prefix_hits') - last(old, 'routing_prefix_hits'))}")
print(f"num_datums={[r.get('num_datums') for r in windows]} num_tokens={[r.get('num_tokens') for r in windows]}")
print(f"grad_norm={[r.get('grad_norm') for r in windows]} mfu={[r.get('actor_train_mfu') for r in windows]}")
print(f"routing_cache_bytes={[r.get('routing_cache_bytes') for r in windows]}")
checkpoints = rows(ckpt)
recovery = [r for r in checkpoints if str(r.get("name", "")).startswith("recovery-") and r.get("state_path")]
print(f"recovery_checkpoint_with_state_path={recovery[-1]['name'] if recovery else 'none'}"
      f" ({len(recovery)} of {len(checkpoints)} records in {ckpt})")
print(f"permanent_checkpoints={[r.get('name') for r in checkpoints if not str(r.get('name', '')).startswith('recovery-')]}")
# Every trajectory that reached a terminal record, and how many died before their first agent step because the
# MCP bridge would not spawn.  This is the 15-node regression (run ta15-rec-q36-r3-v1 lost 968 of 1060
# trajectories to it): OpenHands' programmatic tool calling swapped the process-global sys.stderr for an
# io.StringIO while a sibling rollout was spawning its bridge, so Popen could not get a stderr fd.  It only
# appears once rollouts overlap, hence the production-concurrency knobs above.
finished = errored = mcp = 0
for path in glob.glob(os.path.join(rollouts, "*", "events", "events_*.jsonl")):
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("type") != "trajectory_finished":
                continue
            finished += 1
            message = record.get("error_message") or ""
            errored += bool(message)
            mcp += "MCP Connection Failure" in message
# "none" (not 0) when nothing finished, so the < 10 % check cannot pass on an empty run.
rate = round(100.0 * mcp / finished) if finished else "none"
print(f"trajectories_finished={finished} trajectories_errored={errored} mcp_connection_failures={mcp}")
print(f"mcp_failure_rate_pct={rate}")
PY
  cat "$FACTS"
  expect_log "trainer completed >= 2 optim steps with datums" 'optim_windows_with_datums=([2-9]|[0-9]{2,})' "$FACTS"
  # < 10 %: at production concurrency the pre-fix client lost 91 % of trajectories here (see the fact block).
  expect_log "MCP bridges connected (< 10% of trajectories lost at step 0)" 'mcp_failure_rate_pct=[0-9]$' "$FACTS"
  expect_log "R3 covered the run's datums (routing_covered_datums > 0)" 'routing_covered_datums=[1-9]' "$FACTS"
  expect_log "R3 left no datum uncovered (routing_uncovered_datums == 0)" 'routing_uncovered_datums=0$' "$FACTS"
  expect_log "gateway extended a turn-1 prefix (routing_prefix_hits grew)" 'routing_prefix_hits_delta=[1-9]' "$FACTS"
  expect_log "a recovery checkpoint record has a state_path" 'recovery_checkpoint_with_state_path=recovery-[0-9]' "$FACTS"
  expect_files "episodes ran >= 2 agent steps (OpenHands events)" '"step_index": 1' 'events_*.jsonl' "$ROLLOUTS"
  expect_files "episodes emitted tool calls (ActionEvent.tool_name)" '"tool_name":' 'events_*.jsonl' "$ROLLOUTS"
  expect_files "Toolathlon executed a tool (MCP bridge tool_result)" '"type": "tool_result"' 'bridge_events.jsonl' "$ROLLOUTS"
  expect_files "the job's Toolathlon service served the run" "\"session_url\": \"$SESSION_URL\"" 'bridge_events.jsonl' "$ROLLOUTS"
  expect_log "restart_wrapper saw a clean trainer exit" 'Training completed successfully' "$PLATOON_LOG"
  expect_no_log "no watchdog kill" 'WATCHDOG:.*Forcing exit' "$PLATOON_LOG"
  tail -40 "$PLATOON_LOG" > "$SMOKE_LOG_PATH/platoon-step9-tail.log" 2>/dev/null
  expect_no_log "no unhandled exception at exit" 'Traceback \(most recent call last\)' "$SMOKE_LOG_PATH/platoon-step9-tail.log"
  # the service's own access lines (format depends on the container's uvicorn/nginx): context, never a verdict
  for f in "$JOBDIR"/services-*.log; do
    [[ -r $f ]] || continue
    echo "$(basename "$f"): $(grep -cE '(GET|POST|PUT|DELETE) /' "$f" 2>/dev/null || true) request lines, last non-health:"
    grep -E '(GET|POST|PUT|DELETE) /' "$f" 2>/dev/null | grep -v '/health' | tail -3
  done
  if (( FAILED > STEP9_FAILED_BEFORE )); then
    echo "--- tail -40 $PLATOON_LOG"; tail -40 "$PLATOON_LOG" 2>/dev/null
    for f in "$JOBDIR"/services-*.log; do [[ -r $f ]] && { echo "--- tail -30 $f"; tail -30 "$f"; }; done
    echo "(exit 124 = the run exceeded PLATOON_TIMEOUT=${PLATOON_TIMEOUT:-2400} s; the wall clock is dominated by the"
    echo " full ~490 GB kind=both saves - launch, step 1, step 2, final - plus the rollouts. Raise PLATOON_TIMEOUT,"
    echo " or set PLATOON_STEP=0 to skip this step. Rollout artifacts: $ROLLOUTS)"
  fi
  # reclaim the disposable recovery saves of this step (permanent names, `final` included, are never pruned)
  python3 deploy/abci/clients/prune_tinker_checkpoints.py --log-path "$SMOKE_LOG_PATH" \
    --checkpoint-root "$CHECKPOINT_ROOT" --keep 1 --grace-min 1 || true
fi

echo; echo "smoke_q36 done: $FAILED failed step(s)/check(s) ($(date))"
exit $(( FAILED > 0 ))
