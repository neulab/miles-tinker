#!/bin/bash
# deploy/abci/submit_toolathlon.sh - qsub presets for the Qwen3.6-35B-A3B Toolathlon jobs on deploy/abci/tinker_gateway.pbs.
# usage: deploy/abci/submit_toolathlon.sh {smoke|recursive|nonrecursive} [--dry-run] [--run-name NAME] [--walltime H:MM:SS]
#   smoke        5 nodes (4 train + 1 TP8 engine + 1 Toolathlon service), 06:00:00, CLIENT_SCRIPT=clients/smoke_q36.sh
#   recursive    15 nodes (4 + 11), 168:00:00, platoon recursive yaml, WORKERS=3
#   nonrecursive 15 nodes (4 + 11), 168:00:00, platoon non-recursive yaml, WORKERS=3
# Overrides via the environment: HF_SNAPSHOT, MILES_SRC (this checkout), PLATOON_ROOT, WORKERS, LAUNCHER_ARGS, EXTRA_ARGS,
# TRAIN_NODES, WANDB_MODE, KEEP_GATEWAY. --dry-run prints the exact qsub line and exits without submitting.
set -euo pipefail
preset=${1:-}; shift || true
[[ $preset =~ ^(smoke|recursive|nonrecursive)$ ]] || { sed -n '2,8p' "$0" >&2; exit 2; }
DRY=0; RUN_NAME_ARG=; WALLTIME_ARG=
while (( $# )); do
  case $1 in
    --dry-run) DRY=1 ;;
    --run-name) RUN_NAME_ARG=$2; shift ;;
    --walltime) WALLTIME_ARG=$2; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac; shift
done
MILES_SRC=${MILES_SRC:-$(cd "$(dirname "$0")/../.." && pwd)}
PLATOON_ROOT=${PLATOON_ROOT:-/home/aci18912it/source/platoon-tinker}
HF_SNAPSHOT=${HF_SNAPSHOT:-/groups/gai51740/aci18912it/rasp/cache/huggingface/hub/models--apurvaga--Qwen3.6-35B-A3B-preserve-thinking/snapshots/ce815485d640a33461c25e3d5c505b5346e8c857}
LAUNCHER=examples/tinker/run_qwen3_6_35b_a3b_gateway.py
CONFIG_DIR=platoon/openreward/configs/tinker
case $preset in
  smoke)
    NODES=5; WALLTIME=06:00:00; RUN_NAME=${RUN_NAME_ARG:-smoke-q36-$(date +%m%d%H%M)}
    CLIENT_SCRIPT=$MILES_SRC/deploy/abci/clients/smoke_q36.sh
    LAUNCHER_ARGS=${LAUNCHER_ARGS:---check-weight-update-equal}; CONFIG= ;;
  recursive)
    NODES=15; WALLTIME=168:00:00; RUN_NAME=${RUN_NAME_ARG:-ta15-rec-q36-r3-v1}
    CLIENT_SCRIPT=$MILES_SRC/deploy/abci/clients/platoon_toolathlon.sh
    LAUNCHER_ARGS=${LAUNCHER_ARGS:-}; CONFIG=$CONFIG_DIR/abci_toolathlon_15node_recursive_qwen36_miles_tinker_r3_v1.yaml ;;
  nonrecursive)
    NODES=15; WALLTIME=168:00:00; RUN_NAME=${RUN_NAME_ARG:-ta15-nonrec-q36-r3-v1}
    CLIENT_SCRIPT=$MILES_SRC/deploy/abci/clients/platoon_toolathlon.sh
    LAUNCHER_ARGS=${LAUNCHER_ARGS:-}; CONFIG=$CONFIG_DIR/abci_toolathlon_15node_nonrecursive_qwen36_miles_tinker_r3_v1.yaml ;;
esac
WALLTIME=${WALLTIME_ARG:-$WALLTIME}
TRAIN_NODES=${TRAIN_NODES:-4}
[[ -d $HF_SNAPSHOT ]] || { echo "HF_SNAPSHOT is not a directory: $HF_SNAPSHOT" >&2; exit 2; }
[[ -r $CLIENT_SCRIPT ]] || { echo "CLIENT_SCRIPT is unreadable: $CLIENT_SCRIPT" >&2; exit 2; }
[[ -z $CONFIG || -r $PLATOON_ROOT/plugins/openreward/$CONFIG ]] || { echo "platoon config is unreadable: $PLATOON_ROOT/plugins/openreward/$CONFIG" >&2; exit 2; }
for v in "$MILES_SRC" "$PLATOON_ROOT" "$HF_SNAPSHOT" "${LAUNCHER_ARGS}" "${EXTRA_ARGS:-}" "$RUN_NAME"; do
  [[ $v != *,* ]] || { echo "qsub -v splits on commas; value not allowed: $v" >&2; exit 2; }
done
# LAUNCHER_ARGS are typer options of $LAUNCHER (`serve --help`); a serve_tinker.py flag there (e.g. --tinker-r3-check,
# the launcher spells it --r3-check) is rejected by typer only after the whole Ray cluster is up. Serve flags go in
# EXTRA_ARGS (appended verbatim, last wins).
for tok in ${LAUNCHER_ARGS}; do
  [[ $tok != --tinker-* ]] || { echo "LAUNCHER_ARGS token $tok is a serve_tinker.py flag: use the launcher option (see '$LAUNCHER serve --help', e.g. --r3-check) or put it in EXTRA_ARGS" >&2; exit 2; }
done
VARS="RTYPE=rt_HF,MODE=full,RUN_NAME=$RUN_NAME,SERVICES=1,TRAIN_NODES=$TRAIN_NODES,GATEWAY_LAUNCHER=$LAUNCHER,HF_SNAPSHOT=$HF_SNAPSHOT"
VARS+=",MILES_SRC=$MILES_SRC,PLATOON_ROOT=$PLATOON_ROOT,CLIENT_SCRIPT=$CLIENT_SCRIPT,WORKERS=${WORKERS:-3},WANDB_MODE=${WANDB_MODE:-online}"
[[ -n $CONFIG ]] && VARS+=",CONFIG=$CONFIG"
[[ -n $LAUNCHER_ARGS ]] && VARS+=",LAUNCHER_ARGS=$LAUNCHER_ARGS"
[[ -n ${EXTRA_ARGS:-} ]] && VARS+=",EXTRA_ARGS=$EXTRA_ARGS"
[[ -n ${KEEP_GATEWAY:-} ]] && VARS+=",KEEP_GATEWAY=$KEEP_GATEWAY"
# Smoke-mode knobs read by clients/smoke_q36.sh; forwarded only when set (they must reach the client through -v,
# because the CLIENT_SCRIPT inherits the job environment, not this shell's).  The PLATOON_* run-size knobs let
# the 5-node smoke reproduce production rollout concurrency (WORKERS x GROUP_SIZE concurrent trajectories in the
# one client process); OPENREWARD_STREAM_ISOLATION=0 turns off platoon's stdout/stderr isolation (debug only).
for knob in R3_CHECK R3_POLICY FUSED_HEAD PLATOON_STEP SMOKE_WANDB PLATOON_TIMEOUT BENCH_DATUMS BENCH_SEQ_LEN \
           PLATOON_WORKERS PLATOON_GROUP_SIZE PLATOON_BATCH_SIZE PLATOON_QUORUM PLATOON_MIN_SUCCESS \
           PLATOON_TRAIN_STEPS PLATOON_MAX_STEPS PLATOON_ROLLOUT_TIMEOUT PLATOON_STEP_TIMEOUT \
           PLATOON_STRAGGLER_TIMEOUT PLATOON_WATCHDOG OPENREWARD_STREAM_ISOLATION; do
  [[ -n ${!knob:-} ]] || continue
  [[ ${!knob} != *,* ]] || { echo "qsub -v splits on commas; $knob value not allowed: ${!knob}" >&2; exit 2; }
  VARS+=",$knob=${!knob}"
done
QSUB=(qsub -N "miles-ta-$preset" -l "select=$NODES:mpiprocs=8" -l "walltime=$WALLTIME" -v "$VARS" "$MILES_SRC/deploy/abci/tinker_gateway.pbs")
echo "preset=$preset nodes=$NODES ($TRAIN_NODES train + $((NODES - TRAIN_NODES)) engine/service) walltime=$WALLTIME run=$RUN_NAME"
echo "fork: $(git -C "$MILES_SRC" rev-parse --short HEAD 2>/dev/null) $(git -C "$MILES_SRC" branch --show-current 2>/dev/null); platoon: $(git -C "$PLATOON_ROOT" rev-parse --short HEAD 2>/dev/null)"
echo "reservation queue:"; qstat -Q R9920261300 2>&1 || true
echo "NOTE: the reservation's end time is not visible to this account; confirm with the admins (or a running job's"
echo "      Resource_List / qstat -f) that it covers $WALLTIME before a 168 h submission."
printf 'qsub:'; printf ' %q' "${QSUB[@]}"; echo
if (( DRY )); then echo "(dry run: not submitted)"; exit 0; fi
"${QSUB[@]}"
