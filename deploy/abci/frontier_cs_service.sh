#!/bin/bash
# One host-managed Frontier-CS OpenReward server on each rollout node.
set -euo pipefail
umask 077
DRY_RUN=${DRY_RUN:-0}
[[ -n ${SERVICE_LOG:-} && $DRY_RUN != 1 ]] && exec >> "$SERVICE_LOG" 2>&1
source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then module load singularity-ce/4.4.1 2>/dev/null || true; fi

[[ -n ${PBS_JOBID:-} && -d ${PBS_LOCALDIR:-} ]] || { echo "Frontier-CS requires a PBS allocation" >&2; exit 2; }
MILES_SRC=${MILES_SRC:?}
PLATOON_ROOT=${PLATOON_ROOT:?}
export PLATOON_ABCI_ENV_FILE=${FRONTIER_ENV:-$MILES_SRC/deploy/abci/frontier_cs.env}
[[ -r $PLATOON_ABCI_ENV_FILE ]] || { echo "Missing Frontier-CS environment: $PLATOON_ABCI_ENV_FILE" >&2; exit 2; }
SERVER=$PLATOON_ROOT/pbs-scripts/openreward-frontiercs-singularity-server.sh
[[ -r $SERVER ]] || { echo "Missing Frontier-CS server helper: $SERVER" >&2; exit 2; }
GUARD=(/usr/bin/python3 -I "$MILES_SRC/deploy/abci/openreward-node-memory-guard.py"
  --node-limit-bytes 2061584302080 --trigger-percent 85 --fallback-available-bytes 274877906944
  --poll-seconds 2 --telemetry-seconds 60)
echo "[frontier-cs-service] reward=native-score/100 host=$(hostname -s) port=8085"
if [[ $DRY_RUN == 1 ]]; then
  printf 'DRY_RUN server: bash %q\n' "$SERVER"
  printf 'DRY_RUN guard:'; printf ' %q' "${GUARD[@]}"; echo
  exit 0
fi

server_pid=; guard_pid=
cleanup() {
  local status=$? tick
  trap - EXIT INT TERM
  if [[ -n $server_pid ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM "$server_pid" 2>/dev/null || true
    for tick in $(seq 1 100); do kill -0 "$server_pid" 2>/dev/null || break; sleep 1; done
    if kill -0 "$server_pid" 2>/dev/null; then
      echo "[frontier-cs-service] server cleanup exceeded 100 seconds" >&2
      kill -KILL "$server_pid" 2>/dev/null || true
      status=1
    fi
    wait "$server_pid" 2>/dev/null || true
  fi
  if [[ -n $guard_pid ]]; then
    kill -TERM "$guard_pid" 2>/dev/null || true
    wait "$guard_pid" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
bash "$SERVER" & server_pid=$!
"${GUARD[@]}" & guard_pid=$!
status=0
wait -n "$server_pid" "$guard_pid" || status=$?
echo "[frontier-cs-service] server or memory guard exited status=$status" >&2
exit 1
