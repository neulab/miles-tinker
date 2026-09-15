#!/bin/bash
# Reuse Platoon's pinned SWE server, session isolation, and exact-instance cleanup.
set -euo pipefail
umask 077
DRY_RUN=${DRY_RUN:-0}
[[ -n ${SERVICE_LOG:-} && $DRY_RUN != 1 ]] && exec >> "$SERVICE_LOG" 2>&1
source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then module load singularity-ce/4.4.1 2>/dev/null || true; fi

[[ -n ${PBS_JOBID:-} && -d ${PBS_LOCALDIR:-} ]] || { echo "SWE service requires a PBS allocation" >&2; exit 2; }
MILES_SRC=${MILES_SRC:?}
PLATOON_ROOT=${PLATOON_ROOT:?}
export PLATOON_ABCI_ENV_FILE=${SWE_ENV:-$MILES_SRC/deploy/abci/swe_rebench.env}
[[ -r $PLATOON_ABCI_ENV_FILE ]] || { echo "Missing SWE environment: $PLATOON_ABCI_ENV_FILE" >&2; exit 2; }
SERVER=$PLATOON_ROOT/pbs-scripts/openreward-swe-singularity-server.sh
[[ -r $SERVER ]] || { echo "Missing SWE server helper: $SERVER" >&2; exit 2; }
GUARD=(/usr/bin/python3 -I "$MILES_SRC/deploy/abci/openreward-node-memory-guard.py"
  --node-limit-bytes 2061584302080 --trigger-percent 85 --fallback-available-bytes 274877906944
  --poll-seconds 2 --telemetry-seconds 60)
echo "[swe-rebench-service] reward_mode=${OPENREWARD_REWARD_MODE:-binary} host=$(hostname -s) port=8084"
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
    # The reused helper stops its server and cleans only its own UUID sessions.
    for tick in $(seq 1 100); do kill -0 "$server_pid" 2>/dev/null || break; sleep 1; done
    if kill -0 "$server_pid" 2>/dev/null; then
      echo "[swe-rebench-service] server cleanup exceeded 100 seconds" >&2
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
echo "[swe-rebench-service] server or memory guard exited status=$status" >&2
exit 1
