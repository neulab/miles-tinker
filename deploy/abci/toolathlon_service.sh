#!/bin/bash
# deploy/abci/toolathlon_service.sh - one Toolathlon service on one engine node: nginx :8082 -> 8 uvicorn workers
# 28100-28107 + postgres 127.0.0.1:5432 inside the derived Toolathlon SIF, plus the node memory guard
# (deploy/abci/openreward-node-memory-guard.py). Held by one pbsdsh from tinker_gateway.pbs under `env -i` with
# PBS_JOBID PBS_LOCALDIR PLATOON_ROOT TOOLATHLON_ENV MILES_SRC OPENREWARD_REWARD_MODE OPENREWARD_ENVIRONMENT_SET.
# Trimmed from platoon-v8r36-07bebf9-057108f/pbs-scripts/openreward-node-services.sh: same singularity flags,
# container env and cleanup budget; dropped the experiment-binding tables, runtime attestation, receipt/shutdown
# handshake, SWE server and training-SIF staging. Sources deploy/abci/toolathlon.env (C12) only - never abci.env.
# DRY_RUN=1 prints the resolved SIF + sha256, the guard argv and every singularity command line, then exits 0
# without creating directories or starting anything (usable on a login node).
set -euo pipefail
umask 077
DRY_RUN=${DRY_RUN:-0}
# pbsdsh does not forward stdout: log to the file the launcher names (services-<host>.log) instead
[[ -n ${SERVICE_LOG:-} && ${DRY_RUN} != 1 ]] && exec >> "$SERVICE_LOG" 2>&1
# The launcher never passes credentials (env -i); drop the ten accepted names anyway before anything else runs.
for secret_name in WANDB_API_KEY OPENAI_API_KEY OPENAI_BASE_URL LITELLM_API_KEY LITELLM_BASE_URL RUBRIC_DEFAULT_LLM \
                   HF_TOKEN AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN; do unset "$secret_name"; done
unset secret_name
source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then module load singularity-ce/4.4.1 2>/dev/null || true; fi

log() { echo "[toolathlon-service] $(date '+%FT%T') $*"; }
die() { echo "[toolathlon-service] ERROR: $*" >&2; exit 2; }
dry() { [[ $DRY_RUN == 1 ]]; }
show() { local label=$1; shift; printf 'DRY_RUN %s:' "$label"; printf ' %q' "$@"; echo; }

if dry; then PBS_JOBID=${PBS_JOBID:-dryrun}; PBS_LOCALDIR=${PBS_LOCALDIR:-${TMPDIR:-/tmp}/toolathlon-dryrun}; fi
[[ -n ${PBS_JOBID:-} ]] || die "this helper must run inside PBS (PBS_JOBID unset)"
dry || [[ -d ${PBS_LOCALDIR:-} ]] || die "PBS_LOCALDIR is unavailable: ${PBS_LOCALDIR:-unset}"
export PBS_JOBID PBS_LOCALDIR
PLATOON_ROOT=${PLATOON_ROOT:-/home/aci18912it/source/platoon-tinker}; export PLATOON_ROOT
MILES_SRC=${MILES_SRC:-/home/aci18912it/source/miles}
TOOLATHLON_ENV=${TOOLATHLON_ENV:-$MILES_SRC/deploy/abci/toolathlon.env}
[[ -r $TOOLATHLON_ENV ]] || die "TOOLATHLON_ENV is unreadable: $TOOLATHLON_ENV"
# shellcheck disable=SC1090
source "$TOOLATHLON_ENV"

RUNTIME=${PLATOON_CONTAINER_RUNTIME:-/usr/bin/singularity}
MKFS_EXT3=/usr/sbin/mkfs.ext3
MEMORY_GUARD=$MILES_SRC/deploy/abci/openreward-node-memory-guard.py
ENTRYPOINT=$TOOLATHLON_SCRIPTS_ROOT/openreward-toolathlon-resilient-entrypoint.sh
STAGER=$TOOLATHLON_SCRIPTS_ROOT/openreward-toolathlon-stage-pgdata.sh
REWARD_MODE=${OPENREWARD_REWARD_MODE:-partial}
ENVIRONMENT_SET=${OPENREWARD_ENVIRONMENT_SET:-toolathlon_only}
PORT=8082; WORKER_BASE_PORT=28100; WORKERS=8; PG_PORT=5432; PG_MAX_CONNECTIONS=512
PGDATA_CONTAINER=/mnt/state/postgresql/14/main
LOCAL_ROOT=$PBS_LOCALDIR/toolathlon
STATE=$LOCAL_ROOT/state
OVERLAY=$LOCAL_ROOT/rootfs-overlay.img
OVERLAY_SIZE_GB=16   # sparse on the node-local NVMe; session-scoped MCP environments grow into it
LOCAL_SIF=$LOCAL_ROOT/images/openreward-toolathlon.sif
GUARD_ARGV=(/usr/bin/python3 -I "$MEMORY_GUARD"
  --node-limit-bytes "${OPENREWARD_NODE_MEMORY_LIMIT_BYTES:?}" --trigger-percent "${OPENREWARD_NODE_MEMORY_TRIGGER_PERCENT:?}"
  --fallback-available-bytes "${OPENREWARD_NODE_MEMORY_FALLBACK_AVAILABLE_BYTES:?}"
  --poll-seconds "${OPENREWARD_NODE_MEMORY_POLL_SECONDS:?}" --telemetry-seconds "${OPENREWARD_NODE_MEMORY_TELEMETRY_SECONDS:?}")

# ---------------------------------------------------------------- preconditions
[[ $ENVIRONMENT_SET == toolathlon_only ]] || die "OPENREWARD_ENVIRONMENT_SET must be toolathlon_only, got $ENVIRONMENT_SET"
[[ $REWARD_MODE == partial ]] || die "OPENREWARD_REWARD_MODE must be partial, got $REWARD_MODE"
[[ $RUNTIME == /* && -x $RUNTIME ]] || die "container runtime missing or not absolute: $RUNTIME"
[[ -x $MKFS_EXT3 ]] || die "mkfs.ext3 is missing: $MKFS_EXT3"
[[ -r $MEMORY_GUARD ]] || die "memory guard is missing: $MEMORY_GUARD"
[[ -r $ENTRYPOINT ]] || die "Toolathlon entrypoint is missing: $ENTRYPOINT"
[[ -r $STAGER ]] || die "PGDATA stager is missing: $STAGER"
[[ -r ${OPENREWARD_TOOLATHLON_SIF:-} ]] || die "Toolathlon SIF is unreadable: ${OPENREWARD_TOOLATHLON_SIF:-unset}"
[[ ${OPENREWARD_TOOLATHLON_SIF_SHA256:-} =~ ^[0-9a-f]{64}$ && ${OPENREWARD_TOOLATHLON_BASE_SIF_SHA256:-} =~ ^[0-9a-f]{64}$ && \
   ${OPENREWARD_TOOLATHLON_PORTABILITY_PATCH_SHA256:-} =~ ^[0-9a-f]{64}$ && -n ${OPENREWARD_TOOLATHLON_OCI_AMD64_DIGEST:-} ]] || \
  die "derived Toolathlon SIF requires exact sha256 / base / portability / OCI identities in $TOOLATHLON_ENV"
[[ ${OPENREWARD_TOOLATHLON_PLAYWRIGHT_BROWSERS_PATH:-} == /opt/ms-playwright ]] || \
  die "derived Toolathlon SIF requires PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright"
actual_sha=$(sha256sum "$OPENREWARD_TOOLATHLON_SIF" | awk '{print $1}')
[[ $actual_sha == "$OPENREWARD_TOOLATHLON_SIF_SHA256" ]] || die "Toolathlon SIF digest mismatch: $actual_sha != $OPENREWARD_TOOLATHLON_SIF_SHA256"
log "sif=$OPENREWARD_TOOLATHLON_SIF sha256=$actual_sha"
labels=$("$RUNTIME" inspect --labels "$OPENREWARD_TOOLATHLON_SIF")
grep -Fq "org.openreward.toolathlon.base-sif-sha256: $OPENREWARD_TOOLATHLON_BASE_SIF_SHA256" <<<"$labels" || die "Toolathlon derived SIF base digest mismatch"
grep -Fq "org.openreward.toolathlon.base-oci-digest: $OPENREWARD_TOOLATHLON_OCI_AMD64_DIGEST" <<<"$labels" || die "Toolathlon derived SIF base OCI digest mismatch"
grep -Fq "org.openreward.toolathlon.pbs-portability: $OPENREWARD_TOOLATHLON_PORTABILITY_PATCH_SHA256" <<<"$labels" || die "Toolathlon derived SIF portability patch mismatch"
unset labels

# ---------------------------------------------------------------- command lines (release helper lines 296-306, 360-371, 571-596)
PORTABLE_ENV=(PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright UV_PYTHON_INSTALL_DIR=/opt/uv/python UV_PYTHON_PREFERENCE=only-managed UV_PYTHON_DOWNLOADS=never)
STAGE_ARGV=("$RUNTIME" exec --fakeroot --cleanenv --no-home
  --bind "$STATE:/mnt/state,$STAGER:/app/openreward-toolathlon-stage-pgdata.sh:ro"
  "$LOCAL_SIF" /app/openreward-toolathlon-stage-pgdata.sh "$PGDATA_CONTAINER")
MKFS_ARGV=("$MKFS_EXT3" -q -F -d "$LOCAL_ROOT/overlay-layout" "$OVERLAY")
CONTAINER_ARGV=("$RUNTIME" exec --cleanenv --no-home --overlay "$OVERLAY"
  --bind "$STATE:/mnt/state,$ENTRYPOINT:/app/openreward-toolathlon-resilient-entrypoint.sh:ro"
  "$LOCAL_SIF"
  env HOME=/mnt/state/home UV_CACHE_DIR=/mnt/state/cache/uv "${PORTABLE_ENV[@]}"
      XDG_CACHE_HOME=/mnt/state/cache/xdg TMPDIR=/mnt/state/tmp VIRTUAL_ENV=/opt/venv
      PATH=/opt/venv/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin
      OPENREWARD_RUNTIME_MODE=user "OPENREWARD_REWARD_MODE=$REWARD_MODE"
      OPENREWARD_STATE_DIR=/mnt/state "OPENREWARD_PGDATA=$PGDATA_CONTAINER"
      "OPENREWARD_PG_PORT=$PG_PORT" "OPENREWARD_PG_MAX_CONNECTIONS=$PG_MAX_CONNECTIONS"
      "OPENREWARD_PORT=$PORT" "OPENREWARD_WORKER_BASE_PORT=$WORKER_BASE_PORT" "OPENREWARD_WORKERS=$WORKERS"
      OPENREWARD_WORKER_PID_ISOLATION=0
      /app/openreward-toolathlon-resilient-entrypoint.sh)
if dry; then
  echo "DRY_RUN resolved_sif: $OPENREWARD_TOOLATHLON_SIF"
  echo "DRY_RUN resolved_sif_sha256: $actual_sha"
  echo "DRY_RUN local_root: $LOCAL_ROOT (images/, state/{cache/uv,cache/xdg,home,tmp}, $SINGULARITY_CACHEDIR, $SINGULARITY_TMPDIR)"
  show guard "${GUARD_ARGV[@]}"
  show stage_pgdata "${STAGE_ARGV[@]}"
  show overlay truncate -s "${OVERLAY_SIZE_GB}G" "$OVERLAY"
  show mkfs "${MKFS_ARGV[@]}"
  show container "${CONTAINER_ARGV[@]}"
  exit 0
fi

# ---------------------------------------------------------------- node-local staging
mkdir -p "$LOCAL_ROOT/images" "$STATE/cache/uv" "$STATE/cache/xdg" "$STATE/home" "$STATE/tmp" "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"
chmod 0700 "$LOCAL_ROOT" "$LOCAL_ROOT/images" "$STATE/home" "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"
if [[ ! -f $LOCAL_SIF ]] || [[ $(sha256sum "$LOCAL_SIF" | awk '{print $1}') != "$OPENREWARD_TOOLATHLON_SIF_SHA256" ]]; then
  log "staging the SIF to $LOCAL_SIF"
  stage_tmp=$LOCAL_SIF.partial.$$
  cp "$OPENREWARD_TOOLATHLON_SIF" "$stage_tmp"
  [[ $(sha256sum "$stage_tmp" | awk '{print $1}') == "$OPENREWARD_TOOLATHLON_SIF_SHA256" ]] || die "node-local Toolathlon SIF changed while staging"
  chmod 0400 "$stage_tmp"; mv "$stage_tmp" "$LOCAL_SIF"
fi
log "staging PGDATA on $(hostname -s)"
"${STAGE_ARGV[@]}"
mkdir -p "$LOCAL_ROOT/overlay-layout/upper" "$LOCAL_ROOT/overlay-layout/work"
truncate -s "${OVERLAY_SIZE_GB}G" "$OVERLAY"
"${MKFS_ARGV[@]}"
rmdir "$LOCAL_ROOT/overlay-layout/upper" "$LOCAL_ROOT/overlay-layout/work" "$LOCAL_ROOT/overlay-layout"

# ---------------------------------------------------------------- supervision (child or guard dies -> TERM the other, exit 1)
container_pid=; guard_pid=; cleanup_started=0
port_closed() { ! ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)$PORT\$"; }
cleanup() {
  local st=${1:-$?} t
  ((cleanup_started)) && return; cleanup_started=1; trap - EXIT INT TERM
  log "cleanup status=$st"
  if [[ -n $container_pid ]] && kill -0 "$container_pid" 2>/dev/null; then
    kill -TERM "$container_pid" 2>/dev/null || true
    for t in $(seq 1 90); do kill -0 "$container_pid" 2>/dev/null || break; sleep 1; done
    if kill -0 "$container_pid" 2>/dev/null; then log "container ignored TERM for 90 s: KILL"; kill -KILL "$container_pid" 2>/dev/null || true; st=1; fi
    wait "$container_pid" 2>/dev/null || true
  fi
  if [[ -n $guard_pid ]] && kill -0 "$guard_pid" 2>/dev/null; then
    kill -TERM "$guard_pid" 2>/dev/null || true
    for t in $(seq 1 10); do kill -0 "$guard_pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$guard_pid" 2>/dev/null && { kill -KILL "$guard_pid" 2>/dev/null || true; st=1; }
    wait "$guard_pid" 2>/dev/null || true
  fi
  if port_closed; then log "127.0.0.1:$PORT closed"; else log "ERROR: port $PORT still listening after cleanup (the launcher's leftover sweep follows)"; st=1; fi
  log "exit status=$st"; exit "$st"
}
trap cleanup EXIT; trap 'cleanup 143' TERM; trap 'cleanup 130' INT

log "starting memory guard limit_bytes=$OPENREWARD_NODE_MEMORY_LIMIT_BYTES trigger_percent=$OPENREWARD_NODE_MEMORY_TRIGGER_PERCENT"
"${GUARD_ARGV[@]}" & guard_pid=$!
log "starting Toolathlon host=$(hostname -s) port=$PORT workers=$WORKER_BASE_PORT-$((WORKER_BASE_PORT + WORKERS - 1)) reward_mode=$REWARD_MODE image_sha256=$OPENREWARD_TOOLATHLON_SIF_SHA256"
"${CONTAINER_ARGV[@]}" & container_pid=$!

next_ps=$((SECONDS + 300))
while kill -0 "$container_pid" 2>/dev/null && kill -0 "$guard_pid" 2>/dev/null; do
  if (( SECONDS >= next_ps )); then
    log "top rss (kB, comm):"; ps -eo rss,comm --sort=-rss | head -8; next_ps=$((SECONDS + 300))
  fi
  sleep 2
done
if ! kill -0 "$guard_pid" 2>/dev/null; then
  gst=0; wait "$guard_pid" || gst=$?; log "ERROR: memory guard exited status=$gst (2 = contract, 3 = source, 4 = threshold)"
else
  cst=0; wait "$container_pid" || cst=$?; log "ERROR: Toolathlon container exited status=$cst"
fi
cleanup 1
