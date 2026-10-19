#!/usr/bin/env bash
# Mirror a training run's resumable state from node-local /tmp to GCS.
#
#   persist_run.sh <run dir> [interval seconds]       # loop until killed
#   PERSIST_ONCE=1 persist_run.sh <run dir>             # one pass (exit trap)
#
# /tmp is private to the Slurm job and deleted when it ends, so without this a
# cancelled or crashed run cannot be resumed. Each pass uploads, to
# $GCS/runs/<run id>/ with the same layout as the run directory:
#   * the newest checkpoint named by the client's checkpoints.jsonl
#     (checkpoints/model-*/weights/<name>, ~27 GB for Qwen3.5-4B: weights,
#     FP32 masters and Adam state), and deletes older ones from GCS so at most
#     PERSIST_KEEP (default 2) remain;
#   * checkpoints.jsonl itself, after its checkpoint, so the copy on GCS never
#     names a checkpoint that is not there yet (resume reads its last line);
#   * the small files: rollouts, wandb, logs, checkpoints/perf.jsonl.
# deploy/flame/train_openhands.sbatch pulls a run back with RESUME_RUN_ID when its
# directory is not on the node.
set -uo pipefail

RUN="${1:?usage: persist_run.sh <run dir> [interval seconds]}"
INTERVAL="${2:-600}"
KEEP="${PERSIST_KEEP:-2}"
GCS="${GCS:?set GCS to a gs:// prefix you can write to}"
RUN="${RUN%/}"
DEST="$GCS/runs/$(basename "$RUN")"
export CLOUDSDK_STORAGE_TRACKER_FILES_DIRECTORY="${CLOUDSDK_STORAGE_TRACKER_FILES_DIRECTORY:-$(dirname "$RUN")/.gcloud-tracker}"

log() { echo "[persist $(date +%H:%M:%S)] $*"; }

one_pass() {
    local index ckpt rel uploaded
    index=$(ls "$RUN"/*/*/checkpoints.jsonl 2>/dev/null | head -1)
    # Small, growing files first: cheap, and useful even without a checkpoint.
    # gcloud full-matches each alternative against the path relative to $RUN.
    gcloud storage rsync -r -q --exclude='checkpoints/model-[^/]+/weights/.*|.*checkpoints\.jsonl|\.persisted' \
        "$RUN" "$DEST" >/dev/null 2>&1 || log "small-file sync failed"
    [[ -n "$index" && -s "$index" ]] || return 0
    # tinker://model-X/weights/NAME -> checkpoints/model-X/weights/NAME
    ckpt=$(tail -1 "$index" | python3 -c 'import json,sys; print(json.load(sys.stdin)["state_path"])')
    rel="checkpoints/${ckpt#tinker://}"
    if [[ ! -d "$RUN/$rel" ]]; then
        log "newest checkpoint $ckpt not at $RUN/$rel; skipping"
        return 0
    fi
    uploaded="$RUN/.persisted"
    if ! grep -qxF "$rel" "$uploaded" 2>/dev/null; then
        log "uploading $rel ($(du -sh "$RUN/$rel" | cut -f1))"
        if ! gcloud storage rsync -r -q "$RUN/$rel" "$DEST/$rel" >/dev/null 2>&1; then
            log "upload of $rel failed; will retry"
            return 0
        fi
        echo "$rel" >>"$uploaded"
        log "uploaded $rel"
    fi
    # Index last, relative path kept: <client dir>/<run id>/checkpoints.jsonl.
    gcloud storage cp -q "$index" "$DEST/${index#"$RUN"/}" >/dev/null 2>&1 || log "index upload failed"
    # Keep only the newest $KEEP checkpoints on GCS.
    local old
    for old in $(head -n -"$KEEP" "$uploaded" 2>/dev/null); do
        gcloud storage rm -r -q "$DEST/$old" >/dev/null 2>&1 && log "removed old $old from GCS"
        sed -i "\#^$old\$#d" "$uploaded"
    done
}

if [[ "${PERSIST_ONCE:-0}" == 1 ]]; then
    one_pass
    exit 0
fi
log "mirroring $RUN to $DEST every ${INTERVAL}s"
while true; do
    one_pass
    sleep "$INTERVAL"
done
