#!/bin/bash
# Stage everything a FLAME OpenHands RL run reads onto node-local /tmp.
#
#   $ROOT/images/miles-sglang.sif          server/learner container
#   $ROOT/work/models/Qwen3.5-4B/          HF checkpoint (run.py's /root/models)
#   $ROOT/tmax15k/selected.json, tasks/    task inputs
#   $ROOT/tmax15k/sifs-oh/*.sif            one OpenHands-ready image per task
#
# Sources, in order: an existing copy on this node (LOCAL_SRC, hardlinked, so
# instant and free), then the GCS mirror. Each piece is skipped once present, so
# rerunning resumes an interrupted stage. The pool is ~625 GB; /tmp on a FLAME
# node is a 5.9 TB NVMe array.
#
#   ROOT=/tmp/$USER-oh-rl bash deploy/flame/stage.sh
set -euo pipefail

ROOT="${ROOT:-/tmp/$USER-oh-rl}"
GCS="${GCS:?set GCS to a gs:// prefix you can write to}"
POOL="$GCS/pools/tmax-expanded-v1"
LOCAL_SRC="${LOCAL_SRC:-}"
MODEL=Qwen3.5-4B

mkdir -p "$ROOT/images" "$ROOT/work/models" "$ROOT/tmax15k"
# gcloud keeps resumable-download state in ~/.config/gcloud, which is shared
# NFS, under names derived from the source and destination path. Two nodes
# staging at once use identical names and delete each other's files mid-copy
# (job 155331), so keep that state on this node.
export CLOUDSDK_STORAGE_TRACKER_FILES_DIRECTORY="$ROOT/.gcloud-tracker"
same_fs() { [ -e "$1" ] && [ "$(stat -c %d "$1")" = "$(stat -c %d "$ROOT")" ]; }

echo "[stage] root=$ROOT"

# STAGE_SERVER=0 (client-only runs) skips the server image and the model.
# --- Miles image ------------------------------------------------------------
if [ "${STAGE_SERVER:-1}" = 0 ]; then
    echo "[stage] client only: no server image or model"
elif [ ! -s "$ROOT/images/miles-sglang.sif" ]; then
    if same_fs "$LOCAL_SRC/images/miles-sglang.sif"; then
        ln "$LOCAL_SRC/images/miles-sglang.sif" "$ROOT/images/miles-sglang.sif"
    else
        gcloud storage cp "$GCS/images/miles-sglang.sif" "$ROOT/images/miles-sglang.sif.partial"
        mv "$ROOT/images/miles-sglang.sif.partial" "$ROOT/images/miles-sglang.sif"
    fi
fi
echo "[stage] image ok"

# --- Model weights ----------------------------------------------------------
if [ "${STAGE_SERVER:-1}" = 0 ]; then
    :
elif [ ! -f "$ROOT/work/models/$MODEL/config.json" ]; then
    if same_fs "$LOCAL_SRC/miles-work/models/$MODEL/config.json"; then
        cp -al "$LOCAL_SRC/miles-work/models/$MODEL" "$ROOT/work/models/$MODEL"
    else
        tarball="$ROOT/work/models/$MODEL.tar"
        [ -s "$tarball" ] || gcloud storage cp "$GCS/models/$MODEL.tar" "$tarball"
        tar xf "$tarball" -C "$ROOT/work/models"
        rm -f "$tarball"
    fi
fi
echo "[stage] model ok"

# --- Task inputs ------------------------------------------------------------
if [ ! -f "$ROOT/tmax15k/selected.json" ] || [ ! -d "$ROOT/tmax15k/tasks" ]; then
    archive="$ROOT/tmax15k/task-inputs.tar.gz"
    if same_fs "$LOCAL_SRC/task-inputs.tar.gz"; then
        cp "$LOCAL_SRC/task-inputs.tar.gz" "$archive"
    else
        gcloud storage cp "$POOL/task-inputs.tar.gz" "$archive"
    fi
    tar xzf "$archive" -C "$ROOT/tmax15k"
    rm -f "$archive"
fi
echo "[stage] task inputs ok ($(ls "$ROOT/tmax15k/tasks" | wc -l) task dirs)"

# --- Task images ------------------------------------------------------------
mkdir -p "$ROOT/tmax15k/sifs-oh"
if [ -d "$LOCAL_SRC/tmax15k/sifs-oh" ] && same_fs "$LOCAL_SRC/tmax15k/sifs-oh"; then
    for sif in "$LOCAL_SRC"/tmax15k/sifs-oh/*.sif; do
        [ -e "$ROOT/tmax15k/sifs-oh/${sif##*/}" ] || ln "$sif" "$ROOT/tmax15k/sifs-oh/"
    done
fi
# Fetch only the images not yet here. rsync would compare every object, and a
# hardlinked copy's mtimes differ from GCS, so it could re-hash ~625 GB.
missing_list="$ROOT/tmax15k/.sifs-missing"
python3 - "$ROOT/tmax15k" "$POOL/sifs-oh" > "$missing_list" <<'PY'
import json, sys
from pathlib import Path
root, remote = Path(sys.argv[1]), sys.argv[2]
for name in sorted({e["sif_name"] for e in json.loads((root / "selected.json").read_text())}):
    if not (root / "sifs-oh" / name).is_file():
        print(f"{remote}/{name}")
PY
if [ -s "$missing_list" ]; then
    echo "[stage] downloading $(wc -l < "$missing_list") task images from GCS"
    # Downloads land under a temporary name, so a failed copy can simply be
    # rerun; -n skips the images that already arrived.
    for attempt in 1 2 3; do
        gcloud storage cp -n -I "$ROOT/tmax15k/sifs-oh/" < "$missing_list" && break
        [ "$attempt" = 3 ] && exit 1
        echo "[stage] download attempt $attempt failed; retrying"
    done
fi
rm -f "$missing_list"

python3 - "$ROOT/tmax15k" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
entries = json.loads((root / "selected.json").read_text())
missing = sorted({e["sif_name"] for e in entries if not (root / "sifs-oh" / e["sif_name"]).is_file()})
no_task = [e["task_id"] for e in entries if not (root / "tasks" / e["task_id"]).is_dir()]
print(f"[stage] pool: {len(entries)} tasks, {len(missing)} missing images, {len(no_task)} missing task dirs")
if missing or no_task:
    sys.exit(f"[stage] incomplete pool, e.g. {(missing or no_task)[0]}")
PY
# --- Extra pools (EXTRA_POOLS="tb2-v1 ...") -----------------------------------
# Merged into the same data dir: task dirs copied in, selected.json entries
# appended, images fetched from the extra pool's own sifs-oh. Used for eval-only
# sets such as Terminal-Bench 2.0 (<science-agent>/experiments/qwen38_27b_tinker_amd/pool/build_tb2_pool.py); tasks.py
# keeps an eval list out of the training split.
for extra in ${EXTRA_POOLS:-}; do
    dir="$ROOT/pools/$extra"
    if [ ! -f "$dir/selected.json" ]; then
        mkdir -p "$dir"
        gcloud storage cp "$GCS/pools/$extra/task-inputs.tar.gz" "$dir/task-inputs.tar.gz"
        tar xzf "$dir/task-inputs.tar.gz" -C "$dir"
        rm -f "$dir/task-inputs.tar.gz"
    fi
    python3 - "$ROOT/tmax15k" "$dir" "$GCS/pools/$extra/sifs-oh" > "$ROOT/tmax15k/.extra-missing" <<'PY'
import json, shutil, sys
from pathlib import Path
root, extra, remote = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
main = json.loads((root / "selected.json").read_text())
have = {e["task_id"] for e in main}
added = [e for e in json.loads((extra / "selected.json").read_text()) if e["task_id"] not in have]
for e in added:
    dst = root / "tasks" / e["task_id"]
    if not dst.exists():
        shutil.copytree(extra / "tasks" / e["task_id"], dst)
if added:
    (root / "selected.json").write_text(json.dumps(main + added))
for e in json.loads((extra / "selected.json").read_text()):
    if not (root / "sifs-oh" / e["sif_name"]).is_file():
        print(f"{remote}/{e['sif_name']}")
print(f"[stage] {extra.name}: merged {len(added)} new tasks", file=sys.stderr)
PY
    if [ -s "$ROOT/tmax15k/.extra-missing" ]; then
        echo "[stage] $extra: downloading $(sort -u "$ROOT/tmax15k/.extra-missing" | wc -l) images"
        for attempt in 1 2 3; do
            sort -u "$ROOT/tmax15k/.extra-missing" | gcloud storage cp -n -I "$ROOT/tmax15k/sifs-oh/" && break
            [ "$attempt" = 3 ] && exit 1
        done
    fi
    rm -f "$ROOT/tmax15k/.extra-missing"
    echo "[stage] $extra ok"
done
echo "[stage] done"
