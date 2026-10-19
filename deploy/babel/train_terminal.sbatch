#!/bin/bash
# Terminal-agent RL on Babel. Submit from the repository root:
#
#   sbatch deploy/babel/train_terminal.sbatch
#   TASK=tmax-15k sbatch deploy/babel/train_terminal.sbatch --steps 50
#
# Arguments after the template name are passed to deploy/babel/run.py.
# TASK selects the recipe (litecoder-terminal, tmax-15k); WORK_DIR and DATA_DIR
# override the storage locations, and DATA_DIR defaults to the task pool.
#SBATCH --job-name=terminal-rl
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=terminal-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# The model cache, checkpoints and image pool are far too large for a home quota.
TASK="${TASK:-litecoder-terminal}"
WORK_DIR="${WORK_DIR:-/data/user_data/${USER}/miles-babel}"
case "${TASK}" in
    tmax-15k) DEFAULT_POOL="/data/user_data/${USER}/tmax15k" ;;
    *)        DEFAULT_POOL="/data/user_data/${USER}/litecoder" ;;
esac
DATA_DIR="${DATA_DIR:-${DEFAULT_POOL}}"

if [ ! -f "${DATA_DIR}/selected.json" ]; then
    echo "No staged task pool at ${DATA_DIR}." >&2
    echo "Stage one first, with the script for this task:" >&2
    echo "  uv run <science-agent>/experiments/qwen38_27b_tinker_amd/pool/prepare_litecoder.py --output ${DATA_DIR}   # litecoder-terminal" >&2
    echo "  uv run <science-agent>/experiments/qwen38_27b_tinker_amd/pool/prepare_tmax.py --output ${DATA_DIR}        # tmax-15k" >&2
    exit 1
fi

# Rollouts need Apptainer on the host, which is why the client runs outside the
# Miles container. Fail early rather than after the model has loaded.
if ! command -v apptainer >/dev/null 2>&1; then
    echo "apptainer is not on PATH; rollouts cannot start their sandboxes." >&2
    exit 1
fi

# Sandbox scratch on node-local NVMe: overlay writes and image reads are hot,
# and /data is NFS. Falls back to the work directory if /scratch is unusable.
SCRATCH="/scratch/${USER}/apptainer-tmp"
if mkdir -p "${SCRATCH}" 2>/dev/null; then
    export APPTAINER_TMPDIR="${SCRATCH}"
else
    export APPTAINER_TMPDIR="${WORK_DIR}/tmp"
    mkdir -p "${APPTAINER_TMPDIR}"
fi

# Slurm kills the step on scancel or timeout. Reap rollout sandboxes so their
# tmpfs overlays do not outlive the job. Note: no `exec srun`, or this never runs.
cleanup() {
    local instance
    for instance in $(apptainer instance list 2>/dev/null | awk 'NR > 1 && $1 ~ /^(litecoder|terminal)-/ {print $1}'); do
        echo "stopping leftover sandbox: ${instance}" >&2
        apptainer instance stop "${instance}" >/dev/null 2>&1 || true
    done
}
trap cleanup EXIT

echo "task:      ${TASK}"
echo "work dir:  ${WORK_DIR}"
echo "task pool: ${DATA_DIR} ($(ls "${DATA_DIR}/sifs" | wc -l) images)"
echo "scratch:   ${APPTAINER_TMPDIR}"

srun python deploy/babel/run.py \
    --task "${TASK}" \
    --work-dir "${WORK_DIR}" \
    --data-dir "${DATA_DIR}" \
    "$@"
