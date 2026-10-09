#!/bin/bash
# Serve the gateway and train number-search against it in one job. Submit from the repository root:
#   sbatch deploy/babel/train.sbatch [client options], e.g. --steps 2 for a smoke test.
# The only file that uses both deploy/babel/server.py and deploy/babel/client/.
# Export WANDB_API_KEY before submission, or have an existing W&B login.
#SBATCH --job-name=miles-babel
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=babel-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
WORK="${SLURM_SUBMIT_DIR}/.cache/babel"
NAME="number-search-${SLURM_JOB_ID}"
ENDPOINT="${WORK}/endpoints/${NAME}.endpoint"

# Full fine-tuning of Qwen3-4B on two learner and two rollout GPUs.
srun python deploy/babel/server.py --work-dir "${WORK}" --name "${NAME}" --use-fa2 \
  --mode full --tp 1 --max-tokens-per-gpu 1024 \
  --extra-args "--chunked-optimizer-state-offload --optimizer-state-offload-chunk-size-mb 64 --sglang-device cuda" &
SERVER=$!
trap 'kill -TERM "${SERVER}" 2>/dev/null; wait "${SERVER}" || true' EXIT

until [[ -s "${ENDPOINT}" ]]; do
  kill -0 "${SERVER}" 2>/dev/null || { echo "The server exited before becoming healthy." >&2; exit 1; }
  sleep 10
done
python deploy/babel/client/run.py --work-dir "${WORK}/platoon-client" --log-dir "${WORK}/output" \
  --endpoint-file "${ENDPOINT}" "$@"
