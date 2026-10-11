#!/bin/bash
# Run the number-search client against a running gateway. Submit from the repository root:
#   sbatch deploy/babel/client/client.sbatch --endpoint-file .cache/babel/endpoints/<name>.endpoint [options]
# The client needs no GPU. Export WANDB_API_KEY first or have an existing W&B login.
#SBATCH --job-name=miles-client
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=babel-client-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
exec python deploy/babel/client/run.py --work-dir "${SLURM_SUBMIT_DIR}/.cache/babel-client" --wait 3600 "$@"
