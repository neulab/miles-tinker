#!/bin/bash
# Submit from the repository root: sbatch deploy/babel/train.sbatch
# Adjust the resource requests below for your job.
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

# Export WANDB_API_KEY before submission, or have an existing W&B login.
# Keep the environment, downloads and training outputs across allocations.
exec srun python deploy/babel/run.py --work-dir "${SLURM_SUBMIT_DIR}/.cache/babel" "$@"
