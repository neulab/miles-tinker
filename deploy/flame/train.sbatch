#!/bin/bash
# Submit from the repository root: sbatch deploy/flame/train.sbatch
# Set the QoS to your assigned Flame QoS and adjust resources for your job.
#SBATCH --job-name=miles-flame
#SBATCH --partition=flame
#SBATCH --qos=flame-16gpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=flame-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# Export WANDB_API_KEY before submission, or have an existing W&B login.
# Keep the environment, downloads and training outputs across allocations.
exec srun python deploy/flame/run.py --work-dir "${SLURM_SUBMIT_DIR}/.cache/flame" "$@"
