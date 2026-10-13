#!/bin/bash
# Cache the pinned Miles image as a local SIF. Submit from the repository root:
#   sbatch deploy/babel/image.sbatch --output <sif> [--scratch DIR]
# Pass the SIF to server.py with --image.
#SBATCH --job-name=miles-image
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=babel-image-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
exec python deploy/babel/image.py "$@"
