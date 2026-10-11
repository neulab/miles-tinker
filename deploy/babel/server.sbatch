#!/bin/bash
# Serve the Tinker gateway until the time limit or scancel. Submit from the repository root:
#   sbatch deploy/babel/server.sbatch [server.py options] [gateway options]
# The endpoint appears in .cache/babel/endpoints/<name>.endpoint once the gateway is healthy.
# Adjust the resource requests below; keep one task per node.
#SBATCH --job-name=miles-gateway
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=babel-gateway-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
exec srun --kill-on-bad-exit=1 python deploy/babel/server.py --work-dir "${SLURM_SUBMIT_DIR}/.cache/babel" "$@"
