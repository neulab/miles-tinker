#!/bin/bash
# Convert an SFT Megatron checkpoint back to HF format, so it can be served and
# used as the starting weights for RL.
#
# tools/convert_torch_dist_to_hf.py does NOT work on these checkpoints -- it
# expects a classic Megatron dist-ckpt with common.pt, while miles writes bare
# torch.distributed.checkpoint shards (__N_0.distcp + .metadata). tools/
# convert_to_hf.py is the one that matches: it rebuilds the model through
# miles' own Megatron loader and then writes HF, which means it needs the same
# distributed setup and the same tp as the run that produced the checkpoint.
#
#   ITER=65 sbatch sft_to_hf.sbatch      # default: whatever latest points at
#
#SBATCH --job-name=sft2hf
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --output=/data/user_data/%u/miles-babel/slurm-sft2hf-%j.out
#SBATCH --open-mode=append

set -euo pipefail
echo "node=$(hostname)  job=$SLURM_JOB_ID  start=$(date -Is)"

REPO=$HOME/miles-tinker
WORK=/data/user_data/$USER/miles-babel
BASE=Qwen3-4B-Instruct-2507          # supplies config + tokenizer
SFT_JOB=10644250
ITER=${ITER:-}                        # empty => latest_checkpointed_iteration.txt
GPUS=4                                # must equal the SFT run's tp
TP=4
OUTNAME=${OUTNAME:-Qwen3-4B-Instruct-SFT}
IMAGE=$WORK/apptainer-cache/cache/oci-tmp/3828109b1b56225bfba11f4564640b014ec652f240f62a4f572e2df5aeff2dd3

CKPT=$WORK/output/sft/$SFT_JOB/checkpoints
test -d "$CKPT" || { echo "no checkpoint dir: $CKPT" >&2; exit 1; }
echo "checkpoint dir: $CKPT  (latest=$(cat "$CKPT/latest_checkpointed_iteration.txt" 2>/dev/null))"
STEP_ARG=()
if [[ -n "$ITER" ]]; then STEP_ARG=(--ref-ckpt-step "$ITER"); OUTNAME="${OUTNAME}-iter$ITER"; fi
echo "output: $WORK/models/$OUTNAME"

MODEL_ARGS=$(cd "$REPO" && python3 -c "
import sys; sys.path.insert(0, 'miles/utils/external_utils')
from model_args_utils import shell_safe_model_args
print(shell_safe_model_args('qwen3-4B-Instruct-2507'))
")
test -n "$MODEL_ARGS" || { echo "could not resolve model args" >&2; exit 1; }
read -ra MODEL_ARR <<< "$MODEL_ARGS"

for kv in \
  "PYTHONPATH=/root/miles:/root/Megatron-LM" "PYTHONNOUSERSITE=1" "PYTHONUNBUFFERED=1" \
  "UV_CACHE_DIR=/run/miles/.cache/uv" "HF_HOME=/run/miles/hf" "XDG_CACHE_HOME=/run/miles/.cache" \
  "TMPDIR=/tmp" "NCCL_CUMEM_HOST_ENABLE=0" "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"; do
    export "APPTAINERENV_${kv%%=*}=${kv#*=}"
done
export APPTAINER_CACHEDIR=$WORK/apptainer-cache

APPT=(apptainer exec --nv --contain --pid --ipc
      --home "$WORK/home:$HOME" --no-mount cwd --no-eval
      --bind "$REPO:/root/miles" --bind "$WORK:/run/miles" --bind "$WORK/tmp:/tmp"
      --bind "$WORK/models:/root/models" --bind "$WORK/output:/root/shared_data"
      --pwd /root/miles)

BOOTSTRAP='set -e
python deploy/babel/setup.py --venv /run/miles/venv
source /run/miles/venv/bin/activate
export CUDNN_PATH="${CUDNN_PATH:-$(python -c "import nvidia; print(next(iter(nvidia.__path__)))")}"
export LD_LIBRARY_PATH="$(python -c "import nvidia, pathlib; print(\":\".join(str(p) for root in nvidia.__path__ for p in pathlib.Path(root).glob(\"**/lib\")))"):${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/root/miles:/root/Megatron-LM:$(python -c "import sysconfig; print(sysconfig.get_path(\"purelib\"))")"
exec "$@"'

# convert_to_hf.py goes through megatron_utils.init, so it is a torchrun job with
# the same parallelism the checkpoint was written at (tp=4 here).
"${APPT[@]}" "$IMAGE" bash -c "$BOOTSTRAP" bash \
  torchrun --nproc-per-node "$GPUS" tools/convert_to_hf.py \
    "${MODEL_ARR[@]}" \
    --hf-checkpoint "/root/models/$BASE" \
    --ref-load "/root/shared_data/sft/$SFT_JOB/checkpoints" \
    "${STEP_ARG[@]}" \
    --output-dir "/root/models/$OUTNAME" \
    --tensor-model-parallel-size "$TP" \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "$GPUS" \
    --num-gpus-per-node "$GPUS" \
    --attention-backend flash \
    --vocab-size 151936

echo "done=$(date -Is)"
ls -la "$WORK/models/$OUTNAME" 2>/dev/null | head
