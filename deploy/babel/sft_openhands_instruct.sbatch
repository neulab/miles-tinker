#!/bin/bash
# SFT Qwen3-4B-Instruct-2507 on solved openhands trajectories, as a behaviour
# prior before RL.
#
# Why. Job 10635972 trained this model for 70 RL steps with a flat trend
# (slope +0.00006/step, p~0.81) and held-out solve rate stuck at 15-20%. The
# deficit is not knowledge, it is that the policy stops acting: median
# trajectory 11-15 steps against Qwen3.5's 56-80, and ~50% of GRPO groups
# degenerate because every member scores alike. This teaches the missing
# behaviour -- keep working, then call finish -- from trajectories that did it.
#
# Data: 353 trajectories pooled from 7 Qwen3.5-4B runs, each solved (verifier
# reward 1.0), finish-terminated, and filtered to fit the 32768 context. The
# source model reasons inside <think>; Instruct-2507's template has no think
# branch, so the reasoning is folded into visible content -- which is what
# Instruct does natively. 141 tasks, median 21 assistant turns, 8,912 turns.
# Verified: tool-call sequences match the raw event streams 150/150.
#
#   sbatch sft_openhands_instruct.sbatch
#
#SBATCH --job-name=oh-sft
# general + normal qos rather than aasai: this needs no reserved node and
# `general` has far more free capacity.
#
# 4 GPUs at tp=4, which is a memory requirement rather than a speed choice.
# These are multi-turn agent trajectories with a median of 20.7k tokens, and a
# sample is processed whole -- its logits are materialised at seq_len x 151936
# vocab x 4 bytes, so one median row is 11.7 GiB in a single tensor. Jobs
# 10644106 and 10644152 both OOMed at tp=1 proving it: lowering
# --max-tokens-per-gpu from 9216 to 4096 changed nothing, because that budget
# bounds how many samples are packed, not how long one sample may be.
# Tensor parallelism shards the vocab dimension and the optimizer state with it:
#
#     tp=1   static 59.6G   logits 11.73G   -> OOM on a 47.4G card
#     tp=2   static 29.8G   logits  5.87G   -> ~40G, tight
#     tp=4   static 14.9G   logits  2.93G   -> ~22G, comfortable
#
# The RL runs use --tp 4 for the same reason.
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --gres=gpu:4
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --output=/data/user_data/%u/miles-babel/slurm-sft-%j.out
#SBATCH --open-mode=append

set -euo pipefail
echo "node=$(hostname)  job=$SLURM_JOB_ID  start=$(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

REPO=$HOME/miles-tinker
WORK=/data/user_data/$USER/miles-babel
MODEL=Qwen3-4B-Instruct-2507
MODEL_TYPE=qwen3-4B-Instruct-2507
DATASET=oh_solved_finish_fit32k.jsonl      # must live in $WORK/data
GPUS=4                                     # must match --gres above
OUT_REL=sft/$SLURM_JOB_ID                  # under $WORK/output
IMAGE=$WORK/apptainer-cache/cache/oci-tmp/3828109b1b56225bfba11f4564640b014ec652f240f62a4f572e2df5aeff2dd3

mkdir -p "$WORK/output/$OUT_REL" "$WORK/home" "$WORK/tmp"
test -f "$WORK/data/$DATASET" || { echo "missing dataset: $WORK/data/$DATASET" >&2; exit 1; }
echo "dataset: $WORK/data/$DATASET ($(wc -l < "$WORK/data/$DATASET") rows)"

# Megatron architecture flags, resolved on the host. scripts/models/*.py define a
# model_args() function and print nothing when executed, so they have to go
# through load_model_args rather than being run as scripts.
MODEL_ARGS=$(cd "$REPO" && python3 -c "
import sys; sys.path.insert(0, 'miles/utils/external_utils')
from model_args_utils import shell_safe_model_args
print(shell_safe_model_args('$MODEL_TYPE'))
")
test -n "$MODEL_ARGS" || { echo "could not resolve model args for $MODEL_TYPE" >&2; exit 1; }
read -ra MODEL_ARR <<< "$MODEL_ARGS"

# Container invocation copied from deploy/babel/run.py: the deps (ray, megatron,
# transformer-engine) live in a venv at $WORK/venv, which the BOOTSTRAP below
# builds-or-reuses and activates. The image's own python has ray but not the
# rest, so running train_async.py against it fails on import.
for kv in \
  "PYTHONPATH=/root/miles:/root/Megatron-LM" "PYTHONNOUSERSITE=1" "PYTHONUNBUFFERED=1" \
  "UV_CACHE_DIR=/run/miles/.cache/uv" "HF_HOME=/run/miles/hf" "XDG_CACHE_HOME=/run/miles/.cache" \
  "TMPDIR=/tmp" "NCCL_CUMEM_HOST_ENABLE=0" "NETRC=/run/miles/wandb.netrc" \
  "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"; do
    export "APPTAINERENV_${kv%%=*}=${kv#*=}"
done
export APPTAINER_CACHEDIR=$WORK/apptainer-cache
# W&B auth: --contain hides $HOME, so the netrc is bound in and NETRC points at it.
if [[ -f "$HOME/.netrc" ]]; then
    cp -f "$HOME/.netrc" "$WORK/wandb.netrc" && chmod 600 "$WORK/wandb.netrc"
    echo "wandb: netrc staged at \$WORK/wandb.netrc"
else
    echo "wandb: no ~/.netrc; set WANDB_API_KEY or expect offline mode" >&2
fi
[[ -n "${WANDB_API_KEY:-}" ]] && export APPTAINERENV_WANDB_API_KEY="$WANDB_API_KEY"

APPT=(apptainer exec --nv --contain --pid --ipc
      --home "$WORK/home:$HOME"
      --no-mount cwd --no-eval
      --bind "$REPO:/root/miles"
      --bind "$WORK:/run/miles"
      --bind "$WORK/tmp:/tmp"
      --bind "$WORK/models:/root/models"
      --bind "$WORK/data:/root/datasets"
      --bind "$WORK/output:/root/shared_data"
      --pwd /root/miles)

BOOTSTRAP='set -e
python deploy/babel/setup.py --venv /run/miles/venv
source /run/miles/venv/bin/activate
export CUDNN_PATH="${CUDNN_PATH:-$(python -c "import nvidia; print(next(iter(nvidia.__path__)))")}"
export LD_LIBRARY_PATH="$(python -c "import nvidia, pathlib; print(\":\".join(str(p) for root in nvidia.__path__ for p in pathlib.Path(root).glob(\"**/lib\")))"):${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/root/miles:/root/Megatron-LM:$(python -c "import sysconfig; print(sysconfig.get_path(\"purelib\"))")"
exec "$@"'

# Step 1: HF -> Megatron torch_dist, once. --ref-load needs it; the RL launcher
# never produced one. Skipped on resubmit.
if [[ ! -d "$WORK/models/${MODEL}_torch_dist" ]]; then
    echo "converting $MODEL -> ${MODEL}_torch_dist on $GPUS GPUs"
    "${APPT[@]}" "$IMAGE" bash -c "$BOOTSTRAP" bash \
        torchrun --nproc-per-node "$GPUS" tools/convert_hf_to_torch_dist.py \
          "${MODEL_ARR[@]}" \
          --hf-checkpoint "/root/models/$MODEL" \
          --save "/root/models/${MODEL}_torch_dist" \
        2>&1 | tee "$WORK/output/$OUT_REL/convert.log"
else
    echo "reusing existing ${MODEL}_torch_dist"
fi

# Step 2: SFT, through deploy/babel/sft_instruct_launcher.py rather than
# train_async.py directly -- miles runs training as a ray job (execute_train
# starts the cluster and submits), so a bare invocation has nothing to attach to.
"${APPT[@]}" "$IMAGE" bash -c "$BOOTSTRAP" bash \
    python deploy/babel/sft_instruct_launcher.py \
      --num-gpus-per-node "$GPUS" \
      --model-dir /root/models \
      --model-name $MODEL \
      --megatron-model-type $MODEL_TYPE \
      --prompt-data /root/datasets/$DATASET \
      --output-dir /root/shared_data/$OUT_REL \
      --num-epoch 3 \
      --batch-size 16 \
      --max-tokens-per-gpu 4096 \
      --tensor-model-parallel-size 4 \
      --seq-length 32768 \
      --lr 1e-5 \
      --wandb-project tmax-15k-sft \
      --wandb-group "oh-sft-$SLURM_JOB_ID" \
    2>&1 | tee "$WORK/output/$OUT_REL/train.log"

echo "done=$(date -Is); checkpoints in $WORK/output/$OUT_REL/checkpoints"
