#!/bin/bash
# Tinker server for OpenHands terminal RL on one AMD MI350 node (8 GPUs).
#
# Only the trainer and the SGLang engines run here; the Platoon client and its
# Apptainer sandboxes run on FLAME (deploy/flame/train_openhands.sbatch with
# TINKER_BASE_URL), next to the 650 GB task pool. The two talk through the
# authenticated Modal relay that tools/amd_tinker.py starts.
#
# Same model and trainer settings as the single-node FLAME run (4 learner GPUs at
# TP 4, 4 rollout engines, 98k context; the trainer flags of
# deploy/flame/train_openhands.sbatch and deploy/babel/run.py), except:
#   * Qwen3.5 loads from a converted Megatron checkpoint (raw mode) instead of
#     through HF Bridge; that is how tools/amd_tinker.py serves Qwen3.5 on MI350.
#   * --balance-data is on (the AMD launcher always passes it).
#
# One-time setup on an MI350 node (see docs/platforms/amd-tinker.md):
#   export MILES_AMD_ROOT=$WORK/miles_artifacts
#   python3 tools/amd_tinker.py setup --model-family qwen3.5 --root "$MILES_AMD_ROOT"
#   python3 tools/amd_tinker.py convert --root "$MILES_AMD_ROOT" \
#       --model-family qwen3.5 --model-id Qwen/Qwen3.5-4B --model-type qwen3.5-4B
#
# Then, per run (defaults: Qwen3.8-27B on mi3508x for 12 h, see below):
#   sbatch deploy/amd/serve_openhands.sbatch
#   grep -E 'TINKER_(BASE_URL|API_KEY)' oh-tinker-<jobid>.log   # once it is up
#   # on FLAME, with the same KL_COEF:
#   TINKER_BASE_URL=... TINKER_API_KEY=... REPO=<this checkout> \
#       sbatch --gres=none deploy/flame/train_openhands.sbatch
#
# The relay lives at most 24 h (tools/amd_tinker_modal.py --timeout), and the
# server stops with it; mi3508x caps batch jobs at 12 h, so give the client at
# most 11:30 to let the server outlive it. To go on, resubmit this
# job and resume the client with RESUME_RUN_ID and the new URL and key; keep
# CHECKPOINT_ROOT the same so the client's tinker:// paths still resolve.
#
# Trainer switches, as in deploy/flame/train_openhands.sbatch:
#   KL_COEF (0.01; 0 = off; must equal the client's), KL_TYPE (k3),
#   WEIGHT_MOVEMENT (1), FP32_LM_HEAD (0),
#   STOCHASTIC_ROUNDING (0 | 1 | fixed; default fixed), DIST_TIMEOUT_MIN (60).
# TP (4) sets the learner's tensor parallelism; MI350's 288 GB may allow 1 or 2.
# OPTIMIZER_OFFLOAD=1 (default) keeps the optimizer state offloaded in chunks, as on FLAME.
# MAX_TOKENS_PER_GPU (4096) caps a learner micro-batch; larger packs more short
# datums per micro-batch at the cost of activation memory.
# MODEL_ID (Qwen/Qwen3.8-27B) and MODEL_TYPE (qwen3.8-27B) pick the base model; any
# Qwen3.5-architecture model works (e.g. Qwen/Qwen3.5-4B with qwen3.5-4B). The
# client's MODEL_NAME must match MODEL_ID.
# MODELS_DIR (default empty = $MILES_AMD_ROOT/models on $WORK, where Qwen3.8-27B is already
# converted) holds the HF and converted weights. MODELS_DIR=/tmp/$USER-models instead downloads
# and converts the model on the node at the start of each job (about 110 GB, ~4 min).
# CHECKPOINT_ROOT (default $MILES_AMD_ROOT/runs/oh-rl/checkpoints on $WORK) and KEEP_CHECKPOINTS
# (1) keep the latest checkpoint where it survives the job: each job's /tmp is deleted when it
# ends. Give each run its own CHECKPOINT_ROOT so a resume finds its tinker:// paths.
# SAVE_OPTIM=0 (default) saves weights only (~51 GB for the 27B, 2 bytes/param); the optimizer
# state would add several times that. A resume then starts with a fresh optimizer.
# CONTEXT_LENGTH is accepted as another name for CONTEXT_WINDOW, and RECOMPUTE as another way to
# set recompute: full (default), block:N (= RECOMPUTE_NUM_LAYERS=N), selective, or none.
# TRAIN_GPUS (4) and ROLLOUT_GPUS (4) split the GPUs; training beyond TP is data parallel.
# MEM_FRACTION (0.85) is the rollout engines' static GPU memory share (weights + KV cache);
# raise it (e.g. 0.92) when the engines' KV cache fills ("full token usage" near 1 in the
# server log). The engines have their GPUs to themselves, so this does not touch training.
# Context parallelism is not offered: the Tinker gateway requires CP=1.
#
# Several nodes: NODES="hostA hostB" (or sbatch -N 2, which fills NODES from the allocation)
# serves through tools/amd_tinker_cluster.py, GPUS_PER_NODE (8) each. The trainer takes whole
# nodes first in IP order, e.g. TRAIN_GPUS=8 ROLLOUT_GPUS=8 for two nodes. Both nodes must see
# the model and the checkpoints, so this mode needs MODELS_DIR= (the converted model under
# $MILES_AMD_ROOT/models) and a CHECKPOINT_ROOT outside /tmp. CPUS_PER_NODE (64),
# OBJECT_STORE_GB (64) and IFACE (eth0) set up Ray; the mi3508x RoCE NCCL settings are added
# unless NO_ROCE=1; DRY_RUN=1 prints the plan without touching any node.
#SBATCH --job-name=oh-tinker
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=mi3508x
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.log

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# sbatch runs a copy of this script from the spool directory, so prefer the submit directory.
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/tools/amd_tinker.py" ]]; then
    REPO="$SLURM_SUBMIT_DIR"
fi
export MILES_AMD_ROOT="${MILES_AMD_ROOT:-$WORK/miles_artifacts}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$MILES_AMD_ROOT/runs/oh-rl/checkpoints}"
TP="${TP:-4}"
# CONTEXT_WINDOW (98304) bounds a sequence for the engines and a datum for the learner;
# the client must use the same value.
if [[ -n "${CONTEXT_LENGTH:-}" && -n "${CONTEXT_WINDOW:-}" && "$CONTEXT_LENGTH" != "$CONTEXT_WINDOW" ]]; then
    echo "CONTEXT_LENGTH=$CONTEXT_LENGTH and CONTEXT_WINDOW=$CONTEXT_WINDOW disagree; set one" >&2
    exit 1
fi
CONTEXT_WINDOW="${CONTEXT_WINDOW:-${CONTEXT_LENGTH:-98304}}"
TRAIN_GPUS="${TRAIN_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.8-27B}"
MODEL_TYPE="${MODEL_TYPE:-qwen3.8-27B}"
MODELS_DIR="${MODELS_DIR-}"
# sbatch -N 2 (or more) serves across the whole allocation unless NODES says otherwise.
if [[ -z "${NODES:-}" && "${SLURM_JOB_NUM_NODES:-1}" -gt 1 ]]; then
    NODES="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')"
fi

echo "node=$(hostname)  job=${SLURM_JOB_ID:-local}  start=$(date -Is)  repo=$REPO"
echo "model: $MODEL_ID ($MODEL_TYPE)  checkpoints: $CHECKPOINT_ROOT"
echo "Learner: $TRAIN_GPUS GPUs at TP=$TP, $ROLLOUT_GPUS rollout GPUs, max tokens per GPU ${MAX_TOKENS_PER_GPU:-4096}, context $CONTEXT_WINDOW"

case "${RECOMPUTE:-}" in
    "" | full) ;;
    block:*) RECOMPUTE_NUM_LAYERS="${RECOMPUTE#block:}" ;;
    none | off) RECOMPUTE_NUM_LAYERS=0 ;;
    selective) RECOMPUTE_NUM_LAYERS=selective ;;
    *) echo "RECOMPUTE must be full, block:N, selective or none" >&2; exit 1 ;;
esac

SERVER_FLAGS=(
    --bf16
    --use-distributed-optimizer
    --sglang-max-running-requests 512
    # cache_aware sends every agent with the shared system prompt to one engine and rebalances
    # only past a 64-request gap; with ~190 agents on 4 engines one engine ran 72 against ~15.
    --router-balance-abs-threshold "${ROUTER_BALANCE_ABS:-8}"
    --router-balance-rel-threshold "${ROUTER_BALANCE_REL:-1.2}"
    --tinker-max-tokens-per-datum "$CONTEXT_WINDOW"
    --rollout-max-context-len "$CONTEXT_WINDOW"
    --log-probs-chunk-size 1024
    --tinker-lease-timeout-s 172800
    --distributed-timeout-minutes "${DIST_TIMEOUT_MIN:-60}"
)
# ROUTER_POLICY (unset = the router's cache_aware) picks the SGLang router policy, e.g.
# power_of_two (least loaded of two random engines; no prefix affinity, so more re-prefill).
# SEQUENCE_PARALLEL=1 splits the norm/residual activations across the TP ranks instead of
# keeping a full copy on each (the Qwen3.8-27B recipe in docs/models/qwen/qwen3-8.md uses it).
if [[ "${SEQUENCE_PARALLEL:-0}" == 1 ]]; then
    SERVER_FLAGS+=(--sequence-parallel)
    echo "sequence parallel: on"
fi
# RECOMPUTE_NUM_LAYERS (unset = recompute every layer) recomputes only the first N of the
# 64 layers in backward and keeps the rest's activations: less compute, more memory.
# 0 turns activation recompute off.
if [[ -z "${RECOMPUTE_NUM_LAYERS:-}" ]]; then
    SERVER_FLAGS+=(--recompute-granularity full --recompute-method uniform --recompute-num-layers 1)
elif [[ "$RECOMPUTE_NUM_LAYERS" == selective ]]; then
    SERVER_FLAGS+=(--recompute-granularity selective)
    echo "activation recompute: selective (attention internals only)"
elif [[ "$RECOMPUTE_NUM_LAYERS" != 0 ]]; then
    SERVER_FLAGS+=(--recompute-granularity full --recompute-method block --recompute-num-layers "$RECOMPUTE_NUM_LAYERS")
    echo "activation recompute: first $RECOMPUTE_NUM_LAYERS layers only"
else
    echo "activation recompute: off"
fi
if [[ -n "${ROUTER_POLICY:-}" ]]; then
    SERVER_FLAGS+=(--sglang-router-policy "$ROUTER_POLICY")
    echo "router policy: $ROUTER_POLICY"
fi
if [[ "${OPTIMIZER_OFFLOAD:-1}" == 1 ]]; then
    SERVER_FLAGS+=(--chunked-optimizer-state-offload --optimizer-state-offload-chunk-size-mb 64)
fi
if [[ "${SAVE_OPTIM:-0}" == 0 ]]; then
    SERVER_FLAGS+=(--no-save-optim)
    echo "checkpoints: weights only (no optimizer state)"
fi
if [[ "${FP32_LM_HEAD:-0}" == 1 ]]; then
    SERVER_FLAGS+=(--fp32-lm-head)
    echo "LM head in FP32 (trainer and rollout engines)"
fi
if [[ "${WEIGHT_MOVEMENT:-1}" == 1 ]]; then
    SERVER_FLAGS+=(--log-weight-movement)
fi
KL_COEF="${KL_COEF:-0.01}"
if [[ "$KL_COEF" != 0 && "$KL_COEF" != 0.0 ]]; then
    SERVER_FLAGS+=(--tinker-kl-coef "$KL_COEF" --tinker-kl-type "${KL_TYPE:-k3}")
    echo "KL penalty to the starting weights: coef=$KL_COEF type=${KL_TYPE:-k3} (the client needs the same KL_COEF)"
else
    echo "KL penalty: off"
fi
if [[ "${STOCHASTIC_ROUNDING:-fixed}" == 1 ]]; then
    SERVER_FLAGS+=(--stochastic-rounding)
elif [[ "${STOCHASTIC_ROUNDING:-fixed}" == fixed ]]; then
    SERVER_FLAGS+=(--stochastic-rounding --stochastic-rounding-noise fixed)
fi

SERVE_ARGS=(
    --model-family qwen3.5
    --model-id "$MODEL_ID"
    --model-type "$MODEL_TYPE"
    --tp "$TP"
    --context-length "$CONTEXT_WINDOW"
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-4096}"
    --mem-fraction "${MEM_FRACTION:-0.85}"
    --keep-checkpoints "${KEEP_CHECKPOINTS:-1}"
    --modal-relay
)

cd "$REPO"
if [[ -n "${NODES:-}" ]]; then
    # Several nodes: tools/amd_tinker_cluster.py starts Ray over SSH and the server in tmux on the
    # first node and returns once the relay is up; this script then watches the server. If start
    # fails partway, clean up with: python3 tools/amd_tinker_cluster.py stop --root "$MILES_AMD_ROOT"
    read -r -a NODE_LIST <<< "$NODES"
    echo "Cluster: ${NODE_LIST[*]}, ${GPUS_PER_NODE:-8} GPUs each"
    # Every node reads the model and every trainer rank writes checkpoints, so neither may live on
    # one node's /tmp.
    if [[ -n "${MODELS_DIR:-}" ]]; then
        echo "NODES needs the model on shared storage: set MODELS_DIR= (empty) and convert it once into" \
             "$MILES_AMD_ROOT/models" >&2
        exit 1
    fi
    converted="$MILES_AMD_ROOT/models/${MODEL_ID##*/}_torch_dist/latest_checkpointed_iteration.txt"
    if [[ ! -f "$converted" ]]; then
        echo "Missing $converted: convert $MODEL_ID into $MILES_AMD_ROOT/models first (or copy a node's" \
             "/tmp/$USER-models/* there)" >&2
        exit 1
    fi
    if [[ "$CHECKPOINT_ROOT" == /tmp/* ]]; then
        echo "NODES needs CHECKPOINT_ROOT on shared storage (e.g. \$WORK/miles_artifacts/runs/<run>/checkpoints)" >&2
        exit 1
    fi
    CLUSTER=(python3 tools/amd_tinker_cluster.py)
    if [[ -z "${DRY_RUN:-}" ]]; then
        # Stay in the foreground like the single-node server: Ctrl+C or the end of the job stops
        # the cluster (server and relay first, then Ray on every node).
        trap '"${CLUSTER[@]}" stop --root "$MILES_AMD_ROOT"; exit 0' INT TERM
    fi
    "${CLUSTER[@]}" start \
        --root "$MILES_AMD_ROOT" \
        --checkpoint-root "$CHECKPOINT_ROOT" \
        --nodes "${NODE_LIST[@]}" \
        --gpus-per-node "${GPUS_PER_NODE:-8}" \
        --train-gpus "$TRAIN_GPUS" \
        --rollout-gpus "$ROLLOUT_GPUS" \
        --cpus-per-node "${CPUS_PER_NODE:-64}" \
        --object-store-gb "${OBJECT_STORE_GB:-64}" \
        --interface "${IFACE:-eth0}" \
        $([[ -n "${NO_ROCE:-}" ]] || echo --roce) \
        ${DRY_RUN:+--dry-run} \
        -- "${SERVE_ARGS[@]}" -- "${SERVER_FLAGS[@]}"
    [[ -n "${DRY_RUN:-}" ]] && exit 0
    while :; do
        sleep 60 & wait $!
        # A failed status call (e.g. an SSH hiccup) proves nothing; only a clean "serve: False" ends it.
        status=$("${CLUSTER[@]}" status --root "$MILES_AMD_ROOT" 2>&1) || continue
        grep -q "'serve': True" <<< "$status" || break
    done
    echo "Server exited; stopping the cluster" >&2
    "${CLUSTER[@]}" stop --root "$MILES_AMD_ROOT"
    exit 1
fi
if [[ -n "${MODELS_DIR:-}" ]]; then
    export MILES_AMD_MODELS="$MODELS_DIR"
    mkdir -p "$MODELS_DIR"
    converted="$MODELS_DIR/${MODEL_ID##*/}_torch_dist"
    # A conversion cut short by an earlier job on this node blocks convert; it is only a local copy.
    if [[ -d "$converted" && ! -f "$converted/latest_checkpointed_iteration.txt" ]]; then
        rm -rf "$converted"
    fi
    echo "models: $MODELS_DIR ($(df -h --output=avail "$MODELS_DIR" | tail -1 | tr -d ' ') free); download + convert at $(date -Is)"
    python3 tools/amd_tinker.py convert --root "$MILES_AMD_ROOT" \
        --model-family qwen3.5 --model-id "$MODEL_ID" --model-type "$MODEL_TYPE"
    echo "models ready at $(date -Is)"
fi
# exec, so Slurm's SIGTERM reaches the launcher, which stops the server and relay cleanly.
exec python3 tools/amd_tinker.py serve \
    --root "$MILES_AMD_ROOT" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --train-gpus "$TRAIN_GPUS" \
    --rollout-gpus "$ROLLOUT_GPUS" \
    "${SERVE_ARGS[@]}" \
    -- "${SERVER_FLAGS[@]}"
