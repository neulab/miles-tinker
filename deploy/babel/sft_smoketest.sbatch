#!/bin/bash
# Serve the SFT'd model and run one real task through it, end to end.
#
# The point is to answer "did the SFT help" with the same machinery RL uses, not
# a proxy: vLLM serves the HF weights, openhands-agent-server runs inside the
# task's own Apptainer sandbox, the agent gets the real tool set, and the score
# comes from the task's own verifier via ApptainerBashExecutor.score.
#
# vLLM runs from the conda env rather than the training container -- it is a
# one-off smoke test and does not need Megatron, ray or the venv bootstrap.
#
#   TASK=task_000285_3604e31b sbatch sft_smoketest.sbatch
#   MODEL_DIR=.../Qwen3-4B-Instruct-2507 sbatch sft_smoketest.sbatch   # baseline
#
#SBATCH --job-name=sft-test
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --output=/data/user_data/%u/miles-babel/slurm-sfttest-%j.out
#SBATCH --open-mode=append

set -euo pipefail
echo "node=$(hostname)  job=$SLURM_JOB_ID  start=$(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

REPO=$HOME/miles-tinker
WORK=/data/user_data/$USER/miles-babel
POOL=/data/user_data/$USER/tmax15k
VLLM=$HOME/miniconda3/envs/vllm/bin
# Two interpreters on purpose: the conda env has vLLM but not the openhands SDK,
# and the RL client venv has openhands + platoon.terminal_rl but not vLLM.
AGENT_PY=$WORK/client-tmax-15k-openhands/bin/python
MODEL_DIR=${MODEL_DIR:-$WORK/models/Qwen3-4B-Instruct-SFT}
OUT=${OUT:-/data/user_data/$USER/sft-test/$SLURM_JOB_ID}
# Derived from the job id, not a fixed default: two of these can land on the
# same node, and a fixed port means the second vLLM fails to bind while its
# health check still passes against the first one's server -- silently running
# the whole sweep against the wrong model.
PORT=${PORT:-$(( 8100 + SLURM_JOB_ID % 800 ))}
# Served under the same name training used, so litellm treats tool calling as
# native exactly as it did during RL; changing it changes how calls are parsed.
SERVED_NAME=${SERVED_NAME:-gpt-4o}
MAXLEN=${MAXLEN:-32768}

mkdir -p "$OUT"
test -d "$MODEL_DIR" || { echo "no model at $MODEL_DIR" >&2; exit 1; }
test -x "$AGENT_PY" || { echo "no openhands python at $AGENT_PY" >&2; exit 1; }
echo "model: $MODEL_DIR"

# Pick a task that has an openhands image, unless one is named.
TASK=${TASK:-$(python3 - "$POOL" <<'EOF'
import json,os,sys
pool=sys.argv[1]
sel=json.load(open(f"{pool}/selected.json"))
oh=set(os.listdir(f"{pool}/sifs-oh"))
for e in sel:
    if e["sif_name"] in oh and os.path.isdir(f"{pool}/tasks/{e['task_id']}"):
        print(e["task_id"]); break
EOF
)}
SIF=$(python3 - "$POOL" "$TASK" <<'EOF'
import json,sys
pool,task=sys.argv[1],sys.argv[2]
for e in json.load(open(f"{pool}/selected.json")):
    if e["task_id"]==task: print(f"{pool}/sifs-oh/{e['sif_name']}"); break
EOF
)
test -n "$SIF" && test -f "$SIF" || { echo "no openhands sif for task $TASK" >&2; exit 1; }
echo "task: $TASK"
echo "sif:  $SIF"

# --- serve ---
# The conda env is built against a newer libstdc++ than the compute nodes carry:
# its extensions need CXXABI_1.3.15+, /lib64 stops at 1.3.13, and vLLM dies on
# `import sqlite3` before it ever loads the model. The env ships
# libstdc++.so.6.0.36 (CXXABI_1.3.17), so put that ahead of the system one.
export LD_LIBRARY_PATH="$(dirname "$VLLM")/lib:${LD_LIBRARY_PATH:-}"
# The env's bin must be on PATH too, not just used by absolute path: vLLM's
# FlashInfer backend JIT-compiles kernels and shells out to `ninja`, which lives
# in the conda env rather than on the system. Job 10646152 died on
# FileNotFoundError: 'ninja' after the model had already loaded.
export PATH="$VLLM:$PATH"
echo "starting vLLM on :$PORT ..."
if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "port $PORT is already serving something; refusing to start" >&2; exit 1
fi
echo "serving on port $PORT"
"$VLLM/vllm" serve "$MODEL_DIR" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" --host 127.0.0.1 \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  > "$OUT/vllm.log" 2>&1 &
VLLM_PID=$!
cleanup() { kill "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "waiting for the server ..."
for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then echo "server up after ${i}0s"; break; fi
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "vLLM died; see $OUT/vllm.log" >&2; tail -25 "$OUT/vllm.log"; exit 1; }
    sleep 10
done
curl -sf "http://127.0.0.1:$PORT/v1/models" | head -c 300; echo

# --- run one task through the real openhands stack ---
export APPTAINER_CACHEDIR=$WORK/apptainer-cache
export APPTAINER_TMPDIR=$WORK/tmp
export TERMINAL_RL_MAX_INPUT_TOKENS=${TERMINAL_RL_MAX_INPUT_TOKENS:-30000}
cd "$REPO"
"$AGENT_PY" deploy/babel/agent_server_run.py \
  --task "$TASK" \
  --base-url "http://127.0.0.1:$PORT/v1" \
  --model "openai/$SERVED_NAME" \
  --sif "$SIF" \
  --task-dir "$POOL/tasks" \
  --out "$OUT" \
  --max-iterations ${MAX_ITER:-60} \
  2>&1 | tee "$OUT/run.log"

echo "done=$(date -Is); artifacts in $OUT"
