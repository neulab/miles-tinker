#!/bin/bash
# Run a list of tasks against one served model and report each verifier score.
#
# A multi-task version of sft_smoketest.sbatch: vLLM is started once and every
# task reuses it, so N tasks cost one model load rather than N. Each task still
# gets its own Apptainer sandbox, its own openhands-agent-server and its own
# verifier run, exactly as the RL env does it.
#
#   TASKS=/path/to/ids.txt sbatch sft_eval_tasks.sbatch                      # SFT model
#   TASKS=... MODEL_DIR=.../Qwen3-4B-Instruct-2507 sbatch sft_eval_tasks.sbatch   # baseline
#
#SBATCH --job-name=sft-eval
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --output=/data/user_data/%u/miles-babel/slurm-sfteval-%j.out
#SBATCH --open-mode=append

set -euo pipefail
echo "node=$(hostname)  job=$SLURM_JOB_ID  start=$(date -Is)"

REPO=$HOME/miles-tinker
WORK=/data/user_data/$USER/miles-babel
POOL=${POOL:-/data/user_data/$USER/tmax15k-rl2}
VLLM=$HOME/miniconda3/envs/vllm/bin
AGENT_PY=$WORK/client-tmax-15k-openhands/bin/python
MODEL_DIR=${MODEL_DIR:-$WORK/models/Qwen3-4B-Instruct-SFT}
TASKS=${TASKS:?set TASKS to a file of task ids, one per line}
OUT=${OUT:-/data/user_data/$USER/sft-eval/$SLURM_JOB_ID}
# Derived from the job id, not a fixed default: two of these can land on the
# same node, and a fixed port means the second vLLM fails to bind while its
# health check still passes against the first one's server -- silently running
# the whole sweep against the wrong model.
PORT=${PORT:-$(( 8100 + SLURM_JOB_ID % 800 ))}
SERVED_NAME=${SERVED_NAME:-gpt-4o}
MAXLEN=${MAXLEN:-32768}

mkdir -p "$OUT"
test -d "$MODEL_DIR" && test -x "$AGENT_PY" && test -f "$TASKS"
echo "model: $MODEL_DIR"
echo "tasks: $(wc -l < "$TASKS") from $TASKS"

# The conda env needs its own libstdc++ (CXXABI_1.3.15+, /lib64 stops at .13)
# and its own bin on PATH (FlashInfer JIT shells out to `ninja`).
export LD_LIBRARY_PATH="$(dirname "$VLLM")/lib:${LD_LIBRARY_PATH:-}"
export PATH="$VLLM:$PATH"

if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "port $PORT is already serving something; refusing to start" >&2; exit 1
fi
echo "serving on port $PORT"
"$VLLM/vllm" serve "$MODEL_DIR" \
  --served-model-name "$SERVED_NAME" --port "$PORT" --host 127.0.0.1 \
  --max-model-len "$MAXLEN" --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  > "$OUT/vllm.log" 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo "server up (${i}0s)"; break; }
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "vLLM died; see $OUT/vllm.log" >&2; tail -20 "$OUT/vllm.log"; exit 1; }
    sleep 10
done

export APPTAINER_CACHEDIR=$WORK/apptainer-cache
export APPTAINER_TMPDIR=$WORK/tmp
export TERMINAL_RL_MAX_INPUT_TOKENS=${TERMINAL_RL_MAX_INPUT_TOKENS:-30000}
cd "$REPO"

while read -r TASK; do
    [ -z "$TASK" ] && continue
    SIF=$(python3 -c "
import json,sys
for e in json.load(open('$POOL/selected.json')):
    if e['task_id']=='$TASK': print('$POOL/sifs-oh/'+e['sif_name']); break")
    echo; echo "############ $TASK ############"
    # one task failing must not abort the sweep
    "$AGENT_PY" deploy/babel/agent_server_run.py \
        --task "$TASK" --base-url "http://127.0.0.1:$PORT/v1" --model "openai/$SERVED_NAME" \
        --sif "$SIF" --task-dir "$POOL/tasks" --out "$OUT" \
        --max-iterations ${MAX_ITER:-60} \
        > "$OUT/$TASK.log" 2>&1 || echo "  (runner exited non-zero)"
    T="$OUT/${TASK}_served_openhands.txt"
    if [ -f "$T" ]; then
        echo "  $(grep -m1 '^reward:' "$T")  $(grep -m1 '^tool calls:' "$T")  $(grep -m1 '^ended:' "$T")"
    else
        echo "  no trajectory written; see $OUT/$TASK.log"
    fi
done < "$TASKS"

echo; echo "######## SUMMARY ########"
for T in "$OUT"/*_served_openhands.txt; do
    [ -f "$T" ] || continue
    printf "%-26s %s  %s\n" "$(basename "$T" _served_openhands.txt)" \
      "$(grep -m1 '^reward:' "$T" | tr -s ' ')" "$(grep -m1 '^tool calls:' "$T" | tr -s ' ')"
done
echo "done=$(date -Is); artifacts in $OUT"
