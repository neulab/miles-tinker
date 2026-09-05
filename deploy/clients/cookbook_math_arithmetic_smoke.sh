#!/bin/bash
# tinker-cookbook math_rl arithmetic smoke test against a gateway (single-turn, async, runs
# unmodified from the recipe CLI). Validates the raw train/sample/save loop before the
# multi-turn guess_number acceptance run.
#   deploy/clients/cookbook_math_arithmetic_smoke.sh <endpoint-file-or-url> <log_path> [extra key=value ...]
# Env: COOKBOOK_PYTHON (default: the login-node cookbook venv), TINKER_API_KEY (default tml-local),
#      HF_HOME (default: the shared cache holding the Qwen3-4B-Instruct-2507 tokenizer).
set -euo pipefail
ENDPOINT=${1:?endpoint file or http://host:port}; LOG_PATH=${2:?log_path}; shift 2
[[ -f $ENDPOINT ]] && ENDPOINT=$(tr -d '[:space:]' < "$ENDPOINT")
COOKBOOK_PYTHON=${COOKBOOK_PYTHON:-/home/aci18912it/miles-tinker/cookbook-venv/bin/python}
export TINKER_API_KEY=${TINKER_API_KEY:-tml-local}
export HF_HOME=${HF_HOME:-/groups/gai51740/aci18912it/rasp/cache/huggingface}
export no_proxy="${no_proxy:-}${no_proxy:+,}${ENDPOINT#http://}"
"$COOKBOOK_PYTHON" -c "import tinker_cookbook.recipes.math_rl.train" 2>/dev/null || {
  echo "the math_rl recipe needs the cookbook's math-rl extra (sympy, pylatexenc, math-verify):" >&2
  echo "  $COOKBOOK_PYTHON -m pip install -e '/home/aci18912it/source/tinker-cookbook[math-rl]'" >&2; exit 2; }
echo "math_rl arithmetic smoke -> $ENDPOINT (log_path=$LOG_PATH)"
exec "$COOKBOOK_PYTHON" -m tinker_cookbook.recipes.math_rl.train \
  env=arithmetic \
  model_name=Qwen/Qwen3-4B-Instruct-2507 renderer_name=qwen3_instruct \
  base_url="$ENDPOINT" max_steps_off_policy=1 \
  group_size=4 groups_per_batch=32 max_tokens=16 learning_rate=1e-4 \
  eval_every=0 save_every=10 max_steps=20 \
  log_path="$LOG_PATH" behavior_if_log_dir_exists=delete \
  "$@"
