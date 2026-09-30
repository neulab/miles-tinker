# Sourceable client environment for the Platoon OpenReward RL client (design C10/C11 contracts).
# Sourced by BOTH deploy/abci/clients/platoon_toolathlon.sh (the production client) and step 9 of
# deploy/abci/clients/smoke_q36.sh, so the smoke exercises the exact production environment.
# It validates its inputs and exports; it defines no function and starts no process, so a caller
# can source it and keep its own shell semantics.
# Inputs, exported into CLIENT_SCRIPT by deploy/abci/tinker_gateway.pbs (C10): TINKER_BASE_URL,
# the benchmark's OPENREWARD_SESSION_URLS_* pool, PBS_LOCALDIR; optional SERVICE_ENVIRONMENT
# (toolathlon), TINKER_API_KEY (tml-local), WANDB_MODE.
# Secrets (W&B, OpenAI/LiteLLM judges, HF) are sourced HERE ONLY, from
# /home/aci18912it/rasp/config/abci.env (which chains secrets.env, mode 0600) - never in the batch
# shell, the service helper or any container.
# A missing precondition `exit 2`s the sourcing shell: a caller that must survive that (the smoke,
# which has to print its verdict summary) sources this in a subshell.
[[ -n ${TINKER_BASE_URL:-} ]] || { echo "TINKER_BASE_URL is not set (run through tinker_gateway.pbs CLIENT_SCRIPT)" >&2; exit 2; }
PLATOON_TRIAL_SERVICE=${SERVICE_ENVIRONMENT:-toolathlon}
case "$PLATOON_TRIAL_SERVICE" in
  toolathlon) PLATOON_TRIAL_URLS=${OPENREWARD_SESSION_URLS_TOOLATHLON:-} ;;
  swe_rebench) PLATOON_TRIAL_URLS=${OPENREWARD_SESSION_URLS_SWE_REBENCH:-} ;;
  frontier_cs) PLATOON_TRIAL_URLS=${OPENREWARD_SESSION_URLS_FRONTIER_CS:-} ;;
  *) echo "Unsupported SERVICE_ENVIRONMENT=$PLATOON_TRIAL_SERVICE" >&2; exit 2 ;;
esac
[[ -n $PLATOON_TRIAL_URLS ]] || { echo "OpenReward service URLs are empty: launch with SERVICES=1" >&2; exit 2; }
[[ $(stat -c %a /home/aci18912it/rasp/config/secrets.env) == 600 ]] || { echo "secrets.env must be mode 0600" >&2; exit 2; }
set +x
# Preserve the trial's reward mode across the shared credential environment.
PLATOON_TRIAL_REWARD_MODE=${OPENREWARD_REWARD_MODE:-partial}
# shellcheck disable=SC1091
source /home/aci18912it/rasp/config/abci.env   # secrets only here; exports PLATOON_MODEL_SNAPSHOT, HF_HOME, chains secrets.env + .netrc
[[ -z ${HF_TOKEN:-} && -r ~/.cache/huggingface/token ]] && export HF_TOKEN=$(<~/.cache/huggingface/token)
export TINKER_BASE_URL TINKER_API_KEY=${TINKER_API_KEY:-tml-local} TINKER_TELEMETRY=1
export HF_HOME=/groups/gai51740/aci18912it/rasp/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SERVICE_ENVIRONMENT=$PLATOON_TRIAL_SERVICE OPENREWARD_SESSION_URLS=$PLATOON_TRIAL_URLS
case "$SERVICE_ENVIRONMENT" in
  toolathlon) export OPENREWARD_SESSION_URLS_TOOLATHLON=$PLATOON_TRIAL_URLS ;;
  swe_rebench) export OPENREWARD_SESSION_URLS_SWE_REBENCH=$PLATOON_TRIAL_URLS ;;
  frontier_cs)
    export OPENREWARD_SESSION_URLS_FRONTIER_CS=$PLATOON_TRIAL_URLS
    export OPENREWARD_SESSION_URLS_FCS_ALGORITHMIC=$PLATOON_TRIAL_URLS
    export OPENREWARD_SESSION_URLS_FCS_2_0=$PLATOON_TRIAL_URLS
    ;;
esac
export OPENREWARD_REWARD_MODE=$PLATOON_TRIAL_REWARD_MODE OPENREWARD_ENVIRONMENT_SET=${SERVICE_ENVIRONMENT}_only OPENREWARD_DISABLE_UPDATE_CHECK=1
unset PLATOON_TRIAL_REWARD_MODE PLATOON_TRIAL_SERVICE PLATOON_TRIAL_URLS
[[ $OPENREWARD_REWARD_MODE == partial || $OPENREWARD_REWARD_MODE == binary ]] || { echo "OPENREWARD_REWARD_MODE must be partial or binary" >&2; exit 2; }
echo "platoon reward_mode=$OPENREWARD_REWARD_MODE"
export OPENHANDS_SUPPRESS_BANNER=1 OPENREWARD_MCP_TIMEOUT=1800
if [[ $SERVICE_ENVIRONMENT == frontier_cs ]]; then
  # Native final grading can evaluate both current and best artifacts, each
  # with a three-hour budget. Let Platoon's step deadline govern cancellation.
  export OPENREWARD_MCP_TIMEOUT=25200
  export OPENREWARD_MCP_TOOL_TIMEOUT=25200
fi
export WANDB_MODE=${WANDB_MODE:-online}
CLIENT_LOCAL=${PBS_LOCALDIR:?}/platoon-client
mkdir -p "$CLIENT_LOCAL"/{home,tmp,xdg}
export TMPDIR=$CLIENT_LOCAL/tmp XDG_CACHE_HOME=$CLIENT_LOCAL/xdg
