# Source from client launchers: refuse to run heavy clients on an ABCI login node.
# The Tinker clients (rollouts, rendering, per-token logprobs) pin a core for hours; run them on the
# gateway job's head node via `qsub -v CLIENT_SCRIPT=...` (deploy/abci/tinker_gateway.pbs).
if [[ $(hostname -s) == login* && ${MILES_ALLOW_LOGIN_NODE:-0} != 1 ]]; then
  echo "refusing to run on login node $(hostname -s): submit through deploy/abci/tinker_gateway.pbs with CLIENT_SCRIPT=... (or set MILES_ALLOW_LOGIN_NODE=1 for a short smoke)" >&2
  exit 3
fi
