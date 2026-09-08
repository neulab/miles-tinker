#!/bin/bash
# deploy/abci/kill-leftovers.sh - safety-net sweep run by tinker_gateway.pbs on every node at teardown.
# usage: kill-leftovers.sh [logdir]  (with logdir, appends to <logdir>/teardown-<host>.log)
# Killing singularity's starter orphans what it spawned when the container has no PID namespace (the Toolathlon
# service: nginx, postgres, uvicorn workers, MCP node/chromium sessions), and a --containall miles container that
# ignored TERM leaves ray daemons / SGLang engines holding the GPUs; this kills everything of ours that matches.
[[ $(hostname -s) == login* ]] && { echo "refusing to run on $(hostname -s)" >&2; exit 1; }
[[ -n ${1:-} ]] && exec >> "$1/teardown-$(hostname -s).log" 2>&1
# Final GPU snapshot to pair with node-init.sh's baseline (ECC/retired pages are cumulative).
if [[ -n ${1:-} ]] && command -v nvidia-smi >/dev/null; then
  nvidia-smi -q -d ECC,PAGE_RETIREMENT,PERFORMANCE > "$1/gpu-final-$(hostname -s).txt" 2>&1 || true
  dmesg 2>/dev/null | grep -i "xid\|nvrm" | tail -40 > "$1/xid-$(hostname -s).txt" 2>&1 || true
fi
me=$(id -un)
NAMES='python|python3|raylet|gcs_server|starter|starter-suid|sinit|nginx|postgres|uvicorn|node|chrome|chromium|headless_shell|npx|uv'
CMDS='^(ray::|sglang::|/usr/bin/singularity|/opt/venv/bin/|/app/openreward)'
left=$( { pgrep -u "$me" -a -x "$NAMES"; pgrep -u "$me" -a -f "$CMDS"; } 2>/dev/null | sort -u)
n=$(printf '%s' "$left" | grep -c .)
echo "$(hostname -s): $n leftover process(es) $(date +%T)"; printf '%s\n' "$left" | head -12 | cut -c1-160
if (( n > 0 )); then
  pkill -KILL -u "$me" -x "$NAMES" 2>/dev/null; pkill -KILL -u "$me" -f "$CMDS" 2>/dev/null
  sleep 2; echo "$(hostname -s): $( { pgrep -u "$me" -x "$NAMES"; pgrep -u "$me" -f "$CMDS"; } 2>/dev/null | sort -u | wc -l) left after kill"
fi
exit 0
