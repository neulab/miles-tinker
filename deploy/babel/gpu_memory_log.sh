#!/bin/bash
# Sample GPU memory into a CSV so peak learner usage can be compared against
# capacity: recompute buys memory with compute, and the headroom it leaves is
# what says whether a cheaper setting (selective recompute, a bigger condenser
# budget, more concurrent rollouts) would still fit.
OUT="${1:-/data/user_data/$USER/miles-babel/gpu_memory.csv}"
INTERVAL="${2:-20}"
[ -f "$OUT" ] || echo "timestamp,gpu,used_mib,total_mib,util_pct" > "$OUT"
while true; do
    ts=$(date +%s)
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
               --format=csv,noheader,nounits |
        while IFS=', ' read -r idx used total util; do
            echo "$ts,$idx,$used,$total,$util"
        done >> "$OUT"
    sleep "$INTERVAL"
done
