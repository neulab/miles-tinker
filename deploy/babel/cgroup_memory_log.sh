#!/bin/bash
# Sample the job's cgroup memory into a CSV. Point samples mislead here: usage
# swings by >100 GB between rollout waves and the optimiser step, so a trough
# reading looks safe while the peak is near the limit. Logging lets the peak,
# and any drift in the learner's offloaded optimiser state, be seen.
OUT="${1:-/data/user_data/$USER/miles-babel/cgroup_memory.csv}"
INTERVAL="${2:-30}"
CG=$(awk -F: '/^0::/{print $3}' /proc/self/cgroup)
BASE=/sys/fs/cgroup$CG
LIMIT_CG="/sys/fs/cgroup/system.slice/slurmstepd.scope/$(echo "$CG" | cut -d/ -f4)"
[ -f "$OUT" ] || echo "timestamp,current_gb,peak_gb,anon_gb,shmem_gb,limit_gb" > "$OUT"
while true; do
    cur=$(cat $BASE/memory.current 2>/dev/null)
    pk=$(cat $BASE/memory.peak 2>/dev/null)
    lim=$(cat $LIMIT_CG/memory.max 2>/dev/null)
    anon=$(awk '/^anon /{print $2}' $BASE/memory.stat 2>/dev/null)
    shm=$(awk '/^shmem /{print $2}' $BASE/memory.stat 2>/dev/null)
    python3 -c "
import sys
g=lambda v: int(v)/2**30 if str(v).isdigit() else float('nan')
print(f\"$(date +%s),{g('$cur'):.1f},{g('$pk'):.1f},{g('$anon'):.1f},{g('$shm'):.1f},{g('$lim'):.0f}\")" >> "$OUT" 2>/dev/null
    sleep "$INTERVAL"
done
