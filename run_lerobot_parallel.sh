#!/usr/bin/env bash
# Parallel lerobot-mjwarp video build: 8 file-groups x 3 cameras = 24 independent re-encodes.
# Each (group,cam) is one process; the meta/data/norm_stats copy runs ONCE upfront.
# Usage:  bash run_lerobot_parallel.sh [max_parallel]   (default 8)
set -u
PY=/home/justinyu/abc/.venv/bin/python
CD=/home/justinyu/abc
LOG=/tmp/pkg_mjwarp_cache/lerobot_logs
mkdir -p "$LOG"
MAXP="${1:-8}"

echo "[lerobot-par] copying meta/data/norm_stats once (copy_meta)"
# Trigger the one-time meta/data/norm_stats copy via a no-op group build (group 0 top does the copy
# and also builds g0/top). Remaining (group,cam) run with --no-copy-meta.
cd "$CD"
$PY package_mjwarp.py --phase lerobot --groups 0 --cam top > "$LOG/g0_top.log" 2>&1
echo "[lerobot-par] meta copied + g0/top built (exit $?)"

# Launch the remaining 23 (group,cam) jobs, throttled to MAXP concurrent.
running=0
for gi in 0 1 2 3 4 5 6 7; do
  for cam in top left right; do
    [ "$gi" = "0" ] && [ "$cam" = "top" ] && continue   # already done above
    (
      $PY package_mjwarp.py --phase lerobot --groups "$gi" --cam "$cam" --no-copy-meta \
        > "$LOG/g${gi}_${cam}.log" 2>&1
      echo "[done] g${gi}_${cam} exit=$?"
    ) &
    running=$((running+1))
    if [ "$running" -ge "$MAXP" ]; then wait -n; running=$((running-1)); fi
  done
done
wait
echo "[lerobot-par] ALL (group,cam) builds finished. Check $LOG/*.log for per-job status."
grep -L "DONE groups" "$LOG"/*.log 2>/dev/null && echo "^ logs WITHOUT 'DONE' = failures (empty = all OK)" || echo "all logs show DONE"
