#!/usr/bin/env bash
# Convert one LeRobot task dir to the ABC staged layout on /scratch.
#   local/convert_task.sh <TASK_KEY> <LEROBOT_DIRNAME>
# env: WORKERS (48) VALCOUNT (50)
# Idempotent: writes .staged_ok on success; re-running is a no-op. A partial/failed
# stage leaves no marker, so it is wiped and redone on the next call.
set -uo pipefail
TASK_KEY="$1"; DIRNAME="$2"
WORKERS="${WORKERS:-48}"; VALCOUNT="${VALCOUNT:-50}"
SIM_TASK="${DIRNAME%_30hz_gop10}"
ROOT=/home/karimelrafi/abc-rabc
# Scratch root; /scratch/current is a monthly-rotating symlink — override ABC_SCRATCH
# (e.g. /scratch/<older_month>/karimelrafi) after a month rollover so paths resolve to data.
SCRATCH="${ABC_SCRATCH:-/scratch/current/karimelrafi}"
SRC="$SCRATCH/lerobot_mjgl_30hz_full/$DIRNAME"
OUT="$SCRATCH/abc_cache/$TASK_KEY"

if [ -f "$OUT/.staged_ok" ]; then
  echo "[skip] $TASK_KEY already staged ($(ls "$OUT/train_sim" 2>/dev/null | wc -l) train eps)"; exit 0
fi
echo "[convert] $TASK_KEY sim_task=$SIM_TASK workers=$WORKERS ($(date))"
rm -rf "$OUT"                      # clear any partial stage
cd "$ROOT"
PYTHONPATH="$ROOT" uv run python convert_parallel.py \
  --data-dir "$SRC" --out-dir "$OUT" --task-name "$SIM_TASK" \
  --val-count "$VALCOUNT" --workers "$WORKERS"
rc=$?
if [ $rc -ne 0 ]; then echo "[ERR] convert $TASK_KEY rc=$rc"; exit $rc; fi
# Require every train episode to carry a velocity sidecar (RABC needs it).
tr=$(ls "$OUT/train_sim" 2>/dev/null | wc -l)
ve=$(find "$OUT/train_sim" -name velocity_repromo.bin 2>/dev/null | wc -l)
if [ "$tr" -lt 1 ] || [ "$tr" != "$ve" ]; then
  echo "[ERR] $TASK_KEY incomplete: train=$tr velocity=$ve"; exit 1
fi
touch "$OUT/.staged_ok"
echo "[OK] staged $TASK_KEY train=$tr val=$(ls "$OUT/val_sim" 2>/dev/null | wc -l)"
