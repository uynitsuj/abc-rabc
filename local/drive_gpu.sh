#!/usr/bin/env bash
# End-to-end driver: for every task, stage data (if missing) then finetune both
# arms (rabc, bc) on all 8 GPUs, sequentially. No conversion ever overlaps a GPU
# run (conversion happens just-in-time with GPUs idle), which avoids the CPU-RAM
# OOM that killed an earlier overlapped run. Resumable: skips any arm whose
# finetune_checkpoints/30000.pt exists, and any task already staged (.staged_ok).
#
# Order puts put_bottles first = the sample test.
set -uo pipefail
ROOT=/home/karimelrafi/abc-rabc
# Scratch root. /scratch/current is a MONTHLY-ROTATING symlink; after a month
# rollover it points at an empty new month. Override to the month holding the data:
#   ABC_SCRATCH=/scratch/<older_month>/karimelrafi bash local/drive_gpu.sh
# Exported so run_arm.sh / convert_task.sh inherit the same root.
export ABC_SCRATCH="${ABC_SCRATCH:-/scratch/current/karimelrafi}"
RUNS="$ABC_SCRATCH/abc_cache/runs"
DATA="$ABC_SCRATCH/abc_cache"
mkdir -p "$RUNS"

# task_key | sim_task | lerobot_dirname
TASKS=(
  "put_bottles|sim_put_the_plastic_bottles_in_the_bin|sim_put_the_plastic_bottles_in_the_bin_30hz_gop10"
  "throw_bottles|sim_throw_plastic_bottles_in_bin|sim_throw_plastic_bottles_in_bin_30hz_gop10"
  "load_plates|sim_load_the_plates_into_the_dish_rack|sim_load_the_plates_into_the_dish_rack_30hz_gop10"
  "turn_mug|sim_turn_the_mug_right_side_up|sim_turn_the_mug_right_side_up_30hz_gop10"
  "sweep|sim_sweep_away_paper_scraps_from_the_table|sim_sweep_away_paper_scraps_from_the_table_30hz_gop10"
  "hang_mug|sim_hang_the_mug_on_the_mug_rack|sim_hang_the_mug_on_the_mug_rack_30hz_gop10"
)

run_arm() {
  local TK=$1 ST=$2 ARM=$3
  local marker="$RUNS/${TK}_${ARM}/finetune_checkpoints/30000.pt"
  if [ -f "$marker" ]; then echo "[skip] ${TK}_${ARM} (30000.pt exists)"; return 0; fi
  echo "===== START ${TK}_${ARM} ($(date)) ====="
  bash "$ROOT/local/run_arm.sh" "$TK" "$ST" "$ARM" > "$RUNS/${TK}_${ARM}.log" 2>&1
  local rc=$?
  if [ -f "$marker" ]; then
    echo "===== DONE ${TK}_${ARM} ($(date)) ====="
  else
    echo "===== FAIL ${TK}_${ARM} rc=$rc — no 30000.pt; see $RUNS/${TK}_${ARM}.log ($(date)) ====="
  fi
  return 0
}

for entry in "${TASKS[@]}"; do
  IFS='|' read -r TK ST DN <<< "$entry"
  echo "######## TASK $TK ($(date)) ########"
  if [ ! -f "$DATA/$TK/.staged_ok" ]; then
    echo "[stage] $TK (GPUs idle during conversion)"
    bash "$ROOT/local/convert_task.sh" "$TK" "$DN" || { echo "[ERR] stage $TK failed; skipping task"; continue; }
  fi
  run_arm "$TK" "$ST" rabc
  run_arm "$TK" "$ST" bc
done
echo "ALL_GPU_JOBS_DONE $(date)"
