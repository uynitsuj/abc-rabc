#!/usr/bin/env bash
# Stage the 5 non-put_bottles tasks sequentially (CPU-only). Uses a modest worker
# count so it doesn't starve the GPU run's dataloaders while training runs.
set -uo pipefail
ROOT=/home/karimelrafi/abc-rabc
export WORKERS="${WORKERS:-32}"

# task_key : lerobot dirname  (put_bottles already staged)
JOBS=(
  "hang_mug sim_hang_the_mug_on_the_mug_rack_30hz_gop10"
  "load_plates sim_load_the_plates_into_the_dish_rack_30hz_gop10"
  "sweep sim_sweep_away_paper_scraps_from_the_table_30hz_gop10"
  "throw_bottles sim_throw_plastic_bottles_in_bin_30hz_gop10"
  "turn_mug sim_turn_the_mug_right_side_up_30hz_gop10"
)
for job in "${JOBS[@]}"; do
  set -- $job; TK=$1; DN=$2
  echo "===== convert $TK ($(date)) ====="
  bash "$ROOT/local/convert_task.sh" "$TK" "$DN" || { echo "[ERR] convert $TK failed"; }
done
echo "ALL_CONVERSIONS_DONE $(date)"
