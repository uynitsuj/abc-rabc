#!/usr/bin/env bash
# Wait for the mcap re-render to finish, then convert each task to a LeRobot v3 dataset.
set -u
RD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCAP=/scratch/current/karimelrafi/rerender_mcap_30hz
LR=/scratch/current/karimelrafi/rerender_lerobot_30hz
YAM=/home/karimelrafi/yam_sim/.venv/bin/python
CONV=$RD/convert_to_lerobot_30hz.py
LOG=$RD/logs

echo "[convert_all] waiting for re-render DONE ..."
while ! grep -q "DONE totals" "$LOG/full_all5.log" 2>/dev/null; do sleep 60; done
echo "[convert_all] re-render finished; starting conversions $(date -Is)"

mkdir -p "$LR"
cd /home/karimelrafi/yam_sim
for t in sim_hang_the_mug_on_the_mug_rack sim_load_the_plates_into_the_dish_rack sim_sweep_away_paper_scraps_from_the_table sim_throw_plastic_bottles_in_bin sim_turn_the_mug_right_side_up; do
  $YAM "$CONV" --src "$MCAP/$t" --dst "$LR/$t" --episodes-per-file 1000 > "$LOG/convert_${t}.log" 2>&1 &
done
wait
echo "[convert_all] ALL_CONVERSIONS_DONE $(date -Is)"
for t in sim_hang_the_mug_on_the_mug_rack sim_load_the_plates_into_the_dish_rack sim_sweep_away_paper_scraps_from_the_table sim_throw_plastic_bottles_in_bin sim_turn_the_mug_right_side_up; do
  echo "  $t: $(grep -o 'Done. [0-9]* episodes' "$LOG/convert_${t}.log" | tail -1)"
done
