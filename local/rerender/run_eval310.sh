#!/usr/bin/env bash
set -u
RD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA=/home/karimelrafi/rerender_demos
CACHE=$DATA/_cache
YAM=/home/karimelrafi/yam_sim/.venv/bin/python
ABC=/home/karimelrafi/abc-rabc/.venv/bin/python
OUT=$DATA/out_eval310_640
LOG=$DATA/logs
mkdir -p "$OUT" "$LOG"

declare -A EP=(
 [sim_hang_the_mug_on_the_mug_rack]=episode_019e1f11-b012-782a-9c2d-42c33b6b837e
 [sim_load_the_plates_into_the_dish_rack]=episode_019dd8b1-aaa8-71e7-874b-5a340b14f69f
 [sim_put_the_plastic_bottles_in_the_bin]=episode_019e1e0e-4059-7e36-9193-bee766045473
 [sim_sweep_away_paper_scraps_from_the_table]=episode_019dd861-f4a4-7d9e-9663-a4ebc4bb463f
 [sim_throw_plastic_bottles_in_bin]=episode_019dd883-4306-7297-9718-ec3c341e4ad5
 [sim_turn_the_mug_right_side_up]=episode_019e1cab-95b0-7745-bdbd-5ef3684fc15a
)
GPUS=(1 3 4 5 7 1)

render_one() {
  local t=$1 g=$2
  local e=${EP[$t]} d="$CACHE/$t/${EP[$t]}"
  $YAM "$RD/normalize_scene.py" --in-xml "$d/scene_assembled.xml" --out-xml "$d/scene_normalized.xml" >>"$LOG/${t}_eval310.log" 2>&1
  CUDA_VISIBLE_DEVICES=$g $ABC "$RD/eval_faithful_render.py" \
    --scene "$d/scene_normalized.xml" --state "$d/integration_state.npy" \
    --out-dir "$OUT/$t/$e" --width 640 --height 480 --fps 30 --gpu-id 0 >>"$LOG/${t}_eval310.log" 2>&1
  echo "DONE $t (exit $?)"
}

i=0
for t in "${!EP[@]}"; do
  render_one "$t" "${GPUS[$i]}" &
  i=$((i+1))
done
wait
echo "ALL EVAL310 DONE"
