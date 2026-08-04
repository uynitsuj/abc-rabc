#!/usr/bin/env bash
set -u
BASE=s3://xdof-internal-research/data/deliveries/sim_tasks_20260514
CACHE=/home/karimelrafi/rerender_demos/_cache
OUT=/home/karimelrafi/rerender_demos/out
LOG=/home/karimelrafi/rerender_demos/logs
mkdir -p "$LOG"
cd /home/karimelrafi/yam_sim
PY=.venv/bin/python

run() {  # task episode gpu
  local t=$1 ep=$2 gpu=$3
  echo "[start] $t on gpu $gpu ($ep)"
  $PY scripts/rerender_episode.py \
    --episode-s3 $BASE/$t/$ep \
    --cache-dir "$CACHE" --out-dir "$OUT" \
    --backend mjwarp --gpu-id $gpu --batch-size 64 \
    > "$LOG/$t.log" 2>&1
  echo "[done exit=$?] $t"
}

run sim_hang_the_mug_on_the_mug_rack        episode_019e1f11-b012-782a-9c2d-42c33b6b837e 1 &
run sim_load_the_plates_into_the_dish_rack  episode_019dd8b1-aaa8-71e7-874b-5a340b14f69f 3 &
run sim_put_the_plastic_bottles_in_the_bin  episode_019e1e0e-4059-7e36-9193-bee766045473 7 &
run sim_sweep_away_paper_scraps_from_the_table episode_019dd861-f4a4-7d9e-9663-a4ebc4bb463f 5 &
run sim_throw_plastic_bottles_in_bin        episode_019dd883-4306-7297-9718-ec3c341e4ad5 4 &
wait
echo "ALL DONE"
