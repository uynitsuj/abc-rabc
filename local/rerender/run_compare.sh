#!/usr/bin/env bash
set -u
BASE=s3://xdof-internal-research/data/deliveries/sim_tasks_20260514
CACHE=/home/karimelrafi/rerender_demos/_cache
LOG=/home/karimelrafi/rerender_demos/logs
cd /home/karimelrafi/yam_sim
PY=.venv/bin/python

declare -A EP=(
 [sim_hang_the_mug_on_the_mug_rack]=episode_019e1f11-b012-782a-9c2d-42c33b6b837e
 [sim_load_the_plates_into_the_dish_rack]=episode_019dd8b1-aaa8-71e7-874b-5a340b14f69f
 [sim_put_the_plastic_bottles_in_the_bin]=episode_019e1e0e-4059-7e36-9193-bee766045473
 [sim_sweep_away_paper_scraps_from_the_table]=episode_019dd861-f4a4-7d9e-9663-a4ebc4bb463f
 [sim_throw_plastic_bottles_in_bin]=episode_019dd883-4306-7297-9718-ec3c341e4ad5
)
GPUS=(1 3 4 5 7)

# Wave 1: eval-input-res mjwarp (224x168) -- fast, what the DiT policy sees at eval
i=0
for t in "${!EP[@]}"; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  CUDA_VISIBLE_DEVICES=$g $PY scripts/rerender_episode.py --episode-s3 $BASE/$t/${EP[$t]} \
    --cache-dir "$CACHE" --out-dir /home/karimelrafi/rerender_demos/out_eval224 \
    --backend mjwarp --gpu-id 0 --batch-size 64 --render-width 224 --render-height 168 \
    > "$LOG/${t}_eval224.log" 2>&1 &
  i=$((i+1))
done
wait
echo "WAVE1 (eval224 mjwarp) DONE"

# Wave 2: faithful mujoco-GL (640x480) MSAA -- what collected/training data looks like
i=0
for t in "${!EP[@]}"; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  CUDA_VISIBLE_DEVICES=$g MUJOCO_GL=egl $PY scripts/rerender_episode.py --episode-s3 $BASE/$t/${EP[$t]} \
    --cache-dir "$CACHE" --out-dir /home/karimelrafi/rerender_demos/out_mjgl640 \
    --backend mujoco --render-width 640 --render-height 480 \
    > "$LOG/${t}_mjgl640.log" 2>&1 &
  i=$((i+1))
done
wait
echo "WAVE2 (mjgl640) DONE"
echo "ALL COMPARE DONE"
