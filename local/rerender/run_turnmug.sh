#!/usr/bin/env bash
set -u
BASE=s3://xdof-internal-research/data/deliveries/sim_tasks_20260514
CACHE=/home/karimelrafi/rerender_demos/_cache
LOG=/home/karimelrafi/rerender_demos/logs
cd /home/karimelrafi/yam_sim
PY=.venv/bin/python
T=sim_turn_the_mug_right_side_up
EP=episode_019e1cab-95b0-7745-bdbd-5ef3684fc15a

# mjwarp 640 (out) on GPU 1
CUDA_VISIBLE_DEVICES=1 $PY scripts/rerender_episode.py --episode-s3 $BASE/$T/$EP \
  --cache-dir "$CACHE" --out-dir /home/karimelrafi/rerender_demos/out \
  --backend mjwarp --gpu-id 0 --batch-size 64 --render-width 640 --render-height 480 \
  > "$LOG/${T}_mjwarp640.log" 2>&1 &
P1=$!

# mjwarp 224 (out_eval224) on GPU 3
CUDA_VISIBLE_DEVICES=3 $PY scripts/rerender_episode.py --episode-s3 $BASE/$T/$EP \
  --cache-dir "$CACHE" --out-dir /home/karimelrafi/rerender_demos/out_eval224 \
  --backend mjwarp --gpu-id 0 --batch-size 64 --render-width 224 --render-height 168 \
  > "$LOG/${T}_eval224.log" 2>&1 &
P2=$!

# mujoco GL 640 (out_mjgl640) on GPU 4
CUDA_VISIBLE_DEVICES=4 MUJOCO_GL=egl $PY scripts/rerender_episode.py --episode-s3 $BASE/$T/$EP \
  --cache-dir "$CACHE" --out-dir /home/karimelrafi/rerender_demos/out_mjgl640 \
  --backend mujoco --render-width 640 --render-height 480 \
  > "$LOG/${T}_mjgl640.log" 2>&1 &
P3=$!

wait $P1 $P2 $P3
echo "TURNMUG ALL DONE"
