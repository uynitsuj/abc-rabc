#!/usr/bin/env bash
exec /home/karimelrafi/miniforge3/bin/python "$(dirname "$0")"/run_mcap_batch.py \
  --per-task 0 \
  --tasks sim_hang_the_mug_on_the_mug_rack sim_sweep_away_paper_scraps_from_the_table sim_throw_plastic_bottles_in_bin sim_turn_the_mug_right_side_up \
  --gpus 0 1 3 4 5 6 7 --jobs-per-gpu 3 \
  --out-root /scratch/current/karimelrafi/rerender_mcap_30hz
