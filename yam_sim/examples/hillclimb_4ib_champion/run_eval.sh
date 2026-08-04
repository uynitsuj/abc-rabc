#!/usr/bin/env bash
# Run the hill-climb 4-IB champion config (C36) evaluation.
#
# Prerequisites:
#   - Clone abc repo and run setup_dev.sh
#   - Checkpoint downloads automatically via W&B on first run
#
# Usage:
#   cd /path/to/abc
#   MUJOCO_GL=egl bash /path/to/run_eval.sh [--num-runs 25] [--seed-start 0]
#
# Expected results (25 seeds):
#   4-IB: 68%  5-IB: 48%  6-IB: 24%

set -euo pipefail

NUM_RUNS="${1:-25}"
SEED_START="${2:-0}"

MUJOCO_GL="${MUJOCO_GL:-egl}" uv run python environments/mujoco/batch_eval.py \
    --checkpoint-path "far-wandb/FAR-abc/far-abc-XGYkwlRVLgwVCKSU-last:v200" \
    --scene hybrid \
    --num-chunks 250 \
    --execute-chunk-dim 15 \
    --bottle-opacity 0.7 \
    --all-green-bottles \
    --diffusion-steps 5 \
    --jpeg-quality 75 \
    --bottle-mass 0.015 \
    --prefix-length 3 \
    --match-training-resolution \
    --camera-noise 3.0 \
    --gaussian-blur 3 \
    --no-shadows \
    --randomize-bottles \
    --bottle-min-bin-dist 0.25 \
    --num-runs "$NUM_RUNS" \
    --seed-start "$SEED_START" \
    --output-dir experiments/hillclimb_4ib/C36_reproduction
