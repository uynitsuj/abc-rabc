#!/usr/bin/env bash
# Launch the mjgl (MuJoCo-GL, in-distribution) sim eval for ONE finetuned DiT arm
# via the yam_sim harness + local abc_minimal DiT policy (local/dit_sim_eval.py).
#
#   local/run_dit_eval.sh <bc|rabc> <GPU> <OUTPUT_DIR> [extra dit_sim_eval.py args...]
#
# Runs on abc-rabc's venv using the vendored yam_sim package (repo-root
# yam_sim/, mujoco_warp 3.10). camera_gpu_id == device ==
# GPU (torch policy + mjwarp physics + mjgl render all co-located on that GPU).
set -euo pipefail

ARM="${1:?usage: run_dit_eval.sh <bc|rabc> <GPU> <OUTPUT_DIR> [extra args]}"
GPU="${2:?need GPU index}"
OUT="${3:?need OUTPUT_DIR}"
shift 3

ABC=/home/karimelrafi/abc-rabc
PY="$ABC/.venv/bin/python"
RUNS=/scratch/current/karimelrafi/abc_cache/runs
BC="$RUNS/put_bottles_bc/finetune_checkpoints/30000.pt"
RABC="$RUNS/put_bottles_rabc/finetune_checkpoints/30000.pt"

mkdir -p "$OUT"
cd "$ABC"
exec env MUJOCO_GL=egl PYTHONUNBUFFERED=1 PYTHONPATH="$ABC" \
  "$PY" -u "$ABC/local/dit_sim_eval.py" \
    --bc-ckpt "$BC" --rabc-ckpt "$RABC" --only "$ARM" \
    --output-dir "$OUT" \
    --episodes "${EPISODES:-40}" --num-worlds "${WORLDS:-20}" \
    --max-seconds "${MAXSEC:-128}" --execute-chunk-dim "${EXECDIM:-20}" \
    --diffusion-steps "${DIFF:-10}" --force-object-count "${FORCE:-4}" \
    --seed-base "${SEEDBASE:-20260511}" \
    --camera-gpu-id "$GPU" --device "cuda:$GPU" "$@"
