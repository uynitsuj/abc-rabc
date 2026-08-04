#!/usr/bin/env bash
# Finetune the pretrained DiT-L on one sim task, one arm (rabc | bc), sharded
# over all local GPUs with torchrun/DDP. Each arm gets its own run dir (symlinks
# to the shared converted data + checkpoint) so finetune_checkpoints don't collide.
#
#   local/run_arm.sh <TASK_KEY> <SIM_TASK> <rabc|bc>
# env overrides: STEPS (30000) BS (per-gpu, 48) NGPU (8) COMPILE (1) NUM_WORKERS (8)
#   RESUME (auto): auto-continues from the latest numbered ckpt; set RESUME=off to restart clean.
set -euo pipefail

TASK_KEY="$1"; SIM_TASK="$2"; ARM="$3"
STEPS="${STEPS:-30000}"
BS="${BS:-48}"
NGPU="${NGPU:-8}"
COMPILE="${COMPILE:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"    # per-rank dataloader workers; 8*8=64 keeps CPU RAM bounded
                                   # on this shared box (128 workers OOM-killed a run). GPU-bound,
                                   # so fewer workers doesn't hurt throughput.

ROOT=/home/karimelrafi/abc-rabc
# Scratch root. /scratch/current is a MONTHLY-ROTATING symlink (flips 2026_MM at
# month start); after a rollover it points at a fresh empty month, so override
# ABC_SCRATCH to the month that actually holds the data, e.g.
#   ABC_SCRATCH=/scratch/2026_06/karimelrafi bash local/run_arm.sh ...
SCRATCH="${ABC_SCRATCH:-/scratch/current/karimelrafi}"
DATA="$SCRATCH/abc_cache/$TASK_KEY"
SHARED="$SCRATCH/abc_cache/_shared"
CKPT="$SCRATCH/abc_ckpts/lbm_3_5k_dit_l_50000.ckpt"
RUNDIR="$SCRATCH/abc_cache/runs/${TASK_KEY}_${ARM}"

[ -d "$DATA/train_sim" ] || { echo "[ERR] missing $DATA/train_sim — convert first"; exit 1; }
[ -f "$SHARED/ditl_sim_norm_stats.json" ] || { echo "[ERR] missing norm_stats"; exit 1; }

mkdir -p "$RUNDIR"
ln -sfn "$DATA/train_sim" "$RUNDIR/train_sim"
ln -sfn "$DATA/val_sim"   "$RUNDIR/val_sim"
ln -sfn "$SHARED/ditl_sim_norm_stats.json" "$RUNDIR/norm_stats.json"
ln -sfn "$CKPT" "$RUNDIR/abc_dit_xl_200k_model.pt"

EXTRA=""
[ "$ARM" = "rabc" ] && EXTRA="--rabc-enabled --rabc-velocity-file velocity_repromo.bin --rabc-threshold 1.0"
[ "$COMPILE" = "0" ] && EXTRA="$EXTRA --no-compile"

# Auto-resume an interrupted arm from its latest NUMBERED checkpoint (those carry
# optimizer state; last.pt is model-only). Skipped once the arm is finished (30000.pt).
# Disable with RESUME=off. This lets drive_gpu.sh continue partial arms losslessly.
CKDIR="$RUNDIR/finetune_checkpoints"
if [ "${RESUME:-auto}" != "off" ] && [ ! -f "$CKDIR/30000.pt" ]; then
  latest=$(ls "$CKDIR"/[0-9]*.pt 2>/dev/null | sed 's#.*/##; s#\.pt$##' | sort -n | tail -1)
  if [ -n "${latest:-}" ]; then
    EXTRA="$EXTRA --resume $CKDIR/$latest.pt"
    echo "[run] resuming $TASK_KEY $ARM from step $latest"
  fi
fi

export ABC_CACHE="$RUNDIR"
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
# Cap glibc malloc arenas: torchcodec/ffmpeg threads in each of the 64 dataloader
# workers otherwise fragment the heap, leaking ~45MB/batch -> ~1.3TB over a run that
# OOM-kills a worker. ARENA_MAX=2 + TRIM=0 holds worker RSS flat (probed: 25GB->flat).
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=0
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/abc_compile_cache/inductor"
export TRITON_CACHE_DIR="$SCRATCH/abc_compile_cache/triton"

echo "[run] task=$TASK_KEY arm=$ARM steps=$STEPS bs/gpu=$BS ngpu=$NGPU compile=$COMPILE"
echo "[run] rundir=$RUNDIR"
cd "$ROOT"
exec uv run torchrun --standalone --nproc-per-node "$NGPU" train.py \
  --sim-task "$SIM_TASK" --train-steps "$STEPS" --batch-size "$BS" --num-workers "$NUM_WORKERS" \
  --model.hidden-size 1024 --model.depth 24 --model.num-heads 16 --optim.vision-lr-scale 0 \
  --clip.cache-dir "$SHARED/clip" \
  --load-pretrained $EXTRA
