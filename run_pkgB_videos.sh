#!/usr/bin/env bash
# Parallel videos phase for package_B: one process per (file-group, cam).
# Group count is read from the meta_data run's log line "[pkgB] N file-groups".
# --probe-scratch false: groups were probe-verified in the meta_data run; the
# decode/encode asserts (frames == length, encoded == expected) still enforce
# every episode's frame count. Local scratch reads (byte-verified vs S3).
set -euo pipefail
cd "$HOME/abc"
NGROUPS=$(grep -oE '[0-9]+ file-groups' /mnt/data/logs/pkgB_meta.log | grep -oE '^[0-9]+')
[ -n "$NGROUPS" ] || { echo "cannot determine group count" >&2; exit 1; }
echo "[videos] $NGROUPS groups x 3 cams"
mkdir -p /mnt/data/logs/pkgB_videos
pids=()
for cam in top left right; do
  for ((g=0; g<NGROUPS; g++)); do
    PYTHONPATH=$HOME/abc .venv/bin/python package_B.py \
      --manifest render_manifest_B.json \
      --states-source /mnt/data/staged_put_B \
      --scratch-s3 s3://xdof-internal-research/abc/render_scratch/put_bottles_mjwarp \
      --scratch-local /mnt/data/render_B \
      --out-dir /mnt/data/hf_lerobot --dataset-name sim_put_bottles_mjwarp_B_only \
      --index-offset 2438 \
      --object-counts-json /mnt/data/warp_rm/object_counts_AB.json \
      --no-probe-scratch --phase videos --groups "$g" --cam "$cam" \
      > "/mnt/data/logs/pkgB_videos/g${g}_${cam}.log" 2>&1 &
    pids+=($!)
  done
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "[videos] all jobs finished, failures=$fail"
exit $fail
