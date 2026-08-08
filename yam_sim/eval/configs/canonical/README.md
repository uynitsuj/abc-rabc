# Canonical per-task eval configs (fixed object counts)

One yaml per task because `force_object_count` is a global EvalConfig field.
Counts agreed 2026-08-08: put_bottles=6 (paper/release scene), throw_bottles=4
(fixed in data), sweep_paper=4, load_plates=4, hang_mug=3, turn_mug=3.
Poses still randomize per seed; count/variant/scale are pinned (constant nq).
sweep needs the trash_count plumbing added to yam_sim/__init__.py on 2026-08-08.

Horizons are the suite convention: 2x the longest demo at (or near) the pinned
count. For paper-comparable put_bottles numbers use max_seconds: 60 and score
the saved qpos traces offline with release-candidate's score_bottles.py (the
live put_bottles_in_bin evaluator is a looser center-point check).

Run (per task, with an openpi server already up on PORT):
  cd ~/abc-rabc && MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=G .venv/bin/python -u \
    -m yam_sim.examples.run_eval --config yam_sim/eval/configs/canonical/<task>.yaml \
    --models <label>=127.0.0.1:PORT --output-dir <out>
`camera_gpu_id: 0` assumes CUDA_VISIBLE_DEVICES masks to one GPU. video is off
for speed (~2x: rendering for video dominates wall time); re-render videos
offline from the saved seed_*_qpos.npy via the rerender pipeline.
