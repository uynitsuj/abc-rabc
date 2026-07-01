# mjwarp re-render of put-bottles (dataset A, 2438 eps) — progress checkpoint

Goal: re-render the 2438 dataset-A episodes from mjgl -> mjwarp, package into BOTH
(1) ABC-staged and (2) lerobot-v3 forms on S3, verified.

## Key facts established
- Dataset A = the 2438 UUIDs in `staged_put/{train_sim,val_sim}` = mjgl LeRobot episode_index
  0..2437. **val_sim = index 0-99 (contiguous); train_sim = index 100-2437.**
- The mjgl/repromo LeRobot dataset ALREADY is dataset A (correct order, splits, meta, data).
  So the job = **swap only the videos** (mjgl -> mjwarp) + reuse meta/data/norm_stats verbatim.
- Verified for all 2436 renderable eps: `integration_state.npy rows == mjgl length` (0 mismatch).
- **2436/2438 renderable**; 2 absent from the ENTIRE delivery bucket (no scene_assembled.xml):
  - ep_idx **2264** (uuid 019e2814-…, len 366, train) and **2344** (uuid 019e2857-…, len 862, train).
  - Handling: BOTH formats carry these 2 episodes' ORIGINAL mjgl frames (decoded from the mjgl
    v3 videos) so index-contiguity + frame-exactness hold. Recorded as mjgl-rendered.
- Render source deliveries: 2418 in sim_tasks_20260522, 17 in _20260514, 1 in _20260515
  (all in `render_manifest.json`).
- Render timing: **~110s/episode at 640x480 on RTX 5090** (5.5 fps). Cloud L4/A10G similar-ish.

## Destinations (FINAL — note lerobot corrected by coordinator)
- STAGED (DiT-L): `s3://xdof-internal-research/abc/staged/put_bottles_mjwarp/{train_sim,val_sim}/episode_<uuid>/`
  (combined_camera-images-rgb.mp4 224x504 + states_actions.bin + velocity_*.bin + episode_metadata.json)
  + root `norm_stats.json`.
- LEROBOT (RM): `s3://xdof-internal-research/repromo/datasets/sim_put_the_plastic_bottles_in_the_bin_30hz_mjwarp/{meta,data,videos}`
  mirrors the RM reference `.../repromo/datasets/..._30hz_gop10/` (5-digit-padded file-%05d per
  its info.json). meta/ + data/ + norm_stats.json copied verbatim; only videos regenerated.
  `object_counts.json` rides INSIDE `meta/` (reference's complete 2438 file; our computed counts
  match it exactly on all 2436).

## Artifacts built (all in /home/justinyu/abc)
- `rerender_mjwarp.py` — EXTENDED: renders once at 640x480, emits per-cam native mp4s
  (`--percam-dir`) for lerobot AND the 224x504 staged combined.mp4. Hard-asserts frame counts.
- `render_worker.py` — cloud shard worker: reads manifest from S3, renders [shard_start,shard_end),
  uploads per-episode outputs to scratch keyed by zero-padded episode_index.
- `sky/launch_rerender.py` — sharded sky launcher (mirrors launch_eval.py; any_of L4/A10G/A10/L40S/
  A100 across aws us-west-2/us-east-1 + lambda; --async). RUN installs boto3 into synced venv
  then runs worker with .venv/bin/python (NOT uv run).
- `package_mjwarp.py` — local reduce, 3 phases: `object_counts` (verify vs reference), `staged`
  (per-ep dirs + bins + norm_stats), `lerobot` (copy meta/data/norm_stats + regenerate 8 file-group
  videos/cam by concatenating scratch per-ep segments in index order, frame-exact re-encode; 2
  absent eps use mjgl frames).

## S3 state so far
- `s3://xdof-internal-research/abc/render_assets/` — minimal asset set (task_water_bottles +
  i2rt_yam, 904 objects) for cloud workers. DONE.
- `s3://.../abc/render_scratch/put_bottles_mjwarp/render_manifest.json` — episode_index -> {uuid,
  split, length, delivery}. DONE.
- object_counts VERIFIED (our 2436 match reference; reference covers the 2 absent). Ships via
  lerobot meta sync.

## STATUS: FULL RENDER LAUNCHED (do NOT relaunch)
- Validation gates PASSED: cloud smoke (job 378) rendered ep0 640x480/892fr non-black on worker +
  uploaded; concat correspondence check PASSED (frame-exact, same-frame MAE~0.9 vs boundary MAE~8-10).
- Full run = **24 shards, sky job IDs 379-402** over episode_index [0,2438). A duplicate 2nd launch
  (IDs 403-426, coordinator re-fire) was CANCELLED. Exactly one job per shard now.
- Outputs -> `s3://.../abc/render_scratch/put_bottles_mjwarp/{percam,staged}/<epidx06>/`.
- Worker is idempotent (skips already-uploaded indices), so a relaunch of a failed RANGE is safe,
  but do NOT relaunch the whole thing.

## Remaining commands (in order)
1. Poll shards 379-402 to completion (`sky jobs queue`). Re-run only FAILED ranges via
   `sky/launch_rerender.py --shards N --start A --end B --label patchN`.
2. Confirm scratch has all 2436 renderable indices (percam + staged) before packaging.
3. Package:  `.venv/bin/python package_mjwarp.py --phase staged`
             `.venv/bin/python package_mjwarp.py --phase lerobot`
   (lerobot phase is a serial re-encode of ~2.2M frames x3 cams — run the 8 groups x3 cams in
   parallel processes; keep the re-encode, not stream-copy.)
4. Publish lerobot to BOTH prefixes (zero-regret): repromo/datasets/...30hz_mjwarp AND
   lerobot/...30hz_mjwarp (server-side S3 copy of videos; meta/data tiny).
5. Verify: staged counts 2338 train/100 val; lerobot loads via discover_lerobot_episodes; decode a
   HIGH-index episode (~2300 -> file-00007) at its non-zero from_timestamp, variance-check non-black;
   object_counts in meta/.

## Env note
- Use `/home/justinyu/abc/.venv/bin/python` for everything. boto3 was added to the venv via
  `uv pip install` (no uv sync). NEVER `uv run` locally (triggers uv sync, can crash live procs).
