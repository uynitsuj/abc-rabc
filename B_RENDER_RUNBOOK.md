# B-only re-render + A∪B RM retrain runbook

Prepared 2026-07-06. Renders ONLY the B-only put-bottles episodes (dataset-B minus A),
packages them into a standalone LeRobot-v3 dataset, then (deferred) merges with A and
retrains + rescores the winning RM on the expanded A∪B corpus.

**DO NOT LAUNCH the cloud render until `aws sso login` refreshes local creds.** Everything
below the "PREP DONE" line is validated and ready; the sky launch is step 1.

---

## Facts (measured this session)

- **B-only = 3,553 renderable episodes** (delivery `sim_tasks_20260522` has 5,972 uuids;
  2,418 are already in A; 3,554 are B-only, of which **1 lacks `integration_state.npy` and
  is UNRENDERABLE** → excluded, see `render_manifest_B_excluded.json`).
- **Global episode_index 2438..5990** (contiguous, sorted by uuid), continuing A's 0..2437.
  This single index space == render-scratch key == B-standalone-local + 2438 ==
  object_counts_AB key == the eventual **merged** episode_index.
- Total B frames = **3,098,683** (mean 872/ep). Final corpus A∪B = **5,991 eps**.
- B bottle-count distribution (NOT uniform like A): `{2:649, 3:696, 4:763, 5:746, 6:699}`.
  Merged A∪B: `{2:1130, 3:1208, 4:1234, 5:1242, 6:1177}`.
- A's 24-shard cloud render (2438 eps) spanned **~6h wall** (2026-07-01 02:00→08:01);
  NOTES measured **~110 s/ep** (3-cam 640×480, mixed L4/A10G/L40S/A100 on-demand).

## PREP DONE (this session, no launch)

- `render_manifest_B.json` (3,553 eps) built + uploaded →
  `s3://xdof-internal-research/abc/render_scratch/put_bottles_mjwarp/render_manifest_B.json`
- `object_counts_AB.json` (5,991 eps: A verbatim + B via MjModel) + `object_counts_B.json`
  uploaded to the same prefix. Validated 5/5 vs `randomization.json`.
- `sky/launch_rerender.py` gained `--manifest-s3` (defaults to A's; does NOT change A path).
- All scripts validated on the node (CPU): states_actions reconstruction (3 B eps),
  generalized packager (3 A eps, byte-exact vs A's packaged dataset).

Local artifacts (also in the git repo at `/home/justinyu/abc/`): `make_manifest_B.py`,
`states_actions_from_mcap.py`, `build_states_actions_B.py`, `compute_object_counts_B.py`,
`package_B.py`, `render_manifest_B.json`, `object_counts_AB.json`.

---

## Step 1 — cloud render (LAUNCH after SSO refresh)

Outputs land in the SAME scratch prefix as A, keyed by the B global indices 2438..5990
(non-overlapping with A's 0..2437), so nothing of A's is touched.

```bash
cd /home/justinyu/abc
# 24 shards over global episode_index [2438, 5991); ~148 eps/shard.
# Per sky memories: submit in staggered batches of <=4 (controller jam), verify with
# `sky jobs queue` after EACH batch, no spot.
uv run sky/launch_rerender.py \
  --shards 24 --start 2438 --end 5991 \
  --manifest-s3 s3://xdof-internal-research/abc/render_scratch/put_bottles_mjwarp/render_manifest_B.json \
  --label B
```

**Cost/time (24 shards, on-demand, no spot):**
- Render wall: 3,553 eps × ~110 s/ep / 24 shards ≈ **4.5 h/shard**; expect **~5–8 h**
  end-to-end (A's envelope hit ~6h with stragglers). Bump to `--shards 36` for ~3–5 h.
- Compute: 24 single-GPU instances × ~6 h × ~$1.2–2/h (L4/A10G/L40S) to ~$3–4/h (A100)
  ≈ **$170–430** depending on the capacity mix sky lands.

**Poll + patch failed ranges** (worker is idempotent; safe to re-run a range):
```bash
sky jobs queue
# re-render only a failed [A,B): 
uv run sky/launch_rerender.py --shards 1 --start A --end B \
  --manifest-s3 .../render_manifest_B.json --label Bpatch
# gate before packaging: scratch must have all 3553 B indices (percam + staged)
aws s3 ls s3://xdof-internal-research/abc/render_scratch/put_bottles_mjwarp/percam/ | \
  awk '{print $2}' | sed 's#/##' | awk '$1>=2438 && $1<=5990' | wc -l   # == 3553
```
Schedule a +1h post-launch check (catch step-0 failures before $ burns).

## Step 2 — states/actions for all 3,553 B eps (node, CPU; can run DURING the render)

```bash
# node: ssh shadeform@198.145.127.217 ; cd ~/abc
PYTHONPATH=~/abc .venv/bin/python build_states_actions_B.py \
  --manifest render_manifest_B.json --out-root /mnt/data/staged_put_B --workers 24
# -> /mnt/data/staged_put_B/train_sim/episode_<uuid>/{states_actions.bin, episode_metadata.json}
# (velocity_*.bin come later from scoring, exactly as for A.)
```

## Step 3 — package the B-only LeRobot-v3 dataset (node; needs render scratch + step 2)

```bash
# node
PYTHONPATH=~/abc .venv/bin/python package_B.py \
  --manifest render_manifest_B.json \
  --states-source /mnt/data/staged_put_B \
  --scratch-s3 s3://xdof-internal-research/abc/render_scratch/put_bottles_mjwarp \
  --out-dir /mnt/data/lerobot_B --dataset-name sim_put_bottles_mjwarp_B_only \
  --index-offset 2438 \
  --object-counts-json /mnt/data/object_counts_AB.json \
  --phase all
```
- meta_data phase: seconds. videos phase re-encodes 3,098,683 frames × 3 cams
  (~9.3M frame-encodes). Serial ≈ several hours; parallelize by running `--phase videos`
  per file-group (add a `--groups`/`--cam` split like package_mjwarp if wall matters), or
  run on a many-core box. Expect **~2–4 h** on the node.
- Then sync up:
```bash
aws s3 sync /mnt/data/lerobot_B/sim_put_bottles_mjwarp_B_only \
  s3://xdof-internal-research/repromo/datasets/sim_put_bottles_mjwarp_B_only --only-show-errors
```

## Step 4 — MERGE A∪B (deferred; build the merged LeRobot dataset)

The merge appends B-standalone (local 0..3552, uuid order) after A → global 2438..5990.
**Invariant to preserve (else per-object stratification silently misaligns):**
merged `episode_index` k for k>=2438 must map to B-standalone local k-2438, and
`object_counts_AB[str(k)]` is that episode's bottle count.

Verify after merge:
```python
# merged.episodes.source_episode_id at index 2438 == B-standalone at local 0 == first B uuid
assert merged_ep_df.loc[2438,"source_episode_id"] == "episode_019e1cbe-5b06-7980-9900-4ecd7f9334cd"
```
Merge mechanics (renumber, don't rebuild): shift B's `episode_index += 2438`,
`index += A_total_frames`, `data/file_index` and every `videos/*/file_index += 8`
(A has 8 file-groups 0..7), new `chunk-000` file names continue at `file-00008..`,
concatenate the meta/episodes + data parquets, union the videos trees, and recompute
`meta/stats.json` + the 32-dim RM `norm_stats` over the merged corpus. Use
`object_counts_AB.json` verbatim as the merged `meta/object_counts.json`.

## Step 5 — feature precache for the new B eps

A's 2,438 dinov3 features are cached at `/mnt/data/warp_rm/features`. The RM `--mode
precache` extracts the 3,553 new B eps on first run (top-cam, fs1). If pre-extracting
separately, point `WARP_RM_FEATURE_CACHE` at the same dir.

## Step 6 — RM retrain on A∪B (the coordinator's winning config)

`win32, sss15@30 params, --shortest-frac 0.25 --object-counts-json object_counts_AB.json,
20k steps`, tag `wr_AB_rm_perobj_s25_mjwarp_sss15_20k`. Adapts A's `run_sss15.sh`:

```bash
# node, GPU pinned, pi0 JAX not occupying GPUs
export CUDA_VISIBLE_DEVICES=<free_gpu>
TAG=wr_AB_rm_perobj_s25_mjwarp_sss15_20k
REPO=/mnt/data/RORM
DATA=<MERGED A∪B lerobot root>     # from step 4
CKPT_DIR=/mnt/data/warp_rm/ckpts/$TAG
export PYTHONUNBUFFERED=1 UV_NO_SYNC=1
export WARP_RM_FEATURE_CACHE=/mnt/data/warp_rm/features
export HF_HOME=$HOME/.cache/huggingface
mkdir -p "$CKPT_DIR"; [ -e "$CKPT_DIR/checkpoints" ] || ln -s . "$CKPT_DIR/checkpoints"
cd "$CKPT_DIR"
"$REPO/.venv/bin/python" "$REPO/scripts/train.py" \
  --lerobot-repo "$DATA" --tag "$TAG" --mode precache --ablation no_abs \
  --shortest-frac 0.25 \
  --object-counts-json /mnt/data/warp_rm/object_counts_AB.json \
  --source-standard-stride 15 --window-size 32 --feature-stride 1 \
  --max-steps 20000 --sampler ar --attention bidirectional \
  >> /mnt/data/warp_rm/$TAG.log 2>&1
# inherits bs=1024 lr=4e-4 clean_val=True fusion=concat camera=top_camera-images-rgb
# best ckpt -> $CKPT_DIR/best_model_${TAG}_no_abs.pt
```
(Place a copy of `object_counts_AB.json` at `/mnt/data/warp_rm/object_counts_AB.json`.)

## Step 7 — score the full A∪B corpus with the new RM (AB-RABC sidecar)

Adapts `score_sss15.sh` (window-size 32 explicit, feature-stride 1 load-bearing, SSS
auto-read from ckpt):
```bash
export CUDA_VISIBLE_DEVICES=<free_gpu>
TAG=wr_AB_rm_perobj_s25_mjwarp_sss15_20k ; NAME=rm_AB_perobj_sss15
REPO=/mnt/data/RORM ; DATA=<MERGED A∪B lerobot root>
CKPT=/mnt/data/warp_rm/ckpts/$TAG/best_model_${TAG}_no_abs.pt
export PYTHONUNBUFFERED=1 UV_NO_SYNC=1
cd "$REPO"
"$REPO/.venv/bin/python" scripts/data/write_warp_rm_annotations.py \
  --lerobot-repo "$DATA" --checkpoint "$CKPT" --mode score --tag "$NAME" \
  --window-size 32 --feature-stride 1 --camera-key top_camera-images-rgb \
  --cache-dir /mnt/data/warp_rm/features/dinov3_fs1 \
  --no-extract-fallback --no-inject-lerobot --sidecar-dir /mnt/data/warp_rm/sidecars/$NAME \
  >> /mnt/data/warp_rm/score_${NAME}.log 2>&1
# expect log: "[score] ... 5991/5991 eps have cached features"
# then inject -> velocity_rm_AB_perobj_sss15.bin per the RESEARCH_LOG SCORING RECIPE
# (mode inject from sidecar -> local scored lerobot, then score_to_sidecar.py into staged).
```

---

## Structural surprises / notes for the user

1. **1 unrenderable B ep** (`episode_019e1fcb-9e72-72a1-8996-161fc32e3658`) lacks
   `integration_state.npy` — no qpos render source, no mjgl fallback. Excluded → 3,553
   (not 3,554). This is why indices stop at 5990, not 5991.
2. **No `full_mcap` script existed** — A's staged states_actions came from the mjgl
   LeRobot parquet, which I reproduced from the delivery `output.mcap`:
   `state=[left/right-arm-proprio.position(7 each)]`, `actions=[left/right-command-state
   .position(7 each)]`, and `sa = vstack([src[0:1], src])` (n_msgs == n_integration_rows−1,
   prepend first). Reproduces A to **< 1e-6** (float32 noise) across 5 A eps; the B
   invariant `n_msgs == n_rows−1` holds on all 5 B eps probed.
3. **5-digit file naming, not 4.** A's `info.json` uses `file-{i:05d}` (lex-safe <100k
   files) and the A∪B concat REQUIRES schema-identical naming, so the packager matches A
   (5-digit). The task's "4-digit" is superseded by matching A. B has ~12 file-groups.
4. **RM score columns are float32-zero placeholders** in B's data parquets (overwritten by
   step 7's scoring); present only so the schema matches A for the concat.
5. **integration_state column count varies** (457/494/568…) with bottle count (nq differs);
   harmless — states_actions comes only from `output.mcap`, the render slices `[:, 1:1+nq]`.
6. B's bottle distribution skews high (mode 4–5) vs A's ~uniform 2–6; merged A∪B is balanced.
