# 30 Hz mjwarp re-render pipeline for delivered sim tasks

Re-renders the `sim_tasks_20260514` deliveries at 30 Hz with the **mjwarp 3.10
eval-faithful renderer** (identical config to `abc_minimal/eval_policy.py:MJWarpSim`:
`create_render_context(use_textures=True, use_shadows=True)`), writes drop-in
delivery-format mcaps, and converts them to LeRobot v3 at fps=30.

Result (2026-07): 12,039 re-rendered mcaps / 12,031 LeRobot episodes across the
5 non-put_bottles tasks. Verified pixel-faithful to the published
`uynitsuj/sim-bottles-mjwarp-v1` dataset + eval renderer (MAE ~0.4/255 vs
eval-faithful render; matches mjgl within ~3.5/255).

## Environments (nothing here is standalone — three venvs)

| stage | venv | why |
|---|---|---|
| rendering (`render_to_mcap.py`, `eval_faithful_render.py`) | `~/abc-rabc/.venv` | mujoco 3.10 + mujoco_warp with the fixed shader. **The yam_sim venv (mujoco 3.8) renderer is broken** — blue-tinted hemispheric ambient, blown highlights. |
| LeRobot conversion (`convert_to_lerobot_30hz.py`) | `~/yam_sim/.venv` | mcap/pandas/pyarrow deps; converter derived from `yam_sim/scripts/convert_to_lerobot.py` with `FPS = 30`. |
| assets | `~/yam_sim/yam_sim/models/` (checkout, not venv) | correct dark-gray i2rt_yam grippers + full task_dishrack set. **xdof-sim's copy renders the grippers bright blue — do not use.** |

## Data locations

- raw deliveries (source): `/scratch/current/karimelrafi/sim_archive_20260529/mcap/<task>/episode_<uuid>/` (moved out of `2026_05` on 2026-08-04; hop it forward when scratch months rotate)
- re-rendered 30 Hz mcaps: `/scratch/current/karimelrafi/rerender_mcap_30hz/<task>/<episode>/output.mcap`
- LeRobot v3 datasets: `/scratch/current/karimelrafi/rerender_lerobot_30hz/<task>/` (symlinked into `~/datasets/` for the repromo webui)
- scratch preview/cache artifacts (not in repo): `~/rerender_demos/`

## Asset resolution (`render_to_mcap.py:load_render_model`) — CRITICAL

Delivered `scene_assembled.xml` uses absolute `/home/xdof/.../xdof_sim/models/` paths.

1. Rewrite the asset root to the local **yam_sim** checkout (never xdof-sim, see above).
2. Apply dishrack/plate variant-dir aliases from `dishrack_aliases.json`
   (e.g. `DishRack050` → `dish_rack_12`; dumped from
   `yam_sim.randomization._DISHRACK_VARIANT_ALIASES`). Do **not** use yam_sim's
   `_normalize_recorded_scene_xml` — its `legacy_asset_prefixes` step mis-maps
   plate_19/20 + dish_rack_10 into an empty assets_robocasa layout.
3. Repoint any missing mesh/texture **file** at `_stub/cube.obj` / `_stub/white.png`.
   Do **not** remove `<mesh>`/`<geom>`/`<texture>` elements: removal re-indexes
   mjwarp's mesh/texcoord arrays and corrupts texture sampling on *other* geoms
   (the blue-gripper bug). Missing meshes are collision-only (group 3, never drawn
   with `enabled_geom_groups=[0,1,2]`); missing textures are phantom refs on flat
   materials — the stubs don't change a rendered pixel.
4. Fallback `<inertial>` on jointed bodies lacking one (zero-mass compile guard;
   kinematic replay never uses it).

Replay is kinematics-only: `mj_setState(mjSTATE_INTEGRATION)` then
`mjw.kinematics/com_pos/camlight/refit_bvh/render` — collision geometry never
affects pixels.

## Pipeline

```
# 1. batch re-render (resumable; skips existing output.mcap)
./start_all5.sh                      # -> run_mcap_batch.py -> render_to_mcap.py per episode

# 2. convert to LeRobot v3 (waits for render DONE marker, then 5 parallel)
./convert_all.sh

# 3. inspect
./start_viewer.sh                    # side-by-side mp4 viewer :8777
./start_repromo_webui.sh             # repromo webui :8778 (reads ~/datasets symlinks)
python mcap_to_mp4.py --mcap <output.mcap> --out-dir <dir>   # extract cameras
```

Drop-in mcap format: camera topics replaced with re-rendered 30 Hz
`foxglove.CompressedVideo` H264 (one access unit per message, AUD-delimited,
`keyint=10:scenecut=0`); every non-camera topic passed through verbatim; frame
log_times linearly spaced across the source top-camera span.

## Known unrenderable episodes (skipped)

- 7 incomplete source deliveries (missing required files)
- 7 turn_mug scene/state DOF mismatches (`state width != mjSTATE_INTEGRATION`)
- ~3 load_plates episodes referencing `plate_0` (exists only in the old
  `model.obj`+`texture.png` layout, scene expects `visual/model_normalized_*.obj`)

## Historical one-off scripts

`run_all.sh` / `run_compare.sh` / `run_rest.sh` / `run_turnmug.sh` /
`run_eval310.sh` / `normalize_scene.py` are the original mjwarp-vs-mjgl
comparison experiments (single episodes via yam_sim's `rerender_episode.py`,
S3 download path). Kept for provenance; superseded by the batch pipeline above.
