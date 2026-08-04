# Local per-task RABC + BC finetuning of the pretrained DiT-L

Finetunes `/scratch/current/karimelrafi/abc_ckpts/lbm_3_5k_dit_l_50000.ckpt` (DiT-L,
1024/24/16, 0.746B) separately on each sim task, producing **two** checkpoints per task:

- **RABC** — reward-aligned BC: chunks weighted by chunk-end `repromo_signed_magnitude`,
  **final-action, threshold τ=1.0, no max clip** (`--rabc-enabled --rabc-threshold 1.0`).
  A pre-filter keeps only chunks with vel > 1.0 (~19–25% per task), so every batch is full.
- **BC** — vanilla behavior cloning over all frames (no reweighting).

Both train **30k steps**, sharded across **all 8 GPUs** with torchrun/DDP.

Everything lives on `/scratch` (home is full). Runs on the shared box `dgx3`.

---

## Current status (snapshot)

| task | RABC | BC |
|------|------|-----|
| put_bottles   | ✅ 30k | ✅ 30k |
| throw_bottles | ✅ 30k | ✅ 30k |
| load_plates   | ✅ 30k | ✅ 30k |
| turn_mug      | ✅ 30k | ⏸ **10k** (stopped early for eval) |
| sweep         | — not started | — not started |
| hang_mug      | — not started | — not started |

Checkpoints: `/scratch/current/karimelrafi/abc_cache/runs/<task>_<arm>/finetune_checkpoints/`
(`5000.pt … 30000.pt` = model+optimizer+norm_stats ~9 GB each; `last.pt` = model-only ~3 GB).

---

## Resume the remaining training

**One command — resumes the whole sweep, safely:**

```bash
cd /home/karimelrafi/abc-rabc
bash local/drive_gpu.sh              # run in background (nohup or a background shell)
```

The driver is **idempotent / resumable**:
- Skips any arm whose `finetune_checkpoints/30000.pt` already exists (the 7 done arms).
- Skips staging any task with a `.staged_ok` marker (put_bottles…turn_mug already staged).
- **Auto-resumes** any interrupted arm from its latest numbered checkpoint (see below).
- For the remaining work it will: **resume turn_mug_bc from 10k → 30k**, then **stage sweep →
  sweep RABC → sweep BC → stage hang_mug → hang_mug RABC → hang_mug BC**.

Conversion (staging) happens with the GPUs idle — it never overlaps a training run
(overlapping the two OOM-killed an early run; see Gotchas).

> ⚠️ **Run this only when the GPUs are free.** It launches an 8-GPU torchrun per arm and
> uses ~40 GB/GPU. Don't start it while evaluations are using the GPUs.

### Lossless mid-run resume (`--resume`)

`train.py` supports `--resume <ckpt>`: it loads model + optimizer + step from an intermediate
checkpoint (overriding the pretrained init) and continues to `--train-steps`, fast-forwarding the
LR schedule. `run_arm.sh` **auto-detects** this: if an arm has a numbered checkpoint but no
`30000.pt`, it passes `--resume <latest numbered .pt>` automatically. So `drive_gpu.sh` will
continue **turn_mug_bc from `10000.pt` → 30000** with optimizer state intact — no lost steps.

Notes:
- Resume uses the latest **numbered** checkpoint (`10000.pt`, …) — those carry optimizer state.
  `last.pt` is model-only and would resume without optimizer momentum.
- Disable auto-resume with `RESUME=off` (e.g. to force a clean restart of an arm).
- Data-order/seed restart from epoch 0 on resume; fine for finetuning.

---

## Run a single arm or stage a single task

```bash
# one arm on all 8 GPUs (rabc | bc). Creates runs/<task>_<arm>/ with symlinks to shared data+ckpt.
local/run_arm.sh <task_key> <sim_task> <rabc|bc>
#   env overrides: STEPS=30000  BS=48 (per-gpu)  NGPU=8  NUM_WORKERS=8  COMPILE=1  RESUME=off
#   (auto-resumes from the latest numbered ckpt unless RESUME=off or 30000.pt exists)

# stage one task's LeRobot data -> ABC train_sim/val_sim + velocity sidecars (writes .staged_ok)
local/convert_task.sh <task_key> <lerobot_dirname>
#   env: WORKERS=48  VALCOUNT=50
```

### task_key → sim_task → LeRobot dirname

| task_key | sim_task (prompt/task_name) | lerobot dirname |
|----------|------------------------------|-----------------|
| put_bottles   | sim_put_the_plastic_bottles_in_the_bin      | sim_put_the_plastic_bottles_in_the_bin_30hz_gop10 |
| throw_bottles | sim_throw_plastic_bottles_in_bin            | sim_throw_plastic_bottles_in_bin_30hz_gop10 |
| load_plates   | sim_load_the_plates_into_the_dish_rack      | sim_load_the_plates_into_the_dish_rack_30hz_gop10 |
| turn_mug      | sim_turn_the_mug_right_side_up              | sim_turn_the_mug_right_side_up_30hz_gop10 |
| sweep         | sim_sweep_away_paper_scraps_from_the_table  | sim_sweep_away_paper_scraps_from_the_table_30hz_gop10 |
| hang_mug      | sim_hang_the_mug_on_the_mug_rack            | sim_hang_the_mug_on_the_mug_rack_30hz_gop10 |

(sim_task = dirname minus `_30hz_gop10`.)

---

## Gotchas baked into these scripts (don't remove)

- **`MALLOC_ARENA_MAX=2` + `MALLOC_TRIM_THRESHOLD_=0`** (exported in `run_arm.sh`). Without
  this, torchcodec/ffmpeg threads in the dataloader workers fragment the glibc heap and leak
  ~45 MB/batch → ~1.3 TB over a run → the OOM-killer kills a worker and the run dies with
  `RuntimeError: DataLoader worker ... killed by signal: Killed`. With it, worker RSS is flat.
- **Never overlap conversion with training** — combined CPU-RAM pressure OOMs. The driver
  stages just-in-time with GPUs idle.
- **Everything on `/scratch`** — `/home` is 100% full. Compile/triton caches and CLIP assets
  are redirected to `/scratch/current/karimelrafi/abc_compile_cache` and `.../abc_cache/_shared/clip`.
- **Shared box** — other users co-tenant the GPUs; it/s swings ~2–3.4 depending on their load.
  Keep `NUM_WORKERS=8` (8×8=64 workers) to bound RAM. bs=48/gpu uses ~40 GB (safe alongside others).
- **norm_stats** = the checkpoint's embedded `sim_tasks_20260514_mjwarp_224` set, extracted once to
  `_shared/ditl_sim_norm_stats.json` (same for all tasks; matches how the model was pretrained).

---

## Key paths

```
/scratch/current/karimelrafi/abc_ckpts/lbm_3_5k_dit_l_50000.ckpt   # pretrained init (read-only)
/scratch/current/karimelrafi/lerobot_mjgl_30hz_full/<dirname>/     # raw LeRobot v3 data
/scratch/current/karimelrafi/abc_cache/<task_key>/{train_sim,val_sim,.staged_ok}   # staged
/scratch/current/karimelrafi/abc_cache/_shared/ditl_sim_norm_stats.json            # norm stats
/scratch/current/karimelrafi/abc_cache/runs/<task>_<arm>/finetune_checkpoints/     # outputs
/scratch/current/karimelrafi/abc_cache/runs/<task>_<arm>.log                       # per-arm log
```
