#!/usr/bin/env python3
"""Batch-render sim episodes to drop-in 30 Hz mjwarp-3.10 mcaps from the local archive.

Resumable (skips episodes whose output.mcap already exists), GPU-parallel (one job
per GPU). Runs under any python; shells out to yam_sim venv (scene normalize) and
abc-rabc venv (render_to_mcap).
"""
from __future__ import annotations

import argparse
import glob
import os
import queue
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RD = str(Path(__file__).resolve().parent)
ARCH = "/scratch/current/karimelrafi/sim_archive_20260529/mcap"
YAM = "/home/karimelrafi/yam_sim/.venv/bin/python"
ABC = "/home/karimelrafi/abc-rabc/.venv/bin/python"
NORMALIZE = f"{RD}/normalize_scene.py"
RENDER = f"{RD}/render_to_mcap.py"
REQ = ["scene_assembled.xml", "integration_state.npy", "output.mcap", "timestamp.npy"]

TASKS = [
    "sim_hang_the_mug_on_the_mug_rack",
    "sim_load_the_plates_into_the_dish_rack",
    "sim_sweep_away_paper_scraps_from_the_table",
    "sim_throw_plastic_bottles_in_bin",
    "sim_turn_the_mug_right_side_up",
]

_print_lock = threading.Lock()


def log(msg: str):
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def complete(ep_dir: str) -> bool:
    return all(os.path.exists(os.path.join(ep_dir, r)) for r in REQ)


def process_one(ep_dir: str, task: str, out_root: str, gpu_q: queue.Queue, args) -> tuple[str, str]:
    ep = os.path.basename(ep_dir)
    out_dir = Path(out_root) / task / ep
    out_mcap = out_dir / "output.mcap"
    if out_mcap.exists() and out_mcap.stat().st_size > 0:
        return ep, "skip(exists)"
    if not complete(ep_dir):
        return ep, "skip(incomplete-src)"
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = out_dir / "render.log"
    gpu = gpu_q.get()
    try:
        # render_to_mcap handles asset-root rewrite (xdof-sim) + collision strip on
        # the raw scene_assembled.xml itself, so no separate normalize step.
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        with open(logf, "w") as lf:
            r = subprocess.run(
                [ABC, RENDER, "--episode-dir", ep_dir,
                 "--scene", f"{ep_dir}/scene_assembled.xml",
                 "--out-mcap", str(out_mcap), "--width", str(args.width),
                 "--height", str(args.height), "--fps", str(args.fps),
                 "--gpu-id", "0", "--batch", str(args.batch)],
                stdout=lf, stderr=subprocess.STDOUT, env=env)
            if r.returncode != 0 or not out_mcap.exists():
                return ep, "FAIL(render)"
    finally:
        gpu_q.put(gpu)
    return ep, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="/scratch/current/karimelrafi/rerender_mcap_30hz")
    ap.add_argument("--per-task", type=int, default=25, help="0 = all episodes")
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--gpus", nargs="*", type=int, default=[1, 3, 4, 7])
    ap.add_argument("--jobs-per-gpu", type=int, default=1)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    gpu_q: queue.Queue = queue.Queue()
    for g in args.gpus:
        for _ in range(args.jobs_per_gpu):
            gpu_q.put(g)
    workers = len(args.gpus) * args.jobs_per_gpu

    work = []
    for task in args.tasks:
        eps = sorted(glob.glob(f"{ARCH}/{task}/episode_*"))
        eps = [e for e in eps if os.path.isdir(e)]
        if args.per_task > 0:
            eps = eps[: args.per_task]
        for e in eps:
            work.append((e, task))
    log(f"tasks={len(args.tasks)} episodes={len(work)} workers={workers} out={args.out_root}")

    counts: dict[str, int] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process_one, e, t, args.out_root, gpu_q, args) for e, t in work]
        for fut in futs:
            ep, status = fut.result()
            key = status.split("(")[0]
            counts[status] = counts.get(status, 0) + 1
            done += 1
            if status.startswith("FAIL") or done % 10 == 0:
                log(f"{done}/{len(work)}  {ep} -> {status}   totals={counts}")
    log(f"DONE totals={counts}")


if __name__ == "__main__":
    main()
