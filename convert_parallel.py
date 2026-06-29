# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pyarrow", "pandas", "tyro"]
# ///
"""Parallel local LeRobot->ABC conversion: split episodes across N processes into
train_sim/val_sim, then compute per-task norm_stats over the train split.

Each worker process is the validated single-threaded lerobot_to_abc.py over a disjoint
episode range; all write into the same train_sim/ (distinct episode subdirs -> no
collision). norm_stats is computed in a final pass (FT runs use official stats; this
covers the scratch option).

  uv run python convert_parallel.py <lerobot_dir> <out_dir> --task-name <name> --val-count 100 --workers 8
"""
from __future__ import annotations

import glob
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tyro


@dataclass
class Cfg:
    data_dir: str
    out_dir: str
    task_name: str = "sim_put_the_plastic_bottles_in_the_bin"
    val_count: int = 100
    workers: int = 8


def _convert(data_dir, out_sub, eps_range, task_name):
    return subprocess.Popen(
        [sys.executable, "lerobot_to_abc.py", data_dir, out_sub,
         "--episodes", eps_range, "--task-name", task_name, "--no-write-norm-stats"],
        cwd="/home/justinyu/abc",
    )


def compute_norm_stats(train_dir, out_path):
    s = np.zeros(28); ss = np.zeros(28); n = 0
    for f in glob.glob(f"{train_dir}/episode_*/states_actions.bin"):
        a = np.fromfile(f, dtype=np.float64).reshape(-1, 28)
        s += a.sum(0); ss += (a ** 2).sum(0); n += len(a)
    if not n:
        print("[norm_stats] no train episodes; skipped"); return
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean ** 2, 0.0))
    json.dump({"norm_stats": {
        "state": {"mean": mean[:14].tolist(), "std": std[:14].tolist()},
        "actions": {"mean": mean[14:].tolist(), "std": std[14:].tolist()},
    }}, open(out_path, "w"), indent=2)
    print(f"[norm_stats] over {n} frames -> {out_path}")


def main(cfg: Cfg):
    metas = sorted(glob.glob(f"{cfg.data_dir}/meta/episodes/**/*.parquet", recursive=True))
    eps = sorted(int(x) for x in pq.read_table(metas, columns=["episode_index"]).to_pandas()["episode_index"])
    lo, hi = eps[0], eps[-1]
    val_hi = lo + cfg.val_count - 1
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    procs = [_convert(cfg.data_dir, f"{cfg.out_dir}/val_sim", f"{lo}-{val_hi}", cfg.task_name)]
    train_lo = val_hi + 1
    total = hi - train_lo + 1
    chunk = (total + cfg.workers - 1) // cfg.workers
    for i in range(cfg.workers):
        a = train_lo + i * chunk
        b = min(train_lo + (i + 1) * chunk - 1, hi)
        if a > b:
            continue
        procs.append(_convert(cfg.data_dir, f"{cfg.out_dir}/train_sim", f"{a}-{b}", cfg.task_name))
    print(f"{len(eps)} episodes [{lo}..{hi}]; val [{lo}..{val_hi}], train [{train_lo}..{hi}] over {len(procs)-1} workers")
    rc = [p.wait() for p in procs]
    tr = len(glob.glob(f"{cfg.out_dir}/train_sim/episode_*"))
    va = len(glob.glob(f"{cfg.out_dir}/val_sim/episode_*"))
    print(f"CONVERT rc={rc}")
    compute_norm_stats(f"{cfg.out_dir}/train_sim", f"{cfg.out_dir}/norm_stats.json")
    print(f"CONVERT_DONE train={tr} val={va} all_ok={all(r == 0 for r in rc)}")


if __name__ == "__main__":
    main(tyro.cli(Cfg))
