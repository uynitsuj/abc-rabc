# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "tyro"]
# ///
"""Merge sharded sim-eval summaries (from sky/launch_eval.py) into one combined
summary + the WARP-RM continuous metrics. Pulls shard*/summary.json from S3,
concatenates the per-world results, recomputes success_rate/bottles, and runs the
per-bottle-time/throughput analysis over the union.

  uv run merge_eval.py --evals-s3 s3://xdof-internal-research/abc/evals/<label>/
"""
import glob
import json
import subprocess
import tempfile
from dataclasses import dataclass

import numpy as np
import tyro

from eval_metrics import analyze


@dataclass
class Cfg:
    evals_s3: str                      # s3://.../abc/evals/<label>/  (has shardNN/summary.json)
    out: str = "merged_summary.json"


def main(cfg: Cfg):
    tmp = tempfile.mkdtemp()
    subprocess.run(
        f"aws s3 sync {cfg.evals_s3.rstrip('/')}/ {tmp} --exclude '*' --include '*summary.json'",
        shell=True, check=True)
    shards = sorted(glob.glob(f"{tmp}/shard*/summary.json"))
    if not shards:
        raise SystemExit(f"no shard summaries under {cfg.evals_s3}")
    worlds, base = [], None
    for s in shards:
        d = json.load(open(s))
        base = base or d
        worlds += d.get("worlds", [])
    succ = np.array([w["success"] for w in worlds], dtype=bool)
    rew = np.array([w["reward"] for w in worlds], dtype=float)
    mb = np.array([w["final_task_eval"]["max_bottles_in_bin_so_far"] for w in worlds], dtype=float)
    base["worlds"] = worlds
    base["success_rate"] = float(succ.mean())
    base["num_success"] = int(succ.sum())
    base["num_worlds"] = len(worlds)
    base["mean_reward"] = float(rew.mean())
    base["mean_max_bottles_in_bin"] = float(mb.mean())
    json.dump(base, open(cfg.out, "w"))
    print(f"MERGED {len(shards)} shards -> {len(worlds)} worlds")
    print(f"  success_rate = {base['success_rate']:.3f}  ({base['num_success']}/{len(worlds)})")
    print(f"  mean_max_bottles = {base['mean_max_bottles_in_bin']:.2f}/6   mean_reward = {base['mean_reward']:.3f}")
    r = analyze(cfg.out)
    print(f"  bottles_placed = {r['bottles_placed']}/{r['possible']}")
    print(f"  mean time/bottle = {r['mean_time_per_bottle_s']:.1f} s   throughput = {r['throughput_per_hr']:.1f} bottles/hr")


if __name__ == "__main__":
    main(tyro.cli(Cfg))
