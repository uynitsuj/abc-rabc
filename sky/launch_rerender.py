#!/usr/bin/env python3
"""Data-parallel MuJoCo-Warp re-render of the put-bottles dataset on SkyPilot.

Re-renders the 2438 dataset-A episodes (the exact set in staged_put/{train_sim,val_sim},
= mjgl LeRobot episode_index 0..2437) from mjgl into mjwarp at native 640x480, emitting
BOTH per-camera LeRobot videos and the staged 224x504 combined.mp4. Embarrassingly parallel
across episodes: shards the episode-index range across K single-GPU jobs, each running
render_worker.py over its slice and uploading per-episode outputs to the render-scratch S3
prefix. Reduce + package locally (package_mjwarp.py) once all shards finish.

  uv run sky/launch_rerender.py --shards 24
  uv run sky/launch_rerender.py --shards 1 --start 0 --end 20 --label smoke   # cloud smoke
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import tyro
import yaml

from launch_abc import SETUP  # reuse env setup (uv sync brings mujoco-warp + warp-lang)

S3 = "s3://xdof-internal-research/abc"
ABC_ROOT = "/home/justinyu/abc"
MANIFEST_S3 = f"{S3}/render_scratch/put_bottles_mjwarp/render_manifest.json"
SCRATCH_S3 = f"{S3}/render_scratch/put_bottles_mjwarp"
ASSETS_S3 = f"{S3}/render_assets"
TOTAL_EPISODES = 2438  # episode_index 0..2437


@dataclass
class Cfg:
    shards: int = 24
    start: int = 0                 # inclusive episode_index (default: whole dataset)
    end: int = TOTAL_EPISODES      # exclusive
    label: str = "full"
    overwrite: bool = False
    # Single-world mjwarp render is light on VRAM/compute; any modern single GPU works.
    accelerators: List[str] = field(default_factory=lambda: [
        "L4:1", "A10G:1", "A10:1", "L40S:1", "A100:1", "A100-40GB:1"])
    disk_size: int = 128
    dry_run: bool = False


RUN = r"""echo "############ mjwarp re-render shard ############"
source $HOME/.local/bin/env 2>/dev/null || true
echo "[RENDER] shard [$SHARD_START,$SHARD_END) manifest=$MANIFEST_S3 scratch=$SCRATCH_S3"
mkdir -p /home/ubuntu/render_assets
echo "[RENDER] syncing render assets from $ASSETS_S3"
aws s3 sync "$ASSETS_S3" /home/ubuntu/render_assets --only-show-errors
echo "[RENDER] asset dirs: $(ls /home/ubuntu/render_assets)"
# `uv sync` (in SETUP) rebuilds the env from uv.lock, which does NOT include boto3 — add it
# directly into the synced venv (uv pip install does not re-resolve the lock or the cu128
# torch default). Then run the worker with .venv/bin/python: NOT `uv run` (re-syncs to the
# lock and drops the boto3 we just added) and NOT `uv run render_worker.py` (isolated PEP-723
# env breaks the `from export_mcap import ...` inside rerender_mjwarp).
uv pip install --python .venv/bin/python boto3 --quiet
PYTHONUNBUFFERED=1 .venv/bin/python render_worker.py \
  --shard-start "$SHARD_START" --shard-end "$SHARD_END" \
  --manifest-s3 "$MANIFEST_S3" --scratch-s3 "$SCRATCH_S3" \
  --asset-root /home/ubuntu/render_assets $OVERWRITE_FLAG
RW_EXIT=$?
echo "[RENDER] worker exit=$RW_EXIT"
exit $RW_EXIT
"""


def sh(cmd: str):
    print(f"[RUN] {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main(cfg: Cfg):
    aws_regions = {"us-west-2": "ami-067cc81f948e50e06", "us-east-1": "ami-0365bff494b18bf93"}
    total = cfg.end - cfg.start
    # Contiguous episode-index slices; each shard renders a disjoint range.
    per = -(-total // cfg.shards)  # ceil
    launched = 0
    for k in range(cfg.shards):
        s = cfg.start + k * per
        e = min(cfg.start + (k + 1) * per, cfg.end)
        if s >= e:
            break
        candidates = [{"infra": f"aws/{r}", "accelerators": a, "disk_size": cfg.disk_size,
                       "image_id": img}
                      for r, img in aws_regions.items() for a in cfg.accelerators]
        candidates += [{"infra": "lambda", "accelerators": a, "disk_size": cfg.disk_size}
                       for a in cfg.accelerators]
        sky_cfg = {
            "workdir": ABC_ROOT, "num_nodes": 1,
            "envs": {
                "SHARD_START": str(s), "SHARD_END": str(e),
                "MANIFEST_S3": MANIFEST_S3, "SCRATCH_S3": SCRATCH_S3, "ASSETS_S3": ASSETS_S3,
                "OVERWRITE_FLAG": ("--overwrite" if cfg.overwrite else ""),
            },
            "resources": {"any_of": candidates}, "setup": SETUP, "run": RUN,
        }
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(sky_cfg, f, default_flow_style=False, sort_keys=False)
            ypath = f.name
        name = f"rerender_{cfg.label}_s{k:02d}"
        if cfg.dry_run:
            print(f"[dry] shard {k}: episode_index [{s},{e}) -> {name}")
            continue
        sh(f"sky jobs launch '{ypath}' --yes --async -n {name}")
        launched += 1
    print(f"[OK] launched {launched} shards over episode_index [{cfg.start},{cfg.end}); "
          f"outputs -> {SCRATCH_S3}/{{percam,staged}}/<epidx06>/  (reduce with package_mjwarp.py)")


if __name__ == "__main__":
    main(tyro.cli(Cfg))
