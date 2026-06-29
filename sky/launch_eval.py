#!/usr/bin/env python3
"""Data-parallel MuJoCo-Warp sim eval on SkyPilot.

The eval is embarrassingly parallel across worlds (each is an independent seeded
rollout; per-world MJWarp stepping is the bottleneck). This shards `num_worlds`
across K single-GPU jobs with DISJOINT seed ranges, each running eval_policy.py +
uploading its summary.json + world videos to S3. Aggregate with merge_eval.py.
~K-fold wall-clock speedup vs one sequential local eval.

  uv run sky/launch_eval.py s3://.../abc/ckpts/<exp>/15000.pt --small --shards 8 --worlds-per-shard 6
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, List, Optional

import tyro
import yaml

from launch_abc import SETUP  # reuse the env setup (uv sync + CLIP warm)

S3 = "s3://xdof-internal-research/abc"
ABC_ROOT = "/home/justinyu/abc"
SMALL_MODEL_FLAGS = "--model.hidden-size 512 --model.depth 12 --model.num-heads 8"


@dataclass
class Cfg:
    checkpoint: Annotated[str, tyro.conf.Positional]   # s3:// path to the .pt checkpoint
    small: bool = True                                  # small-DiT eval config (must match the trained ckpt)
    shards: int = 8
    worlds_per_shard: int = 6
    seed_base: int = 20260511
    num_chunks: int = 120
    label: Optional[str] = None                         # eval out path label (default from ckpt)
    accelerators: List[str] = field(default_factory=lambda: [
        "A100-80GB:1", "A100:1", "A100-40GB:1", "L40S:1", "A10G:1"])
    dry_run: bool = False


RUN = r"""echo "############ sim eval shard ############"
source $HOME/.local/bin/env 2>/dev/null || true
echo "[EVAL] ckpt=$CKPT seed=$SEED worlds=$WORLDS chunks=$NUM_CHUNKS out=$OUT_S3"
uv run eval_policy.py --checkpoint "$CKPT" $MODEL_FLAGS \
  --num-worlds "$WORLDS" --seed "$SEED" --num-chunks "$NUM_CHUNKS" \
  --save-video --output-dir outputs/eval_shard
EVAL_EXIT=$?
echo "[EVAL] exit=$EVAL_EXIT; uploading to $OUT_S3"
aws s3 sync outputs/eval_shard "$OUT_S3"
echo "[EVAL] done"
exit $EVAL_EXIT
"""


def sh(cmd: str):
    print(f"[RUN] {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main(cfg: Cfg):
    label = cfg.label or Path(cfg.checkpoint).parent.name + "_" + Path(cfg.checkpoint).stem
    aws_regions = {"us-west-2": "ami-067cc81f948e50e06", "us-east-1": "ami-0365bff494b18bf93"}
    model_flags = SMALL_MODEL_FLAGS if cfg.small else ""
    for k in range(cfg.shards):
        seed = cfg.seed_base + k * cfg.worlds_per_shard
        out_s3 = f"{S3}/evals/{label}/shard{k:02d}"
        candidates = [{"infra": f"aws/{r}", "accelerators": a, "disk_size": 256, "image_id": img}
                      for r, img in aws_regions.items() for a in cfg.accelerators]
        candidates += [{"infra": "lambda", "accelerators": a, "disk_size": 256} for a in cfg.accelerators]
        sky_cfg = {
            "workdir": ABC_ROOT, "num_nodes": 1,
            "envs": {"CKPT": cfg.checkpoint, "MODEL_FLAGS": model_flags,
                     "WORLDS": str(cfg.worlds_per_shard), "SEED": str(seed),
                     "NUM_CHUNKS": str(cfg.num_chunks), "OUT_S3": out_s3},
            "resources": {"any_of": candidates}, "setup": SETUP, "run": RUN,
        }
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(sky_cfg, f, default_flow_style=False, sort_keys=False)
            ypath = f.name
        name = f"eval_{label}_s{k:02d}"
        if cfg.dry_run:
            print(f"[dry] shard {k}: seed {seed}..{seed+cfg.worlds_per_shard-1} -> {out_s3}")
            continue
        sh(f"sky jobs launch '{ypath}' --yes --async -n {name}")
    total = cfg.shards * cfg.worlds_per_shard
    print(f"[OK] launched {cfg.shards} eval shards x {cfg.worlds_per_shard} worlds = {total} worlds")
    print(f"     summaries -> {S3}/evals/{label}/shard*/summary.json   (merge with merge_eval.py)")


if __name__ == "__main__":
    main(tyro.cli(Cfg))
