#!/usr/bin/env python3
"""SkyPilot launcher for ABC-DiT training (single node, 8xA100, torchrun).

One job template covers: vanilla BC or RABC (reward-aligned reweighting), and
scratch or FT-from-bottles_75k. Staged data + init weights are assumed to be on
S3 already (staged by the local lerobot_to_abc converter; weights uploaded once
via --upload-weights). The worker pulls them, runs torchrun train.py, and syncs
checkpoints back to S3.

Usage:
  uv run sky/launch_abc.py --upload-weights        # one-time: push bottles_75k + norm_stats
  uv run sky/launch_abc.py put_bottles             # vanilla, FT-from-bottles_75k, 30k
  uv run sky/launch_abc.py put_bottles --rabc      # RABC (velocity_repromo.bin, tau=1.0)
  uv run sky/launch_abc.py put_bottles --dry-run
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

S3 = "s3://xdof-internal-research/abc"
ABC_ROOT = "/home/justinyu/abc"

# task key -> (LeRobot repo, ABC task_name/prompt-source)
TASKS = {
    "put_bottles": ("sim_put_the_plastic_bottles_in_the_bin_30hz_gop10", "sim_put_the_plastic_bottles_in_the_bin"),
    "throw_bottles": ("sim_throw_plastic_bottles_in_bin_30hz_gop10", "sim_throw_plastic_bottles_in_bin"),
    "load_plates": ("sim_load_the_plates_into_the_dish_rack_30hz_gop10", "sim_load_the_plates_into_the_dish_rack"),
    "turn_mug": ("sim_turn_the_mug_right_side_up_30hz_gop10", "sim_turn_the_mug_right_side_up"),
    "sweep": ("sim_sweep_away_paper_scraps_from_the_table_30hz_gop10", "sim_sweep_away_paper_scraps_from_the_table"),
}


# Small-model preset (shared by train + eval): shrink the DiT, keep DINOv3 ViT-B (frozen).
SMALL_DIT = "--model.hidden-size 512 --model.depth 12 --model.num-heads 8 --optim.vision-lr-scale 0"


@dataclass
class Cfg:
    task: Annotated[str, tyro.conf.Positional] = "put_bottles"
    small: bool = False                     # small DiT + frozen pretrained DINOv3 ViT-B, scratch, single-GPU
    rabc: bool = False
    velocity_file: str = "velocity_repromo.bin"
    rabc_threshold: float = 1.0
    load_pretrained: bool = True            # FT from bottles_75k (DINO loads from ckpt)
    norm_stats: str = "official"            # "official" (bottles_75k stats) | "task" (staged per-task)
    train_steps: int = 30000
    batch_size: int = 90
    exp_name: Optional[str] = None
    accelerators: List[str] = field(default_factory=lambda: [
        "A100-80GB:8", "A100-80GB:4", "H100:8", "H100:4", "H200:8", "H200:4", "B200:4"])
    region: str = "us-west-2"
    image_id: str = "ami-067cc81f948e50e06"   # openpi DLAMI (us-west-2); torch reinstalled by uv sync
    disk_size: int = 512
    upload_weights: bool = False            # one-time: push cache/bottles_75k.pt + norm_stats.json to S3
    dry_run: bool = False


SETUP = r"""echo "[SETUP] abc-dit env"
conda deactivate 2>/dev/null; conda deactivate 2>/dev/null; true
if command -v sudo >/dev/null 2>&1; then SUDO="sudo"; else SUDO=""; fi
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update && $SUDO apt-get install -y git curl ffmpeg awscli
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env 2>/dev/null || true
uv python pin 3.12
uv sync
# Warm the CLIP ViT-B/32 cache in a single process so the 8 torchrun ranks
# don't race on the first-use download (~/.cache/clip).
uv run python -c "from abc_minimal.config import ClipConfig; from abc_minimal.dit import CLIPTextEmbedder; CLIPTextEmbedder(ClipConfig(), device='cpu'); print('[SETUP] CLIP cached')"
echo "[SETUP] done"
"""

RUN = r"""echo "############ ABC-DiT train ############"
echo "[INFO] task=$SIM_TASK rabc=$RABC ft=$LOAD_PRETRAINED steps=$TRAIN_STEPS exp=$EXP_NAME"
echo "[INFO] node $SKYPILOT_NODE_RANK/$SKYPILOT_NUM_NODES gpus_per_node=$SKYPILOT_NUM_GPUS_PER_NODE"
source $HOME/.local/bin/env 2>/dev/null || true
mkdir -p cache
echo "[INFO] pulling staged data from $STAGED_S3"
aws s3 sync "$STAGED_S3/train_sim" cache/train_sim
aws s3 sync "$STAGED_S3/val_sim" cache/val_sim
if [ "$NORM_STATS_MODE" = "official" ]; then
  aws s3 cp "$WEIGHTS_S3/norm_stats.json" cache/norm_stats.json
else
  aws s3 cp "$STAGED_S3/norm_stats.json" cache/norm_stats.json
fi
if [ "$LOAD_PRETRAINED" = "true" ]; then
  echo "[INFO] FT init: pulling bottles_75k.pt"
  aws s3 cp "$WEIGHTS_S3/bottles_75k.pt" cache/abc_dit_xl_200k_model.pt
else
  echo "[INFO] scratch init: pulling standalone DINOv3 weights"
  aws s3 cp "$WEIGHTS_S3/dinov3_vitb16_pretrain_lvd1689m.pth" cache/dinov3_vitb16_pretrain_lvd1689m.pth
fi
echo "[INFO] train_sim episodes: $(ls cache/train_sim | wc -l)  val_sim: $(ls cache/val_sim | wc -l)"
EXTRA=""
[ "$LOAD_PRETRAINED" = "true" ] && EXTRA="$EXTRA --load-pretrained"
[ "$RABC" = "true" ] && EXTRA="$EXTRA --rabc-enabled --rabc-velocity-file $VELOCITY_FILE --rabc-threshold $RABC_THRESHOLD"
echo "[INFO] torchrun train.py --sim-task $SIM_TASK --train-steps $TRAIN_STEPS --batch-size $BATCH_SIZE $EXTRA"
# Periodic checkpoint upload so intermediate (5k/10k/...) ckpts are eval-able
# mid-run for an early read, not just at job end.
( while true; do sleep 1200; aws s3 sync cache/finetune_checkpoints "$CKPT_S3" 2>/dev/null; done ) &
SYNC_PID=$!
uv run torchrun --standalone --nproc-per-node "$SKYPILOT_NUM_GPUS_PER_NODE" train.py \
  --sim-task "$SIM_TASK" --train-steps "$TRAIN_STEPS" --batch-size "$BATCH_SIZE" $MODEL_FLAGS $EXTRA
TRAIN_EXIT=$?
kill $SYNC_PID 2>/dev/null
echo "[INFO] train exit=$TRAIN_EXIT; final sync to $CKPT_S3"
aws s3 sync cache/finetune_checkpoints "$CKPT_S3"
echo "[OK] done (exit $TRAIN_EXIT)"
exit $TRAIN_EXIT
"""


def sh(cmd: str):
    print(f"[RUN] {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def upload_weights():
    ck = Path(ABC_ROOT) / "cache" / "bottles_75k.pt"
    ns = Path(ABC_ROOT) / "cache" / "norm_stats.json"
    if not ck.exists():
        raise FileNotFoundError(f"{ck} not found — run prepare.py --checkpoint first")
    if not ns.exists():
        raise FileNotFoundError(f"{ns} not found — run prepare.py first")
    sh(f"aws s3 cp {ck} {S3}/weights/bottles_75k.pt")
    sh(f"aws s3 cp {ns} {S3}/weights/norm_stats.json")
    print(f"[OK] weights uploaded to {S3}/weights/")


def main(cfg: Cfg):
    if cfg.upload_weights:
        upload_weights()
        return
    if cfg.task not in TASKS:
        raise SystemExit(f"unknown task {cfg.task}; choices={list(TASKS)}")
    _, sim_task = TASKS[cfg.task]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    arm = "rabc" if cfg.rabc else "vanilla"
    # Small-model preset: shrink the DiT, keep pretrained DINOv3 ViT-B (frozen), scratch DiT,
    # single GPU — fast vanilla-vs-RABC comparison that dodges the 8-GPU capacity blocker.
    model_flags, accelerators = "", cfg.accelerators
    load_pretrained, norm_stats, batch_size = cfg.load_pretrained, cfg.norm_stats, cfg.batch_size
    if cfg.small:
        model_flags = SMALL_DIT
        load_pretrained, norm_stats, batch_size = False, "task", 64
        accelerators = ["A100-80GB:1", "A100:1", "A100-40GB:1", "L40S:1"]
    init = "small" if cfg.small else ("ft" if load_pretrained else "scratch")
    exp = cfg.exp_name or f"abc_{cfg.task}_{arm}_{init}_{ts}"

    # 8/4-GPU 80GB-class instances are scarce in any single AWS region; spread across
    # AWS regions (each needs its own DLAMI) AND Lambda (GPU-focused, often has capacity
    # when AWS is exhausted — matches openpi's [aws, lambda]). any_of fails over across all.
    aws_regions = {"us-west-2": "ami-067cc81f948e50e06", "us-east-1": "ami-0365bff494b18bf93"}
    candidates = [{"infra": f"aws/{region}", "accelerators": a,
                   "disk_size": cfg.disk_size, "image_id": image}
                  for region, image in aws_regions.items() for a in accelerators]
    candidates += [{"infra": "lambda", "accelerators": a, "disk_size": cfg.disk_size}
                   for a in accelerators]
    resources = {"any_of": candidates}

    sky_cfg = {
        "workdir": ABC_ROOT,
        "num_nodes": 1,
        "envs": {
            "SIM_TASK": sim_task,
            "STAGED_S3": f"{S3}/staged/{cfg.task}",
            "WEIGHTS_S3": f"{S3}/weights",
            "CKPT_S3": f"{S3}/ckpts/{exp}",
            "EXP_NAME": exp,
            "RABC": "true" if cfg.rabc else "false",
            "VELOCITY_FILE": cfg.velocity_file,
            "RABC_THRESHOLD": str(cfg.rabc_threshold),
            "LOAD_PRETRAINED": "true" if load_pretrained else "false",
            "NORM_STATS_MODE": norm_stats,
            "TRAIN_STEPS": str(cfg.train_steps),
            "BATCH_SIZE": str(batch_size),
            "MODEL_FLAGS": model_flags,
        },
        "resources": resources,
        "setup": SETUP,
        "run": RUN,
    }

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(sky_cfg, f, default_flow_style=False, sort_keys=False)
        yaml_path = f.name
    print(f"[INFO] exp={exp}  ckpts -> {S3}/ckpts/{exp}")
    if cfg.dry_run:
        print(yaml.dump(sky_cfg, default_flow_style=False, sort_keys=False))
        return
    sh(f"sky jobs launch '{yaml_path}' --yes --async -n {exp}")
    print(f"[OK] launched {exp}")


if __name__ == "__main__":
    main(tyro.cli(Cfg))
