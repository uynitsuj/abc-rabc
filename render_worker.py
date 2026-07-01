# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mujoco", "mujoco-warp", "warp-lang", "tyro", "boto3"]
# ///
"""Cloud shard worker for the mjwarp put-bottles re-render.

Reads a render manifest (episode_index -> {uuid, split, length, delivery}) from S3,
selects its [shard_start, shard_end) episode-index range, downloads each episode's
scene_assembled.xml + integration_state.npy from the delivery bucket, renders with
rerender_mjwarp.render_episode (per-cam native 640x480 videos + staged combined.mp4),
and uploads per-episode outputs to the render-scratch S3 prefix.

Layout uploaded (keyed by episode_index, zero-padded, so the reduce reads in order):
  <scratch>/percam/<epidx06>/{top,left,right}_camera-images-rgb.mp4   (640x480 CFR-30)
  <scratch>/staged/<epidx06>/combined_camera-images-rgb.mp4          (224x504 strict)
  <scratch>/staged/<epidx06>/episode_metadata.json

Invoke on the worker with the project env (NOT an isolated PEP-723 env, which would break
the `from export_mcap import ...` inside rerender_mjwarp):
    uv run python render_worker.py --shard-start 0 --shard-end 100 ...
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import boto3
import tyro
from botocore.config import Config as BotoConfig

import rerender_mjwarp as R

DELIVERY_BUCKET = "xdof-bair-abc"
SUBTASK = "sim_put_the_plastic_bottles_in_the_bin"


@dataclass
class Cfg:
    shard_start: int          # inclusive episode_index
    shard_end: int            # exclusive episode_index
    manifest_s3: str          # s3://.../render_manifest.json (epidx -> {uuid,split,length,delivery})
    scratch_s3: str           # s3://.../abc/render_scratch/put_bottles_mjwarp
    asset_root: str = "/home/ubuntu/render_assets"  # local dir the launcher syncs assets into
    work_dir: str = "/tmp/rw_work"
    gpu_id: int = 0
    overwrite: bool = False


def _split_s3(uri: str) -> tuple[str, str]:
    assert uri.startswith("s3://"), uri
    b, _, k = uri[5:].partition("/")
    return b, k


def main(cfg: Cfg) -> None:
    s3 = boto3.client("s3", config=BotoConfig(max_pool_connections=32, retries={"max_attempts": 8}))
    mb, mk = _split_s3(cfg.manifest_s3)
    manifest = json.loads(s3.get_object(Bucket=mb, Key=mk)["Body"].read())
    sb, sk = _split_s3(cfg.scratch_s3.rstrip("/"))

    idxs = [i for i in range(cfg.shard_start, cfg.shard_end) if str(i) in manifest]
    idxs = [i for i in idxs if manifest[str(i)].get("delivery")]  # skip un-renderable (no scene)
    print(f"[worker] shard [{cfg.shard_start},{cfg.shard_end}) -> {len(idxs)} renderable episodes",
          flush=True)

    work = Path(cfg.work_dir)
    deliv_root = work / "deliv"
    staged_out = work / "staged"
    percam_out = work / "percam"
    for d in (deliv_root, staged_out, percam_out):
        d.mkdir(parents=True, exist_ok=True)

    base_cfg = R.Config(
        delivery_root=deliv_root, out_dir=staged_out, staged_src=None,
        percam_dir=percam_out, asset_root=cfg.asset_root, gpu_id=cfg.gpu_id,
        overwrite=cfg.overwrite,
    )

    done, failed = 0, []
    for i in idxs:
        info = manifest[str(i)]
        uuid, deliv = info["uuid"], info["delivery"]
        eppad = f"{i:06d}"
        # Idempotency: skip if all 3 percam + combined already uploaded for this index.
        want = [f"{sk}/percam/{eppad}/{R.CAM_KEY[c]}.mp4" for c in R.CAMERAS] + \
               [f"{sk}/staged/{eppad}/combined_camera-images-rgb.mp4"]
        if not cfg.overwrite and all(_exists(s3, sb, k) for k in want):
            print(f"[worker] skip {eppad} ({uuid}) — already on S3", flush=True)
            done += 1
            continue
        # Download the delivery episode (scene + qpos only).
        ep_deliv = deliv_root / f"episode_{uuid}" if not uuid.startswith("episode_") else deliv_root / uuid
        ep_deliv.mkdir(parents=True, exist_ok=True)
        prefix = f"data/deliveries/{deliv}/{SUBTASK}/{uuid}"
        try:
            for fn in ("scene_assembled.xml", "integration_state.npy"):
                s3.download_file(DELIVERY_BUCKET, f"{prefix}/{fn}", str(ep_deliv / fn))
        except Exception as e:
            print(f"[worker][FAIL-DL] {eppad} {uuid}: {e}", flush=True)
            failed.append(i)
            continue
        # Render (percam + combined) via the shared code path.
        t0 = time.time()
        try:
            R.render_episode(ep_deliv, base_cfg)
        except Exception as e:
            print(f"[worker][FAIL-RENDER] {eppad} {uuid}: {type(e).__name__}: {e}", flush=True)
            failed.append(i)
            _rmtree(ep_deliv)
            continue
        # Upload outputs under zero-padded index keys.
        src_ep = staged_out / ep_deliv.name
        pc_ep = percam_out / ep_deliv.name
        try:
            for c in R.CAMERAS:
                s3.upload_file(str(pc_ep / f"{R.CAM_KEY[c]}.mp4"), sb,
                               f"{sk}/percam/{eppad}/{R.CAM_KEY[c]}.mp4")
            s3.upload_file(str(src_ep / "combined_camera-images-rgb.mp4"), sb,
                           f"{sk}/staged/{eppad}/combined_camera-images-rgb.mp4")
            s3.upload_file(str(src_ep / "episode_metadata.json"), sb,
                           f"{sk}/staged/{eppad}/episode_metadata.json")
        except Exception as e:
            print(f"[worker][FAIL-UP] {eppad} {uuid}: {e}", flush=True)
            failed.append(i)
        else:
            done += 1
            print(f"[worker][UP] {eppad} {uuid} in {time.time()-t0:.0f}s "
                  f"({done}/{len(idxs)})", flush=True)
        # Free disk: drop this episode's inputs + outputs immediately.
        _rmtree(ep_deliv); _rmtree(src_ep); _rmtree(pc_ep)

    print(f"[worker] DONE shard [{cfg.shard_start},{cfg.shard_end}): "
          f"uploaded {done}/{len(idxs)}, failed={failed}", flush=True)
    if failed:
        raise SystemExit(f"[worker] {len(failed)} episodes FAILED: {failed}")


def _exists(s3, bucket, key) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def _rmtree(p: Path) -> None:
    import shutil
    shutil.rmtree(p, ignore_errors=True)


if __name__ == "__main__":
    main(tyro.cli(Cfg))
