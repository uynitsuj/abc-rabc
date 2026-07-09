# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mujoco", "mujoco-warp", "warp-lang", "tyro", "boto3"]
# ///
"""Node-local shard worker for the 60Hz TOP-camera-only mjwarp re-render (RM training).

Mirrors render_worker.py (the cloud shard driver) but renders on a local GPU and writes
into a local dataset dir instead of S3 scratch. Per episode it downloads
scene_assembled.xml + integration_state.npy from the delivery bucket, renders the top
camera at fps_mult=2 (one slerp/lerp midpoint frame between each consecutive qpos row;
output = 2N-1 frames, t30[i] -> t60[2i] EXACT), encodes CFR-60, and moves the result to:

  <out_root>/<epidx06>/top_camera-images-rgb.mp4   (640x480 CFR-60, 2N-1 frames)
  <out_root>/<epidx06>/episode_metadata.json       (uuid/split/frame counts/mapping)

keyed by zero-padded episode_index — the same keying the package_mjwarp.py reduce consumes
(scratch percam/<epidx06>/), so later packaging/feature extraction reads episodes in order.
Shared progress log (all shards append single lines, O_APPEND):
  <out_root>/render.log     "<epidx06> uuid=<uuid> frames=<2N-1> src=<N> sec=<t>"

Resumable: an episode is skipped iff its final mp4 already probes to the expected 2N-1
frames (safe against partial writes: renders land in a work dir and are moved into place).

Shards select episodes by INTERLEAVE (episode_index % num_shards == shard) so lanes stay
balanced and a partial fleet still makes uniform progress.

Run with the project venv python (NOT `uv run` — it would re-sync the env under live jobs):
  cd ~/abc && nohup .venv/bin/python render_worker_node.py \
      --shard 0 --num-shards 6 --gpu-id 6 > /mnt/data/rerender60/_logs/shard0.out 2>&1 &
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import boto3
import tyro
from botocore.config import Config as BotoConfig

import rerender_mjwarp as R

DELIVERY_BUCKET = "xdof-bair-abc"
SUBTASK = "sim_put_the_plastic_bottles_in_the_bin"
MANIFEST_S3 = "s3://xdof-internal-research/abc/render_scratch/put_bottles_mjwarp/render_manifest.json"
TOP_KEY = R.CAM_KEY["top"]  # top_camera-images-rgb


@dataclass
class Cfg:
    shard: int = 0            # this worker renders episode_index % num_shards == shard
    num_shards: int = 1
    gpu_id: int = 0
    out_root: Path = Path("/mnt/data/rerender60")
    asset_root: str = "/mnt/data/render_assets"
    cameras: str = "top"
    fps_mult: int = 2
    indices: str | None = None   # comma-separated episode_index list (overrides sharding)
    limit: int | None = None     # cap episodes after selection (validation runs)
    overwrite: bool = False


def log_line(log_path: Path, line: str) -> None:
    """Single-line O_APPEND write; atomic enough for concurrent shard workers."""
    fd = os.open(str(log_path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, (line.rstrip("\n") + "\n").encode())
    finally:
        os.close(fd)


def main(cfg: Cfg) -> None:
    out_root = cfg.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "render.log"
    cam_list = tuple(c.strip() for c in cfg.cameras.split(",") if c.strip())

    s3 = boto3.client("s3", config=BotoConfig(max_pool_connections=16, retries={"max_attempts": 8}))
    manifest_local = out_root / "render_manifest.json"
    if not manifest_local.exists():
        b, _, k = MANIFEST_S3[5:].partition("/")
        s3.download_file(b, k, str(manifest_local))
    manifest = json.loads(manifest_local.read_text())

    if cfg.indices:
        idxs = [int(x) for x in cfg.indices.split(",")]
    else:
        idxs = [i for i in sorted(int(x) for x in manifest) if i % cfg.num_shards == cfg.shard]
    if cfg.limit is not None:
        idxs = idxs[: cfg.limit]
    print(f"[worker s{cfg.shard}/{cfg.num_shards} gpu{cfg.gpu_id}] {len(idxs)} episodes "
          f"cams={cam_list} fps_mult={cfg.fps_mult}", flush=True)

    work = out_root / "_work" / f"shard{cfg.shard:02d}"
    deliv_root, staged_out, percam_out = work / "deliv", work / "staged", work / "percam"
    for d in (deliv_root, staged_out, percam_out):
        d.mkdir(parents=True, exist_ok=True)

    base_cfg = R.Config(
        delivery_root=deliv_root, out_dir=staged_out, staged_src=None,
        percam_dir=percam_out, asset_root=cfg.asset_root, gpu_id=cfg.gpu_id,
        cameras=cfg.cameras, fps_mult=cfg.fps_mult, no_combined=True,
        overwrite=True,  # work dir is wiped per episode; skip logic lives HERE, on the final path
    )

    done, skipped, failed = 0, 0, []
    for i in idxs:
        info = manifest[str(i)]
        uuid, deliv, length = info["uuid"], info.get("delivery"), int(info["length"])
        eppad = f"{i:06d}"
        if not deliv:  # 2 episodes absent from the entire delivery bucket (no scene) — cannot render
            print(f"[s{cfg.shard}] skip {eppad} ({uuid}): no delivery scene", flush=True)
            continue
        expected = 2 * length - 1 if cfg.fps_mult == 2 else length
        final_dir = out_root / eppad
        final_mp4s = {c: final_dir / f"{R.CAM_KEY[c]}.mp4" for c in cam_list}
        if not cfg.overwrite and all(
            p.exists() and R.probe_nframes(str(p)) == expected for p in final_mp4s.values()
        ):
            skipped += 1
            continue
        t0 = time.time()
        ep_name = uuid if uuid.startswith("episode_") else f"episode_{uuid}"
        ep_deliv = deliv_root / ep_name
        ep_deliv.mkdir(parents=True, exist_ok=True)
        prefix = f"data/deliveries/{deliv}/{SUBTASK}/{uuid}"
        try:
            for fn in ("scene_assembled.xml", "integration_state.npy"):
                s3.download_file(DELIVERY_BUCKET, f"{prefix}/{fn}", str(ep_deliv / fn))
        except Exception as e:
            print(f"[s{cfg.shard}][FAIL-DL] {eppad} {uuid}: {e}", flush=True)
            log_line(log_path, f"{eppad} uuid={uuid} FAIL-DL sec={time.time()-t0:.1f}")
            failed.append(i)
            shutil.rmtree(ep_deliv, ignore_errors=True)
            continue
        try:
            R.render_episode(ep_deliv, base_cfg)
            pc_ep = percam_out / ep_name
            final_dir.mkdir(parents=True, exist_ok=True)
            for c in cam_list:
                src = pc_ep / f"{R.CAM_KEY[c]}.mp4"
                got = R.probe_nframes(str(src))
                if got != expected:
                    raise RuntimeError(f"{c}: {got} frames != expected {expected}")
                os.replace(src, final_mp4s[c])
            # Final metadata: the render_episode one + index/split (worker-level identity).
            meta = json.loads((staged_out / ep_name / "episode_metadata.json").read_text())
            meta.update(episode_index=i, split=info["split"],
                        frame_mapping="t30[i] -> t60[2*i] exact; odd t60 = lerp/slerp midpoint")
            (final_dir / "episode_metadata.json").write_text(json.dumps(meta, indent=2))
        except Exception as e:
            print(f"[s{cfg.shard}][FAIL-RENDER] {eppad} {uuid}: {type(e).__name__}: {e}", flush=True)
            log_line(log_path, f"{eppad} uuid={uuid} FAIL-RENDER sec={time.time()-t0:.1f}")
            failed.append(i)
            continue
        finally:
            for d in (ep_deliv, staged_out / ep_name, percam_out / ep_name):
                shutil.rmtree(d, ignore_errors=True)
        done += 1
        dt = time.time() - t0
        log_line(log_path, f"{eppad} uuid={uuid} frames={expected} src={length} sec={dt:.1f}")
        print(f"[s{cfg.shard}] {eppad} done in {dt:.0f}s ({done} done, {skipped} skipped)", flush=True)

    print(f"[worker s{cfg.shard}] DONE: rendered={done} skipped={skipped} failed={failed}", flush=True)
    if failed:
        raise SystemExit(f"[worker s{cfg.shard}] {len(failed)} episodes FAILED: {failed}")


if __name__ == "__main__":
    main(tyro.cli(Cfg))
