# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mujoco", "mujoco-warp", "warp-lang", "tyro", "boto3"]
# ///
"""Node-local shard worker for the dataset-B 3-camera 30Hz mjwarp re-render.

Mirrors render_worker.py (the cloud shard driver that produced dataset A's scratch) but
renders on a local GPU and writes BOTH a local tree and the same S3 scratch layout.
Render config is the cloud default exactly: cameras=top,left,right, fps_mult=1, native
640x480 per-cam CFR-30 videos + the staged 224x504 strict combined.mp4. Per episode it
downloads scene_assembled.xml + integration_state.npy from the delivery bucket, renders
via rerender_mjwarp.render_episode (shared code path with the cloud worker), and writes:

  <out_root>/percam/<epidx06>/{top,left,right}_camera-images-rgb.mp4   (640x480 CFR-30)
  <out_root>/staged/<epidx06>/combined_camera-images-rgb.mp4           (224x504 strict)
  <out_root>/staged/<epidx06>/episode_metadata.json

then uploads the same 5 files to <scratch_s3>/{percam,staged}/<epidx06>/ — the identical
keying render_worker.py used for A (global episode_index, zero-padded), so package_B.py
reads B's shards exactly as it reads A's.

Shared progress log (all shards append single lines, O_APPEND):
  <out_root>/render.log   "<epidx06> uuid=<uuid> frames=<N> src=<N> sec=<t>"
FAIL lines carry FAIL-DL / FAIL-RENDER / FAIL-UP instead of frames=. Skip-verified
episodes (resume) re-log with skip=1; count DISTINCT success indices for progress.

Resumable: an episode is skipped iff all 4 final local mp4s probe to the manifest length
(safe against partial writes: renders land in a work dir and are moved into place). On
skip the worker still HEAD-checks the 5 S3 keys and uploads any that are missing, so a
re-run repairs interrupted uploads.

Shards select episodes by INTERLEAVE (episode_index % num_shards == shard) so lanes stay
balanced and a partial fleet still makes uniform progress.

Run with the project venv python from ~/abc (NOT `uv run`: re-syncs the env and drops
boto3; isolated PEP-723 env breaks `from export_mcap import ...` inside rerender_mjwarp):
  cd ~/abc && nohup .venv/bin/python render_worker_node2.py \
      --shard 0 --num-shards 16 --gpu-id 0 > /mnt/data/render_B/_logs/shard00.out 2>&1 &
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
MANIFEST_S3 = ("s3://xdof-internal-research/abc/render_scratch/put_bottles_mjwarp/"
               "render_manifest_B.json")
SCRATCH_S3 = "s3://xdof-internal-research/abc/render_scratch/put_bottles_mjwarp"
COMBINED = "combined_camera-images-rgb.mp4"
METADATA = "episode_metadata.json"


@dataclass
class Cfg:
    shard: int = 0            # this worker renders episode_index % num_shards == shard
    num_shards: int = 1
    gpu_id: int = 0
    out_root: Path = Path("/mnt/data/render_B")
    asset_root: str = "/mnt/data/render_assets"
    manifest_s3: str = MANIFEST_S3
    scratch_s3: str = SCRATCH_S3
    indices: str | None = None   # comma-separated episode_index list (overrides sharding)
    limit: int | None = None     # cap episodes after selection (validation runs)
    no_upload: bool = False      # local-only (validation runs; NEVER for the fleet)
    overwrite: bool = False


def log_line(log_path: Path, line: str) -> None:
    """Single-line O_APPEND write; atomic enough for concurrent shard workers."""
    fd = os.open(str(log_path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, (line.rstrip("\n") + "\n").encode())
    finally:
        os.close(fd)


def _split_s3(uri: str) -> tuple[str, str]:
    assert uri.startswith("s3://"), uri
    b, _, k = uri[5:].partition("/")
    return b, k


def _exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def main(cfg: Cfg) -> None:
    out_root = cfg.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "render.log"

    s3 = boto3.client("s3", config=BotoConfig(max_pool_connections=16,
                                              retries={"max_attempts": 8}))
    manifest_local = out_root / Path(cfg.manifest_s3).name
    if not manifest_local.exists():
        b, k = _split_s3(cfg.manifest_s3)
        s3.download_file(b, k, str(manifest_local))
    manifest = json.loads(manifest_local.read_text())
    sb, sk = _split_s3(cfg.scratch_s3.rstrip("/"))

    if cfg.indices:
        idxs = [int(x) for x in cfg.indices.split(",")]
    else:
        idxs = [i for i in sorted(int(x) for x in manifest) if i % cfg.num_shards == cfg.shard]
    if cfg.limit is not None:
        idxs = idxs[: cfg.limit]
    print(f"[worker s{cfg.shard}/{cfg.num_shards} gpu{cfg.gpu_id}] {len(idxs)} episodes "
          f"(3-cam 30Hz default; upload={'off' if cfg.no_upload else 'on'})", flush=True)

    work = out_root / "_work" / f"shard{cfg.shard:02d}"
    deliv_root, staged_out, percam_out = work / "deliv", work / "staged", work / "percam"
    for d in (deliv_root, staged_out, percam_out):
        d.mkdir(parents=True, exist_ok=True)

    # EXACTLY the cloud worker's render config for A: default cameras + fps_mult, combined
    # emitted, percam_dir set (render_worker.py:77-81). overwrite=True inside the work dir;
    # the skip logic lives HERE, on the final local path.
    base_cfg = R.Config(
        delivery_root=deliv_root, out_dir=staged_out, staged_src=None,
        percam_dir=percam_out, asset_root=cfg.asset_root, gpu_id=cfg.gpu_id,
        overwrite=True,
    )

    done, skipped, failed = 0, 0, []
    for i in idxs:
        info = manifest[str(i)]
        uuid, deliv, length = info["uuid"], info.get("delivery"), int(info["length"])
        eppad = f"{i:06d}"
        if not deliv:
            print(f"[s{cfg.shard}] skip {eppad} ({uuid}): no delivery scene", flush=True)
            continue

        pc_final = out_root / "percam" / eppad
        st_final = out_root / "staged" / eppad
        final_mp4s = [pc_final / f"{R.CAM_KEY[c]}.mp4" for c in R.CAMERAS] + \
                     [st_final / COMBINED]
        s3_keys = [f"{sk}/percam/{eppad}/{R.CAM_KEY[c]}.mp4" for c in R.CAMERAS] + \
                  [f"{sk}/staged/{eppad}/{COMBINED}", f"{sk}/staged/{eppad}/{METADATA}"]
        local_files = final_mp4s + [st_final / METADATA]

        # Resume: local outputs complete -> only repair missing S3 objects, re-log skip=1.
        if not cfg.overwrite and all(p.exists() for p in local_files) and all(
            R.probe_nframes(str(p)) == length for p in final_mp4s
        ):
            if not cfg.no_upload:
                for key, p in zip(s3_keys, local_files):
                    if not _exists(s3, sb, key):
                        s3.upload_file(str(p), sb, key)
                        print(f"[s{cfg.shard}] {eppad}: repaired missing S3 {key}", flush=True)
            skipped += 1
            log_line(log_path, f"{eppad} uuid={uuid} frames={length} src={length} sec=0.0 skip=1")
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
            # render_episode hard-asserts percam AND combined frame counts == n_frames.
            R.render_episode(ep_deliv, base_cfg)
            src_st, src_pc = staged_out / ep_name, percam_out / ep_name
            pc_final.mkdir(parents=True, exist_ok=True)
            st_final.mkdir(parents=True, exist_ok=True)
            for c in R.CAMERAS:
                os.replace(src_pc / f"{R.CAM_KEY[c]}.mp4", pc_final / f"{R.CAM_KEY[c]}.mp4")
            os.replace(src_st / COMBINED, st_final / COMBINED)
            os.replace(src_st / METADATA, st_final / METADATA)
            # package_B.py hard-asserts scratch frames == manifest length; gate it HERE.
            got = R.probe_nframes(str(pc_final / f"{R.CAM_KEY['top']}.mp4"))
            if got != length:
                raise RuntimeError(f"rendered {got} frames != manifest length {length}")
        except Exception as e:
            print(f"[s{cfg.shard}][FAIL-RENDER] {eppad} {uuid}: {type(e).__name__}: {e}",
                  flush=True)
            log_line(log_path, f"{eppad} uuid={uuid} FAIL-RENDER sec={time.time()-t0:.1f}")
            failed.append(i)
            continue
        finally:
            for d in (ep_deliv, staged_out / ep_name, percam_out / ep_name):
                shutil.rmtree(d, ignore_errors=True)

        if not cfg.no_upload:
            try:
                for key, p in zip(s3_keys, local_files):
                    s3.upload_file(str(p), sb, key)
            except Exception as e:
                print(f"[s{cfg.shard}][FAIL-UP] {eppad} {uuid}: {e}", flush=True)
                log_line(log_path, f"{eppad} uuid={uuid} FAIL-UP sec={time.time()-t0:.1f}")
                failed.append(i)
                continue  # local outputs kept; resume pass repairs the upload

        done += 1
        dt = time.time() - t0
        log_line(log_path, f"{eppad} uuid={uuid} frames={length} src={length} sec={dt:.1f}")
        print(f"[s{cfg.shard}] {eppad} done in {dt:.0f}s ({done} done, {skipped} skipped)",
              flush=True)

    print(f"[worker s{cfg.shard}] DONE: rendered={done} skipped={skipped} failed={failed}",
          flush=True)
    if failed:
        raise SystemExit(f"[worker s{cfg.shard}] {len(failed)} episodes FAILED: {failed}")


if __name__ == "__main__":
    main(tyro.cli(Cfg))
