# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "boto3"]
# ///
"""Validate node2's 3-cam 30Hz mjwarp render against A's cloud-rendered scratch output.

Mode "compare-a" (node2 render of an A index vs A's cloud shard output, pulled from S3):
  1. Same file set: 3 percam mp4s + staged combined + episode_metadata.json.
  2. Frame counts match per file (percam AND combined).
  3. Per-camera MAE(local[i], cloud[i]): mean must be same-frame encode-noise level
     (< 1.0; NOTES benchmark ~0.43-0.9) and max below the adjacent-frame motion
     baseline p99 — the validate_rerender60.py criteria, applied per camera + combined.

Mode "check-b" (fresh B episodes, no reference): 3 percam present + combined present,
every video decodes cleanly, frame counts == manifest length == integration rows.

Run: .venv/bin/python validate_node2_render.py --mode compare-a --index 0 \
         --local-root /mnt/data/render_B_val
     .venv/bin/python validate_node2_render.py --mode check-b --index 2438,2439 \
         --local-root /mnt/data/render_B
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import boto3
import numpy as np
import tyro

SCRATCH = ("xdof-internal-research", "abc/render_scratch/put_bottles_mjwarp")
CAMS = ("top", "left", "right")
CAM_KEY = {c: f"{c}_camera-images-rgb" for c in CAMS}
COMBINED = "combined_camera-images-rgb.mp4"
METADATA = "episode_metadata.json"


@dataclass
class Cfg:
    mode: str = "compare-a"          # compare-a | check-b
    index: str = "0"                 # comma-separated episode indices
    local_root: Path = Path("/mnt/data/render_B_val")
    ref_dir: Path = Path("/mnt/data/render_B_val/_ref")
    manifest: Path = Path("/mnt/data/render_B/render_manifest_B.json")


def probe_nframes(path: str) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
        capture_output=True, text=True).stdout.strip()
    return int(out)


def decode(path: str) -> np.ndarray:
    pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                         "stream=width,height", "-of", "csv=p=0", path],
                        capture_output=True, text=True).stdout.strip()
    w, h = (int(x) for x in pr.split(","))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vsync", "0", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def compare_a(cfg: Cfg, i: int) -> bool:
    s3 = boto3.client("s3")
    eppad = f"{i:06d}"
    ok = True
    ref = cfg.ref_dir / eppad
    ref.mkdir(parents=True, exist_ok=True)
    pairs = []  # (name, local_path, ref_key)
    for c in CAMS:
        pairs.append((c, cfg.local_root / "percam" / eppad / f"{CAM_KEY[c]}.mp4",
                      f"{SCRATCH[1]}/percam/{eppad}/{CAM_KEY[c]}.mp4"))
    pairs.append(("combined", cfg.local_root / "staged" / eppad / COMBINED,
                  f"{SCRATCH[1]}/staged/{eppad}/{COMBINED}"))
    # file set: metadata must exist locally and on S3
    meta_local = cfg.local_root / "staged" / eppad / METADATA
    if not meta_local.exists():
        print(f"[FAIL] {eppad}: local {METADATA} missing")
        ok = False
    try:
        s3.head_object(Bucket=SCRATCH[0], Key=f"{SCRATCH[1]}/staged/{eppad}/{METADATA}")
    except Exception:
        print(f"[WARN] {eppad}: cloud {METADATA} missing on S3 (reference)")
    for name, lp, rk in pairs:
        if not lp.exists():
            print(f"[FAIL] {eppad}/{name}: local file missing: {lp}")
            ok = False
            continue
        rp = ref / Path(rk).name
        if not rp.exists():
            s3.download_file(SCRATCH[0], rk, str(rp))
        f_loc, f_ref = decode(str(lp)), decode(str(rp))
        if len(f_loc) != len(f_ref) or f_loc.shape != f_ref.shape:
            print(f"[FAIL] {eppad}/{name}: shape {f_loc.shape} != ref {f_ref.shape}")
            ok = False
            continue
        per = np.abs(f_loc.astype(np.int16) - f_ref.astype(np.int16)).mean(axis=(1, 2, 3))
        adj = np.abs(f_ref[:-1].astype(np.int16) - f_ref[1:].astype(np.int16)).mean(axis=(1, 2, 3))
        p99 = float(np.percentile(adj, 99))
        this_ok = per.mean() < 1.0 and per.max() < max(2.0, p99)
        print(f"   {eppad}/{name}: frames={len(f_loc)} MAE mean={per.mean():.3f} "
              f"max={per.max():.3f} (adjacent-frame baseline mean={adj.mean():.3f} "
              f"p99={p99:.3f}) -> {'OK' if this_ok else 'FAIL'}")
        ok &= this_ok
    return ok


def check_b(cfg: Cfg, i: int) -> bool:
    manifest = json.loads(cfg.manifest.read_text())
    eppad = f"{i:06d}"
    length = int(manifest[str(i)]["length"])
    ok = True
    files = [cfg.local_root / "percam" / eppad / f"{CAM_KEY[c]}.mp4" for c in CAMS] + \
            [cfg.local_root / "staged" / eppad / COMBINED]
    for p in files:
        if not p.exists():
            print(f"[FAIL] {eppad}: missing {p}")
            ok = False
            continue
        f = decode(str(p))  # full decode = clean-decode check
        n = probe_nframes(str(p))
        this_ok = (n == length == len(f))
        print(f"   {eppad}/{p.name}: probe={n} decoded={len(f)} manifest={length} "
              f"shape={f.shape[1:]} -> {'OK' if this_ok else 'FAIL'}")
        ok &= this_ok
    meta = cfg.local_root / "staged" / eppad / METADATA
    if meta.exists():
        m = json.loads(meta.read_text())
        this_ok = m["num_steps"] == length and m["fps"] == 30 and m["fps_mult"] == 1 \
            and m["cameras"] == list(CAMS)
        print(f"   {eppad}/{METADATA}: num_steps={m['num_steps']} fps={m['fps']} "
              f"cams={m['cameras']} -> {'OK' if this_ok else 'FAIL'}")
        ok &= this_ok
    else:
        print(f"[FAIL] {eppad}: missing {METADATA}")
        ok = False
    return ok


def main(cfg: Cfg) -> None:
    ok = True
    for tok in cfg.index.split(","):
        i = int(tok)
        print(f"== episode {i:06d} ({cfg.mode})")
        ok &= (compare_a(cfg, i) if cfg.mode == "compare-a" else check_b(cfg, i))
    print("VALIDATION", "PASSED" if ok else "FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main(tyro.cli(Cfg))
