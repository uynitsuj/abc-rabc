# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "boto3"]
# ///
"""Validate the 60Hz top-cam re-render against the existing 30Hz mjwarp render.

Per episode index:
  1. ffprobe the 60Hz mp4: fps must be 60/1, frame count must be 2N-1.
  2. Download the 30Hz mjwarp percam top video from S3 scratch (same renderer, N frames).
  3. Even-frame check: MAE(t60[2i], t30[i]) must be tiny (encode noise), far below the
     adjacent-frame MAE(t30[i], t30[i+1]) baseline.
  4. Odd-frame intermediacy: MAE(odd, even_prev) and MAE(odd, even_next) each < MAE(even_prev,
     even_next) at motion-heavy positions.
  5. Save 3 side-by-side PNG triplets [t60[2i] | t60[2i+1] | t60[2i+2]] for human inspection.

Run: .venv/bin/python validate_rerender60.py --indices 0,100
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import boto3
import numpy as np
import tyro

SCRATCH = ("xdof-internal-research", "abc/render_scratch/put_bottles_mjwarp/percam")
TOP = "top_camera-images-rgb"


@dataclass
class Cfg:
    indices: str = "0,100"
    out_root: Path = Path("/mnt/data/rerender60")
    val_dir: Path = Path("/mnt/data/rerender60/_validation")


def probe(path: str) -> tuple[str, int]:
    fps = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=avg_frame_rate", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip()
    n = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                        "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
                       capture_output=True, text=True).stdout.strip()
    return fps, int(n)


def decode(path: str) -> np.ndarray:
    pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                         "stream=width,height", "-of", "csv=p=0", path],
                        capture_output=True, text=True).stdout.strip()
    w, h = (int(x) for x in pr.split(","))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vsync", "0", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def save_png(img: np.ndarray, path: Path) -> None:
    h, w, _ = img.shape
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{w}x{h}", "-i", "-", "-frames:v", "1", str(path)],
                   input=np.ascontiguousarray(img).tobytes(), check=True)


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def main(cfg: Cfg) -> None:
    cfg.val_dir.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    ok = True
    for tok in cfg.indices.split(","):
        i = int(tok)
        eppad = f"{i:06d}"
        p60 = cfg.out_root / eppad / f"{TOP}.mp4"
        fps, n60 = probe(str(p60))
        f60 = decode(str(p60))
        p30 = cfg.val_dir / f"{eppad}_top30.mp4"
        if not p30.exists():
            s3.download_file(SCRATCH[0], f"{SCRATCH[1]}/{eppad}/{TOP}.mp4", str(p30))
        f30 = decode(str(p30))
        n30 = len(f30)
        exp = 2 * n30 - 1
        print(f"== ep {eppad}: 60Hz fps={fps} frames={n60} (expect 60/1, {exp}); "
              f"30Hz frames={n30}")
        ok &= (fps == "60/1" and n60 == exp and len(f60) == exp)

        even = f60[0::2]
        assert len(even) == n30
        per = np.abs(even.astype(np.int16) - f30.astype(np.int16)).mean(axis=(1, 2, 3))
        adj = np.abs(f30[:-1].astype(np.int16) - f30[1:].astype(np.int16)).mean(axis=(1, 2, 3))
        print(f"   even-vs-30hz MAE: mean={per.mean():.3f} max={per.max():.3f}   "
              f"(adjacent-30hz-frame baseline: mean={adj.mean():.3f} p99={np.percentile(adj, 99):.3f})")
        # Byte-close = same-frame encode-noise level: mean MAE < 1.0 (NOTES same-frame
        # benchmark ~0.9) and NO frame drifting to motion-level difference (max < adj p99).
        ok &= per.mean() < 1.0 and per.max() < max(2.0, float(np.percentile(adj, 99)))

        # motion-heavy triplet positions: top-3 adjacent-frame deltas (spread apart)
        order = np.argsort(adj)[::-1]
        picks, taken = [], []
        for j in order:
            if all(abs(int(j) - t) > 30 for t in taken):
                picks.append(int(j)); taken.append(int(j))
            if len(picks) == 3:
                break
        inter_ok = True
        for k, j in enumerate(sorted(picks)):
            a, m, b = f60[2 * j], f60[2 * j + 1], f60[2 * j + 2]
            d_am, d_mb, d_ab = mae(a, m), mae(m, b), mae(a, b)
            inter_ok &= (d_am < d_ab and d_mb < d_ab)
            print(f"   triplet@i30={j}: MAE(prev,mid)={d_am:.2f} MAE(mid,next)={d_mb:.2f} "
                  f"MAE(prev,next)={d_ab:.2f} -> intermediate={d_am < d_ab and d_mb < d_ab}")
            save_png(np.concatenate([a, m, b], axis=1),
                     cfg.val_dir / f"ep{eppad}_triplet{k}_i30_{j}.png")
        ok &= inter_ok
    print("VALIDATION", "PASSED" if ok else "FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main(tyro.cli(Cfg))
