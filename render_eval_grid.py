# /// script
# requires-python = ">=3.10"
# dependencies = ["tyro"]
# ///
"""Tile sim-eval rollout videos into one annotated grid montage.

Each eval output dir (from eval_policy.py --save-video) holds world_*.mp4 + summary.json.
This flattens (dir, world) into grid cells, scales+pads each, freezes the last frame so
shorter rollouts hold while longer ones finish, draws a per-cell label (arm | world |
OK/x), and xstacks into a single mp4. Pass multiple --eval-dirs to compare arms
(e.g. vanilla vs RABC) in one grid.

  uv run render_eval_grid.py --eval-dirs outputs/vanilla outputs/rabc --out cmp.mp4
"""
from __future__ import annotations

import glob
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import tyro


@dataclass
class Cfg:
    eval_dirs: List[str] = field(default_factory=list)  # each: world_*.mp4 + summary.json
    out: str = "eval_grid.mp4"
    cols: Optional[int] = None     # default: #worlds in the largest dir
    cell_w: int = 480              # cell width px (world frame is 3 cams = 672x168)
    fps: int = 30
    labels: Optional[List[str]] = None  # per-dir labels; else derived from dir name + success_rate


def _dur(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def main(cfg: Cfg):
    if not cfg.eval_dirs:
        raise SystemExit("pass --eval-dirs DIR [DIR ...]")
    W = cfg.cell_w
    H = int(round(W * 168 / 672))  # preserve the 672x168 world-frame aspect

    cells = []  # (label, path)
    per_dir_cols = []
    for di, d in enumerate(cfg.eval_dirs):
        dp = Path(d)
        summ = json.loads((dp / "summary.json").read_text()) if (dp / "summary.json").exists() else {}
        succ = {w["world_index"]: bool(w["success"]) for w in summ.get("worlds", [])}
        sr = summ.get("success_rate")
        arm = (cfg.labels[di] if cfg.labels and di < len(cfg.labels)
               else dp.name + (f" sr={sr:.2f}" if sr is not None else ""))
        vids = sorted(dp.glob("world_*.mp4"))
        per_dir_cols.append(len(vids))
        for v in vids:
            wi = int(v.stem.split("_")[1])
            tag = "OK" if succ.get(wi) else "x"
            cells.append((f"{arm} w{wi} {tag}", str(v)))

    if not cells:
        raise SystemExit(f"no world_*.mp4 found in {cfg.eval_dirs}")
    # Drop unreadable/incomplete videos (e.g. an eval still writing the file).
    probed = [(c, _dur(c[1])) for c in cells]
    skipped = [c[1] for c, d in probed if d <= 0]
    if skipped:
        print(f"[grid] skipping {len(skipped)} unreadable/incomplete: {skipped}")
    cells = [c for c, d in probed if d > 0]
    if not cells:
        raise SystemExit("no readable videos (all incomplete?)")
    ncols = cfg.cols or max(per_dir_cols or [1])
    n = len(cells)
    maxdur = max(d for _, d in probed if d > 0)

    inputs = []
    parts = []
    for i, (label, path) in enumerate(cells):
        inputs += ["-i", path]
        esc = label.replace(":", r"\:").replace("'", "")
        parts.append(
            f"[{i}:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,"
            f"tpad=stop=-1:stop_mode=clone,"
            f"drawtext=text='{esc}':x=4:y=4:fontsize=14:fontcolor=white:"
            f"box=1:boxcolor=black@0.5[c{i}]"
        )
    layout = "|".join(f"{(i % ncols) * W}_{(i // ncols) * H}" for i in range(n))
    filt = ";".join(parts) + ";" + "".join(f"[c{i}]" for i in range(n)) + \
        f"xstack=inputs={n}:layout={layout}:fill=black[out]"

    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", filt, "-map", "[out]",
           "-t", f"{maxdur:.2f}", "-r", str(cfg.fps), "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-crf", "20", cfg.out]
    print(f"[grid] {n} cells, {ncols} cols x {(n + ncols - 1)//ncols} rows, "
          f"cell {W}x{H}, dur {maxdur:.1f}s -> {cfg.out}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-1500:])
        raise SystemExit("ffmpeg grid failed")
    print(f"[grid] wrote {cfg.out}")


if __name__ == "__main__":
    main(tyro.cli(Cfg))
