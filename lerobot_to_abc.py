# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pyarrow", "pandas", "torchcodec", "tyro"]
# ///
"""Convert a LeRobot v3 sim dataset into the ABC-DiT staged training layout.

Per episode -> <out>/<ep_id>/{states_actions.bin, combined_camera-images-rgb.mp4,
episode_metadata.json}. Reuses export_mcap.py's strict-x264 params + the
vstack/settb/setpts filter so the trainer's synthesized frame mapping (pts=512*k,
GOP30, timebase 1/15360) is byte-true. Frames come from the LeRobot v3 videos
(torchcodec) instead of MCAP h264; everything downstream of the per-cam 224^2 mp4
is identical to export_mcap.py.

State/action are written VERBATIM in the LeRobot order
[L-arm6, L-grip, R-arm6, R-grip] x (state, action) = 28 cols float64. Verified
empirically: grippers are dims 6 & 13 (bounded [0,1]), which matches both the eval
env's gripper indices (eval_policy.py:362-375) and ABC's MCAP column order — so no
remap is needed and train<->eval stay consistent.
"""

from __future__ import annotations

import glob
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import pyarrow.parquet as pq
import tyro
from torchcodec.decoders import VideoDecoder

from export_mcap import FPS, TIMESCALE, TICKS_PER_FRAME, X264, X264_STRICT_FFMPEG_ARGS

# Sim cameras are 4:3 (640x480) -> exact 224x168 with no letterbox. This matches the
# OFFICIAL sim staging (cache/train_sim res [224,168]) and the eval renderer
# (SimEvalConfig camera_width=224, camera_height=168). NOT export_mcap's square 224x224
# (that's the real-robot path). The trainer's resize_with_pad pads 168->224 at load time.
OUT_W, OUT_H = 224, 168

# Stack order in combined.mp4; the trainer slices top->bottom into these names.
CAMERAS = ["top", "left", "right"]
CAM_KEY = {c: f"{c}_camera-images-rgb" for c in CAMERAS}
# Exact letterbox from export_mcap.encode_aligned: fit-inside-224 (bicubic) + center pad.
LETTERBOX_VF = (
    f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease:flags=bicubic,"
    f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2,pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2"
)


@dataclass
class Config:
    data_dir: Annotated[Path, tyro.conf.Positional]  # LeRobot v3 root (meta+data+videos local)
    out_dir: Annotated[Path, tyro.conf.Positional]
    task_name: str = "sim_put_the_plastic_bottles_in_the_bin"
    episodes: str | None = None       # "0" | "0-5" | "0,3,7"; None=all
    max_episodes: int | None = None
    write_norm_stats: bool = True
    # If present in the parquet, dump this per-frame column to velocity_repromo.bin
    # (float64) for RABC reweighting. None disables. Fresh-RM velocity sidecars are
    # written by the separate scoring step.
    velocity_column: str | None = "repromo_signed_magnitude"
    workers: int = 1                  # reserved; conversion is ffmpeg/IO-bound


def _probe_nframes(path: str) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    ).stdout.strip()
    return int(out)


def _select(meta, episodes: str | None, max_episodes: int | None):
    idx = sorted(int(x) for x in meta["episode_index"].tolist())
    if episodes is not None:
        wanted = set()
        for tok in episodes.split(","):
            if "-" in tok:
                a, b = tok.split("-")
                wanted.update(range(int(a), int(b) + 1))
            else:
                wanted.add(int(tok))
        idx = [e for e in idx if e in wanted]
    if max_episodes is not None:
        idx = idx[:max_episodes]
    return idx


def encode_percam(frames_hwc: np.ndarray, out_path: str) -> None:
    """frames_hwc (N,H,W,3) uint8 -> letterboxed 224^2 CFR-30 mp4 (mirrors encode_aligned)."""
    n, h, w, _ = frames_hwc.shape
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-r", str(FPS), "-i", "-", "-vsync", "0", "-vf", LETTERBOX_VF, *X264,
         "-threads", "1", out_path],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    try:
        view = np.ascontiguousarray(frames_hwc)
        for i in range(n):  # chunked writes avoid a single huge blocking write
            enc.stdin.write(view[i].tobytes())
    finally:
        enc.stdin.close()
        if enc.wait() != 0:
            raise RuntimeError(f"per-cam encode failed: {out_path}")


def vstack_strict(percam_paths: list[str], combined: str) -> None:
    """vstack per-cam mp4s + force pts=512*k @ 1/15360 (exact export_mcap.py filter)."""
    filt = (
        "".join(f"[{i}:v]" for i in range(len(percam_paths)))
        + f"vstack=inputs={len(percam_paths)}[v0];"
        + f"[v0]settb=expr=1/{TIMESCALE},setpts=N*{TICKS_PER_FRAME}[out]"
    )
    subprocess.run(
        ["ffmpeg", "-y", *sum((["-i", p] for p in percam_paths), []),
         "-filter_complex", filt, "-map", "[out]", *X264_STRICT_FFMPEG_ARGS, combined],
        capture_output=True, check=True,
    )


def export_episode(data_dir: Path, ep_row, data_df_cache: dict, out_root: Path,
                   task_name: str, stats: dict, velocity_column: str | None) -> str | None:
    ep = int(ep_row["episode_index"])
    length = int(ep_row["length"])
    ep_id = str(ep_row.get("source_episode_id") or f"episode_{ep:06d}")

    # --- state/action rows (verbatim order) ---
    dci, dfi = int(ep_row["data/chunk_index"]), int(ep_row["data/file_index"])
    key = (dci, dfi)
    if key not in data_df_cache:
        p = data_dir / "data" / f"chunk-{dci:03d}" / f"file-{dfi:05d}.parquet"
        cols = ["episode_index", "frame_index", "state", "actions"]
        have = set(pq.ParquetFile(p).schema.names)
        if velocity_column and velocity_column in have:
            cols.append(velocity_column)
        data_df_cache[key] = pq.read_table(p, columns=cols).to_pandas()
    df = data_df_cache[key]
    rows = df[df["episode_index"] == ep].sort_values("frame_index")
    state = np.stack(rows["state"].to_numpy()).astype(np.float64)      # (T,14)
    actions = np.stack(rows["actions"].to_numpy()).astype(np.float64)  # (T,14)
    if len(state) != length:
        print(f"[WARN] {ep_id}: {len(state)} rows != meta length {length}; using rows")
        length = len(state)
    sa = np.concatenate([state, actions], axis=1)  # (T,28) [state14|actions14]

    out_dir = out_root / ep_id
    out_dir.mkdir(parents=True, exist_ok=True)
    sa.tofile(out_dir / "states_actions.bin")
    if velocity_column and velocity_column in rows.columns:
        vel = rows[velocity_column].to_numpy().astype(np.float64).reshape(-1)
        vel.tofile(out_dir / "velocity_repromo.bin")

    # --- combined video: decode v3 frames -> per-cam 224^2 -> vstack strict ---
    import tempfile
    with tempfile.TemporaryDirectory() as work:
        percam = []
        for cam in CAMERAS:
            ci = int(ep_row[f"videos/{CAM_KEY[cam]}/chunk_index"])
            fi = int(ep_row[f"videos/{CAM_KEY[cam]}/file_index"])
            from_ts = float(ep_row[f"videos/{CAM_KEY[cam]}/from_timestamp"])
            vpath = data_dir / "videos" / CAM_KEY[cam] / f"chunk-{ci:03d}" / f"file-{fi:05d}.mp4"
            dec = VideoDecoder(str(vpath))
            start = int(round(from_ts * FPS))
            batch = dec.get_frames_in_range(start=start, stop=start + length)
            frames = batch.data.permute(0, 2, 3, 1).contiguous().numpy()  # (N,H,W,3) uint8
            if len(frames) != length:
                raise RuntimeError(f"{ep_id} {cam}: decoded {len(frames)} != {length}")
            pc = str(Path(work) / f"{cam}.mp4")
            encode_percam(frames, pc)
            percam.append(pc)
        combined = str(out_dir / "combined_camera-images-rgb.mp4")
        vstack_strict(percam, combined)
        n = _probe_nframes(combined)
        if n != length:
            raise RuntimeError(f"{ep_id}: combined has {n} frames, expected {length}")

    meta = {
        "task_name": task_name,
        "cameras": CAMERAS,
        "camera_resolutions": {c: [OUT_W, OUT_H] for c in CAMERAS},
        "alignment": "lerobot_v3_30hz",
        "num_steps": length,
        "source_episode_id": ep_id,
    }
    (out_dir / "episode_metadata.json").write_text(json.dumps(meta, indent=2))
    stats["sum"] += sa.sum(axis=0)
    stats["sumsq"] += (sa ** 2).sum(axis=0)
    stats["count"] += length
    print(f"[OK] {ep_id}: {length} steps")
    return ep_id


def write_norm_stats(out_root: Path, stats: dict) -> None:
    n = stats["count"]
    mean = stats["sum"] / n
    var = np.maximum(stats["sumsq"] / n - mean ** 2, 0.0)
    std = np.sqrt(var)
    blob = {
        "norm_stats": {
            "state": {"mean": mean[:14].tolist(), "std": std[:14].tolist()},
            "actions": {"mean": mean[14:].tolist(), "std": std[14:].tolist()},
        }
    }
    (out_root / "norm_stats.json").write_text(json.dumps(blob, indent=2))
    print(f"[norm_stats] over {n} frames -> {out_root / 'norm_stats.json'}")


def main(cfg: Config) -> None:
    meta_files = sorted(glob.glob(str(cfg.data_dir / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))
    if not meta_files:
        raise FileNotFoundError(f"no meta/episodes parquet under {cfg.data_dir}")
    meta = pq.read_table(meta_files).to_pandas()
    want = _select(meta, cfg.episodes, cfg.max_episodes)
    meta = meta[meta["episode_index"].isin(want)].sort_values("episode_index")
    print(f"{len(meta)} episodes -> {cfg.out_dir}")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"sum": np.zeros(28), "sumsq": np.zeros(28), "count": 0}
    data_cache: dict = {}
    done = 0
    for _, row in meta.iterrows():
        if export_episode(cfg.data_dir, row, data_cache, cfg.out_dir, cfg.task_name,
                          stats, cfg.velocity_column):
            done += 1
    print(f"exported {done}/{len(meta)}")
    if cfg.write_norm_stats and stats["count"]:
        write_norm_stats(cfg.out_dir, stats)


if __name__ == "__main__":
    main(tyro.cli(Config))
