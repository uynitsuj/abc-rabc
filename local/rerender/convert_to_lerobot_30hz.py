#!/usr/bin/env python3
"""Convert sim mcap episodes -> LeRobot v3.0 dataset matching the
   tshirt_folding reference layout.

Usage:
    python convert_to_lerobot.py --src <SRC_DIR> --dst <DST_DIR> [--limit N] [--episodes-per-file 1000]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory as PbDec
from tqdm import tqdm

FPS = 30
EPISODES_PER_FILE = 1000
CHUNKS_SIZE = 1000  # info.json field; we keep one chunk dir
DATA_FILES_SIZE_IN_MB = 100
VIDEO_FILES_SIZE_IN_MB = 200

CAMERA_TOPIC_TO_KEY = {
    "/left-wrist-camera/image-raw": "left_camera-images-rgb",
    "/right-wrist-camera/image-raw": "right_camera-images-rgb",
    "/top-camera/image-raw": "top_camera-images-rgb",
}
PROPRIO_TOPICS = ["/left-arm-proprio", "/right-arm-proprio"]
COMMAND_TOPICS = ["/left-command-state", "/right-command-state"]
INSTRUCTION_TOPIC = "/instruction"


@dataclass
class EpisodeExtract:
    """Per-episode extracted data, ready to be written into a chunk file."""
    n_frames: int
    state: np.ndarray            # (T, 14) float32
    actions: np.ndarray          # (T, 14) float32
    timestamps: np.ndarray       # (T,) float32, seconds from episode start
    task: str
    cam_h264: Dict[str, bytes]   # camera_key -> raw concatenated h264 NAL bytes


def _interp(target_t: np.ndarray, src_t: np.ndarray, src_v: np.ndarray) -> np.ndarray:
    """Per-dim linear interpolation. src_t must be sorted."""
    out = np.empty((target_t.size, src_v.shape[1]), dtype=np.float32)
    for d in range(src_v.shape[1]):
        out[:, d] = np.interp(target_t, src_t, src_v[:, d])
    return out


def extract_episode(ep_dir: Path) -> EpisodeExtract | None:
    mcap_path = ep_dir / "output.mcap"
    if not mcap_path.exists():
        return None

    # First pass: count messages per camera so we can preallocate is unnecessary;
    # collect everything streaming.
    proprio: Dict[str, List[Tuple[int, List[float]]]] = {t: [] for t in PROPRIO_TOPICS}
    command: Dict[str, List[Tuple[int, List[float]]]] = {t: [] for t in COMMAND_TOPICS}
    cam_msgs: Dict[str, List[Tuple[int, bytes]]] = {k: [] for k in CAMERA_TOPIC_TO_KEY.values()}
    task: str = ""

    with open(mcap_path, "rb") as f:
        r = make_reader(f, decoder_factories=[PbDec()])
        for schema, channel, msg, dec in r.iter_decoded_messages():
            topic = channel.topic
            t = msg.log_time  # ns
            if topic in proprio:
                proprio[topic].append((t, list(dec.position)))
            elif topic in command:
                command[topic].append((t, list(dec.position)))
            elif topic in CAMERA_TOPIC_TO_KEY:
                cam_msgs[CAMERA_TOPIC_TO_KEY[topic]].append((t, bytes(dec.data)))
            elif topic == INSTRUCTION_TOPIC:
                task = str(dec.data)

    # Camera reference timeline = top camera (longest / most reliable),
    # fall back to whichever camera has frames.
    ref_key = "top_camera-images-rgb"
    if not cam_msgs[ref_key]:
        for k in cam_msgs:
            if cam_msgs[k]:
                ref_key = k
                break
        else:
            return None

    cam_msgs[ref_key].sort(key=lambda x: x[0])
    ref_times = np.array([t for t, _ in cam_msgs[ref_key]], dtype=np.int64)
    n_frames = ref_times.size
    if n_frames < 2:
        return None

    # Build proprio (state) and command (action) arrays interpolated to ref_times.
    def stack(topic_dict, topics):
        arrays = []
        for tp in topics:
            arr = sorted(topic_dict[tp], key=lambda x: x[0])
            if not arr:
                return None
            ts = np.array([a[0] for a in arr], dtype=np.float64)
            vs = np.array([a[1] for a in arr], dtype=np.float32)
            # interp at ref_times
            interp_vs = _interp(ref_times.astype(np.float64), ts, vs)
            arrays.append(interp_vs)
        return np.concatenate(arrays, axis=1)

    state = stack(proprio, PROPRIO_TOPICS)
    actions = stack(command, COMMAND_TOPICS)
    if state is None or actions is None:
        return None
    if state.shape[1] != 14 or actions.shape[1] != 14:
        print(f"  unexpected widths state={state.shape} actions={actions.shape} in {ep_dir.name}", file=sys.stderr)
        return None

    # Videos are muxed at a fixed FPS below. LeRobot decodes video frames by
    # adding these per-row timestamps to the episode's video from_timestamp, so
    # the tabular timeline must match the encoded video timeline exactly.
    timestamps = (np.arange(n_frames, dtype=np.float32) / FPS).astype(np.float32)

    # Build per-camera h264 streams, ordered by log_time. Resample each camera
    # onto ref_times by nearest-frame indexing so all videos have n_frames.
    cam_h264: Dict[str, bytes] = {}
    for key, msgs in cam_msgs.items():
        msgs.sort(key=lambda x: x[0])
        if not msgs:
            return None
        cam_t = np.array([t for t, _ in msgs], dtype=np.int64)
        # Nearest-neighbor index from each ref_time into this camera's stream.
        # If counts already match exactly, skip resampling.
        if cam_t.size == n_frames and np.all(cam_t == ref_times):
            chosen = list(range(n_frames))
        else:
            idx = np.searchsorted(cam_t, ref_times)
            idx = np.clip(idx, 1, cam_t.size - 1)
            left = cam_t[idx - 1]
            right = cam_t[idx]
            chosen = np.where(np.abs(left - ref_times) <= np.abs(right - ref_times), idx - 1, idx).tolist()
        cam_h264[key] = b"".join(msgs[i][1] for i in chosen)

    return EpisodeExtract(
        n_frames=n_frames,
        state=state,
        actions=actions,
        timestamps=timestamps,
        task=task,
        cam_h264=cam_h264,
    )


def probe_video_frame_count(mp4: Path) -> int:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames,nb_frames",
         "-of", "default=nokey=1:noprint_wrappers=1", str(mp4)],
        capture_output=True, text=True, check=True,
    )
    for line in p.stdout.splitlines():
        line = line.strip()
        if line and line != "N/A":
            return int(line)
    raise RuntimeError(f"Could not determine frame count for {mp4}")


def write_chunk_video(h264_bytes: bytes, out_mp4: Path, expected_frames: int) -> None:
    """Mux raw h264 elementary stream into mp4 at FPS and verify frame count."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    tmp_h264 = out_mp4.with_suffix(".h264")
    tmp_h264.write_bytes(h264_bytes)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "h264", "-r", str(FPS), "-i", str(tmp_h264),
        "-c", "copy",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)
    tmp_h264.unlink()
    actual_frames = probe_video_frame_count(out_mp4)
    if actual_frames != expected_frames:
        raise RuntimeError(
            f"Frame count mismatch for {out_mp4}: expected {expected_frames}, got {actual_frames}"
        )


def probe_video_dims(mp4: Path) -> Tuple[int, int]:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "csv=p=0:s=x", str(mp4)],
        capture_output=True, text=True, check=True,
    )
    w, h = p.stdout.strip().split("x")
    return int(w), int(h)


def build_info_json(total_episodes: int, total_frames: int, total_chunks: int,
                    video_dims: Dict[str, Tuple[int, int]]) -> dict:
    info = {
        "codebase_version": "v3.0",
        "robot_type": "bimanual_sim",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_episodes * len(video_dims),
        "total_chunks": total_chunks,
        "chunks_size": CHUNKS_SIZE,
        "data_files_size_in_mb": DATA_FILES_SIZE_IN_MB,
        "video_files_size_in_mb": VIDEO_FILES_SIZE_IN_MB,
        "fps": FPS,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:05d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:05d}.mp4",
        "features": {},
    }
    for cam_key, (w, h) in video_dims.items():
        info["features"][cam_key] = {
            "dtype": "video",
            "shape": [h, w, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.fps": FPS,
                "video.height": h,
                "video.width": w,
                "video.channels": 3,
                "video.codec": "libx264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    for name, shape in [
        ("frame_index", [1]), ("episode_index", [1]),
        ("timestamp", [1]), ("task_index", [1]), ("index", [1]),
    ]:
        info["features"][name] = {
            "dtype": "float32" if name == "timestamp" else "int64",
            "shape": shape, "names": None, "fps": FPS,
        }
    info["features"]["actions"] = {"dtype": "float32", "shape": [14], "names": ["actions"], "fps": FPS}
    info["features"]["state"] = {"dtype": "float32", "shape": [14], "names": ["state"], "fps": FPS}
    return info


def compute_stats(frames_per_feature: Dict[str, np.ndarray]) -> dict:
    out = {}
    for k, arr in frames_per_feature.items():
        if arr.ndim == 1:
            arr = arr[:, None]
        out[k] = {
            "mean": arr.mean(axis=0).astype(float).tolist(),
            "std": (arr.std(axis=0) + 1e-12).astype(float).tolist(),
            "min": arr.min(axis=0).astype(float).tolist(),
            "max": arr.max(axis=0).astype(float).tolist(),
            "count": [int(arr.shape[0])],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--limit", type=int, default=None, help="Only process first N episodes")
    ap.add_argument("--episodes-per-file", type=int, default=EPISODES_PER_FILE)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if dst.exists():
        print(f"DST exists: {dst} — refusing to overwrite. Remove it first.", file=sys.stderr)
        sys.exit(2)
    dst.mkdir(parents=True)

    ep_dirs = sorted(p for p in src.iterdir() if p.is_dir() and p.name.startswith("episode_"))
    if args.limit:
        ep_dirs = ep_dirs[:args.limit]
    print(f"Found {len(ep_dirs)} episodes -> {dst}")

    # Group episodes into files (within chunk-000).
    per_file = args.episodes_per_file

    # Aggregators
    cam_keys_order = list(CAMERA_TOPIC_TO_KEY.values())
    video_dims: Dict[str, Tuple[int, int]] = {}
    episodes_meta_rows: List[dict] = []
    data_frames_running: List[pd.DataFrame] = []
    global_index = 0
    file_local_state: List[np.ndarray] = []
    file_local_actions: List[np.ndarray] = []
    file_local_timestamps: List[np.ndarray] = []
    # Per-camera h264 byte accumulators per file, and per-episode frame counts
    file_local_cam_bytes: Dict[str, List[bytes]] = {k: [] for k in cam_keys_order}
    file_local_cam_frame_offsets: Dict[str, List[int]] = {k: [] for k in cam_keys_order}
    # All-time stats accumulators
    all_state = []
    all_actions = []
    all_timestamps = []

    task_string: str = ""
    cur_file_idx = 0
    eps_in_file = 0
    eps_total_kept = 0

    def flush_file(file_idx: int):
        nonlocal data_frames_running
        if not data_frames_running:
            return
        # Write data parquet
        df = pd.concat(data_frames_running, ignore_index=True)
        data_dir = dst / "data" / "chunk-000"
        data_dir.mkdir(parents=True, exist_ok=True)
        # Convert state/actions np arrays to list<float> column
        out_table = pa.table({
            "state": pa.array(df["state"].tolist(), type=pa.list_(pa.float32())),
            "actions": pa.array(df["actions"].tolist(), type=pa.list_(pa.float32())),
            "timestamp": pa.array(df["timestamp"].to_numpy(np.float32)),
            "frame_index": pa.array(df["frame_index"].to_numpy(np.int64)),
            "episode_index": pa.array(df["episode_index"].to_numpy(np.int64)),
            "index": pa.array(df["index"].to_numpy(np.int64)),
            "task_index": pa.array(df["task_index"].to_numpy(np.int64)),
        })
        pq.write_table(out_table, data_dir / f"file-{file_idx:05d}.parquet")
        data_frames_running.clear()

        # Write per-camera videos
        for cam_key in cam_keys_order:
            buf = b"".join(file_local_cam_bytes[cam_key])
            out_mp4 = dst / "videos" / cam_key / "chunk-000" / f"file-{file_idx:05d}.mp4"
            expected_frames = file_local_cam_frame_offsets[cam_key][-1]
            write_chunk_video(buf, out_mp4, expected_frames)
            # populate per-episode from/to_timestamp
            offsets = file_local_cam_frame_offsets[cam_key]
            # offsets contain cumulative frame counts; convert to timestamps using FPS
            for ep_meta in episodes_meta_rows:
                if ep_meta.get("_file_idx") != file_idx:
                    continue
                lo_frames = ep_meta[f"_offset_{cam_key}_from"]
                hi_frames = ep_meta[f"_offset_{cam_key}_to"]
                ep_meta[f"videos/{cam_key}/chunk_index"] = 0
                ep_meta[f"videos/{cam_key}/file_index"] = file_idx
                ep_meta[f"videos/{cam_key}/from_timestamp"] = lo_frames / FPS
                ep_meta[f"videos/{cam_key}/to_timestamp"] = hi_frames / FPS
            if cam_key not in video_dims:
                video_dims[cam_key] = probe_video_dims(out_mp4)
            file_local_cam_bytes[cam_key] = []
            file_local_cam_frame_offsets[cam_key] = []

    pbar = tqdm(ep_dirs, desc="episodes", unit="ep")
    for ep_idx, ep_dir in enumerate(pbar):
        try:
            extracted = extract_episode(ep_dir)
        except Exception as e:
            tqdm.write(f"[{ep_idx}] FAILED {ep_dir.name}: {e}", file=sys.stderr)
            continue
        if extracted is None:
            tqdm.write(f"[{ep_idx}] skipped {ep_dir.name} (no usable data)")
            continue
        if not task_string:
            task_string = extracted.task or "sim_throw plastic bottles in bin"

        T = extracted.n_frames
        ep_meta_row = {
            "episode_index": eps_total_kept,
            "length": T,
            "data/chunk_index": 0,
            "data/file_index": cur_file_idx,
            "dataset_from_index": global_index,
            "dataset_to_index": global_index + T,
            "tasks": [task_string],
            "source_episode_id": ep_dir.name,
            "_file_idx": cur_file_idx,
        }
        # Per-camera frame offsets within this file
        for cam_key in cam_keys_order:
            current_frames_in_file = sum(
                # count = bytes count proxy is wrong; track via offsets list
                0 for _ in []
            )
            prev = file_local_cam_frame_offsets[cam_key][-1] if file_local_cam_frame_offsets[cam_key] else 0
            ep_meta_row[f"_offset_{cam_key}_from"] = prev
            ep_meta_row[f"_offset_{cam_key}_to"] = prev + T
            file_local_cam_frame_offsets[cam_key].append(prev + T)
            file_local_cam_bytes[cam_key].append(extracted.cam_h264[cam_key])

        episodes_meta_rows.append(ep_meta_row)

        # Build data df rows
        df = pd.DataFrame({
            "state": list(extracted.state.tolist()),
            "actions": list(extracted.actions.tolist()),
            "timestamp": extracted.timestamps,
            "frame_index": np.arange(T, dtype=np.int64),
            "episode_index": np.full(T, eps_total_kept, dtype=np.int64),
            "index": np.arange(global_index, global_index + T, dtype=np.int64),
            "task_index": np.zeros(T, dtype=np.int64),
        })
        data_frames_running.append(df)
        all_state.append(extracted.state)
        all_actions.append(extracted.actions)
        all_timestamps.append(extracted.timestamps)
        global_index += T
        eps_total_kept += 1
        eps_in_file += 1

        if eps_in_file >= per_file:
            flush_file(cur_file_idx)
            cur_file_idx += 1
            eps_in_file = 0

        pbar.set_postfix(kept=eps_total_kept, frames=global_index)

    flush_file(cur_file_idx)
    total_files = cur_file_idx + (1 if eps_in_file == 0 and cur_file_idx > 0 and not data_frames_running else 0)
    # actually: total chunks = 1 here always
    total_chunks = 1

    # Write meta/tasks.parquet
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    tasks_table = pa.table({
        "task_index": pa.array([0], type=pa.int64()),
        "task": pa.array([task_string], type=pa.string()),
    })
    pq.write_table(tasks_table, dst / "meta" / "tasks.parquet")

    # Write meta/episodes/chunk-000/file-NNN.parquet (one per data file)
    meta_eps_dir = dst / "meta" / "episodes" / "chunk-000"
    meta_eps_dir.mkdir(parents=True, exist_ok=True)
    by_file: Dict[int, List[dict]] = {}
    for row in episodes_meta_rows:
        by_file.setdefault(row["_file_idx"], []).append(row)
    keep_keys = [
        "episode_index", "length", "data/chunk_index", "data/file_index",
        "dataset_from_index", "dataset_to_index",
    ]
    for cam_key in cam_keys_order:
        keep_keys += [
            f"videos/{cam_key}/chunk_index",
            f"videos/{cam_key}/file_index",
            f"videos/{cam_key}/from_timestamp",
            f"videos/{cam_key}/to_timestamp",
        ]
    keep_keys.append("tasks")
    keep_keys.append("source_episode_id")

    for file_idx, rows in by_file.items():
        cols = {k: [r[k] for r in rows] for k in keep_keys}
        meta_table = pa.table({
            "episode_index": pa.array(cols["episode_index"], type=pa.int64()),
            "length": pa.array(cols["length"], type=pa.int64()),
            "data/chunk_index": pa.array(cols["data/chunk_index"], type=pa.int64()),
            "data/file_index": pa.array(cols["data/file_index"], type=pa.int64()),
            "dataset_from_index": pa.array(cols["dataset_from_index"], type=pa.int64()),
            "dataset_to_index": pa.array(cols["dataset_to_index"], type=pa.int64()),
            **{
                f"videos/{ck}/chunk_index": pa.array(cols[f"videos/{ck}/chunk_index"], type=pa.int64())
                for ck in cam_keys_order
            },
            **{
                f"videos/{ck}/file_index": pa.array(cols[f"videos/{ck}/file_index"], type=pa.int64())
                for ck in cam_keys_order
            },
            **{
                f"videos/{ck}/from_timestamp": pa.array(cols[f"videos/{ck}/from_timestamp"], type=pa.float64())
                for ck in cam_keys_order
            },
            **{
                f"videos/{ck}/to_timestamp": pa.array(cols[f"videos/{ck}/to_timestamp"], type=pa.float64())
                for ck in cam_keys_order
            },
            "tasks": pa.array(cols["tasks"], type=pa.list_(pa.string())),
            "source_episode_id": pa.array(cols["source_episode_id"], type=pa.string()),
        })
        pq.write_table(meta_table, meta_eps_dir / f"file-{file_idx:05d}.parquet")

    # Stats
    all_state_arr = np.concatenate(all_state, axis=0)
    all_actions_arr = np.concatenate(all_actions, axis=0)
    all_timestamps_arr = np.concatenate(all_timestamps, axis=0)
    stats = compute_stats({
        "state": all_state_arr,
        "actions": all_actions_arr,
        "timestamp": all_timestamps_arr,
    })
    (dst / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))

    info = build_info_json(
        total_episodes=eps_total_kept,
        total_frames=global_index,
        total_chunks=total_chunks,
        video_dims=video_dims,
    )
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    print(f"\nDone. {eps_total_kept} episodes, {global_index} frames -> {dst}")


if __name__ == "__main__":
    main()
