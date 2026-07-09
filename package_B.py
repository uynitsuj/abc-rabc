# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pyarrow", "pandas", "boto3", "tyro"]
# ///
"""Piece 3: generalized LeRobot-v3 packager for an arbitrary render manifest.

package_mjwarp.py's `lerobot` phase copies A's meta/data VERBATIM (A already existed as a
mjgl LeRobot dataset). B-only eps have NO reference meta/data, so this script BUILDS a
standalone v3 dataset from scratch:

  * episode lengths from ACTUAL rendered frame counts (probe scratch percam top mp4;
    asserted == manifest length == integration_state rows).
  * file grouping by a frames-per-file target (mirrors A's size-based split), 5-digit
    zero-padded file naming `file-{i:05d}` (matches A's info.json data_path/video_path;
    lex-safe for <100k files, and the A+B concat REQUIRES schema-identical naming — the
    "4-digit" ask is superseded by matching A).
  * data parquets: state(14 f32)|actions(14 f32) from the staged states_actions.bin,
    timestamp/frame_index/episode_index/index/task_index, + the 5 RM score columns as
    float32 zeros (PLACEHOLDERS overwritten by the fresh-RM scoring/inject step; present
    only so the schema matches A for the concat).
  * episodes parquet + tasks.parquet + info.json (A conventions) + stats.json + a
    re-keyed meta/object_counts.json.
  * videos: per file-group per cam, concatenate the scratch per-episode percam mp4s in
    episode order (reuses A's X264_LEROBOT encode via package_mjwarp helpers).

Index mapping: the manifest is keyed by GLOBAL episode_index (A: 0.., B: 2438..). The
scratch is keyed by that global index (percam/<global06>/). The standalone dataset uses
LOCAL episode_index = enumerate order (0..N-1); source_episode_id carries the uuid, and
local + `--index-offset` == global (== merged episode_index). object_counts is re-keyed
global->local for the standalone meta.

Not (re)produced here — regenerated downstream over the MERGED corpus, noted in the
runbook: the 32-dim RM norm_stats, repromo_meta.json / warp_rm_meta.json (RM annotation
artifacts written by scoring).

Runs node-side (scratch read creds). Writes a LOCAL dir; sync to S3 separately.
    # validate against A (3 eps):
    .venv/bin/python package_B.py --manifest render_manifest.json \\
        --states-source staged_put --scratch-s3 s3://.../render_scratch/put_bottles_mjwarp \\
        --out-dir /tmp/valA --dataset-name _valA --index-offset 0 --limit 3 --phase all
"""
from __future__ import annotations

import glob
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Reuse A's exact conventions/helpers so B can never drift from A (concat safety).
from package_mjwarp import (
    CAMERAS, CAM_KEY, FPS, X264_LEROBOT, _s3, _s3_exists, _split, probe_nframes, _decode_all,
)

TASK_STR = "sim_put the plastic bottles in the bin"
# Feature schema copied verbatim from A's info.json. Video features carry an `info` codec
# block (matches the X264_LEROBOT encode: libx264/yuv420p/gop10); non-video features carry
# `fps`. Reproduced exactly so the A+B concat is schema-identical.
_VID_INFO = {"video.fps": 30, "video.height": 480, "video.width": 640, "video.channels": 3,
             "video.codec": "libx264", "video.pix_fmt": "yuv420p",
             "video.is_depth_map": False, "has_audio": False, "video.gop_size": 10}


def _vid(): return {"dtype": "video", "shape": [480, 640, 3],
                    "names": ["height", "width", "channel"], "info": dict(_VID_INFO)}


def _scalar(dtype, shape, names): return {"dtype": dtype, "shape": shape, "names": names, "fps": 30}


FEATURES = {
    "left_camera-images-rgb": _vid(),
    "right_camera-images-rgb": _vid(),
    "top_camera-images-rgb": _vid(),
    "frame_index": _scalar("int64", [1], None),
    "episode_index": _scalar("int64", [1], None),
    "timestamp": _scalar("float32", [1], None),
    "task_index": _scalar("int64", [1], None),
    "index": _scalar("int64", [1], None),
    "actions": _scalar("float32", [14], ["actions"]),
    "state": _scalar("float32", [14], ["state"]),
    # RM score columns were appended by annotation WITHOUT an fps field (match exactly).
    "repromo_progress": {"dtype": "float32", "shape": [1], "names": None},
    "repromo_signed_magnitude": {"dtype": "float32", "shape": [1], "names": None},
    "repromo_quality": {"dtype": "float32", "shape": [1], "names": None},
    "warp_rm_progress": {"dtype": "float32", "shape": [1], "names": None},
    "warp_rm_signed_magnitude": {"dtype": "float32", "shape": [1], "names": None},
}
RM_SCORE_COLS = ["repromo_progress", "repromo_signed_magnitude", "repromo_quality",
                 "warp_rm_progress", "warp_rm_signed_magnitude"]


@dataclass
class Cfg:
    manifest: str
    states_source: str              # local staged dir: <split>/episode_<uuid>/states_actions.bin
    scratch_s3: str                 # s3://.../render_scratch/put_bottles_mjwarp
    out_dir: str                    # local output root for the standalone dataset
    dataset_name: str = "sim_put_bottles_mjwarp_B_only"
    index_offset: int = 2438        # global episode_index = local + offset (A:0, B:2438)
    object_counts_json: Optional[str] = None  # global-keyed AB counts to re-key -> local meta
    target_frames_per_group: int = 280000     # ~A's mean group (2228979/8)
    phase: Literal["meta_data", "videos", "all"] = "all"
    limit: Optional[int] = None     # first N eps (validation slice)
    probe_scratch: bool = True      # probe actual frame counts from scratch percam top
    workers: int = 4
    # Optional local mirror of the render scratch (same layout: <dir>/percam/<idx06>/
    # <cam>.mp4). When set, probe + video decode read local files directly instead of
    # downloading from --scratch-s3 (node2 has the full render output on /mnt/data).
    scratch_local: Optional[str] = None
    # Parallelize the videos phase per (file-group, cam), mirroring package_mjwarp's
    # phase_lerobot: comma-separated group ids and/or a single camera name.
    groups: Optional[str] = None
    cam: Optional[str] = None


# --------------------------------------------------------------------------- packing
def load_eps(cfg: Cfg):
    man = json.loads(Path(cfg.manifest).read_text())
    eps = sorted(((int(k), v["uuid"], v.get("split", "train_sim"), int(v["length"]))
                  for k, v in man.items() if v.get("delivery")), key=lambda x: x[0])
    if cfg.limit is not None:
        eps = eps[:cfg.limit]
    return eps


def _local_percam(cfg: Cfg, global_idx: int, cam: str) -> Optional[Path]:
    """Path to the local scratch copy of an episode's percam mp4, if configured+present."""
    if not cfg.scratch_local:
        return None
    p = Path(cfg.scratch_local) / "percam" / f"{global_idx:06d}" / f"{CAM_KEY[cam]}.mp4"
    return p if p.exists() and p.stat().st_size > 0 else None


def frame_count(cfg: Cfg, s3, sb, sk, global_idx: int, manifest_len: int) -> int:
    """Actual rendered frame count (probe scratch percam top); assert == manifest length."""
    if not cfg.probe_scratch:
        return manifest_len
    loc_p = _local_percam(cfg, global_idx, "top")
    if loc_p is not None:
        n = probe_nframes(str(loc_p))
    else:
        key = f"{sk}/percam/{global_idx:06d}/{CAM_KEY['top']}.mp4"
        with tempfile.TemporaryDirectory() as td:
            loc = Path(td) / "top.mp4"
            s3.download_file(sb, key, str(loc))
            n = probe_nframes(str(loc))
    if n != manifest_len:
        raise RuntimeError(f"idx {global_idx}: scratch frames {n} != manifest length {manifest_len}")
    return n


def build_groups(cfg: Cfg, eps, s3, sb, sk):
    """Assign each ep to a file-group by accumulating frames up to the target.

    Returns list of groups; each = list of dicts {local, global, uuid, split, length,
    from_ts, to_ts, ds_from, ds_to} with file_index = group position.
    """
    # Prefetch actual frame counts in parallel (ffprobe -count_frames fully decodes;
    # serial over ~3.5k eps would take ~an hour). Order-independent, assert-identical.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max(cfg.workers, 32)) as ex:
        lengths = list(ex.map(
            lambda e: frame_count(cfg, s3, sb, sk, e[0], e[3]), eps))
    groups, cur, cur_frames = [], [], 0
    ds_cursor = 0
    for local, (gidx, uuid, split, mlen) in enumerate(eps):
        L = lengths[local]
        if cur and cur_frames + L > cfg.target_frames_per_group:
            groups.append(cur); cur, cur_frames = [], 0
        # from_timestamp accumulates WITHIN the current group.
        from_ts = cur_frames / FPS
        cur.append({"local": local, "global": gidx, "uuid": uuid, "split": split,
                    "length": L, "from_ts": from_ts, "to_ts": (cur_frames + L) / FPS,
                    "ds_from": ds_cursor, "ds_to": ds_cursor + L})
        cur_frames += L
        ds_cursor += L
    if cur:
        groups.append(cur)
    return groups


# --------------------------------------------------------------------------- meta+data
def build_meta_data(cfg: Cfg, groups, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    ep_rows = []
    global_index = 0
    stat = {"sum": np.zeros(28), "sumsq": np.zeros(28), "count": 0,
            "min": np.full(28, np.inf), "max": np.full(28, -np.inf)}
    for gi, group in enumerate(groups):
        data_frames = []
        for ep in group:
            sa_path = Path(cfg.states_source) / ep["split"] / ep["uuid"] / "states_actions.bin"
            sa = np.fromfile(sa_path, dtype=np.float64).reshape(-1, 28)
            if sa.shape[0] != ep["length"]:
                raise RuntimeError(f"{ep['uuid']}: states rows {sa.shape[0]} != length {ep['length']}")
            L = ep["length"]
            state = sa[:, :14].astype(np.float32)
            actions = sa[:, 14:].astype(np.float32)
            df = pd.DataFrame({
                "state": list(state),
                "actions": list(actions),
                "timestamp": (np.arange(L) / FPS).astype(np.float32),
                "frame_index": np.arange(L, dtype=np.int64),
                "episode_index": np.full(L, ep["local"], dtype=np.int64),
                "index": np.arange(global_index, global_index + L, dtype=np.int64),
                "task_index": np.zeros(L, dtype=np.int64),
            })
            for c in RM_SCORE_COLS:
                df[c] = np.zeros(L, dtype=np.float32)  # placeholder, overwritten by scoring
            data_frames.append(df)
            global_index += L
            # running stats (28-col)
            stat["sum"] += sa.sum(0); stat["sumsq"] += (sa ** 2).sum(0)
            stat["count"] += L
            stat["min"] = np.minimum(stat["min"], sa.min(0))
            stat["max"] = np.maximum(stat["max"], sa.max(0))
            ep_rows.append({
                "episode_index": ep["local"], "length": L,
                "data/chunk_index": 0, "data/file_index": gi,
                "dataset_from_index": ep["ds_from"], "dataset_to_index": ep["ds_to"],
                **{f"videos/{CAM_KEY[c]}/chunk_index": 0 for c in CAMERAS},
                **{f"videos/{CAM_KEY[c]}/file_index": gi for c in CAMERAS},
                **{f"videos/{CAM_KEY[c]}/from_timestamp": ep["from_ts"] for c in CAMERAS},
                **{f"videos/{CAM_KEY[c]}/to_timestamp": ep["to_ts"] for c in CAMERAS},
                "tasks": np.array([TASK_STR], dtype=object),
                "source_episode_id": ep["uuid"],
            })
        big = pd.concat(data_frames, ignore_index=True)
        # cast columns to match A: state/actions are fixed-size float32 lists -> pyarrow
        # fixed_size_list; use pandas->arrow with explicit schema for the 2 array cols.
        _write_data_parquet(big, out / "data" / "chunk-000" / f"file-{gi:05d}.parquet")
        print(f"[pkgB] data file-{gi:05d}: {len(big)} frames, {len(group)} eps", flush=True)

    _write_episodes_parquet(ep_rows, out / "meta" / "episodes" / "chunk-000" / "file-00000.parquet")

    # tasks.parquet
    pq.write_table(pa.table({"task_index": pa.array([0], pa.int64()),
                             "task": pa.array([TASK_STR], pa.string())}),
                   out / "meta" / "tasks.parquet")

    total_frames = int(sum(ep["length"] for g in groups for ep in g))
    total_eps = int(sum(len(g) for g in groups))
    info = {
        "codebase_version": "v3.0", "robot_type": "bimanual_sim",
        "total_episodes": total_eps, "total_frames": total_frames, "total_tasks": 1,
        "total_videos": total_eps * len(CAMERAS), "total_chunks": 1, "chunks_size": 1000,
        "data_files_size_in_mb": 100, "video_files_size_in_mb": 500, "fps": FPS,
        "splits": {"train": f"0:{total_eps}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:05d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:05d}.mp4",
        "features": FEATURES,
    }
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    # stats.json (state/actions/timestamp): mean/std/min/max/count.
    n = stat["count"]
    mean = stat["sum"] / n
    std = np.sqrt(np.maximum(stat["sumsq"] / n - mean ** 2, 0.0))
    stats = {
        "state": {"mean": mean[:14].tolist(), "std": std[:14].tolist(),
                  "min": stat["min"][:14].tolist(), "max": stat["max"][:14].tolist(),
                  "count": [n]},
        "actions": {"mean": mean[14:].tolist(), "std": std[14:].tolist(),
                    "min": stat["min"][14:].tolist(), "max": stat["max"][14:].tolist(),
                    "count": [n]},
        "timestamp": {"mean": [float(np.mean(np.arange(0)))] if False else [0.0],
                      "std": [0.0], "min": [0.0],
                      "max": [(max(ep["length"] for g in groups for ep in g) - 1) / FPS],
                      "count": [n]},
    }
    (out / "meta" / "stats.json").write_text(json.dumps(stats, indent=4))

    # meta/object_counts.json re-keyed global -> local (subtract offset).
    if cfg.object_counts_json:
        ab = json.loads(Path(cfg.object_counts_json).read_text())["counts"]
        local_counts = {}
        for g in groups:
            for ep in g:
                gk = str(ep["global"])
                if gk in ab:
                    local_counts[str(ep["local"])] = ab[gk]
        from collections import Counter
        (out / "meta" / "object_counts.json").write_text(json.dumps({
            "count_key": "bottle_count", "n_episodes": len(local_counts),
            "distribution": dict(sorted(Counter(local_counts.values()).items())),
            "index_space": "LOCAL episode_index (standalone); merged uses object_counts_AB",
            "counts": local_counts}, indent=0))
    print(f"[pkgB] meta+data DONE: {total_eps} eps, {total_frames} frames, {len(groups)} groups")


def _write_data_parquet(df: pd.DataFrame, path: Path):
    """Write with state/actions as list<float32> (matches A's arrow schema for concat)."""
    arrays, names = [], []
    for col in df.columns:
        if col in ("state", "actions"):
            rows = [np.asarray(r, dtype=np.float32) for r in df[col].to_numpy()]
            arrays.append(pa.array(rows, type=pa.list_(pa.float32())))
        elif col == "timestamp" or col in RM_SCORE_COLS:
            arrays.append(pa.array(df[col].to_numpy().astype(np.float32), pa.float32()))
        else:
            arrays.append(pa.array(df[col].to_numpy().astype(np.int64), pa.int64()))
        names.append(col)
    pq.write_table(pa.Table.from_arrays(arrays, names=names), path)


def _write_episodes_parquet(ep_rows: list, path: Path):
    """Explicit arrow schema matching A: int64 indices, double timestamps, tasks
    list<string>, source_episode_id string."""
    i64, dbl = pa.int64(), pa.float64()
    fields = [("episode_index", i64), ("length", i64), ("data/chunk_index", i64),
              ("data/file_index", i64), ("dataset_from_index", i64), ("dataset_to_index", i64)]
    for c in CAMERAS:
        fields.append((f"videos/{CAM_KEY[c]}/chunk_index", i64))
    for c in CAMERAS:
        fields.append((f"videos/{CAM_KEY[c]}/file_index", i64))
    for c in CAMERAS:
        fields.append((f"videos/{CAM_KEY[c]}/from_timestamp", dbl))
    for c in CAMERAS:
        fields.append((f"videos/{CAM_KEY[c]}/to_timestamp", dbl))
    fields.append(("tasks", pa.list_(pa.string())))
    fields.append(("source_episode_id", pa.string()))
    schema = pa.schema(fields)
    cols = {}
    for name, typ in fields:
        if name == "tasks":
            cols[name] = pa.array([[TASK_STR] for _ in ep_rows], type=pa.list_(pa.string()))
        elif name == "source_episode_id":
            cols[name] = pa.array([r[name] for r in ep_rows], type=pa.string())
        else:
            vals = [r[name] for r in ep_rows]
            cols[name] = pa.array(vals, type=typ)
    pq.write_table(pa.table(cols, schema=schema), path)


# --------------------------------------------------------------------------- videos
def build_videos(cfg: Cfg, groups, out: Path):
    s3 = _s3()
    sb, sk = _split(cfg.scratch_s3)
    W, H = 640, 480
    sel_groups = {int(g) for g in cfg.groups.split(",")} if cfg.groups else None
    sel_cams = (cfg.cam,) if cfg.cam else CAMERAS
    for gi, group in enumerate(groups):
        if sel_groups is not None and gi not in sel_groups:
            continue
        expected = sum(ep["length"] for ep in group)
        for cam in sel_cams:
            key_cam = CAM_KEY[cam]
            dst = out / "videos" / key_cam / "chunk-000"
            dst.mkdir(parents=True, exist_ok=True)
            outp = dst / f"file-{gi:05d}.mp4"
            enc = subprocess.Popen(
                ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                 "-r", str(FPS), "-i", "-", "-vsync", "0", *X264_LEROBOT, "-threads", "2",
                 str(outp)], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
            written = 0
            with tempfile.TemporaryDirectory() as td:
                for ep in group:
                    loc_p = _local_percam(cfg, ep["global"], cam)
                    if loc_p is not None:
                        seg = loc_p
                    else:
                        seg = Path(td) / f"seg_{ep['global']}_{cam}.mp4"
                        s3.download_file(sb, f"{sk}/percam/{ep['global']:06d}/{key_cam}.mp4", str(seg))
                    frames = _decode_all(str(seg), ep["length"])
                    if frames.shape[0] != ep["length"]:
                        raise RuntimeError(f"ep {ep['global']} {cam}: {frames.shape[0]} != {ep['length']}")
                    for k in range(ep["length"]):
                        enc.stdin.write(np.ascontiguousarray(frames[k]).tobytes())
                    written += ep["length"]
            enc.stdin.close()
            if enc.wait() != 0:
                raise RuntimeError(f"group{gi} {cam} encode failed")
            got = probe_nframes(str(outp))
            if got != expected or written != expected:
                raise RuntimeError(f"group{gi} {cam}: encoded {got} (wrote {written}) != {expected}")
            print(f"[pkgB] video file-{gi:05d} {cam}: {got} frames", flush=True)


def main(cfg: Cfg):
    out = Path(cfg.out_dir) / cfg.dataset_name
    s3 = _s3()
    sb, sk = _split(cfg.scratch_s3)
    eps = load_eps(cfg)
    print(f"[pkgB] {len(eps)} eps -> {out} (offset={cfg.index_offset})")
    groups = build_groups(cfg, eps, s3, sb, sk)
    print(f"[pkgB] {len(groups)} file-groups, sizes={[len(g) for g in groups]}")
    if cfg.phase in ("meta_data", "all"):
        build_meta_data(cfg, groups, out)
    if cfg.phase in ("videos", "all"):
        build_videos(cfg, groups, out)


if __name__ == "__main__":
    import tyro
    main(tyro.cli(Cfg))
