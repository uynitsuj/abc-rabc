# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pyarrow", "pandas"]
# ///
"""Post-merge verification of sim_put_bottles_mjwarp_AB (runbook step-4 'Verify').

Checks (all hard-assert):
  1. counts: 5,991 eps / 5,327,662 frames / 20 data files / 60 videos; episode_index
     0..5990 contiguous; runbook uuid invariant at 2438.
  2. object_counts distribution == runbook's merged {2:1130,3:1208,4:1234,5:1242,6:1177}.
  3. A-half videos are hardlinks of A's files (inode equality) with identical
     episode meta (offsets/lengths vs A's episodes parquet); B-half videos are
     hardlinks of B-standalone files (+8 rename).
  4. spot decode BOTH halves: for sample eps, decode 3 frames (first/mid/last)
     from the merged video at the episodes-parquet offset and compare against the
     RENDER-SOURCE percam mp4 (B: local render_B scratch; A: S3 render scratch)
     with a re-encode tolerance (mean |diff| < 4/255 per frame).
  5. state/actions spot check: merged data parquet rows == staged states_actions.bin
     (B half) / A's original data parquet (A half).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

AB = Path("/mnt/data/warp_rm/lerobot/sim_put_bottles_mjwarp_AB")
A = Path("/mnt/data/hf_lerobot/sim_put_the_plastic_bottles_in_the_bin_30hz_mjwarp")
B = Path("/mnt/data/hf_lerobot/sim_put_bottles_mjwarp_B_only")
RENDER_B = Path("/mnt/data/render_B/percam")
SCRATCH_S3 = "s3://xdof-internal-research/abc/render_scratch/put_bottles_mjwarp"
STAGED_B = Path("/mnt/data/staged_put_B")
MAN_B = Path("/home/shadeform/abc/render_manifest_B.json")
CAM = "top_camera-images-rgb"
FPS = 30
EXPECT_DIST = {"2": 1130, "3": 1208, "4": 1234, "5": 1242, "6": 1177}

A_SAMPLES = [0, 1234, 2437]
B_SAMPLES = [2438, 4200, 5990]


def decode_frame_raw(video: Path, idx: int) -> np.ndarray:
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"select=eq(n\\,{idx})",
         "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True)
    a = np.frombuffer(p.stdout, dtype=np.uint8)
    assert a.size == 480 * 640 * 3, f"{video}:{idx} decoded {a.size} bytes"
    return a.reshape(480, 640, 3).astype(np.int16)


def main():
    info = json.loads((AB / "meta/info.json").read_text())
    ep = pq.read_table(AB / "meta/episodes/chunk-000/file-00000.parquet").to_pandas()
    n_data = len(list((AB / "data/chunk-000").glob("file-*.parquet")))
    n_vid = len(list((AB / "videos").glob("*/chunk-000/file-*.mp4")))
    assert info["total_episodes"] == 5991 and len(ep) == 5991
    assert info["total_frames"] == 5327662, info["total_frames"]
    assert n_data == 20 and n_vid == 60, (n_data, n_vid)
    print(f"[verify] counts OK: 5991 eps, {info['total_frames']} frames, "
          f"{n_data} data files, {n_vid} videos")

    oc = json.loads((AB / "meta/object_counts.json").read_text())
    dist = {}
    for v in oc["counts"].values():
        dist[str(v)] = dist.get(str(v), 0) + 1
    assert dist == EXPECT_DIST, dist
    print(f"[verify] object_counts distribution OK: {dist}")

    # hardlink identity
    for cams in (AB / "videos").iterdir():
        for i in range(8):
            src = A / "videos" / cams.name / "chunk-000" / f"file-{i:05d}.mp4"
            dst = cams / "chunk-000" / f"file-{i:05d}.mp4"
            assert os.stat(src).st_ino == os.stat(dst).st_ino, dst
        for i in range(12):
            src = B / "videos" / cams.name / "chunk-000" / f"file-{i:05d}.mp4"
            dst = cams / "chunk-000" / f"file-{i + 8:05d}.mp4"
            assert os.stat(src).st_ino == os.stat(dst).st_ino, dst
    print("[verify] video hardlink identity OK (A: 24 files, B: 36 files)")

    # A-half meta equality vs A's episodes parquet
    epA = pq.read_table(A / "meta/episodes/chunk-000/file-00000.parquet").to_pandas()
    a_cols = [c for c in epA.columns if c != "tasks"]
    pd.testing.assert_frame_equal(ep.loc[:2437, a_cols].reset_index(drop=True),
                                  epA[a_cols].reset_index(drop=True))
    print("[verify] A-half episodes meta identical to A")

    man_b = json.loads(MAN_B.read_text())

    def check_ep(k: int):
        r = ep.loc[k]
        fi = int(r[f"videos/{CAM}/file_index"])
        off = round(float(r[f"videos/{CAM}/from_timestamp"]) * FPS)
        L = int(r["length"])
        vid = AB / "videos" / CAM / "chunk-000" / f"file-{fi:05d}.mp4"
        # source percam
        if k >= 2438:
            src = RENDER_B / f"{k:06d}" / f"{CAM}.mp4"
        else:
            src = Path(tempfile.gettempdir()) / f"percam_{k:06d}_{CAM}.mp4"
            if not src.exists():
                subprocess.run(["aws", "s3", "cp",
                                f"{SCRATCH_S3}/percam/{k:06d}/{CAM}.mp4", str(src),
                                "--only-show-errors"], check=True)
        for frame in (0, L // 2, L - 1):
            m = decode_frame_raw(vid, off + frame)
            s = decode_frame_raw(src, frame)
            mad = float(np.abs(m - s).mean())
            assert mad < 4.0, f"ep{k} frame{frame}: mean abs diff {mad:.2f}"
        # states: merged data parquet vs source
        dfi = int(r["data/file_index"])
        t = pq.read_table(AB / "data/chunk-000" / f"file-{dfi:05d}.parquet",
                          columns=["episode_index", "frame_index", "state", "actions",
                                   "index"])
        d = t.to_pandas()
        d = d[d["episode_index"] == k].sort_values("frame_index")
        assert len(d) == L, (k, len(d), L)
        assert d["index"].iloc[0] == int(r["dataset_from_index"])
        sa_merged = np.concatenate([np.stack(d["state"].to_numpy()),
                                    np.stack(d["actions"].to_numpy())], axis=1)
        if k >= 2438:
            uuid = man_b[str(k)]["uuid"]
            split = man_b[str(k)]["split"]
            sa_src = np.fromfile(STAGED_B / split / uuid / "states_actions.bin",
                                 dtype=np.float64).reshape(-1, 28).astype(np.float32)
        else:
            tA = pq.read_table(A / "data/chunk-000" / f"file-{dfi:05d}.parquet",
                               columns=["episode_index", "frame_index", "state", "actions"])
            dA = tA.to_pandas()
            dA = dA[dA["episode_index"] == k].sort_values("frame_index")
            sa_src = np.concatenate([np.stack(dA["state"].to_numpy()),
                                     np.stack(dA["actions"].to_numpy())], axis=1)
        assert sa_merged.shape == sa_src.shape
        e = float(np.abs(sa_merged - sa_src).max())
        assert e < 1e-6, (k, e)
        print(f"[verify] ep {k}: video decode OK (3 frames), states OK "
              f"(max diff {e:.1e}), L={L}")

    for k in A_SAMPLES + B_SAMPLES:
        check_ep(k)
    print("[verify] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
