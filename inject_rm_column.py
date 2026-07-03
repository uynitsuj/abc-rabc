#!/usr/bin/env python3
"""Invert score_to_sidecar: build a videos-symlinked (here: videos-on-S3) LeRobot
copy whose `warp_rm_signed_magnitude` column == a given RM's per-frame .bin values.

Source base:  /home/justinyu/RORM/scored_lerobot/sim_put_the_plastic_bottles_in_the_bin_30hz_mjwarp
              (14-dim state/actions; has data/ + meta/; NO videos locally)
RM .bin sets: /home/justinyu/abc/staged/put_bottles_mjwarp/{train_sim,val_sim}/<episode_*>/<binfile>
Join:         meta/episodes source_episode_id -> staged dir name; per data parquet
              (data/chunk_index, data/file_index); rows filtered by episode_index,
              sorted by frame_index. (Exactly the score_to_sidecar.py join, inverted.)

Writes:       $HF_LEROBOT_HOME/<repo_id>/{data,meta}   (videos handled separately via S3)
Rewrites the EXISTING float32 `warp_rm_signed_magnitude` column in place (preserves
arrow schema/type). Validates float32(.bin) == new column, all episodes, both splits.

Usage:
  python inject_rm_column.py --repo-id sim_put_bottles_mjwarp_rmperobj  --binfile velocity_rm_perobj.bin
  python inject_rm_column.py --repo-id sim_put_bottles_mjwarp_rmoverall --binfile velocity_rm_overall.bin
"""
import argparse
import glob
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BASE = "/home/justinyu/RORM/scored_lerobot/sim_put_the_plastic_bottles_in_the_bin_30hz_mjwarp"
STAGED = "/home/justinyu/abc/staged/put_bottles_mjwarp"
COLUMN = "warp_rm_signed_magnitude"
HF_HOME = Path.home() / ".cache" / "huggingface" / "lerobot"


def build_sid_to_bin(binfile: str) -> dict[str, np.ndarray]:
    out = {}
    for sub in ("train_sim", "val_sim"):
        for ep in Path(STAGED, sub).glob("episode_*"):
            p = ep / binfile
            if p.exists():
                out[ep.name] = np.frombuffer(p.read_bytes(), dtype=np.float64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--binfile", required=True)
    ap.add_argument("--validate-only", action="store_true",
                    help="skip copy/inject; just re-validate an existing dest copy")
    args = ap.parse_args()

    dest = HF_HOME / args.repo_id
    sid2bin = build_sid_to_bin(args.binfile)
    print(f"[inject] {args.repo_id}  <- {args.binfile}  ({len(sid2bin)} staged .bin files)")

    if not args.validate_only:
        # 1. Copy data/ + meta/ (NOT videos) into the dest repo.
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        for sub in ("data", "meta"):
            shutil.copytree(Path(BASE, sub), dest / sub)
        print(f"[inject] copied data/ + meta/ -> {dest}")

    # meta join
    metas = sorted(glob.glob(f"{dest}/meta/episodes/**/*.parquet", recursive=True))
    meta = pq.read_table(metas).to_pandas()

    # Group episodes by their data parquet (chunk,file) so we rewrite each file once.
    file_groups: dict[tuple, list] = {}
    for _, row in meta.iterrows():
        key = (int(row["data/chunk_index"]), int(row["data/file_index"]))
        file_groups.setdefault(key, []).append(row)

    n_ep_injected = 0
    nomatch = 0
    for (dci, dfi), rows in sorted(file_groups.items()):
        ppath = dest / "data" / f"chunk-{dci:03d}" / f"file-{dfi:05d}.parquet"
        table = pq.read_table(ppath)
        pdf = table.to_pandas()
        # Build a new column array aligned to the table's existing row order.
        new_col = pdf[COLUMN].to_numpy().astype(np.float32).copy()
        # index rows by (episode_index, frame_index) -> position in table
        # (frame ordering within an episode is monotonic here, but be robust)
        for row in rows:
            sid = str(row["source_episode_id"])
            b = sid2bin.get(sid)
            if b is None:
                nomatch += 1
                continue
            ei = int(row["episode_index"])
            mask = (pdf["episode_index"].to_numpy() == ei)
            fi = pdf["frame_index"].to_numpy()[mask]
            order = np.argsort(fi, kind="stable")
            # positions in the full table for this episode, in frame_index order
            positions = np.nonzero(mask)[0][order]
            if len(positions) != len(b):
                raise SystemExit(
                    f"[FATAL] len mismatch ep {ei} sid {sid}: table {len(positions)} vs bin {len(b)}")
            new_col[positions] = b.astype(np.float32)
            n_ep_injected += 1
        if args.validate_only:
            continue
        # Replace column preserving arrow type (float32)
        col_idx = table.schema.get_field_index(COLUMN)
        field = table.schema.field(col_idx)
        new_arr = pa.array(new_col, type=field.type)
        table = table.set_column(col_idx, field, new_arr)
        pq.write_table(table, ppath)

    print(f"[inject] injected {n_ep_injected} episodes ({nomatch} sid nomatch)")

    # ---- VALIDATION: read back dest, compare column vs .bin per episode ----
    print("[validate] reading back dest and comparing to .bin ...")
    val_stats = {"n_ep": 0, "exact": 0, "len_mismatch": 0, "max_abs_diff": 0.0,
                 "n_frames": 0, "sum_sq": 0.0, "train_ep": 0, "val_ep": 0}
    # split membership from staged dirs
    train_sids = {p.name for p in Path(STAGED, "train_sim").glob("episode_*")}
    val_sids = {p.name for p in Path(STAGED, "val_sim").glob("episode_*")}
    cache = {}
    for _, row in meta.iterrows():
        sid = str(row["source_episode_id"])
        b = sid2bin.get(sid)
        if b is None:
            continue
        dci, dfi = int(row["data/chunk_index"]), int(row["data/file_index"])
        key = (dci, dfi)
        if key not in cache:
            p = dest / "data" / f"chunk-{dci:03d}" / f"file-{dfi:05d}.parquet"
            cache[key] = pq.read_table(
                p, columns=["episode_index", "frame_index", COLUMN]).to_pandas()
        df = cache[key]
        r = df[df["episode_index"] == int(row["episode_index"])].sort_values("frame_index")
        col = r[COLUMN].to_numpy().astype(np.float64)
        bb = b.astype(np.float32).astype(np.float64)  # what got written (float32) upcast
        val_stats["n_ep"] += 1
        if sid in train_sids:
            val_stats["train_ep"] += 1
        elif sid in val_sids:
            val_stats["val_ep"] += 1
        if len(col) != len(bb):
            val_stats["len_mismatch"] += 1
            continue
        d = np.abs(col - bb)
        val_stats["n_frames"] += len(col)
        val_stats["max_abs_diff"] = max(val_stats["max_abs_diff"], float(d.max()) if len(d) else 0.0)
        val_stats["sum_sq"] += float(np.sum(d * d))
        if len(d) == 0 or d.max() == 0.0:
            val_stats["exact"] += 1

    rmse = (val_stats["sum_sq"] / val_stats["n_frames"]) ** 0.5 if val_stats["n_frames"] else float("nan")
    print(f"[validate] episodes: {val_stats['n_ep']} (train {val_stats['train_ep']}, val {val_stats['val_ep']})")
    print(f"[validate] exact-match (float32): {val_stats['exact']}/{val_stats['n_ep']}  len_mismatch: {val_stats['len_mismatch']}")
    print(f"[validate] max|col-bin32|: {val_stats['max_abs_diff']:.3e}  RMSE: {rmse:.3e}")
    ok = (val_stats["exact"] == val_stats["n_ep"] and val_stats["len_mismatch"] == 0
          and val_stats["n_ep"] == 2438)
    print(f"[validate] RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
