#!/usr/bin/env python3
"""Task 1: determine which RM (perobj/overall) the current scored_lerobot
`warp_rm_signed_magnitude` column came from, by comparing per-episode against
each staged .bin (matched by source_episode_id, sorted by frame_index).

Reuses the exact join from /home/justinyu/abc/score_to_sidecar.py (inverted).
"""
import glob
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

DATA_DIR = "/home/justinyu/RORM/scored_lerobot/sim_put_the_plastic_bottles_in_the_bin_30hz_mjwarp"
STAGED_DIR = "/home/justinyu/abc/staged/put_bottles_mjwarp"
COLUMN = "warp_rm_signed_magnitude"
BINS = {"perobj": "velocity_rm_perobj.bin", "overall": "velocity_rm_overall.bin"}


def load_bin(ep_dir: Path, name: str) -> np.ndarray:
    return np.frombuffer((ep_dir / name).read_bytes(), dtype=np.float64)


def main():
    metas = sorted(glob.glob(f"{DATA_DIR}/meta/episodes/**/*.parquet", recursive=True))
    meta = pq.read_table(metas).to_pandas()
    staged = {p.name: p for sub in ("train_sim", "val_sim")
              for p in Path(STAGED_DIR, sub).glob("episode_*")}

    cache: dict = {}
    stats = {k: {"n_ep": 0, "n_frames": 0, "max_abs_diff": 0.0, "sum_sq": 0.0,
                 "len_mismatch": 0, "exact_match_ep": 0} for k in BINS}
    nomatch = 0
    n_checked = 0
    # sample a spread of episodes across train and val for a full pass; do ALL.
    for _, row in meta.iterrows():
        sid = str(row["source_episode_id"])
        ep_dir = staged.get(sid)
        if ep_dir is None:
            nomatch += 1
            continue
        dci, dfi = int(row["data/chunk_index"]), int(row["data/file_index"])
        key = (dci, dfi)
        if key not in cache:
            p = Path(DATA_DIR) / "data" / f"chunk-{dci:03d}" / f"file-{dfi:05d}.parquet"
            cache[key] = pq.read_table(
                p, columns=["episode_index", "frame_index", COLUMN]).to_pandas()
        df = cache[key]
        rows = df[df["episode_index"] == int(row["episode_index"])].sort_values("frame_index")
        col = rows[COLUMN].to_numpy().astype(np.float64).reshape(-1)
        n_checked += 1
        for rmname, binfile in BINS.items():
            b = load_bin(ep_dir, binfile)
            s = stats[rmname]
            s["n_ep"] += 1
            if len(b) != len(col):
                s["len_mismatch"] += 1
                continue
            d = np.abs(col - b)
            s["n_frames"] += len(col)
            s["max_abs_diff"] = max(s["max_abs_diff"], float(d.max()) if len(d) else 0.0)
            s["sum_sq"] += float(np.sum(d * d))
            if len(d) == 0 or d.max() == 0.0:
                s["exact_match_ep"] += 1

    print(f"episodes checked: {n_checked}  (nomatch source_episode_id: {nomatch})")
    for rmname, s in stats.items():
        rmse = (s["sum_sq"] / s["n_frames"]) ** 0.5 if s["n_frames"] else float("nan")
        print(f"\n[{rmname}]  ({BINS[rmname]})")
        print(f"  episodes compared: {s['n_ep']}  len_mismatch: {s['len_mismatch']}")
        print(f"  exact-match episodes (max|diff|==0): {s['exact_match_ep']}/{s['n_ep']}")
        print(f"  global max|col - bin|: {s['max_abs_diff']:.3e}")
        print(f"  RMSE(col, bin):        {rmse:.3e}")


if __name__ == "__main__":
    main()
