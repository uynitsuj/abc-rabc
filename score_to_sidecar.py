# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pyarrow", "pandas", "tyro"]
# ///
"""Write per-frame reward-velocity sidecars into the ABC staged dir from a scored
LeRobot v3 dataset (e.g. the warp_rm_signed_magnitude column injected by a fresh WARP-RM).

Phase 2 glue for fresh-RM RABC: matches staged episodes to LeRobot episodes by
source_episode_id and writes <staged>/{train_sim,val_sim}/<ep>/<out_name> (float64,
per-frame, aligned to states_actions.bin). Then `aws s3 sync` the sidecars up and train
RABC with --rabc-velocity-file <out_name>.

  uv run score_to_sidecar.py --data-dir <scored_lerobot> --staged-dir cache/put_bottles \
    --column warp_rm_signed_magnitude --out-name velocity_sss15.bin
"""
import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tyro


@dataclass
class Cfg:
    data_dir: str                              # scored LeRobot v3 root
    staged_dir: str                            # cache/<task> (has train_sim/ + val_sim/)
    column: str = "warp_rm_signed_magnitude"
    out_name: str = "velocity_sss15.bin"


def main(cfg: Cfg):
    metas = sorted(glob.glob(f"{cfg.data_dir}/meta/episodes/**/*.parquet", recursive=True))
    meta = pq.read_table(metas).to_pandas()
    staged = {p.name: p for sub in ("train_sim", "val_sim")
              for p in Path(cfg.staged_dir, sub).glob("episode_*")}
    cache: dict = {}
    written = mismatch = nomatch = 0
    for _, row in meta.iterrows():
        sid = str(row["source_episode_id"])
        ep_dir = staged.get(sid)
        if ep_dir is None:
            nomatch += 1
            continue
        dci, dfi = int(row["data/chunk_index"]), int(row["data/file_index"])
        key = (dci, dfi)
        if key not in cache:
            p = Path(cfg.data_dir) / "data" / f"chunk-{dci:03d}" / f"file-{dfi:05d}.parquet"
            cache[key] = pq.read_table(
                p, columns=["episode_index", "frame_index", cfg.column]).to_pandas()
        df = cache[key]
        rows = df[df["episode_index"] == int(row["episode_index"])].sort_values("frame_index")
        vel = rows[cfg.column].to_numpy().astype(np.float64).reshape(-1)
        n = (ep_dir / "states_actions.bin").stat().st_size // (28 * 8)
        if len(vel) != n:
            print(f"[WARN] {sid}: velocity {len(vel)} != states_actions rows {n}")
            mismatch += 1
        (ep_dir / cfg.out_name).write_bytes(vel.tobytes())
        written += 1
    print(f"wrote {written} '{cfg.out_name}' sidecars ({mismatch} length mismatches, "
          f"{nomatch} LeRobot episodes not in staged set)")


if __name__ == "__main__":
    main(tyro.cli(Cfg))
