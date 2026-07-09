# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pyarrow", "pandas"]
# ///
"""Span-fit stat for the AB RM train subset (pre-training report).

Replicates scripts/train.py:filter_shortest_stratified EXACTLY (group by
object_counts[str(episode_index)], keep shortest max(1, round(frac*len(group))) by
n_frames) on the merged episodes parquet, then reports the fraction of KEPT episodes
whose length covers the win32 @ sss15 context span:
    span = (32 - 1) * 15 = 465 frames = 15.5 s @30Hz
    fit  = n_frames >= 466  (window fully inside the episode; also reported >= 465)
Reference: A @ sss15 was 86% fit.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

FRAC = 0.25
SPAN = (32 - 1) * 15  # 465


def main(ds_dir: str, counts_json: str):
    ep = pq.read_table(Path(ds_dir) / "meta/episodes/chunk-000/file-00000.parquet",
                       columns=["episode_index", "length"]).to_pandas()
    raw = json.loads(Path(counts_json).read_text())
    counts = raw["counts"] if "counts" in raw else raw

    groups: dict = {}
    ungrouped = []
    for _, r in ep.iterrows():
        c = counts.get(str(int(r["episode_index"])))
        (groups.setdefault(c, []) if c is not None else ungrouped).append(
            (int(r["episode_index"]), int(r["length"])))
    kept = []
    for c in sorted(groups):
        g = sorted(groups[c], key=lambda x: x[1])
        n_keep = max(1, int(round(FRAC * len(g))))
        kept += g[:n_keep]
        print(f"  count={c}: {len(g)} -> keep {n_keep} "
              f"(len range {g[0][1]}..{g[n_keep-1][1]})")
    if ungrouped:
        g = sorted(ungrouped, key=lambda x: x[1])
        kept += g[:max(1, int(round(FRAC * len(g))))]
        print(f"  count=<ungrouped>: {len(g)}")

    lens = np.array([l for _, l in kept])
    fit466 = float((lens >= SPAN + 1).mean())
    fit465 = float((lens >= SPAN).mean())
    print(f"[span-fit] kept {len(kept)}/{len(ep)} (frac={FRAC})")
    print(f"[span-fit] span={SPAN} frames (win32 @ sss15) = {SPAN/30:.1f}s")
    print(f"[span-fit] kept-ep lengths: min={lens.min()} med={int(np.median(lens))} "
          f"max={lens.max()} mean={lens.mean():.0f}")
    print(f"[span-fit] FIT (n_frames >= {SPAN+1}): {100*fit466:.1f}%   "
          f"(>= {SPAN}: {100*fit465:.1f}%)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
