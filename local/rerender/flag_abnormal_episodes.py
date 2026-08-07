#!/usr/bin/env python3
"""Flag episodes with abnormal durations across the mjwarp 30hz LeRobot datasets.

Episode length scales strongly with object count, so durations are compared
within (task, object_count) buckets (counts from meta/object_counts.json; the
bottles dataset has a fixed count and forms one bucket). Flags:

  short           duration < Q1 - 1.5*IQR of its bucket  (suspiciously quick —
                  likely an incomplete/degenerate demo)
  long            duration > Q3 + 1.5*IQR of its bucket  (struggle / retry /
                  timeout-ish demo)
  no_object_count episode missing from the sidecar (incomplete source delivery)

Output: <task>/meta/abnormal_episodes.json sidecar per dataset + stdout summary.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/local/karimelrafi")
DATASETS = [
    ROOT / "rerender_lerobot_30hz" / t for t in (
        "sim_hang_the_mug_on_the_mug_rack",
        "sim_load_the_plates_into_the_dish_rack",
        "sim_sweep_away_paper_scraps_from_the_table",
        "sim_throw_plastic_bottles_in_bin",
        "sim_turn_the_mug_right_side_up",
    )
] + [ROOT / "sim-bottles-mjwarp-v1"]


def load_episodes(ds: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(ds / "meta" / "episodes" / "*" / "*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    cols = ["episode_index", "length"]
    if "source_episode_id" in df.columns:
        cols.append("source_episode_id")
    return df[cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iqr-mult", type=float, default=1.5)
    ap.add_argument("--write-sidecars", action="store_true")
    args = ap.parse_args()

    grand_total = 0
    for ds in DATASETS:
        info = json.load(open(ds / "meta" / "info.json"))
        fps = info["fps"]
        df = load_episodes(ds)
        df["duration_s"] = df["length"] / fps

        oc_path = ds / "meta" / "object_counts.json"
        counts = {}
        if oc_path.exists():
            counts = {int(k): int(v) for k, v in
                      json.load(open(oc_path))["counts"].items()}
        df["object_count"] = df["episode_index"].map(counts)

        flagged, stats = [], {}
        groups = df.groupby(df["object_count"].fillna(-1))
        for cnt, g in groups:
            cnt = int(cnt)
            if cnt == -1 and counts:  # sidecar exists but these eps missing
                for _, r in g.iterrows():
                    flagged.append(dict(
                        episode_index=int(r.episode_index),
                        source_episode_id=str(r.get("source_episode_id", "")),
                        object_count=None,
                        duration_s=round(float(r.duration_s), 1),
                        reason="no_object_count"))
                continue
            q1, med, q3 = np.percentile(g["duration_s"], [25, 50, 75])
            iqr = q3 - q1
            lo, hi = q1 - args.iqr_mult * iqr, q3 + args.iqr_mult * iqr
            key = "all" if cnt == -1 else cnt
            stats[str(key)] = dict(
                n=int(len(g)), median_s=round(float(med), 1),
                q1_s=round(float(q1), 1), q3_s=round(float(q3), 1),
                lo_fence_s=round(float(lo), 1), hi_fence_s=round(float(hi), 1),
                max_s=round(float(g["duration_s"].max()), 1))
            for _, r in g.iterrows():
                if r.duration_s < lo or r.duration_s > hi:
                    flagged.append(dict(
                        episode_index=int(r.episode_index),
                        source_episode_id=str(r.get("source_episode_id", "")),
                        object_count=None if cnt == -1 else cnt,
                        duration_s=round(float(r.duration_s), 1),
                        bucket_median_s=round(float(med), 1),
                        reason="short" if r.duration_s < lo else "long"))

        flagged.sort(key=lambda x: (x["reason"], x["duration_s"]))
        n_short = sum(f["reason"] == "short" for f in flagged)
        n_long = sum(f["reason"] == "long" for f in flagged)
        n_nc = sum(f["reason"] == "no_object_count" for f in flagged)
        grand_total += len(flagged)

        print(f"\n=== {ds.name} ({len(df)} eps, fps={fps}) ===")
        for k in sorted(stats, key=lambda x: (x != "all", int(x) if x != "all" else 0)):
            s = stats[k]
            print(f"  count={k:>3}: n={s['n']:<5} median={s['median_s']:>6.1f}s "
                  f"fences=[{s['lo_fence_s']:.1f}, {s['hi_fence_s']:.1f}]s max={s['max_s']:.1f}s")
        print(f"  FLAGGED: {len(flagged)} ({n_short} short, {n_long} long, {n_nc} no-count)")
        for f in flagged[:8]:
            print(f"    ep {f['episode_index']:>5} cnt={f['object_count']} "
                  f"{f['duration_s']:>7.1f}s ({f['reason']})")
        if len(flagged) > 8:
            print(f"    ... and {len(flagged) - 8} more")

        if args.write_sidecars:
            out = ds / "meta" / "abnormal_episodes.json"
            json.dump(dict(iqr_mult=args.iqr_mult, bucket_stats=stats,
                           flagged=flagged), open(out, "w"), indent=1)
            print(f"  -> {out}")

    print(f"\nTOTAL flagged across datasets: {grand_total}")


if __name__ == "__main__":
    main()
