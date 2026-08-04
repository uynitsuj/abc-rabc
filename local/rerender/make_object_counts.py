#!/usr/bin/env python3
"""Generate per-episode object_counts.json for each rerendered LeRobot task.

Joins each LeRobot v3.0 episode's `source_episode_id` to the original sim
archive's randomization.json and extracts the object count from metadata
(any `*_count` key: plate_count, mug_count, trash_count). Falls back to
counting `bottle_*` keys in object_states for the bottles task, whose
randomization metadata is empty (always 4 bottles). Prefers actual placed
counts over `requested_*_count` — requested can exceed what fit in the
scene (e.g. turn_the_mug requested_mug_count=4 with mug_count=2).

Output: <lerobot_root>/<task>/meta/object_counts.json in the format expected
by WARP-RM scripts/train.py --object-counts-json (per-object-count stratified
shortest-N% episode filtering):
    {"counts": {"<episode_index>": <int>, ...}, "provenance": {...}}

Episodes whose archive delivery lacks randomization.json (see "Known
unrenderable episodes" in README.md — some incomplete deliveries did render)
are listed in provenance and land in WARP-RM's 'ungrouped' stratum.

Usage (defaults match the data locations in README.md):
    python make_object_counts.py
    python make_object_counts.py --tasks sim_load_the_plates_into_the_dish_rack
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

BOTTLE_KEY_RE = re.compile(r"^bottle_\d+")


def count_for_episode(rand_path: Path) -> tuple[int, str]:
    with open(rand_path) as f:
        rand = json.load(f)
    meta = rand.get("metadata") or {}
    count_keys = [k for k in meta if k.endswith("_count") and not k.startswith("requested_")]
    if len(count_keys) > 1:
        raise ValueError(f"{rand_path}: multiple *_count keys {count_keys}")
    if count_keys:
        return int(meta[count_keys[0]]), count_keys[0]
    n_bottles = len({k for k in rand.get("object_states", {}) if BOTTLE_KEY_RE.match(k)})
    if n_bottles == 0:
        raise ValueError(f"{rand_path}: no *_count metadata and no bottle_* joints")
    return n_bottles, "bottle_joints"


def process_task(task_dir: Path, archive_task: Path) -> bool:
    """Write meta/object_counts.json for one task. Returns True if any episode
    was missing archive metadata."""
    ep_files = sorted((task_dir / "meta" / "episodes").rglob("*.parquet"))
    df = pd.concat(
        [pd.read_parquet(f, columns=["episode_index", "length", "source_episode_id"]) for f in ep_files],
        ignore_index=True,
    )
    counts: dict[str, int] = {}
    key_sources: Counter = Counter()
    missing: list[str] = []
    for row in df.itertuples(index=False):
        rand_path = archive_task / row.source_episode_id / "randomization.json"
        if not rand_path.is_file():
            missing.append(row.source_episode_id)
            continue
        c, src = count_for_episode(rand_path)
        counts[str(row.episode_index)] = c
        key_sources[src] += 1

    out = {
        "counts": counts,
        "provenance": {
            "generated_by": "local/rerender/make_object_counts.py",
            "lerobot_root": str(task_dir),
            "archive_root": str(archive_task),
            "count_source": dict(key_sources),
            "n_episodes": int(len(df)),
            "n_matched": len(counts),
            "n_missing_archive": len(missing),
            "missing_source_episode_ids": missing,
        },
    }
    out_path = task_dir / "meta" / "object_counts.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)

    dist = Counter(counts.values())
    dist_str = ", ".join(f"count={k}: {v}" for k, v in sorted(dist.items()))
    print(f"[{task_dir.name}] {len(counts)}/{len(df)} episodes -> {out_path.name}")
    print(f"  source keys: {dict(key_sources)}")
    print(f"  distribution: {dist_str}")
    if missing:
        print(f"  WARNING: {len(missing)} episodes missing archive randomization.json")
    return bool(missing)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--lerobot-root", type=Path,
        default=Path("/scratch/current/karimelrafi/rerender_lerobot_30hz"),
        help="Root containing one LeRobot v3 dataset dir per task",
    )
    ap.add_argument(
        "--archive-root", type=Path,
        default=Path("/scratch/current/karimelrafi/sim_archive_20260529/mcap"),
        help="Raw delivery archive with <task>/episode_<uuid>/randomization.json",
    )
    ap.add_argument(
        "--tasks", nargs="*", default=None,
        help="Task dir names to process (default: every dir under --lerobot-root with a meta/)",
    )
    args = ap.parse_args()

    task_dirs = (
        [args.lerobot_root / t for t in args.tasks]
        if args.tasks
        else sorted(p for p in args.lerobot_root.iterdir() if (p / "meta").is_dir())
    )
    any_missing = False
    for task_dir in task_dirs:
        archive_task = args.archive_root / task_dir.name
        if not archive_task.is_dir():
            print(f"[{task_dir.name}] SKIP: no archive dir {archive_task}")
            any_missing = True
            continue
        any_missing |= process_task(task_dir, archive_task)
    return 1 if any_missing else 0


if __name__ == "__main__":
    sys.exit(main())
