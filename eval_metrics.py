# /// script
# requires-python = ">=3.10"
# dependencies = ["tyro"]
# ///
"""Per-bottle placement time + throughput from sim-eval summary.json(s).

Matches the WARP-RM paper's bottle-in-bin metrics (Table 4 / Fig 5), adapted to sim:
- Mean Time / Bottle = mean interval between consecutive drops, over placed bottles only
  (drop_k = first chunk where #bottles >= k; reconstructed from the per-chunk `bottles`
  trace in chunk_metrics).
- Throughput = bottles placed per (sim) HOUR, with non-completing worlds charged their
  FULL rollout budget to the denominator (mirrors the paper charging timed-out trials
  their full timeout) and completing worlds charged their actual completion time.
- Time base is SIM seconds (control_decimation * timestep * actions) = the policy's task
  speed, independent of GPU/inference latency. Pass multiple summaries to compare arms.

  uv run eval_metrics.py --summaries outputs/vanilla/summary.json outputs/rabc/summary.json \
    --labels vanilla rabc
"""
import json
from dataclasses import dataclass, field
from typing import List, Optional

import tyro


@dataclass
class Cfg:
    summaries: List[str] = field(default_factory=list)
    labels: Optional[List[str]] = None


def analyze(path: str) -> dict:
    s = json.load(open(path))
    c = s["config"]; sc = c["scene"]
    per_action = sc["control_decimation"] * sc["timestep"]
    chunk_s = c["execute_chunk_dim"] * per_action
    nb = sc["bottle_count"]
    per_bottle, total_placed, total_time, worlds = [], 0, 0.0, 0
    for w in s["worlds"]:
        cm = w["chunk_metrics"]
        if not cm:
            continue
        worlds += 1
        trace = [m["bottles"] for m in cm]
        # placed = canonical peak (max_bottles_in_bin_so_far); drop TIMES from the
        # per-chunk trace (first chunk reaching each k, up to the trace's own peak).
        peak = int((w.get("final_task_eval") or {}).get("max_bottles_in_bin_so_far")
                   or (max(trace) if trace else 0))
        total_placed += peak
        prev = 0
        for k in range(1, (max(trace) if trace else 0) + 1):
            dc = next((i for i, b in enumerate(trace) if b >= k), None)
            if dc is not None:
                per_bottle.append((dc - prev) * chunk_s)
                prev = dc
        total_time += len(cm) * chunk_s  # full budget for non-complete; actual for complete
    return {
        "worlds": worlds,
        "bottles_placed": total_placed,
        "possible": worlds * nb,
        "mean_time_per_bottle_s": (sum(per_bottle) / len(per_bottle)) if per_bottle else 0.0,
        "median_time_per_bottle_s": (sorted(per_bottle)[len(per_bottle) // 2]) if per_bottle else 0.0,
        "throughput_per_hr": (total_placed / total_time * 3600) if total_time else 0.0,
        "success_rate": s.get("success_rate"),
        "total_sim_time_s": total_time,
        "chunk_s": chunk_s,
    }


def main(cfg: Cfg):
    if not cfg.summaries:
        raise SystemExit("pass --summaries path/to/summary.json [...]")
    rows = []
    for i, p in enumerate(cfg.summaries):
        r = analyze(p)
        lbl = cfg.labels[i] if cfg.labels and i < len(cfg.labels) else p.split("/")[-2]
        rows.append((lbl, r))
        print(f"\n=== {lbl} ===")
        print(f"  success_rate (all binned): {r['success_rate']:.2f}")
        print(f"  bottles placed:            {r['bottles_placed']}/{r['possible']}")
        print(f"  mean time / bottle:        {r['mean_time_per_bottle_s']:.1f} s  (median {r['median_time_per_bottle_s']:.1f} s)")
        print(f"  throughput:                {r['throughput_per_hr']:.1f} bottles/hr  (sim-time)")
        print(f"  (worlds={r['worlds']}, total sim-time {r['total_sim_time_s']:.0f}s, {r['chunk_s']:.2f}s/chunk)")
    if len(rows) > 1:
        print("\n=== comparison ===")
        print(f"{'arm':<16}{'succ':>6}{'placed':>9}{'s/bottle':>10}{'btl/hr':>9}")
        for lbl, r in rows:
            print(f"{lbl:<16}{r['success_rate']:>6.2f}{r['bottles_placed']:>5}/{r['possible']:<3}{r['mean_time_per_bottle_s']:>10.1f}{r['throughput_per_hr']:>9.1f}")


if __name__ == "__main__":
    main(tyro.cli(Cfg))
