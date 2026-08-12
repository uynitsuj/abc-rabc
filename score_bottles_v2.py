"""score_bottles + the composite metric and bin-tip diagnostics.

Composite ("tight-upright-ever", DEFAULT headline per Justin 2026-08-12):
a bottle scores iff it completes the tight-anypart persistence window while the
bin is upright (relative tilt < 20 deg from its own initial pose). Ever-credit:
placed-then-bin-tipped keeps credit; never-truly-in never earns it.

Validated against human ground truth on the 4 highest-disagreement worlds of
bs128conv: exact on 2 (incl. both-metrics-wrong s20260536), +1 on 2. Known +1
bias: ever-credit counts a bottle that passes tight+persist mid-episode and is
later displaced without the bin tipping (s20260525).

Also reports, per arm: paper-rule count (comparability with all campaign
numbers), tip rate, and median tip step. The paper rule's known flaw stands:
its acceptance cylinder is world-aligned at the bin POSITION (quat ignored), so
tipped-bin worlds can earn phantom partial credit.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import score_bottles as SB
except ImportError:  # running from a deploy dir next to abc_rabc
    sys.path.insert(0, "/scratch/warprm_eval/abc_rabc")
    import score_bottles as SB

TILT_DEG = 20.0
COS_LIM = math.cos(math.radians(TILT_DEG))


def _rotmat(q4):
    w, x, y, z = q4
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def composite_world(trace, rule="tight-anypart"):
    R, LO, HI, mode, ns = SB.RULES[rule]
    n = len(SB.BOTTLE_ADDRS)
    q0 = trace[0]
    v0 = _rotmat(q0[SB.BIN_ADDR + 3:SB.BIN_ADDR + 7]).T @ np.array([0.0, 0.0, 1.0])
    streak = np.zeros(n, int)
    credited = np.zeros(n, bool)
    tip_step = None
    for i, q in enumerate(trace):
        step = i + 1
        up = float((_rotmat(q[SB.BIN_ADDR + 3:SB.BIN_ADDR + 7]) @ v0)[2])
        upright = up >= COS_LIM
        if not upright and tip_step is None:
            tip_step = step
        inb = SB._in_bin_frame(q, R, LO, HI, mode, ns)
        streak = np.where(inb, streak + 1, 0)
        if upright and step >= SB.GRACE:
            credited |= (streak >= SB.PERSIST)
    return int(credited.sum()), tip_step


def score_arm(trace_dir, arm, arm_glob):
    out = {}
    for seed, npz in SB.collect(trace_dir, arm, arm_glob).items():
        tr = np.load(npz)["qpos"].astype(np.float64)
        comp, tip = composite_world(tr)
        R, LO, HI, mode, ns = SB.RULES[SB.PAPER_RULE]
        tight = SB.score_world(tr, R, LO, HI, mode, ns)["count"]
        out[seed] = dict(comp=comp, tight=tight, tip=tip)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--arm-glob", default="fullhz_{arm}_sh*")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--arm", required=True)
    a = ap.parse_args()

    base = score_arm(a.trace_dir, a.baseline, a.arm_glob)
    arm = score_arm(a.trace_dir, a.arm, a.arm_glob)
    common = sorted(set(base) & set(arm))
    for name, d in ((a.baseline, base), (a.arm, arm)):
        c = [d[s]["comp"] for s in common]
        t = [d[s]["tight"] for s in common]
        tips = [d[s]["tip"] for s in common if d[s]["tip"] is not None]
        print(f"{name:>14s}: composite={np.mean(c):.3f}  paper-rule={np.mean(t):.3f}  "
              f"tip-rate={len(tips)}/{len(common)}  median-tip-step={int(np.median(tips)) if tips else '-'}")
    d = np.array([arm[s]["comp"] - base[s]["comp"] for s in common], float)
    t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 and d.std() > 0 else float("nan")
    p = math.erfc(abs(t) / math.sqrt(2)) if not math.isnan(t) else float("nan")
    print(f"\ncomposite paired: diff={d.mean():+.3f}  t({len(d)-1})={t:.2f}  p={p:.2e}  n={len(d)}")


if __name__ == "__main__":
    main()
