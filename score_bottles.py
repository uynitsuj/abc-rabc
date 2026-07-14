#!/usr/bin/env python
"""Score full-horizon bottle-in-bin rollout traces into the paper's metrics.

This is the OFFLINE scorer that produces the paper's simulated bottle-in-bin
numbers. It is deliberately separate from ``abc_minimal.eval_policy``:

  Two-step eval (see the WARP-RM release README):
    1. abc_minimal.eval_policy --no-early-stop   -> saves fp16 qpos traces
       (qpos_trace_*.npz) for the full 60 s horizon of every world.
    2. score_bottles.py over those traces         -> the paper metrics.

  IMPORTANT: eval_policy's LIVE in-bin count is a loose *center-point* check
  used only for rollout control (early_stop); it is NOT the paper metric and
  will not match these numbers. The paper scores here, OFFLINE: a bottle counts
  as placed iff *any part of it* (5 points sampled along its long axis from the
  free-joint pose) lies inside a rim-tight cylinder, for >=0.5 s continuously
  (persistence), and it is still inside at the 60 s horizon (final-standing).

Metrics (all per paper): bottles/scene (paired t vs baseline), pooled from-0
per-bottle placement interval, throughput (Sum count / Sum effective-time * 3600,
effective-time = last-placement-time if all FULL placed else the 60 s horizon),
and >=k placed rates. ``--all-rules`` additionally emits the appendix robustness
ladder (loose-center / tight-center / tight-lowest / tight-anypart).

Usage
-----
    python score_bottles.py --trace-dir local_eval_out --arm-glob 'fullhz_{arm}_sh*'
    python score_bottles.py --trace-dir local_eval_out --all-rules
    python score_bottles.py --trace-dir local_eval_out --self-test   # reproduce the locked paper table
"""
from __future__ import annotations
import argparse, glob, json, math, re, sys
from pathlib import Path
import numpy as np

# Fixed free-joint qpos offsets for the 6-bottle put_bottles scene. Constant
# across all procedurally-generated scenes (verified over both arms x all
# shards). Re-derive if the scene XML changes, via:
#   from abc_minimal.view_primitive import build_model, _freejoint_adr, bottle_joint_names
#   m,_ = build_model(record, scene_xml)
#   BOTTLE_ADDRS = [_freejoint_adr(m, n) for n in bottle_joint_names(record)]
#   BIN_ADDR     = _freejoint_adr(m, "bin_joint")
BOTTLE_ADDRS = (23, 30, 37, 44, 51, 58)
BIN_ADDR = 0
HALF = 0.083                # bottle half-length along local +z (~17 cm bottle)
HZ, DT = 30.0, 1.0 / 30.0
PERSIST, GRACE = 15, 30     # >=0.5 s continuous contact to qualify; 1 s startup grace
HORIZON_S, FULL = 60.0, 6

# name -> (radial R [m], rel-z low [m], rel-z high [m], mode, n_axis_samples)
RULES = {
    "loose-center":  (0.155, -0.06, 0.26, "center", 1),
    "tight-center":  (0.10, -0.06, 0.14, "center", 1),
    "tight-lowest":  (0.10, -0.06, 0.14, "lowest", 2),
    "tight-anypart": (0.10, -0.06, 0.14, "anypart", 5),   # <-- PAPER metric
}
PAPER_RULE = "tight-anypart"


def _quat_axis(qw, qx, qy, qz):
    """World-frame direction of the bottle's local +z (long) axis."""
    return np.array([2 * (qx * qz + qw * qy),
                     2 * (qy * qz - qw * qx),
                     1 - 2 * (qx * qx + qy * qy)])


def _in_region(rel, R, LO, HI):
    return (math.hypot(rel[0], rel[1]) <= R) and (LO <= rel[2] <= HI)


def _in_bin_frame(q, R, LO, HI, mode, ns):
    binp = q[BIN_ADDR:BIN_ADDR + 3]
    out = np.zeros(len(BOTTLE_ADDRS), bool)
    for j, a in enumerate(BOTTLE_ADDRS):
        pos = q[a:a + 3]
        if mode == "center":
            out[j] = _in_region(pos - binp, R, LO, HI)
        else:
            ax = _quat_axis(*q[a + 3:a + 7])
            if mode == "lowest":
                base, top = pos - HALF * ax, pos + HALF * ax
                lo = base if base[2] < top[2] else top
                out[j] = _in_region(lo - binp, R, LO, HI)
            else:  # anypart: any of ns points along the long axis
                out[j] = any(_in_region(pos + f * HALF * ax - binp, R, LO, HI)
                             for f in np.linspace(-1, 1, ns))
    return out


def score_world(trace, R, LO, HI, mode, ns):
    """One rollout trace -> (count, from-0 per-bottle intervals, eff-time, throughput)."""
    n = len(BOTTLE_ADDRS)
    streak = np.zeros(n, int)
    qual = np.full(n, -1)
    last_inb = last_streak = None
    for i, q in enumerate(trace):
        step = i + 1
        inb = _in_bin_frame(q, R, LO, HI, mode, ns)
        streak = np.where(inb, streak + 1, 0)
        newly = (streak >= PERSIST) & (qual < 0)
        if step >= GRACE:
            qual[newly] = step - PERSIST + 1
        last_inb, last_streak = inb, streak
    final_standing = last_inb & (last_streak >= PERSIST)
    count = int(final_standing.sum())
    qt = sorted(qual[b] * DT for b in range(n) if final_standing[b] and qual[b] > 0)
    ivs = [qt[0]] + [qt[k] - qt[k - 1] for k in range(1, len(qt))] if qt else []
    last_s = qt[-1] if qt else None
    eff = last_s if (count == FULL and last_s) else HORIZON_S
    return dict(count=count, ivs=ivs, mean_iv=(float(np.mean(ivs)) if ivs else None),
                eff=eff, thru=count / eff * 3600.0)


def collect(trace_dir, arm, arm_glob):
    """Return {world_seed: trace_path} for one arm. Globs qpos_trace_*.npz directly
    (robust to summary.json schema differences across eval_policy versions); the
    world seed is parsed from the filename qpos_trace_<idx>_s<seed>.npz."""
    out = {}
    pat = arm_glob.format(arm=arm)
    for shard in sorted(glob.glob(str(Path(trace_dir) / pat))):
        for npz in sorted(glob.glob(str(Path(shard) / "qpos_trace_*.npz"))):
            m = re.search(r"_s(\d+)\.npz$", Path(npz).name)
            seed = int(m.group(1)) if m else Path(npz).name
            out[seed] = npz
    return out


def paired_t(vd, wd):
    c = sorted(set(vd) & set(wd))
    v = np.array([vd[s] for s in c]); w = np.array([wd[s] for s in c])
    d = w - v
    t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 and d.std() > 0 else float("nan")
    p = math.erfc(abs(t) / math.sqrt(2)) if not math.isnan(t) else float("nan")
    return v.mean(), w.mean(), d.mean(), t, p, len(c)


def score_arm(trace_dir, arm, arm_glob, rule):
    R, LO, HI, mode, ns = RULES[rule]
    worlds = collect(trace_dir, arm, arm_glob)
    per = {}
    for seed, tp in worlds.items():
        z = np.load(tp); tr = z[z.files[0]].astype(np.float64)
        per[seed] = score_world(tr, R, LO, HI, mode, ns)
    return per


def report(trace_dir, arm_glob, baseline, arm, rule):
    v = score_arm(trace_dir, baseline, arm_glob, rule)
    w = score_arm(trace_dir, arm, arm_glob, rule)
    c = sorted(set(v) & set(w))
    vm, wm, dm, t, p, n = paired_t({s: v[s]["count"] for s in v}, {s: w[s]["count"] for s in w})
    vp = [iv for s in c for iv in v[s]["ivs"]]; wp = [iv for s in c for iv in w[s]["ivs"]]
    _, _, _, tt, tpp, _ = paired_t({s: v[s]["mean_iv"] for s in v if v[s]["mean_iv"] is not None},
                                   {s: w[s]["mean_iv"] for s in w if w[s]["mean_iv"] is not None})
    agv = sum(v[s]["count"] for s in c) / sum(v[s]["eff"] for s in c) * 3600
    agw = sum(w[s]["count"] for s in c) / sum(w[s]["eff"] for s in c) * 3600
    _, _, _, tth, thp, _ = paired_t({s: v[s]["thru"] for s in v}, {s: w[s]["thru"] for s in w})
    res = dict(n=n, rule=rule,
               van_count=sum(v[s]["count"] for s in c), warp_count=sum(w[s]["count"] for s in c),
               van_per=vm, warp_per=wm, diff=dm, t=t, p=p,
               tpb_van=float(np.mean(vp)), tpb_warp=float(np.mean(wp)), tpb_t=tt, tpb_p=tpp,
               thru_van=agv, thru_warp=agw, thru_t=tth, thru_p=thp,
               geq={k: (100 * np.mean([v[s]["count"] >= k for s in c]),
                        100 * np.mean([w[s]["count"] >= k for s in c])) for k in (4, 5, 6)})
    return res


def print_res(r, label=""):
    print(f"\n===== {r['rule']}  n={r['n']}  {label} =====")
    print(f"  bottles placed:  {r['van_count']}/{r['n']*6} -> {r['warp_count']}/{r['n']*6}")
    print(f"  bottles/scene:   {r['van_per']:.3f} -> {r['warp_per']:.3f}   diff {r['diff']:+.3f}  t({r['n']-1})={r['t']:.2f}  p={r['p']:.1e}")
    print(f"  time/bottle:     {r['tpb_van']:.1f} -> {r['tpb_warp']:.1f} s   per-scene paired t={r['tpb_t']:.2f} p={r['tpb_p']:.1e}")
    print(f"  throughput:      {r['thru_van']:.0f} -> {r['thru_warp']:.0f} /hr   per-scene paired t={r['thru_t']:.2f} p={r['thru_p']:.1e}")
    for k in (4, 5, 6):
        print(f"  >= {k} placed:     {r['geq'][k][0]:.1f}% -> {r['geq'][k][1]:.1f}%")


def self_test(trace_dir, arm_glob):
    """Assert every published (locked) n=128 cell. Non-zero exit on mismatch."""
    r = report(trace_dir, arm_glob, "vanilla", "warp", PAPER_RULE)
    print_res(r, "[PAPER — self-test]")
    checks = [
        ("n", r["n"], 128, 0), ("van_count", r["van_count"], 509, 0), ("warp_count", r["warp_count"], 598, 0),
        ("van_per", r["van_per"], 3.98, 0.01), ("warp_per", r["warp_per"], 4.67, 0.01),
        ("diff", r["diff"], 0.695, 0.01), ("t", r["t"], 4.47, 0.05),
        ("tpb_van", r["tpb_van"], 10.7, 0.1), ("tpb_warp", r["tpb_warp"], 8.8, 0.1), ("tpb_t", abs(r["tpb_t"]), 5.01, 0.1),
        ("thru_van", r["thru_van"], 243, 2), ("thru_warp", r["thru_warp"], 301, 2), ("thru_t", r["thru_t"], 5.30, 0.1),
        ("all6_van", r["geq"][6][0], 12.5, 0.5), ("all6_warp", r["geq"][6][1], 28.1, 0.5),
    ]
    # appendix robustness ladder (bottles/scene diff + t)
    ladder = {"loose-center": (0.766, 4.77), "tight-center": (0.664, 4.60),
              "tight-lowest": (0.383, 2.43), "tight-anypart": (0.695, 4.47)}
    print("\n===== appendix robustness ladder =====")
    for rule, (exp_d, exp_t) in ladder.items():
        rr = report(trace_dir, arm_glob, "vanilla", "warp", rule)
        print(f"  {rule:15s} {rr['van_per']:.3f} -> {rr['warp_per']:.3f}  diff {rr['diff']:+.3f}  t={rr['t']:.2f}")
        checks += [(f"{rule}:diff", rr["diff"], exp_d, 0.01), (f"{rule}:t", rr["t"], exp_t, 0.05)]
    print("\n===== self-test assertions =====")
    ok = True
    for name, got, exp, tol in checks:
        good = abs(got - exp) <= tol
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {name}: got {got:.3f}  expect {exp}±{tol}")
    print("\n" + ("ALL CELLS REPRODUCED — scorer matches the locked paper table." if ok
                  else "*** MISMATCH — scorer does NOT reproduce the paper table ***"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--arm-glob", default="fullhz_{arm}_sh*",
                    help="subdir glob per arm; '{arm}' is substituted (default n=128 set)")
    ap.add_argument("--baseline", default="vanilla")
    ap.add_argument("--arm", default="warp")
    ap.add_argument("--rule", default=PAPER_RULE, choices=list(RULES))
    ap.add_argument("--all-rules", action="store_true", help="emit the appendix robustness ladder")
    ap.add_argument("--self-test", action="store_true", help="assert the locked paper table")
    ap.add_argument("--bottle-addrs", default=None,
                    help="comma-sep freejoint qpos offsets, overrides baked-in (use if the scene's qpos layout differs)")
    ap.add_argument("--bin-addr", type=int, default=None, help="bin freejoint qpos offset override")
    a = ap.parse_args()
    global BOTTLE_ADDRS, BIN_ADDR
    if a.bottle_addrs:
        BOTTLE_ADDRS = tuple(int(x) for x in a.bottle_addrs.split(","))
        print(f"[override] BOTTLE_ADDRS={BOTTLE_ADDRS}")
    if a.bin_addr is not None:
        BIN_ADDR = a.bin_addr
        print(f"[override] BIN_ADDR={BIN_ADDR}")
    if a.self_test:
        sys.exit(self_test(a.trace_dir, a.arm_glob))
    if a.all_rules:
        for rule in RULES:
            print_res(report(a.trace_dir, a.arm_glob, a.baseline, a.arm, rule))
    else:
        print_res(report(a.trace_dir, a.arm_glob, a.baseline, a.arm, a.rule), "[PAPER]" if a.rule == PAPER_RULE else "")


if __name__ == "__main__":
    main()
