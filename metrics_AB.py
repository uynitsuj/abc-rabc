# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "pyarrow"]
# ///
"""GT gate metrics for the AB-retrained RM, over the full A∪B GT corpus.

Adapted from gt_validation/code/metrics.py (same math: per-episode progress
Spearman, pooled event AUC, across-episode pace Spearman, magnitude/kept-frac) with:
  * manifest = A's render_manifest.json (0..2437) + render_manifest_B.json
    (2438..5990) merged in memory;
  * signal = a dense_predictions.parquet sidecar path given on the CLI
    (episode_index keyed in the MERGED index space);
  * pace Spearman additionally reported per scope: all, val_sim-only (A-champion's
    0.703 reference scope), A-half (idx<2438), B-half (idx>=2438).

Usage: metrics_AB.py <name> <sidecar_parquet> [<scale_to_45eq>]
Writes summary_<name>.json + per_episode_<name>.tsv into gt_validation/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/data/warp_rm/gt_validation")
GT = ROOT / "gt"
MAN_A = Path("/mnt/data/warp_rm/render_manifest_A.json")
MAN_B = Path("/home/shadeform/abc/render_manifest_B.json")

sys.path.insert(0, str(ROOT / "code"))
from metrics import rank_avg, spearman, auc  # noqa: E402  (reuse the validated math)

PRE_S, IDLE_S = 3.0, 5.0


def load_manifest_ab() -> dict:
    man = json.loads(MAN_A.read_text())
    man_b = json.loads(MAN_B.read_text())
    overlap = set(man) & set(man_b)
    assert not overlap, f"manifest key overlap: {sorted(overlap)[:5]}"
    man.update(man_b)
    return man


def main() -> None:
    name, sidecar = sys.argv[1], sys.argv[2]
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0 / 45.0

    manifest = load_manifest_ab()
    idx2uuid = {int(k): v["uuid"] for k, v in manifest.items()}
    uuid2meta = {v["uuid"]: (int(k), v["split"]) for k, v in manifest.items()}
    print(f"[{name}] manifest AB: {len(manifest)} eps")

    import pandas as pd
    df = pd.read_parquet(sidecar, columns=["episode_index", "frame_index",
                                           "warp_rm_signed_magnitude"])
    vel: dict[str, np.ndarray] = {}
    for ep_idx, g in df.groupby("episode_index"):
        u = idx2uuid.get(int(ep_idx))
        if u is not None:
            g = g.sort_values("frame_index")
            vel[u] = g["warp_rm_signed_magnitude"].to_numpy(np.float64)
    print(f"[{name}] velocity sidecars for {len(vel)} episodes")

    rows = []
    pre_pool, idle_pool, all_v_raw = [], [], []
    mismatches = 0
    for gtf in sorted(GT.glob("*.npz")):
        uuid = gtf.stem
        if uuid not in vel or uuid not in uuid2meta:
            continue
        z = np.load(gtf)
        t, n_in = z["t"], z["n_in_bin"].astype(np.int16)
        v = vel[uuid]
        nT = min(len(t), len(v))
        mism = abs(len(t) - len(v))
        if mism:
            mismatches += 1
        t, n_in, v = t[:nT], n_in[:nT], v[:nT]

        ev_t = t[np.flatnonzero(np.diff(n_in) > 0) + 1]
        sp_prog = spearman(np.cumsum(v), n_in.astype(np.float64)) if len(ev_t) else float("nan")
        if len(ev_t):
            dt_signed = ev_t[None, :] - t[:, None]
            pre = np.any((dt_signed > 0) & (dt_signed <= PRE_S), axis=1)
            idle = np.min(np.abs(dt_signed), axis=1) > IDLE_S
        else:
            pre = np.zeros(nT, bool)
            idle = np.ones(nT, bool)
        pre_pool.append(v[pre]); idle_pool.append(v[idle]); all_v_raw.append(v)

        dur = float(t[-1] - t[0])
        idx, split = uuid2meta[uuid]
        rows.append(dict(
            idx=idx, uuid=uuid, split=split, n_frames_gt=len(z["t"]),
            n_frames_sidecar=len(vel[uuid]), len_mismatch=mism,
            n_events=len(ev_t), spearman_cumsum=round(sp_prog, 4),
            mean_v_raw=round(float(v.mean()), 4),
            kept_frac_raw=round(float((v > 1.0).mean()), 4),
            gt_final=int(n_in[-1]),
            gt_pace_bps=round(float(n_in[-1]) / dur, 5) if dur > 0 else 0.0,
        ))

    sp = np.array([r["spearman_cumsum"] for r in rows], dtype=np.float64)
    sp_ok = sp[~np.isnan(sp)]
    pre_v, idle_v = np.concatenate(pre_pool), np.concatenate(idle_pool)
    vv = np.concatenate(all_v_raw)
    mean_v = np.array([r["mean_v_raw"] for r in rows])
    pace = np.array([r["gt_pace_bps"] for r in rows])
    is_val = np.array([r["split"] == "val_sim" for r in rows])
    is_a = np.array([r["idx"] < 2438 for r in rows])

    def pace_sp(mask):
        return round(spearman(mean_v[mask], pace[mask]), 4) if mask.sum() >= 3 else None

    def kept_frac(pred):
        vs = [v for r, v in zip(rows, all_v_raw) if pred(r["idx"])]
        if not vs:
            return None
        return round(float((np.concatenate(vs) > 1.0).mean()), 4)

    summary = dict(
        signal=name, n_episodes=len(rows), n_len_mismatch=mismatches,
        scale_to_45eq=scale,
        progress_spearman_mean=round(float(sp_ok.mean()), 4),
        progress_spearman_median=round(float(np.median(sp_ok)), 4),
        progress_spearman_p10=round(float(np.percentile(sp_ok, 10)), 4),
        progress_n_valid=int(len(sp_ok)),
        event_auc=round(auc(pre_v, idle_v), 4),
        n_pre_frames=int(len(pre_v)), n_idle_frames=int(len(idle_v)),
        pace_spearman=pace_sp(np.ones(len(rows), bool)),
        pace_spearman_val_sim=pace_sp(is_val),
        pace_spearman_A_half=pace_sp(is_a),
        pace_spearman_B_half=pace_sp(~is_a),
        pace_spearman_val_sim_A=pace_sp(is_val & is_a),
        n_val_sim=int(is_val.sum()), n_A=int(is_a.sum()), n_B=int((~is_a).sum()),
        mean_v_raw=round(float(vv.mean()), 4),
        median_v_raw=round(float(np.median(vv)), 4),
        kept_frac_raw_thr1=round(float((vv > 1.0).mean()), 4),
        kept_frac_raw_thr1_A=kept_frac(lambda i: i < 2438),
        kept_frac_raw_thr1_B=kept_frac(lambda i: i >= 2438),
        n_frames_total=int(len(vv)),
    )
    (ROOT / f"summary_{name}.json").write_text(json.dumps(summary, indent=2))
    cols = list(rows[0])
    with open(ROOT / f"per_episode_{name}.tsv", "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in sorted(rows, key=lambda r: r["idx"]):
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
