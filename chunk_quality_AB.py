# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "pyarrow"]
# ///
"""Kept-chunk GT-quality diagnostic for the AB RM (node2 port of warp_rm60/chunk_quality.py).

Same math as node1's chunk_quality.py (H=30 final-action gate at raw thr=1.0, GT
look-progress over chunk+1s), over the merged A∪B GT corpus, with:
  * manifest = A + B merged (idx 0..5990);
  * the AB RM sidecar given on the CLI;
  * reference kept-set = A-champion sss15_30 sidecar (A episodes only) for Jaccard
    on the common A half.

A-champion reference (node1 chunk_quality_summary.json, sss15_30): kept 31.52%,
ratio_look 1.337, ratio_chunk 1.41.

Usage: chunk_quality_AB.py <name> <sidecar_parquet>
Writes chunk_quality_<name>.tsv + merges into chunk_quality_AB_summary.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/data/warp_rm/gt_validation")
GT_DIR = ROOT / "gt"
MAN_A = Path("/mnt/data/warp_rm/render_manifest_A.json")
MAN_B = Path("/home/shadeform/abc/render_manifest_B.json")
REF_SIDECAR = Path("/mnt/data/warp_rm/sidecars/rm_perobj_sss15/dense_predictions.parquet")
SUMMARY_JSON = ROOT / "chunk_quality_AB_summary.json"

H, LOOK, THR, PRE_S = 30, 60, 1.0, 3.0


def load_manifest_ab() -> dict:
    man = json.loads(MAN_A.read_text())
    man.update(json.loads(MAN_B.read_text()))
    return man


def load_parquet_signal(path, idx2uuid) -> dict[str, np.ndarray]:
    df = pd.read_parquet(path, columns=["episode_index", "frame_index",
                                        "warp_rm_signed_magnitude"])
    out = {}
    for ep_idx, g in df.groupby("episode_index"):
        u = idx2uuid.get(int(ep_idx))
        if u is not None:
            g = g.sort_values("frame_index")
            out[u] = g["warp_rm_signed_magnitude"].to_numpy(np.float64)
    return out


def episode_arrays(t, stair, v):
    L = min(len(t), len(v))
    t, stair, v = t[:L], stair[:L], v[:L]
    idx = np.arange(L)
    kept = v[np.minimum(idx + H - 1, L - 1)] > THR
    prog_look = stair[np.minimum(idx + LOOK - 1, L - 1)] - stair
    prog_chunk = stair[np.minimum(idx + H - 1, L - 1)] - stair
    ev_t = t[np.flatnonzero(np.diff(stair) > 0) + 1]
    if len(ev_t):
        dt = ev_t[None, :] - t[:, None]
        pre = np.any((dt > 0) & (dt <= PRE_S), axis=1)
    else:
        pre = np.zeros(L, bool)
    return L, kept, prog_look.astype(np.float64), prog_chunk.astype(np.float64), pre


def main():
    name, sidecar = sys.argv[1], sys.argv[2]
    manifest = load_manifest_ab()
    idx2uuid = {int(k): v["uuid"] for k, v in manifest.items()}
    uuid2idx = {v["uuid"]: int(k) for k, v in manifest.items()}

    gt = {}
    for f in sorted(GT_DIR.glob("*.npz")):
        z = np.load(f)
        gt[f.stem] = (z["t"].astype(np.float64), z["n_in_bin"].astype(np.int32))
    print(f"[gt] {len(gt)} episodes")

    vel = load_parquet_signal(sidecar, idx2uuid)
    # Reference kept-sets on the A half (A-champion consumed A's own index space,
    # which is IDENTICAL to the merged space for idx<2438).
    ref_vel = load_parquet_signal(REF_SIDECAR, idx2uuid) if REF_SIDECAR.exists() else {}
    print(f"[sig] {name}: {len(vel)} eps; ref(A-champion): {len(ref_vel)} eps")

    def pooled(subset_uuids, tag):
        if not subset_uuids:
            return [], dict(scope=tag, n_episodes=0)
        kept_all, look_all, chunk_all, pre_all = [], [], [], []
        j_inter = j_union = 0
        n_ref = 0
        rows = []
        for uuid in subset_uuids:
            t, stair = gt[uuid]
            L, kept, pl, pc, pre = episode_arrays(t, stair, vel[uuid])
            kept_all.append(kept); look_all.append(pl); chunk_all.append(pc); pre_all.append(pre)
            rows.append(dict(uuid=uuid, idx=uuid2idx[uuid], L=L,
                             kept_frac=round(float(kept.mean()), 4)))
            if uuid in ref_vel:
                _, rk, _, _, _ = episode_arrays(t, stair, ref_vel[uuid])
                n = min(len(rk), len(kept))
                a, b = kept[:n], rk[:n]
                j_inter += int((a & b).sum()); j_union += int((a | b).sum())
                n_ref += 1
        kept = np.concatenate(kept_all); look = np.concatenate(look_all)
        chunk = np.concatenate(chunk_all); pre = np.concatenate(pre_all)
        kl, kr = look[kept], look[~kept]
        cl, cr = chunk[kept], chunk[~kept]
        return rows, dict(
            scope=tag, n_episodes=len(rows), n_chunks=int(len(kept)),
            kept_frac=round(float(kept.mean()), 4),
            mean_prog_look_kept=round(float(kl.mean()), 5),
            mean_prog_look_rej=round(float(kr.mean()), 5),
            ratio_look=round(float(kl.mean() / kr.mean()), 3) if kr.mean() else None,
            mean_prog_chunk_kept=round(float(cl.mean()), 5),
            mean_prog_chunk_rej=round(float(cr.mean()), 5),
            ratio_chunk=round(float(cl.mean() / cr.mean()), 3) if cr.mean() else None,
            zero_frac_kept_look=round(float((kl == 0).mean()), 4),
            neg_frac_kept_look=round(float((kl < 0).mean()), 4),
            pre_frac_kept=round(float(pre[kept].mean()), 4),
            pre_frac_all=round(float(pre.mean()), 4),
            pre_capture_rate=round(float(kept[pre].mean()), 4),
            jaccard_vs_Achampion=round(j_inter / j_union, 4) if j_union else None,
            n_common_ref_eps=n_ref,
        )

    covered = [u for u in sorted(gt) if u in vel]
    a_half = [u for u in covered if uuid2idx[u] < 2438]
    b_half = [u for u in covered if uuid2idx[u] >= 2438]
    out = {}
    rows, out["AB"] = pooled(covered, "AB")
    _, out["A_half"] = pooled(a_half, "A_half")
    _, out["B_half"] = pooled(b_half, "B_half")

    summary = json.loads(SUMMARY_JSON.read_text()) if SUMMARY_JSON.exists() else {}
    summary[name] = out
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    with open(ROOT / f"chunk_quality_{name}.tsv", "w") as f:
        cols = list(rows[0])
        f.write("\t".join(cols) + "\n")
        for r in sorted(rows, key=lambda r: r["idx"]):
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
