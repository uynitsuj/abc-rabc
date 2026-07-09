# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mujoco", "boto3", "tyro"]
# ///
"""Per-episode bottle counts for the B-only eps + a merged object_counts_AB.json.

The winning RM is retrained on the expanded A+B corpus with per-object-count shortest-25%
stratification, which needs a bottle count for EVERY episode keyed by the merged
episode_index. This script derives B's counts the SAME way the GT harness did
(gt_validation/code/gt_compute.py): build the MjModel from scene_assembled.xml and count
joints matching `bottle_(\\d+)_joint` (verified identical to the delivery
randomization.json bottle_count on a stratified sample).

Index space (single, consistent — see make_manifest_B.py): the B manifest is contiguous
2438..(2438+N-1). Those ARE the merged episode_index for B (A occupies 0..2437, then B
appends in manifest/uuid order). So object_counts_AB is keyed 0..(2437+N):
    * 0..2437       -> copied VERBATIM from A's object_counts.json (never recomputed)
    * 2438..2437+N  -> B counts at their manifest global index

Run node-side (delivery read creds + /mnt/data/render_assets):
    .venv/bin/python compute_object_counts_B.py --manifest render_manifest_B.json \\
        --a-object-counts /mnt/data/warp_rm/object_counts.json \\
        --asset-root /mnt/data/render_assets --out-ab object_counts_AB.json
"""
from __future__ import annotations

import json
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import tyro

DELIVERY_BUCKET = "xdof-bair-abc"
SUBTASK = "sim_put_the_plastic_bottles_in_the_bin"
BOTTLE_JOINT = re.compile(r"bottle_(\d+)_joint")

_S3 = None
_ASSET_ROOT = None


@dataclass
class Cfg:
    manifest: str = "render_manifest_B.json"
    a_object_counts: str = "/mnt/data/warp_rm/object_counts.json"
    asset_root: str = "/mnt/data/render_assets"
    out_b: str = "object_counts_B.json"     # B-only, keyed by manifest global index
    out_ab: str = "object_counts_AB.json"   # A (verbatim) + B, keyed 0..2437+N
    workers: int = 48
    validate: int = 5                       # cross-check N eps vs randomization.json


def _init(asset_root: str):
    global _S3, _ASSET_ROOT
    import os
    import boto3
    from botocore.config import Config as BotoConfig
    os.nice(10)
    _S3 = boto3.client("s3", config=BotoConfig(max_pool_connections=4,
                                               retries={"max_attempts": 8}))
    _ASSET_ROOT = asset_root


def count_bottles(job) -> tuple[int, str, int | None, str]:
    """(global_idx, uuid) -> (idx, uuid, n_bottles_from_model, error)."""
    import mujoco
    from rerender_mjwarp import build_scene_xml
    idx, uuid = job[0], job[1]
    try:
        with tempfile.TemporaryDirectory() as td:
            xmlp = Path(td) / "scene.xml"
            _S3.download_file(DELIVERY_BUCKET, job[2] + "/scene_assembled.xml", str(xmlp))
            xml = build_scene_xml(xmlp, _ASSET_ROOT)
            model = mujoco.MjModel.from_xml_string(xml)
            n = sum(1 for j in range(model.njnt)
                    if BOTTLE_JOINT.fullmatch(model.jnt(j).name or ""))
        return idx, uuid, n, ""
    except Exception as e:  # noqa: BLE001
        return idx, uuid, None, f"{type(e).__name__}: {e}"


def randomization_count(s3, delivery: str, uuid: str) -> tuple[int | None, int | None]:
    key = f"data/deliveries/{delivery}/{SUBTASK}/{uuid}/randomization.json"
    r = json.loads(s3.get_object(Bucket=DELIVERY_BUCKET, Key=key)["Body"].read())
    obj = r.get("object_states", {})
    n = sum(1 for k in obj if k.startswith("bottle_") and k.endswith("_joint"))
    meta_n = r.get("metadata", {}).get("bottle_count")
    return n, meta_n


def main(cfg: Cfg) -> None:
    man = json.loads(Path(cfg.manifest).read_text())
    # job carries the full delivery prefix so the worker needs no manifest.
    jobs = []
    for k, info in sorted(man.items(), key=lambda kv: int(kv[0])):
        prefix = f"data/deliveries/{info['delivery']}/{SUBTASK}/{info['uuid']}"
        jobs.append((int(k), info["uuid"], prefix))
    print(f"[oc-B] {len(jobs)} B eps -> bottle counts via MjModel")

    counts_b: dict[str, int] = {}
    errors = []
    with Pool(cfg.workers, initializer=_init, initargs=(cfg.asset_root,)) as pool:
        for n, (idx, uuid, nb, err) in enumerate(pool.imap_unordered(count_bottles, jobs, chunksize=8)):
            if err:
                errors.append((idx, uuid, err))
            else:
                counts_b[str(idx)] = nb
            if (n + 1) % 500 == 0:
                print(f"[oc-B] {n + 1}/{len(jobs)} ...", flush=True)
    print(f"[oc-B] computed {len(counts_b)}/{len(jobs)} (errors={len(errors)})")
    if errors:
        print(f"[oc-B][WARN] errors: {errors[:5]}")
    distB = dict(sorted(Counter(counts_b.values()).items()))
    print(f"[oc-B] B distribution: {distB}")

    # Cross-check a stratified sample vs randomization.json.
    import boto3
    from botocore.config import Config as BotoConfig
    s3 = boto3.client("s3", config=BotoConfig(retries={"max_attempts": 8}))
    by_count: dict[int, list] = {}
    for k, info in sorted(man.items(), key=lambda kv: int(kv[0])):
        c = counts_b.get(k)
        if c is not None and len(by_count.get(c, [])) < 1:
            by_count.setdefault(c, []).append((k, info))
    n_checked = ok = 0
    for c, lst in sorted(by_count.items()):
        if n_checked >= cfg.validate:
            break
        for k, info in lst:
            rn, mn = randomization_count(s3, info["delivery"], info["uuid"])
            model_n = counts_b[k]
            match = (model_n == rn == mn)
            ok += match
            n_checked += 1
            print(f"[oc-B][validate] idx={k} {info['uuid'][8:24]} model={model_n} "
                  f"rand={rn} meta={mn} match={match}")
    print(f"[oc-B] validation: {ok}/{n_checked} match randomization.json")

    Path(cfg.out_b).write_text(json.dumps({
        "count_key": "bottle_count",
        "method": "mjmodel_bottle_joint (== gt_compute.py; == randomization.json on sample)",
        "n_episodes": len(counts_b), "distribution": distB,
        "index_space": "manifest global episode_index (== merged A+B index)",
        "counts": counts_b}, indent=0))
    print(f"[oc-B] wrote {cfg.out_b}")

    # Merge with A (verbatim) -> object_counts_AB.json keyed 0..2437+N.
    a = json.loads(Path(cfg.a_object_counts).read_text())
    a_counts = a["counts"]
    assert set(a_counts) == {str(i) for i in range(len(a_counts))}, "A counts not 0..N-1 contiguous"
    ab = dict(a_counts)             # 0..2437 verbatim
    overlap = set(ab) & set(counts_b)
    assert not overlap, f"A/B index overlap in object_counts: {sorted(overlap)[:5]}"
    ab.update(counts_b)             # 2438..2437+N
    ab_idxs = sorted(int(x) for x in ab)
    assert ab_idxs == list(range(len(ab))), "object_counts_AB not contiguous 0..M-1"
    distAB = dict(sorted(Counter(ab.values()).items()))
    Path(cfg.out_ab).write_text(json.dumps({
        "count_key": "bottle_count",
        "source": f"A verbatim ({len(a_counts)}) + B via MjModel ({len(counts_b)})",
        "n_episodes": len(ab), "n_A": len(a_counts), "n_B": len(counts_b),
        "distribution_A": a.get("distribution"),
        "distribution_B": distB, "distribution_AB": distAB,
        "index_space": "merged episode_index: A[0..%d] ++ B[uuid-sorted]" % (len(a_counts) - 1),
        "counts": ab}, indent=0))
    print(f"[oc-B] wrote {cfg.out_ab}: {len(ab)} eps (A={len(a_counts)} + B={len(counts_b)})")
    print(f"[oc-B] AB distribution: {distAB}")


if __name__ == "__main__":
    main(tyro.cli(Cfg))
