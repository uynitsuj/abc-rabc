# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "boto3"]
# ///
"""Extend the GT staircase corpus to the B-only episodes (node2).

Phase 'download': fetch scene_assembled.xml + integration_state.npy for every
render_manifest_B.json episode into the SAME deliv layout gt_compute.py expects
(<root>/<uuid>/...). Resumable.

Phase 'compute': run the UNMODIFIED gt_compute.py episode_gt() logic over the B
manifest by importing it with its module-level path constants overridden for node2:
    MANIFEST   -> /home/shadeform/abc/render_manifest_B.json  (B-only, global 2438..5990)
    OBJ_COUNTS -> /mnt/data/warp_rm/object_counts_AB.json     (global keys, covers B)
    DELIV      -> /mnt/data/warp_rm/gt_validation/deliv       (uuid-keyed, no collision w/ A)
    GT_OUT     -> /mnt/data/warp_rm/gt_validation/gt          (uuid-keyed npz, joins A's 2,436)
    AUDIT_TSV  -> /mnt/data/warp_rm/gt_validation/gt_audit_B.tsv
    ASSET_ROOT -> /mnt/data/render_assets                     (already on node2 from the render)
gt_compute.py itself is not edited — same geometry/validation code that produced A's GT.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CODE = Path("/mnt/data/warp_rm/gt_validation/code")
MANIFEST_B = Path("/home/shadeform/abc/render_manifest_B.json")
DELIV = Path("/mnt/data/warp_rm/gt_validation/deliv")
BUCKET, SUB = "xdof-bair-abc", "sim_put_the_plastic_bottles_in_the_bin"
FILES = ("scene_assembled.xml", "integration_state.npy")


def download(workers: int = 48) -> None:
    import boto3
    from botocore.config import Config as BotoConfig
    man = json.loads(MANIFEST_B.read_text())
    jobs = [(int(k), v["uuid"], v["delivery"]) for k, v in man.items() if v.get("delivery")]
    s3 = boto3.client("s3", config=BotoConfig(max_pool_connections=workers * 2,
                                              retries={"max_attempts": 8}))

    def fetch(job):
        idx, uuid, deliv = job
        d = DELIV / uuid
        d.mkdir(parents=True, exist_ok=True)
        if all((d / f).exists() and (d / f).stat().st_size > 0 for f in FILES):
            return "cached"
        for fn in FILES:
            tmp = d / (fn + ".part")
            s3.download_file(BUCKET, f"data/deliveries/{deliv}/{SUB}/{uuid}/{fn}", str(tmp))
            os.replace(tmp, d / fn)
        return "ok"

    ok = cached = fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, st in enumerate(ex.map(fetch, jobs)):
            ok += st == "ok"; cached += st == "cached"
            if (i + 1) % 400 == 0:
                print(f"[deliv-B] {i+1}/{len(jobs)} ok={ok} cached={cached}", flush=True)
    print(f"[deliv-B] DONE ok={ok} cached={cached} fail={fail} of {len(jobs)}")


def compute(n_validate: int = 25) -> None:
    """Run gt_compute.main() with B paths patched onto the module."""
    sys.path.insert(0, str(CODE))
    sys.path.insert(0, "/home/shadeform/abc")
    import gt_compute as g
    g.MANIFEST = MANIFEST_B
    g.OBJ_COUNTS = Path("/mnt/data/warp_rm/object_counts_AB.json")
    g.DELIV = DELIV
    g.GT_OUT = Path("/mnt/data/warp_rm/gt_validation/gt")
    g.AUDIT_TSV = Path("/mnt/data/warp_rm/gt_validation/gt_audit_B.tsv")
    g.ASSET_ROOT = "/mnt/data/render_assets"
    sys.argv = ["gt_compute.py", str(n_validate)]
    g.main()


if __name__ == "__main__":
    if sys.argv[1] == "download":
        download()
    elif sys.argv[1] == "compute":
        compute(int(sys.argv[2]) if len(sys.argv) > 2 else 25)
