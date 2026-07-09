# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mcap", "mcap-protobuf-support", "boto3", "tyro"]
# ///
"""Piece 2: generate staged states_actions.bin + episode_metadata.json for B-only eps.

The mjwarp render (render_worker.py) produces the per-episode video only; the staged
format also needs states_actions.bin. A gets these by copying from the local staged_put/
(built from the mjgl LeRobot dataset). B-only eps have NO LeRobot index, so we
reconstruct states_actions directly from each delivery episode's output.mcap using the
method validated against A (states_actions_from_mcap.reconstruct_states_actions).

Output mirrors A's local staged_put/ layout so the generalized packager's staged phase
can consume it identically:
    <out_root>/<split>/episode_<uuid>/states_actions.bin      (n_frames x 28 float64)
    <out_root>/<split>/episode_<uuid>/episode_metadata.json

velocity_*.bin are NOT produced here — those are RM-score sidecars written later by the
scoring step (score_to_sidecar.py), exactly as for A.

Run on a host with delivery-bucket read creds (the node), CPU-only:
    .venv/bin/python build_states_actions_B.py --manifest render_manifest_B.json \\
        --out-root staged_put_B --limit 3      # validation slice
"""
from __future__ import annotations

import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import boto3
import numpy as np
import tyro
from botocore.config import Config as BotoConfig

from states_actions_from_mcap import reconstruct_states_actions

DELIVERY_BUCKET = "xdof-bair-abc"
SUBTASK = "sim_put_the_plastic_bottles_in_the_bin"
TASK_NAME = SUBTASK


@dataclass
class Cfg:
    manifest: str = "render_manifest_B.json"
    out_root: str = "staged_put_B"
    limit: int | None = None      # cap eps (validation)
    start: int = 0                # slice manifest-sorted eps [start, start+limit) for sharding
    workers: int = 16
    overwrite: bool = False


def _fetch(s3, uuid: str, delivery: str, dst: Path) -> None:
    prefix = f"data/deliveries/{delivery}/{SUBTASK}/{uuid}"
    for fn in ("output.mcap", "integration_state.npy"):
        p = dst / fn
        if not p.exists() or p.stat().st_size == 0:
            s3.download_file(DELIVERY_BUCKET, f"{prefix}/{fn}", str(p))


def build_one(s3, idx: int, info: dict, out_root: Path, tmp_root: Path,
              overwrite: bool) -> tuple[int, str]:
    uuid, split = info["uuid"], info["split"]
    out_ep = out_root / split / uuid
    sa_path = out_ep / "states_actions.bin"
    if sa_path.exists() and not overwrite:
        return idx, "cached"
    tmp = tmp_root / uuid
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        _fetch(s3, uuid, info["delivery"], tmp)
        integ = np.load(tmp / "integration_state.npy")
        n_rows = int(integ.shape[0])
        # qpos-verified alignment (k = n_rows - n_msgs in {0,1}); raises on mismatch.
        sa = reconstruct_states_actions(tmp / "output.mcap", n_rows,
                                        qpos=integ[:, 1:])  # (n,28) f64
        out_ep.mkdir(parents=True, exist_ok=True)
        sa.astype(np.float64).tofile(sa_path)
        meta = {
            "task_name": TASK_NAME,
            "cameras": ["top", "left", "right"],
            "camera_resolutions": {c: [224, 168] for c in ("top", "left", "right")},
            "alignment": "mcap_proprio_command_align_tail_prepend_first_k",
            "num_steps": int(n_rows),
            "source_episode_id": uuid.replace("episode_", ""),
            "global_index": idx,
            "renderer": "mjwarp",
        }
        (out_ep / "episode_metadata.json").write_text(json.dumps(meta, indent=2))
        return idx, "ok"
    except Exception as e:  # collected by the caller; do not kill sibling futures
        return idx, f"fail: {type(e).__name__}: {e}"
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def main(cfg: Cfg) -> None:
    man = json.loads(Path(cfg.manifest).read_text())
    items = sorted(((int(k), v) for k, v in man.items() if v.get("delivery")),
                   key=lambda x: x[0])
    if cfg.limit is not None:
        items = items[cfg.start:cfg.start + cfg.limit]
    print(f"[sa-B] {len(items)} renderable B-only eps -> {cfg.out_root}")
    s3 = boto3.client("s3", config=BotoConfig(max_pool_connections=cfg.workers * 2,
                                              retries={"max_attempts": 8}))
    out_root = Path(cfg.out_root)
    ok, cached, fail = 0, 0, []
    with tempfile.TemporaryDirectory() as td:
        tmp_root = Path(td)

        def work(item):
            idx, info = item
            return build_one(s3, idx, info, out_root, tmp_root, cfg.overwrite)

        with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
            for i, (idx, status) in enumerate(ex.map(work, items)):
                if status == "ok":
                    ok += 1
                elif status == "cached":
                    cached += 1
                elif status.startswith("fail"):
                    fail.append((idx, status))
                    print(f"[sa-B] FAIL idx={idx}: {status}", flush=True)
                if (i + 1) % 200 == 0:
                    print(f"[sa-B] {i + 1}/{len(items)} (ok={ok} cached={cached} "
                          f"fail={len(fail)})", flush=True)
    print(f"[sa-B] DONE: ok={ok} cached={cached} fail={len(fail)} -> {cfg.out_root}")
    for idx, status in fail:
        print(f"[sa-B] FAILED idx={idx}: {status}")


if __name__ == "__main__":
    main(tyro.cli(Cfg))
