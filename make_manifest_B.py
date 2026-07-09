# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "boto3", "tyro"]
# ///
"""Build render_manifest_B.json for the B-only put-bottles episodes.

B = all uuids under the sim_tasks_20260522 delivery for the put-bottles subtask.
A = the 2438 episodes already rendered (keys in render_manifest.json). B-only = B - A.

Each B-only uuid is assigned a NEW global episode_index continuing A's numbering
(A occupies 0..2437, so B-only occupies 2438..2438+N-1), sorted deterministically by
uuid. The manifest schema mirrors A's render_manifest.json:

  { "<global_index>": {"uuid": "episode_<uuid>", "split": "train_sim",
                        "length": <int frames>, "delivery": "sim_tasks_20260522"}, ... }

`length` = integration_state row count (== render frame count == states_actions rows),
read cheaply from each npy's header (ranged GET, no full download). `split` is
train_sim for every B-only ep (they are new training data; A held out val=0..99).

Run on a host with delivery-bucket read creds (the node):
    .venv/bin/python make_manifest_B.py --out render_manifest_B.json
"""
from __future__ import annotations

import ast
import json
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import boto3
import tyro
from botocore.config import Config as BotoConfig

DELIVERY_BUCKET = "xdof-bair-abc"
DELIVERY = "sim_tasks_20260522"
SUBTASK = "sim_put_the_plastic_bottles_in_the_bin"
DELIV_PREFIX = f"data/deliveries/{DELIVERY}/{SUBTASK}/"
A_MANIFEST_S3 = "s3://xdof-internal-research/abc/render_scratch/put_bottles_mjwarp/render_manifest.json"
A_START = 0
A_COUNT = 2438  # A occupies episode_index 0..2437; B-only continues at 2438


@dataclass
class Cfg:
    out: str = "render_manifest_B.json"
    a_manifest_s3: str = A_MANIFEST_S3
    b_start_index: int = A_START + A_COUNT  # 2438
    workers: int = 48


def _split_s3(uri: str) -> tuple[str, str]:
    b, _, k = uri[5:].partition("/")
    return b, k


def _npy_rows(s3, bucket: str, key: str) -> int:
    """Read integration_state.npy row count from just its header via a ranged GET.

    .npy format: 6-byte magic \\x93NUMPY, 2 version bytes, then header length
    (uint16 for v1.0 / uint32 for v2.0) + an ASCII dict literal with 'shape'.
    """
    head = s3.get_object(Bucket=bucket, Key=key, Range="bytes=0-255")["Body"].read()
    assert head[:6] == b"\x93NUMPY", f"not a npy: {key}"
    major = head[6]
    if major == 1:
        hlen = struct.unpack("<H", head[8:10])[0]
        off = 10
    else:
        hlen = struct.unpack("<I", head[8:12])[0]
        off = 12
    hdr = head[off:off + hlen]
    if len(hdr) < hlen:  # header spilled past 256B (rare); refetch enough
        head = s3.get_object(Bucket=bucket, Key=key,
                             Range=f"bytes=0-{off + hlen + 8}")["Body"].read()
        hdr = head[off:off + hlen]
    meta = ast.literal_eval(hdr.decode("latin1").strip())
    return int(meta["shape"][0])


def list_delivery_uuids(s3) -> set[str]:
    """All episode_<uuid> names under the delivery prefix (paginated)."""
    uuids = set()
    tok = None
    while True:
        kw = dict(Bucket=DELIVERY_BUCKET, Prefix=DELIV_PREFIX, Delimiter="/")
        if tok:
            kw["ContinuationToken"] = tok
        r = s3.list_objects_v2(**kw)
        for cp in r.get("CommonPrefixes", []):
            name = cp["Prefix"][len(DELIV_PREFIX):].rstrip("/")  # episode_<uuid>
            if name.startswith("episode_"):
                uuids.add(name)
        if r.get("IsTruncated"):
            tok = r["NextContinuationToken"]
        else:
            break
    return uuids


def main(cfg: Cfg) -> None:
    s3 = boto3.client("s3", config=BotoConfig(max_pool_connections=64,
                                              retries={"max_attempts": 8}))
    ab, ak = _split_s3(cfg.a_manifest_s3)
    a_man = json.loads(s3.get_object(Bucket=ab, Key=ak)["Body"].read())
    a_uuids = {v["uuid"] for v in a_man.values()}  # episode_<uuid> form
    print(f"[B] A manifest: {len(a_man)} entries, {len(a_uuids)} unique uuids")

    b_all = list_delivery_uuids(s3)
    print(f"[B] delivery {DELIVERY}: {len(b_all)} uuids")
    a_in_deliv = a_uuids & b_all
    print(f"[B] of A's uuids, {len(a_in_deliv)} are in this delivery "
          f"({len(a_uuids) - len(a_in_deliv)} A uuids come from other deliveries)")

    b_only = sorted(b_all - a_uuids)  # deterministic by uuid string
    print(f"[B] B-only = delivery - A = {len(b_only)} uuids")

    # Read integration_state row counts (=lengths) in parallel from npy headers.
    # An episode missing integration_state.npy is UNRENDERABLE (like A's 2 delivery-absent
    # eps): record length=None + delivery=None so the render worker skips it, mirroring A's
    # convention (worker filters on `delivery` being truthy).
    def rows(name: str):
        key = f"{DELIV_PREFIX}{name}/integration_state.npy"
        try:
            return name, _npy_rows(s3, DELIVERY_BUCKET, key), None
        except s3.exceptions.NoSuchKey:
            return name, None, "no_integration_state"
        except Exception as e:  # noqa: BLE001 - surface any other read error
            return name, None, f"{type(e).__name__}: {e}"

    lengths: dict[str, int | None] = {}
    missing: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        for i, (name, n, err) in enumerate(ex.map(rows, b_only)):
            lengths[name] = n
            if err:
                missing[name] = err
            if (i + 1) % 500 == 0:
                print(f"[B] read {i + 1}/{len(b_only)} lengths ...", flush=True)
    if missing:
        print(f"[B][WARN] {len(missing)} B-only eps lack integration_state.npy -> UNRENDERABLE "
              f"(no qpos source, no mjgl fallback); EXCLUDED from the numbered manifest: "
              f"{list(missing)[:10]}")

    # Number ONLY renderable eps, contiguously from b_start_index. This keeps a SINGLE
    # index space across: render-scratch key == B-standalone-local + b_start_index ==
    # object_counts_AB key == eventual merged episode_index. Excluded (unrenderable) eps
    # are recorded in a companion file, never assigned an index (a gap here would silently
    # misalign the per-object stratification against the merged dataset).
    renderable = [n for n in b_only if n not in missing]
    manifest = {}
    for off, name in enumerate(renderable):
        idx = cfg.b_start_index + off
        manifest[str(idx)] = {
            "uuid": name,                 # episode_<uuid> (A's exact form)
            "split": "train_sim",
            "length": lengths[name],
            "delivery": DELIVERY,
        }

    # Sanity gates.
    assert len(manifest) == len(renderable)
    man_uuids = {v["uuid"] for v in manifest.values()}
    assert man_uuids.isdisjoint(a_uuids), "B-only overlaps A!"
    idxs = sorted(int(k) for k in manifest)
    assert idxs[0] == cfg.b_start_index and idxs == list(range(idxs[0], idxs[-1] + 1)), "non-contiguous"
    print(f"[B] indices {idxs[0]}..{idxs[-1]} ({len(idxs)}), zero overlap with A: OK; "
          f"renderable={len(renderable)}, excluded_unrenderable={len(missing)}")

    Path(cfg.out).write_text(json.dumps(manifest, indent=0))
    excluded_path = Path(cfg.out).with_name(Path(cfg.out).stem + "_excluded.json")
    excluded_path.write_text(json.dumps(
        {"reason": "missing integration_state.npy (unrenderable)",
         "count": len(missing), "uuids": sorted(missing)}, indent=2))
    total_frames = sum(lengths[n] for n in renderable)
    print(f"[B] wrote {cfg.out}: {len(manifest)} eps, total_frames={total_frames}, "
          f"mean_len={total_frames/max(len(renderable),1):.0f}")
    print(f"[B] wrote {excluded_path}: {len(missing)} excluded uuid(s)")


if __name__ == "__main__":
    main(tyro.cli(Cfg))
