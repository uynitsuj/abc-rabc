# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "boto3", "pyarrow", "mujoco"]
# ///
"""Per-frame ground-truth oracle rewards for the sim_tasks_20260507 delivery.

Second (and third) exact-reward task families beyond sim_put_bottles_mjwarp,
following the oracle_rewards.py playbook but delegating the success GEOMETRY to
the *canonical* xdof_sim task evaluators (the exact code that defined task
success during data collection) instead of re-implementing it:

    throw   -> xdof_sim.task_eval.bottles.BottlesInBinEvaluator   (bottle_in_bin_mask)
    sweep   -> xdof_sim.task_eval.sweep.SweepAwayEvaluator         (trash_in_bin_mask)
    plates  -> xdof_sim.task_eval.dishrack.DishrackEvaluator       (EXPERIMENTAL/GATED)

For every episode we stream scene_assembled.xml + integration_state.npy from the
delivery, load the MjModel (with a variant-aware asset resolver, since the
delivery scenes embed collection-time absolute mesh paths that don't resolve on
this host), instantiate the canonical evaluator, feed the ENTIRE qpos trajectory
as the evaluator's world batch, and read the per-frame in-target boolean mask
straight from the returned metrics. We then apply the SAME persistence-window +
startup-grace debounce as oracle_rewards.py (PERSIST=15, GRACE=30 at 30 Hz) to
turn the instantaneous mask into a stable per-frame placed_count:

    qualified[t, o] = mask[t, o] AND streak[t, o] >= PERSIST AND (t+1) >= GRACE
    placed_count[t] = sum_o qualified[t, o]

integration_state[:, 0] is sim time; integration_state[:, 1:1+nq] is qpos, EXACTLY
one row per staged render frame (verified 1:1 vs the lerobot mirror episode length,
same as the bottles pipeline). Free-joint qpos addresses are derived per episode by
a pure-XML parse (delivery scenes reference assets by absolute paths that don't
resolve on this host; a pure parse works because free joints are 7-dim in declared
body order) and cross-checked against MjModel.jnt_qposadr at load.

Output (matches icrrt.data.lerobot_v3_oracle.LeRobotV3OracleSource schema exactly,
so the join is ZERO code change):
    <out-dir>/<episode_uuid>.parquet   (frame_index, placed_count, event)
    <out-dir>/summary.json             (episodes[i] = {uuid, n_bottles, final_count, ...})
The denominator field is named "n_bottles" (the source hardcodes that key) but holds
the per-scene object count: bottles for throw, active scraps for sweep.

TASK STATUS
-----------
  throw : SHIPPED  - validated 18/18 final==object_count, position-only geometry.
  sweep : SHIPPED  - validated 15/15 instant_max==object_count + 100%
                     persist_final==instant_final; final<count cases confirmed
                     genuine (scrap swept off the table, not binned).
  plates: GATED    - the canonical reward gates on a rack-CONTACT flag that flickers
                     frame-to-frame (narrow-phase artifact), scoring genuinely-seated
                     plates as failures; a contact-free kinematic rule (R3: spatial-
                     in-rack AND upright, latched) is implemented here but could NOT
                     be cross-checked at the 18/20 bar (plates demos are ~22%
                     genuinely incomplete, so object_count is not a clean success
                     oracle, and there is no logged per-frame reward). See
                     SECOND_ORACLE_TASK_FAMILY.md. NOT run at scale / NOT joined.

Usage
-----
    # acceptance gate (validate 20 episodes vs object_count):
    uv run oracle_rewards_sim_tasks.py --task throw --validate --count 20 --workers 12
    uv run oracle_rewards_sim_tasks.py --task sweep --validate --count 20 --workers 12

    # full extraction:
    uv run oracle_rewards_sim_tasks.py --task throw --out-dir out/throw --workers 16
    uv run oracle_rewards_sim_tasks.py --task sweep --out-dir out/sweep --workers 16
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# ---- persistence rule (mirrors oracle_rewards.py / PutBottlesEvaluator) ------
PERSIST, GRACE = 15, 30  # >=0.5 s continuous; 1 s startup grace (30 Hz)

DELIVERY_BUCKET = "xdof-internal-research"
DELIVERY_PREFIX = "sim/sim_tasks_20260507"

# xdof_sim canonical evaluators + collection-time asset root -> local checkouts.
XDOF_SIM_ROOT = os.environ.get("XDOF_SIM_ROOT", "/home/justinyu/xdof-sim")
COLLECT_ASSET_PREFIX = (
    "/home/xdof/lab42/packages/market42/submodules/xdof-sim/xdof_sim/models/assets/"
)
ASSET_ROOTS = [
    "/home/justinyu/xdof-sim/xdof_sim/models/assets",
    "/home/justinyu/xdof-sim_abc/xdof_sim/models/assets",
]

TASKS = {
    "throw": {
        "subdir": "sim_throw_plastic_bottles_in_bin",
        "spec": "sim_throw_plastic_bottles_in_bin",
        "evaluator": "bottles_in_bin",
        "mask_key": "bottle_in_bin_mask",
        "object_re": re.compile(r"bottle_(\d+)_joint"),
    },
    "sweep": {
        "subdir": "sim_sweep_away_paper_scraps_from_the_table",
        "spec": "sim_sweep_away_paper_scraps_from_the_table",
        "evaluator": "sweep_away",
        "mask_key": "trash_in_bin_mask",
        "object_re": re.compile(r"trash_(\d+)_jnt"),
    },
    "plates": {  # EXPERIMENTAL / GATED - not run at scale, not joined
        "subdir": "sim_load_the_plates_into_the_dish_rack",
        "spec": "sim_load_the_plates_into_the_dish_rack",
        "evaluator": "dishrack_plates_in_rack",
        "mask_key": None,  # custom kinematic rule (R3), see plates_inrack_mask()
        "object_re": re.compile(r"plate_joint(?:_(\d+))?"),
    },
}


# ---- variant-aware asset resolver -------------------------------------------
# Delivery scenes embed collection-time absolute mesh paths whose on-disk layout
# differs from the local checkouts (some plate variants are flat, others have
# collision/ subdirs; dish racks use mesh-name vs variant folders). We resolve
# every `file=` ref by basename, disambiguating multi-candidate basenames by
# shared variant token (plate_N / dish_rack_N) then longest trailing path match.
# Validated: reproduces exact-path loads (identical nq/ngeom/geom_rbound) on
# every scene where the exact path also resolves.
_VARIANT_RE = re.compile(r"/(plate_\d+|dish_rack_\d+)/")
_ASSET_INDEX: dict[str, list[str]] = {}


def _build_asset_index() -> None:
    if _ASSET_INDEX:
        return
    for root in ASSET_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                _ASSET_INDEX.setdefault(fn, []).append(os.path.join(dirpath, fn))


def _resolve_asset(ref_abs: str) -> str | None:
    base = os.path.basename(ref_abs)
    cands = _ASSET_INDEX.get(base)
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    ref_tok = set(_VARIANT_RE.findall(ref_abs))

    def score(cand: str) -> int:
        s = 100 * len(ref_tok & set(_VARIANT_RE.findall(cand)))
        for a, b in zip(reversed(ref_abs.split("/")), reversed(cand.split("/"))):
            if a == b:
                s += 1
            else:
                break
        return s

    return max(cands, key=score)


def resolve_scene_xml(xml_text: str) -> str:
    """Rewrite every mesh/texture file= to a resolvable local path; neutralize
    meshdir/texturedir (paths become absolute)."""
    _build_asset_index()

    def repl(m: re.Match) -> str:
        ref = m.group(1)
        ref_abs = ref if ref.startswith("/") else COLLECT_ASSET_PREFIX + ref
        found = _resolve_asset(ref_abs)
        return f'file="{found}"' if found else m.group(0)

    fixed = re.sub(r'file="([^"]+)"', repl, xml_text)
    return re.sub(r'(meshdir|texturedir)="[^"]*"', r'\1=""', fixed)


# ---- pure-XML free-joint qpos address derivation (from oracle_rewards.py) ----
def _default_joint_types(root: ET.Element) -> dict:
    out: dict = {}

    def walk(el: ET.Element, inherited):
        cls = el.get("class")
        j = el.find("joint")
        typ = j.get("type") if j is not None and j.get("type") else inherited
        out[cls] = typ
        for child in el.findall("default"):
            walk(child, typ)

    for d in root.findall("default"):
        walk(d, None)
    return out


def _joint_qpos_dims(el: ET.Element, cls_types: dict, childclass) -> int:
    if el.tag == "freejoint":
        return 7
    typ = el.get("type")
    if typ is None:
        typ = cls_types.get(el.get("class") or childclass) or "hinge"
    return {"free": 7, "ball": 4}.get(typ, 1)


def derive_freejoint_addrs(xml_text: str) -> tuple[dict[str, int], int]:
    root = ET.fromstring(xml_text)
    cls_types = _default_joint_types(root)
    world = root.find("worldbody")
    if world is None:
        raise ValueError("no <worldbody> in scene XML")
    addrs: dict[str, int] = {}
    nq = 0

    def walk(body: ET.Element, childclass):
        nonlocal nq
        childclass = body.get("childclass", childclass)
        for el in body:
            if el.tag in ("joint", "freejoint"):
                dim = _joint_qpos_dims(el, cls_types, childclass)
                if dim == 7 and el.get("name"):
                    addrs[el.get("name")] = nq
                nq += dim
            elif el.tag == "body":
                walk(el, childclass)

    walk(world, None)
    return addrs, nq


# ---- per-frame persistence scoring (from oracle_rewards.py) -------------------
def score_frames(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(T,O) in-target bools -> (placed_count (T,), event (T,), first_qual (O,))."""
    T, O = mask.shape
    streak = np.zeros(O, np.int32)
    placed = np.zeros(T, np.int16)
    event = np.zeros(T, np.int16)
    first_qual = np.full(O, -1, np.int32)
    for t in range(T):
        streak = np.where(mask[t], streak + 1, 0)
        q = mask[t] & (streak >= PERSIST) & (t + 1 >= GRACE)
        placed[t] = q.sum()
        newly = q & (first_qual < 0)
        first_qual[newly] = t
        event[t] = newly.sum()
    return placed, event, first_qual


def score_frames_latch(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Latching variant (used for plates R3): an object qualifies after PERSIST
    consecutive in-target frames (past GRACE) and STAYS qualified until it leaves
    the target for PERSIST consecutive frames. Robust to a flickering signal."""
    T, O = mask.shape
    on = np.zeros(O, np.int32)
    off = np.zeros(O, np.int32)
    latched = np.zeros(O, bool)
    placed = np.zeros(T, np.int16)
    event = np.zeros(T, np.int16)
    first_qual = np.full(O, -1, np.int32)
    for t in range(T):
        on = np.where(mask[t], on + 1, 0)
        off = np.where(~mask[t], off + 1, 0)
        newly = (on >= PERSIST) & (t + 1 >= GRACE) & ~latched
        latched |= (on >= PERSIST) & (t + 1 >= GRACE)
        latched &= ~(off >= PERSIST)
        placed[t] = latched.sum()
        first_qual[newly & (first_qual < 0)] = t
        event[t] = newly.sum()
    return placed, event, first_qual


# ---- evaluator wiring --------------------------------------------------------
def _import_xdof():
    if XDOF_SIM_ROOT not in sys.path:
        sys.path.insert(0, XDOF_SIM_ROOT)
    import mujoco  # noqa
    from xdof_sim.task_specs import get_task_spec
    from xdof_sim.task_eval.bottles import BottlesInBinEvaluator
    from xdof_sim.task_eval.sweep import SweepAwayEvaluator
    from xdof_sim.task_eval.dishrack import DishrackEvaluator
    return {
        "mujoco": mujoco,
        "get_task_spec": get_task_spec,
        "bottles_in_bin": BottlesInBinEvaluator,
        "sweep_away": SweepAwayEvaluator,
        "dishrack_plates_in_rack": DishrackEvaluator,
    }


def plates_inrack_mask(metrics: dict) -> np.ndarray:
    """Plates R3 kinematic in-target mask: spatially-in-rack AND upright, with NO
    contact flags (contact flags flicker). EXPERIMENTAL - see module docstring."""
    spatial = np.asarray(metrics["plate_spatially_in_rack_mask"])
    upright = np.asarray(metrics["plate_upright_mask"])
    return spatial & upright


def compute_masks(task: str, qpos: np.ndarray, xml_text: str, xd: dict, active_trash=None):
    """Return (in_target_mask (T,O), object_names, n_objects)."""
    mujoco = xd["mujoco"]
    spec = xd["get_task_spec"](TASKS[task]["spec"])
    model = mujoco.MjModel.from_xml_string(resolve_scene_xml(xml_text))
    if qpos.shape[1] != model.nq:
        raise ValueError(f"qpos nq={qpos.shape[1]} != model.nq={model.nq}")
    ev_cls = xd[TASKS[task]["evaluator"]]
    ev = ev_cls(model=model, spec=spec, **spec.evaluator_kwargs())
    ev.reset(nworld=qpos.shape[0])
    if task == "sweep" and active_trash is not None:
        ev.set_active_trash_joints([active_trash] * qpos.shape[0])
    res = ev.evaluate_qpos_batch(qpos.astype(np.float32))
    if task == "plates":
        mask = plates_inrack_mask(res.metrics)
        names = list(res.metrics["plate_names"][-1]) if isinstance(res.metrics["plate_names"], list) else []
        n_obj = int(np.asarray(res.metrics["plate_count"])[-1])
    else:
        mask = np.asarray(res.metrics[TASKS[task]["mask_key"]])
        names = list(res.metrics["bottle_names" if task == "throw" else "trash_names"])
        if task == "sweep":
            # object count = number of ACTIVE scraps (authoritative: metadata list)
            n_obj = len(active_trash) if active_trash else int(np.asarray(res.metrics["num_active_scraps"])[-1])
        else:
            n_obj = len(names)
    return mask, names, n_obj, res


# ---- S3 ----------------------------------------------------------------------
def make_s3(workers: int):
    import boto3
    from botocore.config import Config
    return boto3.client("s3", config=Config(
        connect_timeout=10, read_timeout=120, retries={"max_attempts": 4},
        max_pool_connections=max(2 * workers, 10)))


def s3_get(s3, key: str) -> bytes:
    return s3.get_object(Bucket=DELIVERY_BUCKET, Key=key)["Body"].read()


def list_episode_uuids(s3, subdir: str) -> list[str]:
    prefix = f"{DELIVERY_PREFIX}/{subdir}/"
    uuids, tok = [], None
    while True:
        kw = dict(Bucket=DELIVERY_BUCKET, Prefix=prefix, Delimiter="/")
        if tok:
            kw["ContinuationToken"] = tok
        r = s3.list_objects_v2(**kw)
        for cp in r.get("CommonPrefixes", []):
            name = cp["Prefix"][len(prefix):].rstrip("/")
            if name.startswith("episode_"):
                uuids.append(name)
        if r.get("IsTruncated"):
            tok = r["NextContinuationToken"]
        else:
            break
    return sorted(uuids)


# ---- per-episode pipeline ----------------------------------------------------
def process_episode(s3, xd, task: str, uuid: str, out_dir: Path | None) -> dict:
    t0 = time.monotonic()
    base = f"{DELIVERY_PREFIX}/{TASKS[task]['subdir']}/{uuid}"
    xml_text = s3_get(s3, f"{base}/scene_assembled.xml").decode()
    state = np.load(io.BytesIO(s3_get(s3, f"{base}/integration_state.npy")))
    addrs, nq = derive_freejoint_addrs(xml_text)
    if state.ndim != 2 or state.shape[1] < 1 + nq:
        raise ValueError(f"integration_state {state.shape} too narrow for nq={nq}")
    qpos = state[:, 1:1 + nq]

    # structural check: derived free-joint quaternions must be unit-norm
    for name, a in addrs.items():
        norms = np.linalg.norm(qpos[::max(1, len(qpos) // 8), a + 3:a + 7], axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(f"{name}@{a}: quat norms {norms[:4]} - bad qpos addr")

    active_trash = None
    if task == "sweep":
        # The active-scrap set MUST come from randomization metadata: z-inference
        # is unreliable here because ~5% of episodes (those lacking
        # randomization.json) spawn all 7 trash on the table with the unused ones
        # physics-ejected to z>100, so z>active_z_min flags all 7 as active and
        # corrupts the denominator. Require metadata; skip the episode otherwise.
        try:
            rand = json.loads(s3_get(s3, f"{base}/randomization.json"))
            active_trash = rand.get("metadata", {}).get("trash_joints")
        except Exception as e:
            raise ValueError(f"sweep requires randomization.json (missing): {e}")
        if not active_trash:
            raise ValueError("sweep randomization.json has no metadata.trash_joints")

    # compute_masks asserts qpos.shape[1] == model.nq (address-alignment gate);
    # the unit-quat check above confirms the free-joint addresses are correct.
    mask, names, n_obj, _res = compute_masks(task, qpos, xml_text, xd, active_trash)

    placed, event, first_qual = (score_frames_latch if task == "plates" else score_frames)(mask)
    # instantaneous canonical count (for validation cross-check)
    instant = mask.sum(axis=1).astype(np.int16)

    if out_dir is not None:
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.table({
            "frame_index": pa.array(np.arange(len(placed), dtype=np.int32)),
            "placed_count": pa.array(placed),
            "event": pa.array(event),
        })
        pq.write_table(table, out_dir / f"{uuid}.parquet")

    return {
        "uuid": uuid,
        "n_frames": int(len(placed)),
        "n_bottles": int(n_obj),          # denominator key the source reads
        "n_objects": int(n_obj),          # clear alias
        "object_names": names,
        "final_count": int(placed[-1]),
        "max_count": int(placed.max()),
        "instant_final": int(instant[-1]),
        "instant_max": int(instant.max()),
        "active_trash": active_trash,
        "seconds": round(time.monotonic() - t0, 2),
    }


# ---- main --------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--count", type=int, default=None, help="limit #episodes (default all)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--validate", action="store_true",
                    help="don't write parquets; report match rates vs object_count")
    args = ap.parse_args()

    if args.task == "plates" and not args.validate:
        print("[WARN] plates is GATED (see module docstring / SECOND_ORACLE_TASK_FAMILY.md); "
              "running for inspection only.", flush=True)

    xd = _import_xdof()
    s3 = make_s3(args.workers)
    uuids = list_episode_uuids(s3, TASKS[args.task]["subdir"])
    if args.count:
        uuids = uuids[:args.count]
    if args.out_dir is not None and not args.validate:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    out_dir = None if args.validate else args.out_dir
    print(f"[oracle:{args.task}] {len(uuids)} episodes, workers={args.workers}, "
          f"validate={args.validate}", flush=True)

    results, errors = {}, {}
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_episode, s3, xd, args.task, u, out_dir): u for u in uuids}
        for n, fut in enumerate(as_completed(futs)):
            u = futs[fut]
            try:
                results[u] = fut.result()
            except Exception as e:  # noqa: BLE001
                errors[u] = f"{type(e).__name__}: {e}"
                if len(errors) <= 20:
                    print(f"[oracle][ERR] {u}: {errors[u]}", flush=True)
            if (n + 1) % 100 == 0:
                print(f"[oracle] {n + 1}/{len(uuids)} "
                      f"({(time.monotonic() - t0) / (n + 1):.2f} s/ep)", flush=True)

    ok = list(results.values())
    n_complete = sum(1 for r in ok if r["final_count"] == r["n_objects"])
    n_maxhit = sum(1 for r in ok if r["instant_max"] == r["n_objects"])
    n_faithful = sum(1 for r in ok if r["final_count"] == r["instant_final"])
    if args.validate:
        print(f"\n{'uuid':22} {'T':>5} {'nobj':>4} {'final':>5} {'max':>4} "
              f"{'i_final':>7} {'i_max':>5}")
        for u in sorted(results):
            r = results[u]
            print(f"{u[:22]:22} {r['n_frames']:>5} {r['n_objects']:>4} {r['final_count']:>5} "
                  f"{r['max_count']:>4} {r['instant_final']:>7} {r['instant_max']:>5}")
    N = max(len(ok), 1)
    print(f"\n[oracle:{args.task}] n={len(ok)} errors={len(errors)}")
    print(f"  final_count==object_count      : {n_complete}/{len(ok)} "
          f"({100*n_complete/N:.0f}%)  [complete demos]")
    print(f"  instant_max==object_count      : {n_maxhit}/{len(ok)} "
          f"({100*n_maxhit/N:.0f}%)  [geometry detects all objects at peak]")
    print(f"  persist_final==instant_final   : {n_faithful}/{len(ok)} "
          f"({100*n_faithful/N:.0f}%)  [faithful reproduction of canonical verdict]")

    if out_dir is not None:
        summary = {
            "task": args.task,
            "delivery": f"{DELIVERY_BUCKET}/{DELIVERY_PREFIX}/{TASKS[args.task]['subdir']}",
            "rule": {"persist": PERSIST, "grace": GRACE,
                     "evaluator": TASKS[args.task]["evaluator"],
                     "scorer": "latch" if args.task == "plates" else "persist"},
            "n_episodes": len(ok), "n_errors": len(errors), "errors": errors,
            "n_complete_demos": n_complete,
            "episodes": {str(i): results[u] for i, u in enumerate(sorted(results))},
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        dt = time.monotonic() - t0
        print(f"[oracle:{args.task}] wrote {len(ok)} parquets + summary.json to {out_dir} "
              f"in {dt:.1f}s ({dt/N:.2f}s/ep)")
    return 1 if errors and not args.validate else 0


if __name__ == "__main__":
    sys.exit(main())
