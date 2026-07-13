"""Visualize the in-bin reward primitive on the put-bottles sim scene.

The reward's "in bin" test (PutBottlesEvaluator.evaluate) is a CYLINDER around
the bin's XY, using each bottle's freejoint (body-center) position:

    rel = bottle_pos - bin_pos
    in_bin = (norm(rel[:2]) <= eval_bin_radius)
             and (eval_min_rel_z <= rel[z] <= eval_max_rel_z)

then a bottle must stay in_bin for PERSIST_STEPS (0.5s@30Hz) to count, and
placed-ever latches monotonically. This tool draws that cylinder as a
translucent volume so you can SEE how loose it is vs the physical bin, and
tune (eval_bin_radius, eval_min_rel_z, eval_max_rel_z).

Self-contained: only needs mujoco + abc_minimal.{config,eval_policy.scene_xml}.
No warp/GPU, no policy — works on a laptop clone.

TWO MODES
  GUI (default):    python -m abc_minimal.view_primitive --world-json <summary.json> --trace <npz>
      Opens the MuJoCo interactive viewer (needs a display; over SSH use `ssh -X`).
      Live tuning keys (printed on launch):
        [ ]   radius -/+ 0.005      t/T   top z -/+ 0.01
        b/B   bottom z -/+ 0.01     g/G   compare a tightened preset (green)
        p     print current (radius, min_rel_z, max_rel_z) + live in-bin count
  Snapshot:  add  --snapshot out.jpg  [--camera overhead]
      Offscreen EGL render of one frame with the cylinder overlaid — headless-safe.

Examples
  # GUI on the seed-20260519 WARP world, last trace frame:
  python -m abc_minimal.view_primitive \
    --world-json local_eval_out/hard_warp_sh0/summary.json \
    --trace     local_eval_out/hard_warp_sh0/qpos_trace_world8_seed20260519.npz
  # JPG of a specific frame with a tightened cylinder:
  python -m abc_minimal.view_primitive --world-json ... --trace ... \
    --frame 1799 --radius 0.10 --max-rel-z 0.14 --snapshot out.jpg
"""

import argparse
import json
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer


def _freejoint_adr(model, name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise SystemExit(f"joint not found: {name}")
    return int(model.jnt_qposadr[jid])


def load_record(world_json, trace_path, world_index):
    data = json.loads(Path(world_json).read_text())
    scene_cfg = (data.get("config") or {}).get("scene")
    if "worlds" not in data:
        return data, scene_cfg
    worlds = data["worlds"]
    if world_index is not None:
        m = [w for w in worlds if w["world_index"] == world_index]
        if not m:
            raise SystemExit(f"world_index {world_index} not found")
        return m[0], scene_cfg
    base = Path(trace_path).name
    m = [w for w in worlds if w.get("qpos_trace_path") and Path(w["qpos_trace_path"]).name == base]
    if len(m) != 1:
        raise SystemExit(f"trace {base}: {len(m)} matches; pass --world-index")
    return m[0], scene_cfg


def build_model(record, scene_cfg):
    from abc_minimal.config import PutBottlesSimConfig
    from abc_minimal.eval_policy import scene_xml
    scene = PutBottlesSimConfig(**scene_cfg) if scene_cfg else PutBottlesSimConfig()
    rand = record["randomization"]
    xml = scene_xml(scene,
                    np.asarray(rand["bottle_scales"], dtype=np.float64),
                    float(rand["bin_scale"]))
    return mujoco.MjModel.from_xml_string(xml), scene


def bottle_joint_names(record):
    return list(record["randomization"]["bottle_states"].keys())


def measure_bin(model, data, bin_adr):
    """Empirical bin geometry in rel-z: iterate the bin body's geoms, transform
    their mesh AABB to world, report rim (max z), floor (min z), opening radius."""
    bin_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bin")
    if bin_body < 0:  # bin body name may differ; find body owning bin_joint
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bin_joint")
        bin_body = int(model.jnt_bodyid[jid])
    bin_z = float(data.qpos[bin_adr + 2])
    zs, rs = [], []
    for gi in range(model.ngeom):
        if int(model.geom_bodyid[gi]) != bin_body:
            continue
        pos = data.geom_xpos[gi]
        # geom_rbound is a bounding-sphere radius; use it as a coarse extent
        rb = float(model.geom_rbound[gi])
        zs += [pos[2] + rb, pos[2] - rb]
        rs.append(np.linalg.norm(pos[:2] - data.qpos[bin_adr:bin_adr + 2]) + rb)
    if not zs:
        return None
    return dict(rim_rel_z=max(zs) - bin_z, floor_rel_z=min(zs) - bin_z,
                approx_radius=max(rs) if rs else float("nan"))


def add_cylinder(scn, center, radius, half_h, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_CYLINDER,
        np.array([radius, radius, half_h], dtype=np.float64),
        np.asarray(center, dtype=np.float64),
        np.eye(3).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )
    scn.ngeom += 1


def add_sphere(scn, center, r, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([r, r, r], dtype=np.float64),
        np.asarray(center, dtype=np.float64),
        np.eye(3).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )
    scn.ngeom += 1


def bottle_status(model, data, bin_adr, bnames, radius, min_z, max_z):
    """Per-bottle (center_xyz, in_bin_bool)."""
    binp = data.qpos[bin_adr:bin_adr + 3]
    out = []
    for nm in bnames:
        a = _freejoint_adr(model, nm)
        p = np.asarray(data.qpos[a:a + 3])
        rel = p - binp
        inb = (np.linalg.norm(rel[:2]) <= radius) and (min_z <= rel[2] <= max_z)
        out.append((p, bool(inb)))
    return out


def decorate(scn, model, data, bin_adr, bnames, radius, min_z, max_z,
             tightened=None):
    binp = data.qpos[bin_adr:bin_adr + 3]
    center = [float(binp[0]), float(binp[1]), float(binp[2] + (max_z + min_z) / 2)]
    add_cylinder(scn, center, radius, (max_z - min_z) / 2, (1.0, 0.45, 0.0, 0.22))
    if tightened is not None:
        tr, tmin, tmax = tightened
        tc = [float(binp[0]), float(binp[1]), float(binp[2] + (tmax + tmin) / 2)]
        add_cylinder(scn, tc, tr, (tmax - tmin) / 2, (0.1, 0.9, 0.2, 0.28))
    for p, inb in bottle_status(model, data, bin_adr, bnames, radius, min_z, max_z):
        rgba = (0.1, 0.9, 0.2, 0.9) if inb else (0.95, 0.1, 0.1, 0.9)
        add_sphere(scn, p, 0.02, rgba)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--world-json", required=True)
    p.add_argument("--trace", required=True)
    p.add_argument("--world-index", type=int, default=None)
    p.add_argument("--frame", type=int, default=-1, help="trace frame to show (default last)")
    p.add_argument("--radius", type=float, default=None, help="override eval_bin_radius")
    p.add_argument("--min-rel-z", type=float, default=None, help="override eval_min_rel_z")
    p.add_argument("--max-rel-z", type=float, default=None, help="override eval_max_rel_z")
    p.add_argument("--tightened", default="", help="'r,minz,maxz' green comparison cylinder")
    p.add_argument("--snapshot", default="", help="write a JPG (headless) instead of GUI")
    p.add_argument("--camera", default="overhead")
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--width", type=int, default=960)
    args = p.parse_args()

    record, scene_cfg = load_record(args.world_json, args.trace, args.world_index)
    model, scene = build_model(record, scene_cfg)
    data = mujoco.MjData(model)

    trace = np.load(args.trace)
    trace = trace[trace.files[0]] if hasattr(trace, "files") else trace
    fr = args.frame if args.frame >= 0 else len(trace) - 1
    data.qpos[:] = trace[fr].astype(np.float64)
    mujoco.mj_forward(model, data)

    bin_adr = _freejoint_adr(model, "bin_joint")
    bnames = bottle_joint_names(record)
    radius = args.radius if args.radius is not None else scene.eval_bin_radius
    min_z = args.min_rel_z if args.min_rel_z is not None else scene.eval_min_rel_z
    max_z = args.max_rel_z if args.max_rel_z is not None else scene.eval_max_rel_z
    tightened = None
    if args.tightened:
        tightened = tuple(float(x) for x in args.tightened.split(","))

    meas = measure_bin(model, data, bin_adr)
    print(f"[primitive] eval cylinder: radius={radius:.3f}  rel_z=[{min_z:.3f}, {max_z:.3f}]")
    if meas:
        print(f"[physical]  bin rim_rel_z={meas['rim_rel_z']:.3f}  floor_rel_z={meas['floor_rel_z']:.3f}  "
              f"approx_radius={meas['approx_radius']:.3f}  (rbound-based, coarse)")
    stat = bottle_status(model, data, bin_adr, bnames, radius, min_z, max_z)
    print(f"[frame {fr}] in-bin now: {sum(s for _, s in stat)}/{len(stat)}  "
          f"placed_ever(record)={record['final_task_eval'].get('placed_ever')}")

    if args.snapshot:
        r = mujoco.Renderer(model, height=args.height, width=args.width)
        r.update_scene(data, camera=args.camera)
        decorate(r.scene, model, data, bin_adr, bnames, radius, min_z, max_z, tightened)
        img = r.render()
        try:
            from PIL import Image
            Image.fromarray(img).save(args.snapshot, quality=92)
        except ImportError:
            import imageio.v2 as imageio
            imageio.imwrite(args.snapshot, img)
        print(f"[snapshot] wrote {args.snapshot}")
        return

    # Interactive GUI with live tuning.
    state = {"r": radius, "min": min_z, "max": max_z, "cmp": False}
    print("keys:  [ ] radius -/+   t/T top-z -/+   b/B bottom-z -/+   "
          "g toggle tightened(0.10,-0.06,0.14)   p print")

    def key_cb(key):
        c = chr(key) if 0 <= key < 0x110000 else ""
        if c == "[":
            state["r"] = max(0.02, state["r"] - 0.005)
        elif c == "]":
            state["r"] += 0.005
        elif c == "t":
            state["max"] -= 0.01
        elif c == "T":
            state["max"] += 0.01
        elif c == "b":
            state["min"] -= 0.01
        elif c == "B":
            state["min"] += 0.01
        elif c in ("g", "G"):
            state["cmp"] = not state["cmp"]
        elif c in ("p", "P"):
            n = sum(s for _, s in bottle_status(model, data, bin_adr, bnames,
                                                state["r"], state["min"], state["max"]))
            print(f"radius={state['r']:.3f} rel_z=[{state['min']:.3f},{state['max']:.3f}] in-bin={n}/{len(bnames)}")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as v:
        while v.is_running():
            v.user_scn.ngeom = 0
            tcmp = (0.10, -0.06, 0.14) if state["cmp"] else tightened
            decorate(v.user_scn, model, data, bin_adr, bnames,
                     state["r"], state["min"], state["max"], tcmp)
            v.sync()


if __name__ == "__main__":
    main()
