"""Render a side-by-side rollout comparison from two published qpos traces.

Reconstructs the scene from the trace's own `randomization` record (bottle
scales + bin scale, which change the model geometry), replays the recorded qpos
frame by frame, and composes the two arms horizontally into one mp4.

This replaces a renderer that was lost with a decommissioned node. It depends
only on published artifacts: the trace npz files and their summary.json.

Usage
-----
    python render_rollout_pair.py \
        --left-dir  <traces>/fullhz_vanilla_sh07     --left-label  "Vanilla BC" \
        --right-dir <traces>/fullhz_paperwarp512_sh07 --right-label "WARP-BC" \
        --seed 20260906 --out rollout_s20260906.mp4

Labels are drawn in a single neutral colour on purpose: the bottle counts
should carry the comparison, not the styling.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


LABEL_RGB = (235, 235, 235)      # same colour for both arms, deliberately
PANEL_RGB = (28, 28, 28)


def find_world(trace_dir: str, seed: int) -> tuple[str, dict]:
    """Locate the trace file and its world record for one seed."""
    summary = json.load(open(os.path.join(trace_dir, "summary.json")))
    for w in summary.get("worlds", []):
        path = w.get("qpos_trace_path", "")
        stem = path.rsplit("/", 1)[-1]
        if stem.endswith("_s%d.npz" % seed):
            local = os.path.join(trace_dir, stem)
            if not os.path.exists(local):
                raise FileNotFoundError(local)
            return local, w
    raise KeyError("seed %d not in %s" % (seed, trace_dir))


# Scene construction is inlined rather than imported from
# abc_minimal.eval_policy, which pulls in torch at module import. This keeps the
# renderer runnable from a small venv (mujoco + opencv + imageio).
ROOT = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.path.join(ROOT, "assets", "put_bottles", "put_bottle.xml")
BOTTLE_COUNT = 6
TIMESTEP = 0.002


def _fmt(values) -> str:
    return " ".join("%.8g" % float(v) for v in values)


def scene_xml(bottle_scales: np.ndarray, bin_scale: float) -> str:
    """Apply this world's per-bottle and bin mesh scales to the base scene.

    Mirrors abc_minimal.eval_policy.scene_xml. The scales change model geometry,
    so a trace only replays correctly against its own randomization.
    """
    import xml.etree.ElementTree as ET

    assets = os.path.join(ROOT, "assets", "put_bottles", "assets")
    root = ET.parse(SCENE_XML).getroot()
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", assets)
        compiler.set("texturedir", assets)
    for mesh in root.findall("./asset/mesh"):
        name = mesh.get("name", "")
        scale = np.asarray([float(v) for v in mesh.get("scale", "1 1 1").split()],
                           dtype=np.float64)
        for idx in range(BOTTLE_COUNT):
            if name.startswith("bottle_%d_" % idx):
                mesh.set("scale", _fmt(scale * float(bottle_scales[idx])))
                break
        if name.startswith("water_bottle_"):
            mesh.set("scale", _fmt(scale * float(bin_scale)))
    return ET.tostring(root, encoding="unicode")


def build_env(world: dict):
    """Rebuild the MuJoCo model for this world's randomized geometry."""
    import mujoco

    rnd = world["randomization"]
    xml = scene_xml(np.asarray(rnd["bottle_scales"], dtype=np.float64),
                    float(rnd["bin_scale"]))
    model = mujoco.MjModel.from_xml_string(xml)
    model.opt.timestep = TIMESTEP
    return model, mujoco.MjData(model), None


def render_arm(trace_path: str, world: dict, width: int, height: int,
               camera: str, stride: int) -> list[np.ndarray]:
    import mujoco

    model, data, _ = build_env(world)
    qpos = np.load(trace_path)["qpos"].astype(np.float64)
    if qpos.shape[1] != model.nq:
        raise ValueError("trace nq=%d but model nq=%d — scene mismatch"
                         % (qpos.shape[1], model.nq))

    renderer = mujoco.Renderer(model, height=height, width=width)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cam_id < 0:
        raise KeyError("camera %r not in model" % camera)

    frames = []
    for i in range(0, qpos.shape[0], stride):
        data.qpos[:] = qpos[i]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam_id)
        frames.append(renderer.render().copy())
    renderer.close()
    return frames


def label(frame: np.ndarray, text: str, sub: str) -> np.ndarray:
    """Draw a neutral caption bar. Same colour for every arm, by design."""
    import cv2

    h, w = frame.shape[:2]
    bar = np.full((46, w, 3), PANEL_RGB, np.uint8)
    out = np.vstack([bar, frame])
    cv2.putText(out, text, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                LABEL_RGB, 1, cv2.LINE_AA)
    cv2.putText(out, sub, (12, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                (170, 170, 170), 1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left-dir", required=True)
    ap.add_argument("--right-dir", required=True)
    ap.add_argument("--left-label", default="Vanilla BC")
    ap.add_argument("--right-label", default="WARP-BC")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--height", type=int, default=336)
    ap.add_argument("--camera", default="top")
    ap.add_argument("--stride", type=int, default=2,
                    help="sample every Nth control step (29.4 Hz native)")
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()

    import cv2
    import imageio.v2 as imageio

    lp, lw = find_world(a.left_dir, a.seed)
    rp, rw = find_world(a.right_dir, a.seed)

    def counts(w):
        t = w.get("placement_times_s") or []
        n = len(t)
        thr = (n / (max(t) / 3600.0)) if t and max(t) > 0 else 0.0
        return n, thr

    ln, lthr = counts(lw)
    rn, rthr = counts(rw)
    print("seed %d | %s %d bottles %.0f/hr | %s %d bottles %.0f/hr"
          % (a.seed, a.left_label, ln, lthr, a.right_label, rn, rthr))

    lf = render_arm(lp, lw, a.width, a.height, a.camera, a.stride)
    rf = render_arm(rp, rw, a.width, a.height, a.camera, a.stride)
    n = min(len(lf), len(rf))
    print("rendered %d frames per arm" % n)

    lsub = "%d bottles  %.0f/hr" % (ln, lthr)
    rsub = "%d bottles  %.0f/hr" % (rn, rthr)
    writer = imageio.get_writer(a.out, fps=a.fps, macro_block_size=1)
    for i in range(n):
        left = label(lf[i], a.left_label, lsub)
        right = label(rf[i], a.right_label, rsub)
        gap = np.full((left.shape[0], 6, 3), PANEL_RGB, np.uint8)
        writer.append_data(np.hstack([left, gap, right]))
    writer.close()
    print("wrote %s (%.1f MB)" % (a.out, os.path.getsize(a.out) / 1e6))


if __name__ == "__main__":
    main()
