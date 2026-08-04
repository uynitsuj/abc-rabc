#!/usr/bin/env python3
"""Render one episode through the EXACT abc-rabc eval renderer (MJWarpSim, mjwarp 3.10).

Replicates abc_minimal/eval_policy.py:MJWarpSim.create_render_context config
(use_textures=True, use_shadows=True; render_skybox/ambient at defaults) and its
BGRA->RGB conversion, driving it from the dataset's supplemental full-state trace.
"""
import argparse, subprocess
import numpy as np
import mujoco
import warp as wp
import mujoco_warp as mjw

CAM_NAMES = ("top", "left", "right")


def resolve_cams(m, mujoco):
    """Map desired camera names to their indices in this scene."""
    cams = {}
    for name in CAM_NAMES:
        idx = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if idx < 0:
            raise ValueError(f"camera {name!r} not found; scene cams: "
                             f"{[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)]}")
        cams[name] = idx
    return cams


def open_writer(path, w, h, fps):
    return subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "14", path],
        stdin=subprocess.PIPE,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--gpu-id", type=int, default=0)
    args = ap.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    m = mujoco.MjModel.from_xml_path(args.scene)
    d = mujoco.MjData(m)
    cams = resolve_cams(m, mujoco)
    states = np.load(args.state)  # (T, mjSTATE_INTEGRATION)
    T = states.shape[0]
    ssz = mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION)
    if states.shape[1] != ssz:
        raise ValueError(f"state width {states.shape[1]} != mjSTATE_INTEGRATION {ssz} for {args.scene}")

    wp.set_device(f"cuda:{args.gpu_id}")
    m_warp = mjw.put_model(m)
    # Raw scene model has nconmax/njmax = -1 (auto); mjwarp put_data needs >= 0.
    # We render via kinematics only (geom/cam/light poses), so collision buffers are unused.
    nconmax = max(int(m.nconmax), 4096)
    njmax = max(int(m.njmax), 4096)
    d_warp = mjw.put_data(m, d, nworld=1, nconmax=nconmax, njmax=njmax)
    rc = mjw.create_render_context(
        mjm=m, nworld=1, cam_res=(args.width, args.height),
        render_rgb=[True] * m.ncam, render_depth=[False] * m.ncam,
        use_textures=True, use_shadows=True,
    )

    writers = {name: open_writer(f"{args.out_dir}/{name}_camera.mp4", args.width, args.height, args.fps)
               for name in cams}
    try:
        for i in range(T):
            mujoco.mj_setState(m, d, states[i], mujoco.mjtState.mjSTATE_INTEGRATION)
            mujoco.mj_forward(m, d)
            wp.copy(d_warp.qpos, wp.from_numpy(np.asarray(d.qpos, np.float32)[None], dtype=wp.float32))
            if m.nv > 0:
                wp.copy(d_warp.qvel, wp.from_numpy(np.asarray(d.qvel, np.float32)[None], dtype=wp.float32))
            if hasattr(d_warp, "time"):
                wp.copy(d_warp.time, wp.from_numpy(np.asarray([d.time], np.float32), dtype=wp.float32))
            # Position stage only: kinematics + com + camera/light poses (no collision solve).
            mjw.kinematics(m_warp, d_warp)
            mjw.com_pos(m_warp, d_warp)
            mjw.camlight(m_warp, d_warp)
            mjw.refit_bvh(m_warp, d_warp, rc)
            mjw.render(m_warp, d_warp, rc)
            rgba = rc.rgb_data.numpy().view(np.uint8).reshape(1, m.ncam, args.height, args.width, 4)
            frame = rgba[0, :, :, :, :3][..., ::-1]  # BGR->RGB, matches MJWarpSim.render
            for name, idx in cams.items():
                writers[name].stdin.write(np.ascontiguousarray(frame[idx]).tobytes())
            if (i + 1) % 100 == 0:
                print(f"  rendered {i+1}/{T}", flush=True)
    finally:
        for wtr in writers.values():
            wtr.stdin.close()
            wtr.wait()
    print(f"Done: {T} frames -> {args.out_dir}")


if __name__ == "__main__":
    main()
