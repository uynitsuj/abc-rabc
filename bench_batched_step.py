"""Lockstep-batching benchmark for the abc_minimal eval loop.

Measures world-steps/s at nworld=B with the SAME per-step obligations the real
evaluator has: ctrl upload, control_decimation physics substeps, and a full
qpos readback each control step (the evaluator + trace both need it).

No policy, no render: the chunk profile shows those at 2.9% + 1.4%; this bench
targets the 96.7%.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "warprm_eval/abc_rabc"))

import mujoco  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 64, 128])
    ap.add_argument("--steps", type=int, default=300, help="control steps to time")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    # Build the real scene exactly as the evaluator does.
    from abc_minimal.eval_policy import PutBottlesSimConfig, scene_xml
    scene = PutBottlesSimConfig()
    rng = np.random.default_rng(20260511)
    bottle_scales = rng.uniform(*scene.bottle_scale_range, size=scene.bottle_count).astype(np.float32)
    bin_scale = float(rng.uniform(*scene.bin_scale_range))
    xml = scene_xml(scene, bottle_scales, bin_scale)
    model = mujoco.MjModel.from_xml_string(xml)
    model.opt.timestep = scene.timestep
    data = mujoco.MjData(model)
    decim = scene.control_decimation
    print(f"scene: nq={model.nq} nu={model.nu} decimation={decim}", flush=True)

    import mujoco_warp as mjw
    import warp as wp
    wp.set_device(f"cuda:{args.gpu}")

    for B in args.batches:
        try:
            m_warp = mjw.put_model(model)
            d_warp = mjw.put_data(model, data, nworld=B,
                                  nconmax=model.nconmax, njmax=model.njmax)
        except Exception as e:
            print(f"B={B}: put_data failed: {e}", flush=True)
            continue
        ctrl = np.zeros((B, model.nu), dtype=np.float32)
        ctrl_wp = wp.from_numpy(ctrl, dtype=wp.float32)

        def control_step():
            wp.copy(d_warp.ctrl, ctrl_wp)
            for _ in range(decim):
                mjw.step(m_warp, d_warp)
            q = d_warp.qpos.numpy()      # [B, nq] readback + sync, as the evaluator needs
            return q

        for _ in range(args.warmup):
            control_step()
        wp.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.steps):
            control_step()
        wp.synchronize()
        dt = time.perf_counter() - t0
        sps = args.steps / dt
        print(f"B={B:4d}  control-steps/s={sps:8.2f}  world-steps/s={sps*B:10.1f}  "
              f"ms/step={1000*dt/args.steps:7.2f}  "
              f"sim-worlds-realtime={sps*B/30:8.1f}x", flush=True)
        del m_warp, d_warp


if __name__ == "__main__":
    main()
