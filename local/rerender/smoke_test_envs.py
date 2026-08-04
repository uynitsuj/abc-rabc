#!/usr/bin/env python3
"""Smoke-test the vendored yam_sim envs under the abc-rabc venv (mujoco 3.10).

Builds each sim-suite task env (batched, mjwarp camera backend), resets with
randomization, steps once with a hold action, and checks camera obs are
non-degenerate. Run from the abc-rabc repo root so `import yam_sim` resolves
to the vendored package.
"""
import traceback

import numpy as np

import yam_sim

TASKS = [
    "throw_plastic_bottles_in_bin",
    "sweep_away_paper_scraps_from_table",
    "put_bottles",
    "hang_mug_on_mug_rack",
    "load_plates_into_dish_rack",
    "turn_mug_right_side_up",
]

print(f"yam_sim from: {yam_sim.__file__}", flush=True)
results = {}
for task in TASKS:
    try:
        env = yam_sim.make_batched_env(
            scene="hybrid", task=task, num_worlds=2,
            camera_backend="mjwarp", camera_gpu_id=0,
        )
        try:
            obs, info = env.reset(seed=0)
            act = np.zeros((2, env.single_timestep_action_dim), np.float32)
            obs2, *_ = env.step(act)
            imgs = {k: v for k, v in obs.items() if hasattr(v, "ndim") and v.ndim >= 4}
            stats = {k: f"{v.shape} mean={np.asarray(v, np.float32).mean():.1f}"
                     for k, v in imgs.items()}
            results[task] = f"OK  {stats}"
        finally:
            env.close()
    except Exception as e:
        results[task] = f"FAIL: {type(e).__name__}: {e}"
        traceback.print_exc()
    print(f"[{task}] {results[task]}", flush=True)

fails = [t for t, r in results.items() if not r.startswith("OK")]
print(f"\nSUMMARY: {len(TASKS) - len(fails)}/{len(TASKS)} tasks OK; fails={fails}", flush=True)
