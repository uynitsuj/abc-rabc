"""Launch the MuJoCo interactive viewer for the YAM bimanual scene.

Usage:
    python -m yam_sim.examples.view
    python -m yam_sim.examples.view --scene eval
    python -m yam_sim.examples.view --animate
"""

from __future__ import annotations

import argparse

import mujoco
import mujoco.viewer
import numpy as np

import yam_sim


def main():
    parser = argparse.ArgumentParser(description="View YAM scene in MuJoCo viewer")
    parser.add_argument(
        "--scene",
        default="hybrid",
        choices=["eval", "training", "hybrid"],
        help="Scene variant (default: hybrid)",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Move the arms with sinusoidal joint motions",
    )
    args = parser.parse_args()

    env = yam_sim.make_env(scene=args.scene, render_cameras=False)
    env.reset()

    if not args.animate:
        mujoco.viewer.launch(env.model, env.data)
        return

    init_q = env.get_init_q()

    # Per-joint amplitude and frequency — small enough to stay in range
    rng = np.random.default_rng(0)
    amplitude = rng.uniform(0.15, 0.35, size=14)
    frequency = rng.uniform(0.3, 0.8, size=14)
    phase = rng.uniform(0, 2 * np.pi, size=14)
    # Keep grippers still
    for gi in env._gripper_indices:
        amplitude[gi] = 0.0

    gripper_set = env._gripper_set
    ctrl_indices = env._ctrl_indices
    from yam_sim.env import _GRIPPER_CTRL_MAX

    def controller(model, data):
        t = data.time
        target = init_q + amplitude * np.sin(2 * np.pi * frequency * t + phase)
        for i in range(14):
            val = target[i]
            if i in gripper_set:
                val = val * _GRIPPER_CTRL_MAX
            data.ctrl[ctrl_indices[i]] = val

    mujoco.set_mjcb_control(controller)
    mujoco.viewer.launch(env.model, env.data)


if __name__ == "__main__":
    main()
