"""Quick render test for the MuJoCo YAM environment.

Usage:
    MUJOCO_GL=egl python -m yam_sim.examples.test_env
"""

from __future__ import annotations

import os

import numpy as np


def main():
    import yam_sim
    from yam_sim.viewer import save_camera_images, save_camera_grid

    output_dir = "/tmp/yam_sim_test"
    os.makedirs(output_dir, exist_ok=True)

    print("Creating environment...")
    env = yam_sim.make_env(scene="hybrid")

    # 1. Reset and inspect observation
    print("\n--- Reset ---")
    obs, info = env.reset()
    print(f"Obs keys: {list(obs.keys())}")
    print(f"State shape: {obs['state'].shape}, dtype: {obs['state'].dtype}")
    print(f"State values: {obs['state']}")
    for cam_name, img in obs["images"].items():
        print(f"  Camera '{cam_name}': shape={img.shape}, dtype={img.dtype}")
    print(f"Prompt: {obs['prompt']}")

    # Save initial camera views
    save_camera_images(env, output_dir, step=0)
    save_camera_grid(env, os.path.join(output_dir, "grid_initial.png"))
    print(f"\nSaved initial camera images to {output_dir}/")

    # 2. Step with init_q (robot should stay still)
    print("\n--- Step with init_q ---")
    init_q = env.get_init_q()
    print(f"init_q: {init_q}")
    action_chunk = np.tile(init_q, (env.chunk_dim, 1))
    final_obs, chunk_history, reward, terminated, truncated, step_info = env.step(
        action_chunk
    )
    print(f"Final state: {final_obs['state']}")
    print(f"State diff from init: {np.abs(final_obs['state'] - init_q).max():.6f}")

    # 3. Step with small random perturbations
    print("\n--- Step with random perturbations ---")
    env.reset()
    perturbation = np.random.uniform(-0.05, 0.05, size=(env.chunk_dim, 14))
    perturbed_action = np.tile(init_q, (env.chunk_dim, 1)) + perturbation
    final_obs2, chunk_history2, _, _, _, _ = env.step(perturbed_action)
    print(f"State diff from init: {np.abs(final_obs2['state'] - init_q).max():.6f}")
    save_camera_grid(env, os.path.join(output_dir, "grid_perturbed.png"))

    # 4. Summary
    print("\n--- Summary ---")
    print(f"Action dim: {env.action_dim} (single step: {env.single_timestep_action_dim})")
    print(f"Chunk dim: {env.chunk_dim}")
    print(f"Camera names: {env.camera_names}")
    print(f"Robot names: {env.robot_names}")
    print(
        f"Effective control rate: "
        f"{1.0 / (env.model.opt.timestep * env._control_decimation):.1f} Hz"
    )
    print(f"\nAll images saved to {output_dir}/")
    print("Test complete!")

    env.close()


if __name__ == "__main__":
    main()
