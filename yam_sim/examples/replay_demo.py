"""Replay a recorded action trajectory in the MuJoCo sim.

Loads a pre-recorded action trajectory and replays it step-by-step,
rendering and saving video. Demonstrates deterministic replay without
needing the policy checkpoint.

The included episode data was captured with the Ki14 LBM checkpoint
(seed 56) using: training scene, green bottles, alpha=0.7, mass=0.015,
ttRTC (prefix_length=3), diffusion_steps=5, jpeg_quality=75,
execute_dim=30. Truncated at 74 chunks (2220 steps) showing 2 bottles
deposited in bin. Full 135-chunk episode (3-IB) available in actions_full.npy.

Usage:
    # Replay with video output
    MUJOCO_GL=egl python -m yam_sim.examples.replay_demo \
        --output /tmp/replay.mp4

    # Replay full episode (3-IB, all 135 chunks)
    MUJOCO_GL=egl python -m yam_sim.examples.replay_demo --full \
        --output /tmp/replay_full.mp4

    # Replay with custom episode data
    MUJOCO_GL=egl python -m yam_sim.examples.replay_demo \
        --episode-dir /path/to/episode_data \
        --output /tmp/custom_replay.mp4

    # Just verify (no video, print final metrics)
    MUJOCO_GL=egl python -m yam_sim.examples.replay_demo --verify-only
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


# Path to bundled episode data
_EXAMPLES_DIR = Path(__file__).parent
_DEFAULT_EPISODE_DIR = _EXAMPLES_DIR / "episode_seed56"


# ── Scene setup helpers ──────────────────────────────────────────────


def apply_bottle_mass(model: mujoco.MjModel, mass: float) -> None:
    """Override bottle body mass (scales inertia proportionally)."""
    for i in range(1, 7):
        body_name = f"bottle_{i}"
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id >= 0:
            old_mass = model.body_mass[body_id]
            if old_mass > 0:
                model.body_inertia[body_id] *= mass / old_mass
            model.body_mass[body_id] = mass


def apply_bottle_alpha(model: mujoco.MjModel, alpha: float) -> None:
    """Override bottle body geom alpha/transparency."""
    for i in range(1, 7):
        gname = f"b{i}_body"
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        if gid >= 0:
            model.geom_rgba[gid][3] = alpha


def apply_green_bottles(model: mujoco.MjModel) -> None:
    """Recolor all bottles to green (matches training distribution)."""
    green_body = np.array([0.30, 0.70, 0.20], dtype=np.float32)
    green_neck = np.array([0.35, 0.75, 0.25], dtype=np.float32)
    green_cap = np.array([0.20, 0.60, 0.15], dtype=np.float32)
    for i in range(1, 7):
        for suffix, color in [
            ("body", green_body),
            ("neck", green_neck),
            ("cap", green_cap),
        ]:
            gname = f"b{i}_{suffix}"
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
            if gid >= 0:
                model.geom_rgba[gid][:3] = color


# ── Bin detection ────────────────────────────────────────────────────

BIN_APOTHEM = 0.120
BIN_WALL_CENTER_Z = 0.063
BIN_WALL_HALF_HEIGHT = 0.06
BIN_TOP_Z_REL = BIN_WALL_CENTER_Z + BIN_WALL_HALF_HEIGHT  # 0.123


def is_bottle_in_bin(
    bottle_xy: np.ndarray, bottle_z: float, bin_pos: np.ndarray
) -> bool:
    """Check if a bottle is inside the bin volume."""
    d_xy = np.linalg.norm(bottle_xy - bin_pos[:2])
    if d_xy > BIN_APOTHEM:
        return False
    z_bottom = bin_pos[2]
    z_top = bin_pos[2] + BIN_TOP_Z_REL
    return z_bottom <= bottle_z <= z_top


# ── Main ─────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay a recorded 2-bottles-in-bin episode"
    )
    parser.add_argument(
        "--episode-dir",
        type=str,
        default=None,
        help="Directory with actions.npy, config.json (default: bundled episode)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/yam_replay_2ib.mp4",
        help="Output video path",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Video frames per second"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip video, just verify bottles end up in bin",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Record every N-th frame (1=all, 5=every 5th). "
        "Reduces memory for long episodes.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full episode (actions_full.npy) instead of truncated version",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Replay in the interactive MuJoCo viewer (macOS/Linux GUI)",
    )
    parser.add_argument(
        "--trim",
        type=float,
        nargs=2,
        metavar=("START_S", "END_S"),
        default=None,
        help="Trim video to a time window in seconds, e.g. --trim 54 70",
    )
    parser.add_argument(
        "--highlight",
        action="store_true",
        help="Auto-trim to show 3s before first bottle-in-bin through 3s after last",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load episode data
    episode_dir = Path(args.episode_dir) if args.episode_dir else _DEFAULT_EPISODE_DIR
    if not episode_dir.exists():
        print(f"ERROR: Episode directory not found: {episode_dir}")
        print("Run with --episode-dir /path/to/your/episode_data")
        return

    actions_file = "actions_full.npy" if args.full else "actions.npy"
    actions_path = episode_dir / actions_file
    if not actions_path.exists():
        actions_path = episode_dir / "actions.npy"
    actions = np.load(str(actions_path))
    with open(episode_dir / "config.json") as f:
        config = json.load(f)

    print(f"Episode: {episode_dir.name}")
    print(f"  Seed: {config['seed']}")
    print(f"  Scene: {config['scene']}")
    print(f"  Actions: {actions.shape} ({actions.shape[0]} timesteps x {actions.shape[1]}D)")
    print(f"  Config: execute_dim={config['execute_dim']}, "
          f"num_chunks={config['num_chunks']}, "
          f"bottle_mass={config['bottle_mass']}, "
          f"bottle_alpha={config['bottle_alpha']}")

    # Create environment with matching config
    import yam_sim
    from yam_sim.scene_variants import apply_scene_variant

    env_config = yam_sim.get_i2rt_sim_config()
    env = yam_sim.MuJoCoYAMEnv(
        config=env_config,
        chunk_dim=config["execute_dim"],
        prompt=config["prompt"],
        render_cameras=not args.verify_only and not args.highlight,
    )

    # Apply scene variant
    apply_scene_variant(env.model, config["scene"])

    # Apply bottle overrides
    apply_bottle_mass(env.model, config["bottle_mass"])
    apply_bottle_alpha(env.model, config["bottle_alpha"])
    if config.get("all_green_bottles", False):
        apply_green_bottles(env.model)

    # Reset environment
    env.reset()

    # Set initial state from config
    init_q = np.array(config["init_q"], dtype=np.float32)
    env._set_qpos_from_state(init_q)
    mujoco.mj_forward(env.model, env.data)

    # Find bottles and bin for metrics
    bottle_names = [f"bottle_{i}" for i in range(1, 7)]
    bottle_info = {}
    for name in bottle_names:
        jnt_name = f"{name}_joint"
        jnt_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
        if jnt_id >= 0:
            bottle_info[name] = env.model.jnt_qposadr[jnt_id]

    bin_body_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_BODY, "bin_container"
    )
    bin_pos = env.data.xpos[bin_body_id].copy() if bin_body_id >= 0 else None

    print(f"\nBin position: [{bin_pos[0]:.3f}, {bin_pos[1]:.3f}, {bin_pos[2]:.3f}]")
    print("Initial bottle positions:")
    for name, addr in bottle_info.items():
        pos = env.data.qpos[addr : addr + 3]
        print(f"  {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

    # Auto-detect highlight window
    if args.highlight and not args.trim:
        highlight_cfg = config.get("highlight")
        if highlight_cfg:
            # Use pre-computed window from config
            args.trim = [highlight_cfg["trim_start_s"], highlight_cfg["trim_end_s"]]
            print(f"\nHighlight from config: {args.trim[0]:.1f}s - {args.trim[1]:.1f}s")
        else:
            # Scan episode to find events
            print("\nScanning for bottle-in-bin events...")
            _hz = 1.0 / (env.model.opt.timestep * env._control_decimation)
            first_in_step = None
            last_in_step = None
            for t in range(len(actions)):
                env._step_single(actions[t])
                for _name, addr in bottle_info.items():
                    if is_bottle_in_bin(
                        env.data.qpos[addr : addr + 2],
                        env.data.qpos[addr + 2],
                        bin_pos,
                    ):
                        if first_in_step is None:
                            first_in_step = t
                        last_in_step = t
            if first_in_step is not None:
                pad = 3.0
                args.trim = [
                    max(0, first_in_step / _hz - pad),
                    min(len(actions) / _hz, last_in_step / _hz + pad),
                ]
                print(f"  Found events: {first_in_step / _hz:.1f}s - {last_in_step / _hz:.1f}s")
                print(f"  Auto-trim: {args.trim[0]:.1f}s - {args.trim[1]:.1f}s")
            else:
                print("  No bottles entered bin, showing full episode")
            # Reset for actual replay
            env.reset()
            env._set_qpos_from_state(init_q)
            mujoco.mj_forward(env.model, env.data)

        # Enable renderer now if needed
        if not args.verify_only and not args.view and not hasattr(env, "renderer"):
            env._render_cameras_flag = True
            env.renderer = mujoco.Renderer(
                env.model,
                height=env._camera_height,
                width=env._camera_width,
            )

    # Interactive viewer mode
    if args.view:
        from yam_sim.env import _GRIPPER_CTRL_MAX

        ctrl_indices = env._ctrl_indices
        gripper_set = env._gripper_set
        decimation = env._control_decimation
        control_hz = 1.0 / (env.model.opt.timestep * decimation)

        # Trim support
        if args.trim:
            view_start = max(0, int(args.trim[0] * control_hz))
            view_end = min(len(actions), int(args.trim[1] * control_hz))
            print(f"\nFast-forwarding to {args.trim[0]:.1f}s (step {view_start})...")
            for t in range(view_start):
                env._step_single(actions[t])
            print(f"Playing {args.trim[0]:.1f}s - {args.trim[1]:.1f}s in viewer")
        else:
            view_start = 0
            view_end = len(actions)

        action_idx = [view_start]
        substep = [0]

        def controller(model, data):
            i = action_idx[0]
            if i >= view_end:
                return
            action = actions[i]
            for j in range(env.single_timestep_action_dim):
                val = float(action[j])
                if j in gripper_set:
                    val = val * _GRIPPER_CTRL_MAX
                data.ctrl[ctrl_indices[j]] = val
            substep[0] += 1
            if substep[0] >= decimation:
                substep[0] = 0
                action_idx[0] = i + 1

        mujoco.set_mjcb_control(controller)
        remaining = view_end - view_start
        print(f"\nLaunching viewer — replaying {remaining} steps...")
        mujoco.viewer.launch(env.model, env.data)
        env.close()
        return

    # Trim range
    control_hz = 1.0 / (env.model.opt.timestep * env._control_decimation)
    if args.trim:
        trim_start = max(0, int(args.trim[0] * control_hz))
        trim_end = min(len(actions), int(args.trim[1] * control_hz))
        print(f"\nTrimming to {args.trim[0]:.1f}s - {args.trim[1]:.1f}s "
              f"(steps {trim_start}-{trim_end})")
    else:
        trim_start = 0
        trim_end = len(actions)

    # Replay actions
    frames = []
    render = not args.verify_only
    frame_skip = args.frame_skip
    step_count = 0

    if render:
        from yam_sim.viewer import render_cameras

        if trim_start == 0:
            images = render_cameras(env)
            grid = np.concatenate(
                [images[name] for name in env.camera_names], axis=1
            )
            frames.append(grid)

    print(f"\nReplaying {trim_end} action steps...")
    for t in range(trim_end):
        env._step_single(actions[t])
        step_count += 1

        in_window = t >= trim_start
        if render and in_window and (t % frame_skip == 0 or t == trim_end - 1):
            images = render_cameras(env)
            grid = np.concatenate(
                [images[name] for name in env.camera_names], axis=1
            )
            frames.append(grid)

        if t % 500 == 0 and t > 0:
            print(f"  Step {t}/{len(actions)}")

    # Check final metrics
    print(f"\n{'='*50}")
    print("REPLAY RESULTS")
    print(f"{'='*50}")
    print(f"Total steps: {step_count}")

    bottles_in_bin = []
    for name, addr in bottle_info.items():
        pos = env.data.qpos[addr : addr + 3]
        in_bin = (
            is_bottle_in_bin(pos[:2], pos[2], bin_pos) if bin_pos is not None else False
        )
        status = " ** IN BIN **" if in_bin else ""
        print(f"  {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]{status}")
        if in_bin:
            bottles_in_bin.append(name)

    print(f"\nBottles in bin: {len(bottles_in_bin)}")
    if len(bottles_in_bin) >= 2:
        print("SUCCESS: 2+ bottles deposited in bin!")
    elif len(bottles_in_bin) == 1:
        print("Partial success: 1 bottle deposited.")
    else:
        print("No bottles deposited.")

    expected_key = "bottles_in_bin_full" if args.full else "bottles_in_bin"
    expected = config.get(expected_key, config.get("bottles_in_bin", []))
    if expected:
        print(f"Expected: {expected}")
        if set(bottles_in_bin) == set(expected):
            print("MATCH: Replay matches original results!")
        else:
            print("MISMATCH: Different bottles in bin than original run.")
            print("  (This can happen due to floating-point differences across hardware.)")

    # Save video/gif
    if render and frames:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        is_gif = args.output.lower().endswith(".gif")

        try:
            import imageio.v3 as iio

            if is_gif:
                from PIL import Image

                # Downscale for reasonable GIF size
                scale = 0.5
                resized = []
                for frame in frames:
                    img = Image.fromarray(frame)
                    new_size = (int(img.width * scale), int(img.height * scale))
                    resized.append(np.array(img.resize(new_size, Image.LANCZOS)))
                duration_ms = 1000.0 / args.fps
                iio.imwrite(
                    args.output,
                    np.stack(resized),
                    duration=duration_ms,
                    loop=0,
                )
            else:
                iio.imwrite(args.output, np.stack(frames), fps=args.fps)
            print(f"\nSaved {len(frames)} frames to {args.output}")
        except ImportError:
            from PIL import Image

            frames_dir = Path(args.output).with_suffix("")
            frames_dir.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(frames):
                Image.fromarray(frame).save(frames_dir / f"frame_{i:05d}.png")
            print(f"\nSaved {len(frames)} frames to {frames_dir}/")

    env.close()


if __name__ == "__main__":
    main()
