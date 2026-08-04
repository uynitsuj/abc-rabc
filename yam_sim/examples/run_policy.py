"""Run a trained LBM policy in yam-sim.

The default rollout path uses the batched ``mjwarp`` renderer because it works
headless on machines without EGL/OSMesa. The legacy single-world MuJoCo camera
path is still available via ``--render-backend mujoco``.

Examples:
    MUJOCO_GL=glfw python -m yam_sim.examples.run_policy \
        --checkpoint-path /path/to/checkpoint.pt \
        --task bottles \
        --render-backend mjwarp \
        --output /tmp/policy_rollout.mp4

    MUJOCO_GL=egl python -m yam_sim.examples.run_policy \
        --checkpoint-path /path/to/checkpoint.pt \
        --render-backend mujoco \
        --output /tmp/policy_rollout.mp4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mujoco
import numpy as np
import torch

from yam_sim.eval.rollout import (
    grid_from_obs as _grid_from_obs,
    prepare_policy_obs as _prepare_policy_obs,
    slice_execute_actions,
    ttrtc_prefix,
)


def _task_eval_arrays(env) -> tuple[np.ndarray | None, np.ndarray | None]:
    result = env.evaluate_task()
    if result is None:
        return None, None
    reward = np.asarray(result.reward, dtype=np.float32).reshape(-1)
    success = np.asarray(result.success, dtype=np.float32).reshape(-1)
    return reward, success


def parse_args():
    parser = argparse.ArgumentParser(description="Run an LBM policy in yam-sim")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to a local LBM checkpoint. Omit when using --serve-host (openpi server).",
    )
    parser.add_argument(
        "--serve-host",
        type=str,
        default=None,
        help="If set, fetch actions from an openpi websocket server at this host "
        "(see openpi/scripts/serve_policy.py) instead of loading a local checkpoint.",
    )
    parser.add_argument("--serve-port", type=int, default=8000)
    parser.add_argument("--serve-api-key", type=str, default=None)
    parser.add_argument("--task", type=str, default="bottles")
    parser.add_argument("--prompt", type=str, default="throw plastic bottles in bin")
    parser.add_argument("--output", type=str, default="/tmp/yam_policy_rollout.mp4")
    parser.add_argument("--num-worlds", type=int, default=1)
    parser.add_argument("--num-chunks", type=int, default=10)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="If set, overrides --num-chunks. Sim time per chunk = "
        "execute_chunk_dim * physics_dt * control_decimation (~0.67 s at defaults).",
    )
    parser.add_argument("--execute-chunk-dim", type=int, default=20)
    parser.add_argument("--prefix-length", type=int, default=3)
    parser.add_argument("--diffusion-steps", type=int, default=3)
    parser.add_argument("--model-size", type=str, default="dit_xL")
    parser.add_argument(
        "--render-backend",
        type=str,
        default="mjwarp",
        choices=["mjwarp", "mujoco"],
        help="Use mjwarp for headless rollout/video or mujoco for the legacy renderer path.",
    )
    parser.add_argument(
        "--camera-gpu-id",
        type=int,
        default=None,
        help="GPU index for mjwarp rendering.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Policy GPU index. Ignored if CUDA_VISIBLE_DEVICES is already set.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-ttrtc", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=0)
    parser.add_argument(
        "--scene",
        type=str,
        default="hybrid",
        choices=["eval", "training", "hybrid"],
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--bottle-mass", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gpu is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    import yam_sim
    from yam_sim.task_specs import maybe_get_task_spec

    task_spec = maybe_get_task_spec(args.task)
    env_task = task_spec.env_task if task_spec is not None else args.task
    prompt = task_spec.prompt if task_spec is not None else args.prompt

    if args.serve_host is not None:
        from yam_sim.policy.openpi_policy import OpenPIPolicy

        policy = OpenPIPolicy(
            host=args.serve_host,
            port=args.serve_port,
            api_key=args.serve_api_key,
        )
    else:
        if args.checkpoint_path is None:
            raise ValueError(
                "Provide --checkpoint-path for local inference or --serve-host for a remote openpi server."
            )
        from yam_sim.policy.lbm_policy import LBMPolicy, LBMPolicyConfig

        policy = LBMPolicy(
            LBMPolicyConfig(
                ckpt_path=args.checkpoint_path,
                policy_type="lbm",
                model_size=args.model_size,
                diffusion_steps=args.diffusion_steps,
            )
        )

    if args.num_worlds < 1:
        raise ValueError(f"--num-worlds must be >= 1, got {args.num_worlds}")
    if args.render_backend == "mujoco" and args.num_worlds != 1:
        raise ValueError("--render-backend mujoco only supports --num-worlds 1.")

    if args.render_backend == "mjwarp":
        env = yam_sim.make_batched_env(
            scene=args.scene,
            task=env_task,
            prompt=prompt,
            num_worlds=args.num_worlds,
            camera_backend="mjwarp",
            camera_gpu_id=args.camera_gpu_id,
        )
    else:
        env = yam_sim.make_env(
            scene=args.scene,
            task=env_task,
            prompt=prompt,
        )

    if args.bottle_mass is not None:
        for i in range(1, 7):
            body_name = f"bottle_{i}"
            body_id = mujoco.mj_name2id(
                env.model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            if body_id >= 0:
                old_mass = env.model.body_mass[body_id]
                if old_mass > 0:
                    env.model.body_inertia[body_id] *= args.bottle_mass / old_mass
                env.model.body_mass[body_id] = args.bottle_mass

    use_ttrtc = not args.no_ttrtc
    execute_dim = args.execute_chunk_dim if use_ttrtc else 30
    prefix_len = args.prefix_length if use_ttrtc else 0

    base = env if not hasattr(env, "_base_env") else env._base_env
    step_dt = float(getattr(base, "_physics_dt", 0.002)) * int(
        getattr(base, "_control_decimation", 17)
    )
    seconds_per_chunk = execute_dim * step_dt
    num_chunks = args.num_chunks
    if args.max_seconds is not None:
        num_chunks = max(1, int(np.ceil(args.max_seconds / seconds_per_chunk)))

    print(
        f"Rollout: backend={args.render_backend}, worlds={args.num_worlds}, "
        f"{num_chunks} chunks ({num_chunks * seconds_per_chunk:.1f}s sim), "
        f"execute {execute_dim}/30 per chunk ({seconds_per_chunk:.2f}s/chunk)"
    )

    obs, _ = env.reset(seed=args.seed)
    init_q = np.asarray(obs["state"], dtype=np.float32)
    is_batched = init_q.ndim == 2

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    video_writer = None
    frames_dir = None
    try:
        import imageio

        video_writer = imageio.get_writer(
            args.output, fps=args.fps, macro_block_size=1
        )
    except ImportError:
        frames_dir = Path(args.output).with_suffix("")
        frames_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0

    def _emit_frame(frame: np.ndarray) -> None:
        nonlocal frame_count
        if video_writer is not None:
            video_writer.append_data(frame)
        else:
            from PIL import Image
            Image.fromarray(frame).save(frames_dir / f"frame_{frame_count:05d}.png")
        frame_count += 1

    _emit_frame(_grid_from_obs(obs, env.camera_names))
    all_actions = []
    previous_chunk_actions = None

    for chunk_idx in range(num_chunks):
        policy_obs = _prepare_policy_obs(
            obs,
            prompt=prompt,
            jpeg_quality=args.jpeg_quality,
        )

        action_prefix = None
        prefix_length = None
        if use_ttrtc:
            action_prefix, prefix_length = ttrtc_prefix(
                previous_chunk_actions, init_q, prefix_len, is_batched=is_batched
            )

        result = policy.infer(
            policy_obs,
            action_prefix=action_prefix,
            prefix_length=prefix_length,
        )

        predicted_actions = np.asarray(result["actions"], dtype=np.float32)
        if is_batched and predicted_actions.ndim == 2:
            predicted_actions = predicted_actions[None, ...]
        execute_actions = slice_execute_actions(
            predicted_actions,
            prefix_length=prefix_length,
            execute_dim=execute_dim,
            use_ttrtc=use_ttrtc,
            is_batched=is_batched,
        )
        all_actions.append(execute_actions)

        if is_batched:
            action_iter = (execute_actions[:, step_idx, :] for step_idx in range(execute_actions.shape[1]))
        else:
            action_iter = iter(execute_actions)

        for action in action_iter:
            if args.render_backend == "mjwarp":
                if is_batched:
                    obs, _, _, _, _ = env.step(action)
                else:
                    obs, _, _, _, _ = env.step(action[None, :])
            else:
                env._step_single(action)
                obs = env.get_obs()
            _emit_frame(_grid_from_obs(obs, env.camera_names))

        previous_chunk_actions = execute_actions
        reward, success = _task_eval_arrays(env)
        reward_mean = None if reward is None else float(reward.mean())
        success_mean = None if success is None else float(success.mean())
        success_list = None if success is None else success.astype(int).tolist()
        range_min = float(predicted_actions.min())
        range_max = float(predicted_actions.max())
        print(
            f"  Chunk {chunk_idx}: actions range=[{range_min:.3f}, {range_max:.3f}] "
            f"reward_mean={reward_mean} success_mean={success_mean} success={success_list}"
        )

    if video_writer is not None:
        video_writer.close()
        print(f"Saved video ({frame_count} frames) to {args.output}")
    else:
        print(f"Saved {frame_count} frames to {frames_dir}/")

    actions_path = Path(args.output).with_suffix(".npy")
    action_axis = 1 if is_batched else 0
    actions = np.concatenate(all_actions, axis=action_axis)
    np.save(str(actions_path), actions)
    print(f"Saved actions to {actions_path}")

    final_reward, final_success = _task_eval_arrays(env)
    final_reward_list = None if final_reward is None else final_reward.tolist()
    final_success_list = None if final_success is None else final_success.astype(int).tolist()
    print(
        "Final task eval: "
        f"reward={final_reward_list} "
        f"success={final_success_list}"
    )
    env.close()


if __name__ == "__main__":
    main()
