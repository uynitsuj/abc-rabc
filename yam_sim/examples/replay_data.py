"""Replay dataset trajectories in yam-sim via qpos or action playback.

This utility is aimed at debugging dataset consistency. It can:

- replay recorded states by directly writing qpos each frame
- replay recorded actions by initializing from the first state and stepping physics
- compare action playback against the recorded state trajectory

It supports `states_actions.bin` files from cached dataset episodes and detects
the common on-disk dtype automatically.

Examples:
    # Direct qpos playback from an episode directory
    MUJOCO_GL=egl python -m yam_sim.examples.replay_data \
        --episode-dir /tmp/metadata/episode_cache/sim_tasks_20260414_madrona_224/episode_... \
        --playback-mode qpos \
        --output /tmp/qpos_replay.mp4

    # Action playback with drift report, no rendering
    python -m yam_sim.examples.replay_data \
        --episode-dir /tmp/metadata/episode_cache/sim_tasks_20260414_madrona_224/episode_... \
        --playback-mode action \
        --verify-only

    # Compare qpos vs action playback on a cached bottles episode
    python -m yam_sim.examples.replay_data \
        --episode-dir /tmp/metadata/episode_cache/sim_tasks_20260414_madrona_224/episode_... \
        --playback-mode compare \
        --task sim_throw_plastic_bottles_in_bin \
        --scene training \
        --verify-only
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def _resolve_bin_dtype(
    bin_path: str,
    row_dim: int,
    preferred: str = "auto",
) -> np.dtype:
    if preferred != "auto":
        return np.dtype(preferred)

    file_size = os.path.getsize(bin_path)
    if file_size % (row_dim * 8) == 0:
        return np.dtype(np.float64)
    if file_size % (row_dim * 4) == 0:
        return np.dtype(np.float32)
    raise ValueError(
        f"Could not infer dtype for {bin_path}. "
        f"File size {file_size} is not divisible by row_dim={row_dim} for float32/float64."
    )


def load_states_actions_from_bin(
    bin_path: str,
    n_states: int = 14,
    n_actions: int = 14,
    dtype: str = "auto",
) -> tuple[np.ndarray, np.ndarray, np.dtype]:
    """Load state/action arrays from a states_actions.bin file."""
    row_dim = n_states + n_actions
    resolved_dtype = _resolve_bin_dtype(bin_path, row_dim=row_dim, preferred=dtype)
    arr = np.fromfile(bin_path, dtype=resolved_dtype)
    if arr.size % row_dim != 0:
        raise ValueError(
            f"states_actions.bin at {bin_path} has {arr.size} scalars, "
            f"which is not divisible by row_dim={row_dim}"
        )
    arr = arr.reshape(-1, row_dim)
    states = arr[:, :n_states].astype(np.float32)
    actions = arr[:, n_states : n_states + n_actions].astype(np.float32)
    return states, actions, resolved_dtype


def _load_episode_metadata(episode_dir: Path) -> dict[str, Any] | None:
    """Best-effort lookup of episode metadata for cached dataset episodes."""
    episode_id = episode_dir.name
    dataset_name = episode_dir.parent.name

    candidate_paths = [
        Path("/tmp/metadata") / dataset_name / "collected.json",
        episode_dir.parent.parent.parent / dataset_name / "collected.json",
    ]
    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            with open(path) as f:
                collected = json.load(f)
            if isinstance(collected, dict) and episode_id in collected:
                return collected[episode_id]
        except Exception:
            continue
    return None


def _infer_task_prompt(
    *,
    episode_dir: Path | None,
    task_arg: str | None,
    prompt_arg: str | None,
) -> tuple[str, str]:
    """Resolve env task + prompt, preferring explicit args then cached metadata."""
    import yam_sim

    if task_arg:
        spec = yam_sim.maybe_get_task_spec(task_arg)
        if spec is not None:
            env_task = spec.env_task
            prompt = prompt_arg or spec.prompt
            return env_task, prompt
        return task_arg, (prompt_arg or "")

    if episode_dir is not None:
        metadata = _load_episode_metadata(episode_dir)
        if metadata is not None:
            raw_task = str(metadata.get("task_name", "")).strip()
            if raw_task:
                spec = yam_sim.maybe_get_task_spec(raw_task)
                if spec is not None:
                    return spec.env_task, (prompt_arg or spec.prompt)
                if prompt_arg:
                    return raw_task, prompt_arg

    # Safe fallback for generic replay use.
    return "bottles", (prompt_arg or "throw plastic bottles in bin")


def _collect_task_eval(env) -> dict[str, Any] | None:
    result = env.evaluate_task()
    if result is None:
        return None
    return result.to_info(squeeze=True)


def _make_metrics(
    *,
    mode: str,
    replayed_states: np.ndarray,
    recorded_states: np.ndarray,
    final_task_eval: dict[str, Any] | None,
) -> dict[str, Any]:
    common = min(len(replayed_states), len(recorded_states))
    diffs = replayed_states[:common] - recorded_states[:common]
    l1 = np.abs(diffs)

    metrics = {
        "playback_mode": mode,
        "num_recorded_steps": int(len(recorded_states)),
        "num_replayed_steps": int(len(replayed_states)),
        "state_mae_mean": float(l1.mean()) if l1.size else 0.0,
        "state_mae_p50": float(np.percentile(l1, 50)) if l1.size else 0.0,
        "state_mae_p95": float(np.percentile(l1, 95)) if l1.size else 0.0,
        "state_mae_max": float(l1.max()) if l1.size else 0.0,
        "final_state_mae": float(
            np.mean(np.abs(replayed_states[common - 1] - recorded_states[common - 1]))
        )
        if common > 0
        else 0.0,
        "task_eval": final_task_eval,
    }
    return metrics


def _render_frame_grid(env) -> np.ndarray:
    from yam_sim.viewer import render_cameras

    images_hwc = render_cameras(env)
    return np.concatenate([images_hwc[name] for name in env.camera_names], axis=1)


def _run_qpos_playback(
    *,
    env,
    states: np.ndarray,
    render: bool,
    fps: int,
    output_path: str,
) -> dict[str, Any]:
    import mujoco

    env.reset(randomize=False)
    env._set_qpos_from_state(states[0])
    mujoco.mj_forward(env.model, env.data)

    frames: list[np.ndarray] = []
    replayed_states: list[np.ndarray] = []

    for t, state in enumerate(states):
        env._set_qpos_from_state(state)
        mujoco.mj_forward(env.model, env.data)
        replayed_states.append(env.get_state().copy())
        if render:
            frames.append(_render_frame_grid(env))
        if t % 250 == 0:
            print(f"  [qpos] frame {t}/{len(states)}")

    final_task_eval = _collect_task_eval(env)
    metrics = _make_metrics(
        mode="qpos",
        replayed_states=np.asarray(replayed_states, dtype=np.float32),
        recorded_states=states,
        final_task_eval=final_task_eval,
    )
    if render and frames:
        _save_frames(frames, output_path, fps=fps)
    return metrics


def _run_action_playback(
    *,
    env,
    states: np.ndarray,
    actions: np.ndarray,
    render: bool,
    fps: int,
    output_path: str,
) -> dict[str, Any]:
    import mujoco

    env.reset(randomize=False)
    env._set_qpos_from_state(states[0])
    mujoco.mj_forward(env.model, env.data)

    frames: list[np.ndarray] = []
    replayed_states: list[np.ndarray] = [env.get_state().copy()]

    if render:
        frames.append(_render_frame_grid(env))

    for t, action in enumerate(actions):
        env._step_single(action)
        replayed_states.append(env.get_state().copy())
        if render:
            frames.append(_render_frame_grid(env))
        if t % 250 == 0:
            print(f"  [action] step {t}/{len(actions)}")

    # Align against the recorded state trajectory after each applied action.
    final_task_eval = _collect_task_eval(env)
    metrics = _make_metrics(
        mode="action",
        replayed_states=np.asarray(replayed_states, dtype=np.float32),
        recorded_states=states,
        final_task_eval=final_task_eval,
    )
    if render and frames:
        _save_frames(frames, output_path, fps=fps)
    return metrics


def _save_frames(frames: list[np.ndarray], output_path: str, fps: int) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        import imageio.v3 as iio

        iio.imwrite(output_path, np.stack(frames), fps=fps)
        print(f"Saved video ({len(frames)} frames) to {output_path}")
    except ImportError:
        from PIL import Image

        frames_dir = output_path.replace(".mp4", "_frames")
        os.makedirs(frames_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(os.path.join(frames_dir, f"frame_{i:05d}.png"))
        print(f"Saved {len(frames)} frames to {frames_dir}/")


def parse_args():
    parser = argparse.ArgumentParser(description="Replay dataset trajectories in yam-sim")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--episode-dir", type=str, help="Path to cached episode directory")
    group.add_argument("--bin-file", type=str, help="Path to states_actions.bin")
    parser.add_argument("--n-states", type=int, default=14, help="State dimension")
    parser.add_argument("--n-actions", type=int, default=14, help="Action dimension")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float64"],
        help="On-disk dtype for states_actions.bin",
    )
    parser.add_argument(
        "--playback-mode",
        type=str,
        default="action",
        choices=["qpos", "action", "compare"],
        help="Replay via direct qpos writes, action stepping, or both",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task alias/name for env creation. If omitted, try to infer from episode metadata.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt override. If omitted, try to infer from task metadata.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="hybrid",
        choices=["eval", "training", "hybrid"],
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Replay at most this many rows from the trajectory",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not render video. Print metrics only.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/yam_replay.mp4",
        help="Video path. In compare mode, suffixes _qpos/_action are added.",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default=None,
        help="Optional JSON report path",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    episode_dir = Path(args.episode_dir) if args.episode_dir else None
    if episode_dir is not None:
        bin_path = episode_dir / "states_actions.bin"
        if not bin_path.exists():
            raise FileNotFoundError(f"Expected {bin_path} to exist")
    else:
        bin_path = Path(args.bin_file)

    states, actions, resolved_dtype = load_states_actions_from_bin(
        str(bin_path),
        n_states=args.n_states,
        n_actions=args.n_actions,
        dtype=args.dtype,
    )
    if args.max_steps is not None:
        states = states[: args.max_steps]
        actions = actions[: args.max_steps]

    print(
        f"Loaded trajectory from {bin_path}: "
        f"{len(states)} rows, state_dim={states.shape[-1]}, action_dim={actions.shape[-1]}, "
        f"dtype={resolved_dtype}"
    )

    import yam_sim

    env_task, prompt = _infer_task_prompt(
        episode_dir=episode_dir,
        task_arg=args.task,
        prompt_arg=args.prompt,
    )
    print(f"Env task: {env_task}")
    print(f"Prompt: {prompt!r}")

    render = not args.verify_only
    env = yam_sim.make_env(
        scene=args.scene,
        task=env_task,
        prompt=prompt,
        render_cameras=render,
    )

    reports: dict[str, Any] = {
        "source_bin": str(bin_path),
        "scene": args.scene,
        "env_task": env_task,
        "prompt": prompt,
        "dtype": str(resolved_dtype),
        "num_rows": int(len(states)),
    }

    try:
        if args.playback_mode in {"qpos", "compare"}:
            qpos_output = args.output.replace(".mp4", "_qpos.mp4")
            qpos_metrics = _run_qpos_playback(
                env=env,
                states=states,
                render=render,
                fps=args.fps,
                output_path=qpos_output,
            )
            reports["qpos"] = qpos_metrics
            print("\nQpos playback metrics:")
            print(json.dumps(qpos_metrics, indent=2))

        if args.playback_mode in {"action", "compare"}:
            action_output = args.output.replace(".mp4", "_action.mp4")
            action_metrics = _run_action_playback(
                env=env,
                states=states,
                actions=actions,
                render=render,
                fps=args.fps,
                output_path=action_output,
            )
            reports["action"] = action_metrics
            print("\nAction playback metrics:")
            print(json.dumps(action_metrics, indent=2))
    finally:
        env.close()

    if args.report_json is not None:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(reports, f, indent=2)
        print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    main()
