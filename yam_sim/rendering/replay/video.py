"""Headless replay-to-video helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from typing import Sequence

import mujoco
import numpy as np
from PIL import Image

from yam_sim.env import _GRIPPER_CTRL_MAX


def tile_camera_frames(frames: dict[str, np.ndarray], camera_names: Sequence[str]) -> np.ndarray:
    """Tile camera frames horizontally in a stable order."""
    ordered = [frames[name] for name in camera_names if name in frames]
    if not ordered:
        raise ValueError("No camera frames available to tile")
    if len(ordered) == 1:
        return ordered[0]
    return np.concatenate(ordered, axis=1)


def _open_video_writer(output_path: Path, fps: float):
    """Prefer the imageio-ffmpeg writer over pyav in headless environments."""
    import imageio.v2 as iio

    try:
        import imageio_ffmpeg  # noqa: F401
    except ImportError:
        return iio.get_writer(output_path, fps=fps)
    return iio.get_writer(output_path, fps=fps, format="FFMPEG")


def write_video(output_path: Path, frames: list[np.ndarray], fps: float) -> None:
    """Write frames to an MP4, or fall back to PNGs if imageio is unavailable."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as iio  # noqa: F401
    except ImportError:
        frames_dir = output_path.with_suffix("")
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(frames_dir / f"frame_{i:05d}.png")
        print(f"imageio not available. Saved {len(frames)} PNG frames to {frames_dir}/")
        return

    with _open_video_writer(output_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)


def _tile_render_batch(img, camera_indices: Sequence[int]):
    """Tile a GPU render batch into (B, H, ncam*W, 3) frames."""
    selected = img[:, camera_indices]
    actual, ncam, height, width, channels = selected.shape
    return (
        selected.permute(0, 2, 1, 3, 4)
        .contiguous()
        .view(actual, height, ncam * width, channels)
    )


def _actions_to_ctrl_batch(
    actions: np.ndarray,
    *,
    nu: int,
    ctrl_indices: Sequence[int],
    gripper_indices: Sequence[int],
    gripper_ctrl_max: float = _GRIPPER_CTRL_MAX,
) -> np.ndarray:
    """Map 14D policy actions into full actuator ctrl vectors."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"Expected actions shape (T, D), got {actions.shape}")
    if actions.shape[1] != len(ctrl_indices):
        raise ValueError(
            f"Expected action dim {len(ctrl_indices)} from ctrl index mapping, got {actions.shape[1]}"
        )

    ctrl = np.zeros((actions.shape[0], nu), dtype=np.float32)
    gripper_set = set(int(i) for i in gripper_indices)
    for i, ctrl_idx in enumerate(ctrl_indices):
        values = actions[:, i]
        if i in gripper_set:
            values = values * float(gripper_ctrl_max)
        ctrl[:, int(ctrl_idx)] = values
    return ctrl


def _infer_physics_substeps_per_action(session) -> int:
    """Infer whether replay actions represent control steps or already-expanded sim steps."""
    nominal_control_dt = float(session.env.model.opt.timestep * session.env._control_decimation)
    if len(session.grid_ts) <= 1:
        return int(session.env._control_decimation)
    median_dt = float(np.median(np.diff(np.asarray(session.grid_ts, dtype=np.float64))))
    if median_dt < nominal_control_dt * 0.5:
        return 1
    return int(session.env._control_decimation)


def _infer_qpos_frame_timestamps(
    session,
    *,
    frame_count: int,
    include_initial_frame: bool,
) -> np.ndarray:
    """Infer timestamps for a qpos frame sequence captured from this replay session."""
    grid_ts = np.asarray(session.grid_ts, dtype=np.float64)
    if len(grid_ts) == frame_count:
        return grid_ts
    if not include_initial_frame and len(grid_ts) == frame_count + 1:
        return grid_ts[1:]
    if include_initial_frame and len(grid_ts) == frame_count - 1:
        dt = float(session.step_dt)
        start = float(grid_ts[0] - dt) if len(grid_ts) > 0 else 0.0
        return start + np.arange(frame_count, dtype=np.float64) * dt
    dt = float(session.step_dt)
    return np.arange(frame_count, dtype=np.float64) * dt


def _source_frame_limit_for_output(
    source_ts: np.ndarray,
    *,
    fps: float,
    max_output_frames: int | None,
) -> int:
    """How many source frames are needed to cover the requested output duration."""
    if len(source_ts) == 0:
        return 0
    if max_output_frames is None:
        return len(source_ts)
    if max_output_frames <= 0:
        raise ValueError("max_output_frames must be positive when provided")
    end_t = float(source_ts[0] + (max_output_frames - 1) / fps)
    return max(1, min(len(source_ts), int(np.searchsorted(source_ts, end_t, side="right"))))


def _sample_qpos_frames_for_video(
    qpos_frames: np.ndarray,
    source_ts: np.ndarray,
    *,
    fps: float,
    max_output_frames: int | None,
) -> np.ndarray:
    """Resample qpos frames onto a regular output-video timeline using zero-order hold."""
    qpos_frames = np.asarray(qpos_frames, dtype=np.float32)
    source_ts = np.asarray(source_ts, dtype=np.float64)
    if len(qpos_frames) != len(source_ts):
        raise ValueError(
            f"qpos frame/timestamp length mismatch: {len(qpos_frames)} vs {len(source_ts)}"
        )
    if len(qpos_frames) == 0:
        raise RuntimeError("No qpos frames available for video sampling")
    if fps <= 0:
        raise ValueError("fps must be positive")

    duration = float(source_ts[-1] - source_ts[0]) if len(source_ts) > 1 else 0.0
    full_output_frames = max(1, int(np.floor(duration * fps)) + 1)
    output_frames = (
        full_output_frames
        if max_output_frames is None
        else min(full_output_frames, int(max_output_frames))
    )
    output_ts = source_ts[0] + np.arange(output_frames, dtype=np.float64) / fps
    idx = np.searchsorted(source_ts, output_ts, side="right") - 1
    idx = np.clip(idx, 0, len(source_ts) - 1)
    return qpos_frames[idx]


def _qpos_frames_from_integration_states(session, integration_states: np.ndarray) -> np.ndarray:
    """Materialize MuJoCo qpos frames from recorded mjSTATE_INTEGRATION states."""
    if session.sim_state_spec is None:
        raise RuntimeError("Integration-state replay requires a MuJoCo state spec")
    states = np.asarray(integration_states, dtype=np.float64)
    if states.ndim != 2:
        raise RuntimeError(f"Expected integration states shape (T, D), got {states.shape}")
    qpos = np.empty((len(states), session.model.nq), dtype=np.float32)
    for idx, state in enumerate(states):
        mujoco.mj_setState(session.model, session.data, state, session.sim_state_spec)
        mujoco.mj_forward(session.model, session.data)
        qpos[idx] = np.asarray(session.data.qpos, dtype=np.float32).copy()
    return qpos


def _batched_sim_camera_indices(session) -> list[int]:
    camera_names = tuple(session.env.camera_names)
    available_names = tuple(session.model.cam(i).name for i in range(session.model.ncam))
    camera_indices = [available_names.index(name) for name in camera_names if name in available_names]
    if len(camera_indices) != len(camera_names):
        missing = [name for name in camera_names if name not in available_names]
        raise RuntimeError(f"Requested sim cameras missing from model: {missing}")
    return camera_indices


def _render_batched_qpos_frames(
    session,
    *,
    qpos_frames: np.ndarray,
    output_path: Path,
    fps: float,
    sim_backend: Literal["mjwarp", "madrona"],
    render_width: int,
    render_height: int,
    batch_size: int,
    gpu_id: int | None = None,
    progress_every: int = 100,
) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(qpos_frames) == 0:
        raise RuntimeError("No qpos frames available for batched export")

    from yam_sim.rendering.replay.renderer import RendererWrapper, WarpReplayRuntime

    qpos_frames = np.asarray(qpos_frames, dtype=np.float32)
    camera_indices = _batched_sim_camera_indices(session)
    actual_batch_size = min(batch_size, len(qpos_frames))
    runtime = WarpReplayRuntime(session.model, session.data, nworld=actual_batch_size, gpu_id=gpu_id)
    qpos_frames_device = runtime.upload_qpos_frames(qpos_frames)
    renderer = RendererWrapper(
        backend=sim_backend,
        runtime=runtime,
        cam_res=(render_width, render_height),
        gpu_id=gpu_id,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as iio  # noqa: F401
    except ImportError:
        iio = None

    frames_dir = None
    writer = None
    if iio is None:
        frames_dir = output_path.with_suffix("")
        frames_dir.mkdir(parents=True, exist_ok=True)
    else:
        writer = _open_video_writer(output_path, fps=fps)

    written = 0
    next_progress = progress_every if progress_every > 0 else None
    first_batch = True

    try:
        for start in range(0, len(qpos_frames), actual_batch_size):
            stop = min(start + actual_batch_size, len(qpos_frames))
            actual = stop - start

            runtime.load_qpos_batch_from_device(qpos_frames_device, start=start, stop=stop)
            runtime.forward()
            img = (
                renderer.reset(actual_batch=actual)
                if first_batch
                else renderer.render(actual_batch=actual)
            )
            first_batch = False

            tiled = _tile_render_batch(img, camera_indices).cpu().numpy()
            if writer is not None:
                for frame in tiled:
                    writer.append_data(frame)
            else:
                for idx, frame in enumerate(tiled, start=written):
                    Image.fromarray(frame).save(frames_dir / f"frame_{idx:05d}.png")

            written += actual
            if next_progress is not None:
                while written >= next_progress:
                    print(f"  Rendered {min(next_progress, len(qpos_frames))}/{len(qpos_frames)}")
                    next_progress += progress_every
    finally:
        renderer.close()
        if writer is not None:
            writer.close()

    if frames_dir is not None:
        print(f"imageio not available. Saved {written} PNG frames to {frames_dir}/")

    return written


def collect_physics_rollout_qpos_frames(
    session,
    *,
    include_initial_frame: bool = True,
    max_frames: int | None = None,
    progress_every: int = 100,
) -> np.ndarray:
    """Run physics replay and capture the resulting qpos trajectory for batched rendering."""
    if session.mode != "physics":
        raise RuntimeError("Physics rollout qpos capture requires a physics replay session")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided")

    session.reset()
    frames: list[np.ndarray] = []
    if include_initial_frame and (max_frames is None or len(frames) < max_frames):
        frames.append(np.asarray(session.data.qpos, dtype=np.float32).copy())

    steps = 0
    while (max_frames is None or len(frames) < max_frames) and session.step():
        frames.append(np.asarray(session.data.qpos, dtype=np.float32).copy())
        steps += 1
        if progress_every > 0 and steps % progress_every == 0:
            print(f"  Rolled out {steps}/{session.total_steps} physics steps")

    if not frames:
        raise RuntimeError("Physics rollout produced no qpos frames")
    return np.stack(frames, axis=0)


def collect_mjwarp_rollout_qpos_frames(
    session,
    *,
    include_initial_frame: bool = True,
    max_frames: int | None = None,
    progress_every: int = 100,
    gpu_id: int | None = None,
) -> np.ndarray:
    """Run physics replay with native MJWarp stepping and capture the qpos trajectory."""
    if session.mode != "physics":
        raise RuntimeError("MJWarp rollout qpos capture requires a physics replay session")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided")

    from yam_sim.rendering.replay.renderer import WarpReplayRuntime

    session.reset()
    runtime = WarpReplayRuntime(session.model, session.data, nworld=1, gpu_id=gpu_id)
    runtime.reset_from_mujoco()

    if session.replay_ctrls is not None:
        ctrl_frames = np.asarray(session.replay_ctrls, dtype=np.float32)
    else:
        ctrl_frames = _actions_to_ctrl_batch(
            session.actions,
            nu=session.model.nu,
            ctrl_indices=session.env._ctrl_indices,
            gripper_indices=session.env._gripper_indices,
        )
    physics_substeps = _infer_physics_substeps_per_action(session)

    frames: list[np.ndarray] = []
    if include_initial_frame and (max_frames is None or len(frames) < max_frames):
        frames.append(np.asarray(runtime.d_warp.qpos.numpy()[0], dtype=np.float32).copy())

    steps = 0
    for ctrl in ctrl_frames:
        if max_frames is not None and len(frames) >= max_frames:
            break
        runtime.set_ctrl_batch(ctrl[None, :])
        runtime.step(nstep=physics_substeps)
        frames.append(np.asarray(runtime.d_warp.qpos.numpy()[0], dtype=np.float32).copy())
        steps += 1
        if progress_every > 0 and steps % progress_every == 0:
            print(f"  Rolled out {steps}/{session.total_steps} mjwarp physics steps")

    if not frames:
        raise RuntimeError("MJWarp physics rollout produced no qpos frames")
    return np.stack(frames, axis=0)


def export_batched_qpos_sim_video(
    session,
    *,
    output_path: Path,
    fps: float,
    sim_backend: Literal["mjwarp", "madrona"],
    render_width: int,
    render_height: int,
    batch_size: int,
    gpu_id: int | None = None,
    include_initial_frame: bool = True,
    progress_every: int = 100,
    max_frames: int | None = None,
) -> int:
    """Render exact-qpos replay in batches with MJWarp or Madrona."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if (
        session.mode != "qpos"
        or not session.has_exact_qpos
        or session.sim_state_alignment != "initial"
    ):
        raise RuntimeError("Batched sim export requires qpos replay mode with aligned exact states")

    session.reset()
    if session.sim_integration_states is not None:
        qpos_frames = _qpos_frames_from_integration_states(
            session,
            session.sim_integration_states,
        )
    elif session.sim_states is not None:
        qpos_frames = session.sim_states
    else:
        raise RuntimeError("Batched sim export requires aligned sim states")
    qpos_frames = qpos_frames if include_initial_frame else qpos_frames[1:]
    source_ts = _infer_qpos_frame_timestamps(
        session,
        frame_count=len(qpos_frames),
        include_initial_frame=include_initial_frame,
    )
    source_limit = _source_frame_limit_for_output(
        source_ts,
        fps=fps,
        max_output_frames=max_frames,
    )
    qpos_frames = qpos_frames[:source_limit]
    source_ts = source_ts[:source_limit]
    qpos_frames = _sample_qpos_frames_for_video(
        qpos_frames,
        source_ts,
        fps=fps,
        max_output_frames=max_frames,
    )
    return _render_batched_qpos_frames(
        session,
        qpos_frames=qpos_frames,
        output_path=output_path,
        fps=fps,
        sim_backend=sim_backend,
        render_width=render_width,
        render_height=render_height,
        batch_size=batch_size,
        gpu_id=gpu_id,
        progress_every=progress_every,
    )


def export_batched_physics_sim_video(
    session,
    *,
    output_path: Path,
    fps: float,
    sim_backend: Literal["mjwarp", "madrona"],
    physics_backend: Literal["mujoco", "mjwarp"] = "mujoco",
    render_width: int,
    render_height: int,
    batch_size: int,
    gpu_id: int | None = None,
    include_initial_frame: bool = True,
    progress_every: int = 100,
    max_frames: int | None = None,
) -> int:
    """Run physics replay once, capture qpos at each step, and render via batched sim rendering."""
    if physics_backend == "mujoco":
        source_ts = _infer_qpos_frame_timestamps(
            session,
            frame_count=(len(session.grid_ts) if include_initial_frame else max(0, len(session.grid_ts) - 1)),
            include_initial_frame=include_initial_frame,
        )
        source_limit = _source_frame_limit_for_output(
            source_ts,
            fps=fps,
            max_output_frames=max_frames,
        )
        qpos_frames = collect_physics_rollout_qpos_frames(
            session,
            include_initial_frame=include_initial_frame,
            max_frames=source_limit,
            progress_every=progress_every,
        )
    elif physics_backend == "mjwarp":
        source_ts = _infer_qpos_frame_timestamps(
            session,
            frame_count=(len(session.grid_ts) if include_initial_frame else max(0, len(session.grid_ts) - 1)),
            include_initial_frame=include_initial_frame,
        )
        source_limit = _source_frame_limit_for_output(
            source_ts,
            fps=fps,
            max_output_frames=max_frames,
        )
        qpos_frames = collect_mjwarp_rollout_qpos_frames(
            session,
            include_initial_frame=include_initial_frame,
            max_frames=source_limit,
            progress_every=progress_every,
            gpu_id=gpu_id,
        )
    else:
        raise ValueError(f"Unsupported physics backend: {physics_backend!r}")
    source_ts = _infer_qpos_frame_timestamps(
        session,
        frame_count=len(qpos_frames),
        include_initial_frame=include_initial_frame,
    )
    qpos_frames = _sample_qpos_frames_for_video(
        qpos_frames,
        source_ts,
        fps=fps,
        max_output_frames=max_frames,
    )
    return _render_batched_qpos_frames(
        session,
        qpos_frames=qpos_frames,
        output_path=output_path,
        fps=fps,
        sim_backend=sim_backend,
        render_width=render_width,
        render_height=render_height,
        batch_size=batch_size,
        gpu_id=gpu_id,
        progress_every=progress_every,
    )


def export_replay_video(
    session,
    camera_provider,
    *,
    output_path: Path,
    fps: float,
    include_initial_frame: bool = True,
    progress_every: int = 100,
    max_frames: int | None = None,
) -> int:
    """Step a replay session and write a tiled camera video."""
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided")

    session.reset()
    camera_provider.reset()

    frames: list[np.ndarray] = []
    if include_initial_frame and (max_frames is None or len(frames) < max_frames):
        initial_frames = camera_provider.initial_frames()
        if not initial_frames:
            raise RuntimeError("No camera frames available for the initial replay frame")
        frames.append(tile_camera_frames(initial_frames, camera_provider.camera_names))

    steps = 0
    while (max_frames is None or len(frames) < max_frames) and session.step():
        frame_dict = camera_provider.frames_for_step(session.current_frame_idx, session.current_timestamp)
        if not frame_dict:
            raise RuntimeError("Camera provider returned no frames during replay")
        frames.append(tile_camera_frames(frame_dict, camera_provider.camera_names))
        steps += 1
        if progress_every > 0 and steps % progress_every == 0:
            print(f"  Step {steps}/{session.total_steps}")

    write_video(output_path, frames, fps=fps)
    return len(frames)
