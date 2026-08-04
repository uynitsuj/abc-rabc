"""Render aligned qpos trajectories into dataset camera videos."""

from __future__ import annotations

from contextlib import ExitStack
import os
from pathlib import Path

# Headless dataset export should default to EGL before MuJoCo is imported.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from yam_sim.dataset_export.types import ExportConfig, ExportTrajectory
from yam_sim.dataset_export.video_io import RawRGBVideoWriter, write_frame_mappings
from yam_sim.dataset_export.writer import camera_video_name, combined_video_name


def _resolve_camera_indices(model, camera_names: tuple[str, ...]) -> list[int]:
    available_names = tuple(model.cam(i).name for i in range(model.ncam))
    missing = [name for name in camera_names if name not in available_names]
    if missing:
        raise RuntimeError(f"Requested cameras missing from model: {missing}")
    return [available_names.index(name) for name in camera_names]


def render_episode_videos(
    trajectory: ExportTrajectory,
    env,
    output_dir: Path,
    *,
    config: ExportConfig,
) -> dict[str, Path]:
    """Render per-camera and combined videos for one aligned trajectory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    video_paths = {
        name: output_dir / camera_video_name(name) for name in trajectory.camera_names
    }
    video_paths["combined"] = output_dir / combined_video_name()

    with ExitStack() as stack:
        writers = {
            name: stack.enter_context(
                RawRGBVideoWriter(
                    video_paths[name],
                    fps=config.fps,
                    width=config.image_width,
                    height=config.image_height,
                )
            )
            for name in trajectory.camera_names
        }
        combined_writer = stack.enter_context(
            RawRGBVideoWriter(
                video_paths["combined"],
                fps=config.fps,
                width=config.image_width,
                height=config.image_height * len(trajectory.camera_names),
            )
        )

        if config.render_backend == "mujoco":
            _render_with_mujoco(trajectory, env, writers, combined_writer)
        else:
            _render_with_gpu_backend(trajectory, env, writers, combined_writer, config=config)

    for path in video_paths.values():
        write_frame_mappings(path)
    return video_paths


def _render_with_mujoco(
    trajectory: ExportTrajectory,
    env,
    writers: dict[str, RawRGBVideoWriter],
    combined_writer: RawRGBVideoWriter,
) -> None:
    if not hasattr(env, "renderer"):
        raise RuntimeError("MuJoCo rendering requires an env with render_cameras=True")
    for qpos in trajectory.qpos:
        env.data.qpos[: len(qpos)] = qpos
        mujoco.mj_forward(env.model, env.data)
        ordered_frames = []
        for name in trajectory.camera_names:
            env.renderer.update_scene(env.data, camera=name)
            frame = env.renderer.render().copy()
            writers[name].write_frame(frame)
            ordered_frames.append(frame)
        combined_writer.write_frame(np.concatenate(ordered_frames, axis=0))


def _render_with_gpu_backend(
    trajectory: ExportTrajectory,
    env,
    writers: dict[str, RawRGBVideoWriter],
    combined_writer: RawRGBVideoWriter,
    *,
    config: ExportConfig,
) -> None:
    from yam_sim.rendering.replay.renderer import RendererWrapper, WarpReplayRuntime

    batch_worlds = min(max(1, config.sim_batch_size), len(trajectory.qpos))
    runtime = WarpReplayRuntime(env.model, env.data, nworld=batch_worlds, gpu_id=config.gpu_id)
    qpos_frames_device = runtime.upload_qpos_frames(trajectory.qpos)
    renderer = RendererWrapper(
        backend=config.render_backend,
        runtime=runtime,
        cam_res=(config.image_width, config.image_height),
        gpu_id=config.gpu_id,
    )
    camera_indices = _resolve_camera_indices(env.model, trajectory.camera_names)
    first_batch = True

    try:
        for start in range(0, len(trajectory.qpos), batch_worlds):
            stop = min(start + batch_worlds, len(trajectory.qpos))
            actual = stop - start
            runtime.load_qpos_batch_from_device(qpos_frames_device, start=start, stop=stop)
            runtime.forward()
            rendered = (
                renderer.reset_numpy(actual_batch=actual)
                if first_batch
                else renderer.render_numpy(actual_batch=actual)
            )
            first_batch = False
            selected = rendered[:, camera_indices]
            for idx, name in enumerate(trajectory.camera_names):
                writers[name].write_batch(selected[:, idx])
            combined_writer.write_batch(
                np.concatenate([selected[:, idx] for idx in range(len(trajectory.camera_names))], axis=1)
            )
    finally:
        renderer.close()
