"""Headless rendering utilities for the MuJoCo YAM environment."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from yam_sim.env import MuJoCoYAMEnv


def render_cameras(env: MuJoCoYAMEnv) -> dict[str, np.ndarray]:
    """Render all cameras, return {name: (H, W, 3) uint8}."""
    images = {}
    for name in env.camera_names:
        env.renderer.update_scene(env.data, camera=name)
        img = env.renderer.render()  # (H, W, 3) uint8
        images[name] = img.copy()
    return images


def save_camera_images(env: MuJoCoYAMEnv, output_dir: str, step: int = 0):
    """Render and save each camera as a PNG file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    images = render_cameras(env)
    for name, img in images.items():
        Image.fromarray(img).save(output_path / f"{name}_step{step:04d}.png")


def save_camera_grid(env: MuJoCoYAMEnv, output_path: str):
    """Tile all camera views into a single image and save."""
    images = render_cameras(env)
    imgs = [images[name] for name in env.camera_names]
    grid = np.concatenate(imgs, axis=1)
    Image.fromarray(grid).save(output_path)


def record_episode(
    env: MuJoCoYAMEnv,
    actions: np.ndarray,
    output_path: str,
    fps: int = 30,
):
    """Step through actions and save all camera views as MP4 video.

    Args:
        env: The MuJoCo YAM environment (should already be reset).
        actions: Array of shape (num_chunks, chunk_dim, 14) or (num_chunks, chunk_dim * 14).
        output_path: Path to save the output MP4 video.
        fps: Frames per second for the video.
    """
    try:
        import imageio.v3 as iio
    except ImportError:
        print("imageio not available. Install with: pip install imageio[ffmpeg]")
        output_dir = Path(output_path).with_suffix("")
        output_dir.mkdir(parents=True, exist_ok=True)
        step = 0
        for chunk_idx in range(len(actions)):
            chunk = actions[chunk_idx].reshape(
                env.chunk_dim, env.single_timestep_action_dim
            )
            for t in range(env.chunk_dim):
                env._step_single(chunk[t])
                save_camera_images(env, str(output_dir), step=step)
                step += 1
        print(f"Saved {step} frames to {output_dir}/")
        return

    frames = []
    for chunk_idx in range(len(actions)):
        chunk = actions[chunk_idx].reshape(
            env.chunk_dim, env.single_timestep_action_dim
        )
        for t in range(env.chunk_dim):
            env._step_single(chunk[t])
            images = render_cameras(env)
            grid = np.concatenate(
                [images[name] for name in env.camera_names], axis=1
            )
            frames.append(grid)

    iio.imwrite(output_path, np.stack(frames), fps=fps)
    print(f"Saved video ({len(frames)} frames) to {output_path}")
