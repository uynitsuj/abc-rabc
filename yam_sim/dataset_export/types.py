"""Typed containers for dataset export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

RenderBackend = Literal["mujoco", "mjwarp", "madrona"]

DEFAULT_IMAGE_WIDTH = 224
DEFAULT_IMAGE_HEIGHT = 168


@dataclass(frozen=True)
class ExportConfig:
    """Config shared across single-episode and batch dataset export."""

    batch_name: str
    fps: float = 30.0
    image_width: int = DEFAULT_IMAGE_WIDTH
    image_height: int = DEFAULT_IMAGE_HEIGHT
    render_backend: RenderBackend = "mjwarp"
    sim_batch_size: int = 32
    gpu_id: int | None = None
    state_pad_size: int = 32
    action_pad_size: int = 32


@dataclass(frozen=True)
class ExportTrajectory:
    """Canonical aligned trajectory ready for video/data export."""

    episode_id: str
    source_episode_dir: Path
    source_delivery: str
    scene: str
    task: str
    task_name: str
    instruction: str | None
    data_date: str
    camera_names: tuple[str, ...]
    timestamps: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    qpos: np.ndarray
    initial_qpos: np.ndarray
    scene_source: str | None = None
    state_source: str | None = None


@dataclass(frozen=True)
class EpisodeArtifacts:
    """Filesystem outputs written for one exported episode."""

    episode_dir: Path
    states_actions_path: Path
    states_actions_bin_path: Path
    initial_qpos_path: Path
    episode_metadata_path: Path
    video_paths: dict[str, Path]
