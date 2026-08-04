"""Typed replay data containers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Literal

import numpy as np


EpisodeFormat = Literal["raw", "delivered", "dataset", "recorded"]
ReplayStateKind = Literal["qpos", "policy_state"]
ReplayStateAlignment = Literal["initial", "post_step"]


@dataclass(frozen=True)
class EpisodeStreams:
    """Loaded episode streams used to build a replay."""

    episode_dir: Path
    actions_left: np.ndarray
    ts_left: np.ndarray
    actions_right: np.ndarray
    ts_right: np.ndarray
    camera_frames: dict[str, np.ndarray]
    camera_ts: dict[str, np.ndarray]


@dataclass(frozen=True)
class EpisodeContext:
    """Episode data plus sim-specific metadata."""

    streams: EpisodeStreams
    episode_format: EpisodeFormat
    scene: str
    task: str
    instruction: str | None
    rand_state: Any | None
    raw_sim_states: np.ndarray | None
    raw_sim_timestamps: np.ndarray | None
    scene_xml_string: str | None = None
    scene_source: str | None = None
    raw_sim_integration_states: np.ndarray | None = None
    raw_sim_integration_timestamps: np.ndarray | None = None
    raw_sim_integration_wallclock_timestamps: np.ndarray | None = None
    raw_sim_state_spec: int | None = None
    raw_sim_ctrls: np.ndarray | None = None
    initial_scene_qpos: np.ndarray | None = None
    initial_scene_integration_state: np.ndarray | None = None
    raw_sim_qvels: np.ndarray | None = None
    raw_sim_acts: np.ndarray | None = None
    raw_sim_mocap_pos: np.ndarray | None = None
    raw_sim_mocap_quat: np.ndarray | None = None
    initial_scene_qvel: np.ndarray | None = None
    initial_scene_act: np.ndarray | None = None
    initial_scene_mocap_pos: np.ndarray | None = None
    initial_scene_mocap_quat: np.ndarray | None = None
    replay_actions: np.ndarray | None = None
    replay_ctrls: np.ndarray | None = None
    replay_timestamps: np.ndarray | None = None
    replay_state_kind: ReplayStateKind = "qpos"
    replay_state_alignment: ReplayStateAlignment = "initial"
    physics_overrides: dict[str, Any] | None = None


@dataclass(frozen=True)
class ReplayTimeline:
    """Aligned replay arrays on a single playback clock."""

    actions: np.ndarray
    grid_ts: np.ndarray
    sim_states: np.ndarray | None
    sim_integration_states: np.ndarray | None = None
    sim_state_kind: ReplayStateKind = "qpos"
    sim_state_alignment: ReplayStateAlignment = "initial"
