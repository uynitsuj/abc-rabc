"""Canonical trajectory building for dataset export."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from yam_sim.dataset_export.metadata import (
    data_date_from_timestamps,
    episode_id_from_dir,
    normalize_task_name,
)
from yam_sim.dataset_export.types import ExportTrajectory
from yam_sim.rendering.replay.timeline import (
    normalize_delivered_integration_timestamps,
    sample_hold_align,
)
from yam_sim.rendering.replay.types import EpisodeContext


def normalize_sim_timestamps(context: EpisodeContext) -> np.ndarray:
    """Shift raw sim timestamps onto the action clock when needed."""
    if context.raw_sim_timestamps is None:
        raise ValueError("Episode context does not contain raw sim timestamps")
    raw_ts = np.asarray(context.raw_sim_timestamps, dtype=np.float64)
    if len(raw_ts) == 0:
        raise ValueError("Episode context contains no raw sim timestamps")
    if (
        context.episode_format == "delivered"
        and raw_ts[0] > 1e9
        and len(context.streams.ts_left) > 0
        and context.streams.ts_left[0] < 1e6
    ):
        return raw_ts - raw_ts[0]
    return raw_ts


def normalize_integration_timestamps(context: EpisodeContext) -> np.ndarray | None:
    """Shift integration-state timestamps onto the action clock when present."""
    if context.raw_sim_integration_states is None:
        return None
    return normalize_delivered_integration_timestamps(context)


def build_export_grid(
    *,
    starts: list[float],
    ends: list[float],
    fps: float,
) -> np.ndarray:
    """Build an evenly sampled export grid over the shared valid window."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not starts or not ends:
        raise ValueError("starts and ends must be non-empty")

    window_start = float(max(starts))
    window_end = float(min(ends))
    if window_end < window_start:
        raise ValueError(
            f"No overlapping export window: start={window_start}, end={window_end}"
        )

    duration = window_end - window_start
    if duration <= 0:
        return np.array([window_start], dtype=np.float64)

    num_steps = max(1, int(np.floor(duration * fps + 1e-9)))
    return window_start + np.arange(num_steps, dtype=np.float64) / float(fps)


def _extract_qpos_from_integration_states(
    env,
    integration_states: np.ndarray,
    state_spec: int | None,
) -> np.ndarray:
    if state_spec is None:
        raise ValueError("integration_state.npy is present but the MuJoCo state spec is missing")

    states = np.asarray(integration_states, dtype=np.float64)
    expected_size = mujoco.mj_stateSize(env.model, state_spec)
    if states.ndim != 2 or states.shape[1] != expected_size:
        raise ValueError(
            f"integration_state.npy shape {states.shape} does not match "
            f"mj_stateSize(model, {state_spec})={expected_size}"
        )

    qpos = np.empty((len(states), env.model.nq), dtype=np.float64)
    for idx, state in enumerate(states):
        mujoco.mj_setState(env.model, env.data, state, state_spec)
        mujoco.mj_forward(env.model, env.data)
        qpos[idx] = np.asarray(env.data.qpos, dtype=np.float64).copy()
    return qpos


def build_export_trajectory(
    context: EpisodeContext,
    env,
    *,
    fps: float,
    episode_id: str | None = None,
    source_delivery: str | None = None,
) -> ExportTrajectory:
    """Build aligned states/actions/qpos on the requested export clock."""
    integration_ts = normalize_integration_timestamps(context)
    if context.raw_sim_integration_states is None:
        raise ValueError(
            "Dataset export/render requires integration_state.npy; "
            "refusing to render from sim_state.mcap qpos."
        )
    if integration_ts is None:
        raise ValueError(
            "Dataset export/render requires integration_state.npy timestamps; "
            "refusing to render from sim_state.mcap qpos."
        )
    state_ts = integration_ts

    grid_ts = build_export_grid(
        starts=[
            float(context.streams.ts_left[0]),
            float(context.streams.ts_right[0]),
            float(state_ts[0]),
        ],
        ends=[
            float(context.streams.ts_left[-1]),
            float(context.streams.ts_right[-1]),
            float(state_ts[-1]),
        ],
        fps=fps,
    )

    left = sample_hold_align(context.streams.actions_left, context.streams.ts_left, grid_ts)
    right = sample_hold_align(context.streams.actions_right, context.streams.ts_right, grid_ts)
    actions = np.concatenate(
        [left.astype(np.float32, copy=False), right.astype(np.float32, copy=False)],
        axis=1,
    )
    aligned_integration_states = sample_hold_align(
        np.asarray(context.raw_sim_integration_states, dtype=np.float64),
        integration_ts,
        grid_ts,
    )
    qpos = _extract_qpos_from_integration_states(
        env,
        aligned_integration_states,
        context.raw_sim_state_spec,
    )
    state_source = "integration_state.npy"

    aligned_qpos = qpos.astype(np.float32, copy=False)
    states = np.stack(
        [env.project_state_from_qpos(frame) for frame in aligned_qpos],
        axis=0,
    ).astype(np.float32, copy=False)

    source_episode_dir = Path(context.streams.episode_dir)
    export_episode_id = episode_id or episode_id_from_dir(source_episode_dir)
    task_name = normalize_task_name(context.instruction, context.task)
    return ExportTrajectory(
        episode_id=export_episode_id,
        source_episode_dir=source_episode_dir,
        source_delivery=source_delivery or source_episode_dir.parent.name,
        scene=context.scene,
        task=context.task,
        task_name=task_name,
        instruction=context.instruction,
        data_date=data_date_from_timestamps(context.raw_sim_timestamps),
        camera_names=tuple(env.camera_names),
        timestamps=grid_ts,
        states=states,
        actions=actions,
        qpos=aligned_qpos,
        initial_qpos=aligned_qpos[0].copy(),
        scene_source=getattr(env, "_replay_scene_source", context.scene_source),
        state_source=state_source,
    )
