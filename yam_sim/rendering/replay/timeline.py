"""Replay timeline alignment helpers."""

from __future__ import annotations

import numpy as np

from yam_sim.rendering.replay.types import EpisodeContext, ReplayTimeline


def _as_timestamp_array(timestamps: np.ndarray, *, label: str) -> np.ndarray:
    ts = np.asarray(timestamps, dtype=np.float64)
    if ts.ndim != 1:
        raise ValueError(f"Expected 1D {label} timestamps, got {ts.shape}")
    if len(ts) == 0:
        raise ValueError(f"Episode context contains no {label} timestamps")
    if not np.all(np.isfinite(ts)):
        raise ValueError(f"{label} timestamps contain non-finite values")
    if np.any(np.diff(ts) < -1e-9):
        raise ValueError(f"{label} timestamps must be monotonically non-decreasing")
    return ts


def _relative_action_clock_origin(context: EpisodeContext) -> float | None:
    if context.episode_format != "delivered":
        return None
    starts = []
    for ts in (context.streams.ts_left, context.streams.ts_right):
        arr = np.asarray(ts, dtype=np.float64)
        if len(arr) > 0:
            starts.append(float(arr[0]))
    if not starts:
        return None
    origin = min(starts)
    if origin >= 1e6:
        return None
    return origin


def _validate_elapsed_consistency(
    sim_timestamps: np.ndarray,
    wallclock_timestamps: np.ndarray,
    *,
    label: str,
) -> None:
    wallclock = _as_timestamp_array(wallclock_timestamps, label=f"{label} wallclock")
    if len(wallclock) != len(sim_timestamps):
        raise ValueError(
            f"{label} wallclock length mismatch: {len(wallclock)} vs {len(sim_timestamps)}"
        )
    sim_elapsed = sim_timestamps - sim_timestamps[0]
    wallclock_elapsed = wallclock - wallclock[0]
    positive_steps = np.diff(sim_timestamps)
    positive_steps = positive_steps[positive_steps > 1e-9]
    median_step = float(np.median(positive_steps)) if len(positive_steps) else 0.0
    tolerance_s = max(0.05, 2.0 * median_step)
    max_error_s = float(np.max(np.abs(sim_elapsed - wallclock_elapsed)))
    if max_error_s > tolerance_s:
        raise ValueError(
            f"{label} sim-time and wallclock elapsed clocks disagree: "
            f"max_error={max_error_s:.6f}s, tolerance={tolerance_s:.6f}s"
        )


def normalize_delivered_integration_timestamps(context: EpisodeContext) -> np.ndarray | None:
    """Normalize integration_state_sim_time.npy onto the delivered action clock.

    Delivered action MCAP timestamps are relative to the episode start. The
    integration-state sim-time clock may start at an arbitrary simulator offset,
    so for delivered episodes we zero-base it against its own first sample and
    place it on the same relative action clock. Missing timestamps are not
    inferred from qpos streams or array lengths.
    """
    if context.raw_sim_integration_states is None:
        return None

    raw_ts = context.raw_sim_integration_timestamps
    if raw_ts is None:
        raise ValueError(
            "Delivered integration_state.npy requires integration_state_sim_time.npy; "
            "refusing to infer timestamps from qpos streams or array lengths"
        )

    ts = _as_timestamp_array(raw_ts, label="integration-state")
    action_origin = _relative_action_clock_origin(context)
    if action_origin is None:
        return ts

    if context.raw_sim_integration_wallclock_timestamps is None:
        raise ValueError(
            "Delivered integration_state.npy requires integration_state_wallclock.npy "
            "for clock validation"
        )
    _validate_elapsed_consistency(
        ts,
        context.raw_sim_integration_wallclock_timestamps,
        label="integration_state.npy",
    )
    return ts - ts[0] + action_origin


def sample_hold_align(values: np.ndarray, timestamps: np.ndarray, query_ts: np.ndarray) -> np.ndarray:
    """Align samples to query timestamps using zero-order hold."""
    if len(values) != len(timestamps):
        raise ValueError("values and timestamps must have the same length")
    if len(query_ts) == 0:
        return np.empty((0, *values.shape[1:]), dtype=values.dtype)
    idx = np.searchsorted(timestamps, query_ts, side="right") - 1
    idx = np.clip(idx, 0, len(values) - 1)
    return values[idx]


def build_action_timeline(
    actions_left: np.ndarray,
    ts_left: np.ndarray,
    actions_right: np.ndarray,
    ts_right: np.ndarray,
    control_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample async left/right action streams onto a regular control grid."""
    t_start = min(ts_left[0], ts_right[0])
    t_end = min(ts_left[-1], ts_right[-1])
    grid_ts = np.arange(t_start, t_end, 1.0 / control_hz)

    left_aligned = sample_hold_align(actions_left, ts_left, grid_ts)
    right_aligned = sample_hold_align(actions_right, ts_right, grid_ts)
    actions = np.concatenate([left_aligned.astype(np.float32), right_aligned.astype(np.float32)], axis=1)
    return actions, grid_ts


def align_sim_states(
    context: EpisodeContext,
    grid_ts: np.ndarray,
) -> np.ndarray | None:
    """Align raw qpos samples to the replay clock."""
    raw_qposes = context.raw_sim_states
    raw_timestamps = context.raw_sim_timestamps
    if raw_qposes is None or raw_timestamps is None:
        return None
    aligned_ts = raw_timestamps
    # Delivered episodes use a relative action clock in output.mcap but absolute
    # Unix timestamps in sim_state.mcap. Shift the qpos stream onto the same
    # zero-based timebase before sample-and-hold alignment.
    if (
        context.episode_format == "delivered"
        and len(aligned_ts) > 0
        and len(grid_ts) > 0
        and aligned_ts[0] > 1e9
        and grid_ts[0] < 1e6
    ):
        aligned_ts = aligned_ts - aligned_ts[0]
    return sample_hold_align(raw_qposes, aligned_ts, grid_ts)


def align_sim_integration_states(
    context: EpisodeContext,
    grid_ts: np.ndarray,
) -> np.ndarray | None:
    """Align mjSTATE_INTEGRATION snapshots to the replay clock."""
    states = context.raw_sim_integration_states
    if states is None:
        return None

    aligned_ts = normalize_delivered_integration_timestamps(context)
    if aligned_ts is None:
        return None
    return sample_hold_align(states, aligned_ts, grid_ts)


def build_replay_timeline(context: EpisodeContext, control_hz: float) -> ReplayTimeline:
    """Build the aligned action/qpos timeline used by the viewer."""
    if context.replay_actions is not None and context.replay_timestamps is not None:
        return ReplayTimeline(
            actions=context.replay_actions.astype(np.float32, copy=False),
            grid_ts=np.asarray(context.replay_timestamps, dtype=np.float64),
            sim_states=context.raw_sim_states,
            sim_integration_states=context.raw_sim_integration_states,
            sim_state_kind=context.replay_state_kind,
            sim_state_alignment=context.replay_state_alignment,
        )

    actions, grid_ts = build_action_timeline(
        context.streams.actions_left,
        context.streams.ts_left,
        context.streams.actions_right,
        context.streams.ts_right,
        control_hz=control_hz,
    )
    sim_states = align_sim_states(context, grid_ts)
    sim_integration_states = align_sim_integration_states(context, grid_ts)
    return ReplayTimeline(
        actions=actions,
        grid_ts=grid_ts,
        sim_states=sim_states,
        sim_integration_states=sim_integration_states,
        sim_state_kind=context.replay_state_kind,
        sim_state_alignment=context.replay_state_alignment,
    )
