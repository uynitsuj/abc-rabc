"""Episode loading helpers for raw, delivered, recorded, and dataset episodes."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from yam_sim.rendering.replay.types import EpisodeContext, EpisodeFormat, EpisodeStreams
from yam_sim.rendering.replay.timeline import sample_hold_align
from yam_sim.task_specs import maybe_get_task_spec

_SWEEP_OLD_QPOS_DIM = 79
_SWEEP_CURRENT_QPOS_DIM = 86
_SWEEP_TRASH7_QPOS_ADR = 63
_SWEEP_PARKED_TRASH7_QPOS = np.array([-1.5, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
_SCENE_ASSET_ROOT_RE = re.compile(r"/(?:[^/\"'<>\s]+/)*(?:yam_sim|xdof_sim)/models/assets/?")


def detect_episode_format(episode_dir: Path) -> EpisodeFormat:
    """Detect whether an episode uses the raw, delivered, recorded, or dataset layout."""
    if (episode_dir / "output.mcap").exists():
        return "delivered"
    if (
        ((episode_dir / "actions_full.npy").exists() or (episode_dir / "actions.npy").exists())
        and (episode_dir / "qpos.npy").exists()
        and (episode_dir / "config.json").exists()
    ):
        return "recorded"
    if (
        (episode_dir / "states_actions.bin").exists()
        or (episode_dir / "states_actions.npy").exists()
        or (episode_dir / "episode_metadata.json").exists()
    ):
        return "dataset"
    return "raw"


def _resolve_bin_dtype(bin_path: Path, row_dim: int) -> np.dtype:
    file_size = bin_path.stat().st_size
    if file_size % (row_dim * 8) == 0:
        return np.dtype(np.float64)
    if file_size % (row_dim * 4) == 0:
        return np.dtype(np.float32)
    raise ValueError(
        f"Could not infer dtype for {bin_path}. "
        f"File size {file_size} is not divisible by row_dim={row_dim} for float32/float64."
    )


def _load_dataset_states_actions(
    episode_dir: Path,
    *,
    n_states: int = 14,
    n_actions: int = 14,
) -> tuple[np.ndarray, np.ndarray]:
    row_dim = n_states + n_actions
    npy_path = episode_dir / "states_actions.npy"
    bin_path = episode_dir / "states_actions.bin"

    if npy_path.exists():
        arr = np.load(npy_path)
    elif bin_path.exists():
        dtype = _resolve_bin_dtype(bin_path, row_dim=row_dim)
        arr = np.fromfile(bin_path, dtype=dtype)
        if arr.size % row_dim != 0:
            raise ValueError(
                f"states_actions.bin at {bin_path} has {arr.size} scalars, "
                f"which is not divisible by row_dim={row_dim}"
            )
        arr = arr.reshape(-1, row_dim)
    else:
        raise FileNotFoundError(
            f"Expected {npy_path.name} or {bin_path.name} in dataset episode directory {episode_dir}"
        )

    if arr.ndim != 2 or arr.shape[1] < row_dim:
        raise ValueError(f"Expected states/actions with shape (T, >= {row_dim}), got {arr.shape}")

    states = np.asarray(arr[:, :n_states], dtype=np.float32)
    actions = np.asarray(arr[:, n_states : n_states + n_actions], dtype=np.float32)
    return states, actions


def _is_sweep_task(task: str | None) -> bool:
    if not task:
        return False
    if task == "sweep":
        return True
    spec = maybe_get_task_spec(task)
    return spec is not None and spec.env_task == "sweep"


def _upgrade_sweep_qpos_layout(qpos: np.ndarray) -> np.ndarray:
    """Map old 79D sweep qpos into the current 86D layout by inserting parked trash_7."""
    if qpos.ndim == 1:
        width = len(qpos)
    elif qpos.ndim == 2:
        width = qpos.shape[1]
    else:
        raise ValueError(f"Expected sweep qpos rank 1 or 2, got shape {qpos.shape}")

    if width == _SWEEP_CURRENT_QPOS_DIM:
        return qpos
    if width != _SWEEP_OLD_QPOS_DIM:
        return qpos

    parked = _SWEEP_PARKED_TRASH7_QPOS.astype(qpos.dtype, copy=False)
    if qpos.ndim == 1:
        return np.concatenate(
            [
                qpos[:_SWEEP_TRASH7_QPOS_ADR],
                parked,
                qpos[_SWEEP_TRASH7_QPOS_ADR:],
            ],
            axis=0,
        )

    parked_rows = np.broadcast_to(parked, (qpos.shape[0], len(parked)))
    return np.concatenate(
        [
            qpos[:, :_SWEEP_TRASH7_QPOS_ADR],
            parked_rows,
            qpos[:, _SWEEP_TRASH7_QPOS_ADR:],
        ],
        axis=1,
    )


def _maybe_upgrade_exact_qpos_for_task(
    task: str | None,
    qpos: np.ndarray | None,
) -> np.ndarray | None:
    if qpos is None or not _is_sweep_task(task):
        return qpos
    return _upgrade_sweep_qpos_layout(qpos)


def _load_json(path: Path) -> dict[str, Any] | None:
    import json

    if not path.exists():
        return None
    with open(path) as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else None


def _local_scene_asset_root() -> Path:
    import yam_sim

    return Path(yam_sim.__file__).resolve().parent / "models" / "assets"


def _local_scene_models_root() -> Path:
    return _local_scene_asset_root().parent


def _normalize_recorded_scene_xml(xml: str) -> str:
    """Make recorded XML asset paths resolvable in the current checkout."""
    local_asset_root = _local_scene_asset_root().as_posix().rstrip("/") + "/"
    xml = _SCENE_ASSET_ROOT_RE.sub(local_asset_root, xml)
    robocasa_root = (_local_scene_models_root() / "assets_robocasa").as_posix().rstrip("/") + "/"

    # Older delivered XMLs used human-readable dishrack asset directory names
    # from the collection checkout. This repo stores the same assets under the
    # canonical variant directories used by the randomizer.
    try:
        from yam_sim.randomization import _DISHRACK_VARIANT_ALIASES
    except Exception:
        _DISHRACK_VARIANT_ALIASES = {}

    for kind, aliases in _DISHRACK_VARIANT_ALIASES.items():
        for alias, canonical in aliases.items():
            xml = xml.replace(
                f"task_dishrack/{kind}/{alias}/",
                f"task_dishrack/{kind}/{canonical}/",
            )

    # Some delivered episodes reference task-local dishrack variant ids that
    # were not copied into this checkout's task_dishrack namespace. Resolve
    # those ids directly against the local RoboCasa asset checkout.
    legacy_asset_prefixes = {
        "task_dishrack/dish_rack/dish_rack_10/": (
            "objects_lightwheel/lightwheel/dish_rack/DishRack044/"
        ),
        "task_dishrack/plate/plate_19/": (
            "objaverse/objaverse/plate/plate_19/"
        ),
        "task_dishrack/plate/plate_20/": (
            "objaverse/objaverse/plate/plate_20/"
        ),
    }
    for legacy_prefix, robocasa_prefix in legacy_asset_prefixes.items():
        replacement_prefix = robocasa_root + robocasa_prefix
        xml = xml.replace(local_asset_root + legacy_prefix, replacement_prefix)
        xml = xml.replace(f'file="{legacy_prefix}', f'file="{replacement_prefix}')
    return xml


def load_scene_assembled_xml(episode_dir: Path) -> str | None:
    """Load the recorded assembled scene XML, if present."""
    path = episode_dir / "scene_assembled.xml"
    if not path.exists():
        return None
    xml = path.read_text()
    if not xml.strip():
        raise ValueError(f"scene_assembled.xml is empty in {episode_dir}")
    return _normalize_recorded_scene_xml(xml)


def _mj_state_integration_spec() -> int:
    import mujoco

    return int(mujoco.mjtState.mjSTATE_INTEGRATION.value)


def load_integration_states(
    episode_dir: Path,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, int | None]:
    """Load delivered mjSTATE_INTEGRATION snapshots when present."""
    path = episode_dir / "integration_state.npy"
    if not path.exists():
        return None, None, None, None

    states = np.asarray(np.load(path))
    if states.ndim != 2 or len(states) == 0:
        raise ValueError(f"Expected integration_state.npy shape (T, state_size), got {states.shape} from {path}")

    timestamps = None
    sim_time_path = episode_dir / "integration_state_sim_time.npy"
    if sim_time_path.exists():
        timestamps = np.asarray(np.load(sim_time_path), dtype=np.float64)
        if timestamps.ndim != 1:
            raise ValueError(f"Expected 1D integration timestamps in {sim_time_path}, got {timestamps.shape}")
        if len(timestamps) != len(states):
            raise ValueError(
                f"integration_state_sim_time.npy length mismatch: {len(timestamps)} vs {len(states)} "
                f"in {episode_dir}"
            )

    wallclock_timestamps = None
    wallclock_path = episode_dir / "integration_state_wallclock.npy"
    if wallclock_path.exists():
        wallclock_timestamps = np.asarray(np.load(wallclock_path), dtype=np.float64)
        if wallclock_timestamps.ndim != 1:
            raise ValueError(f"Expected 1D integration wallclock timestamps in {wallclock_path}, got {wallclock_timestamps.shape}")
        if len(wallclock_timestamps) != len(states):
            raise ValueError(
                f"integration_state_wallclock.npy length mismatch: {len(wallclock_timestamps)} vs {len(states)} "
                f"in {episode_dir}"
            )

    return states, timestamps, wallclock_timestamps, _mj_state_integration_spec()


def _extract_recorded_physics_overrides(metadata: dict[str, Any]) -> dict[str, Any] | None:
    physics_overrides: dict[str, Any] = {}
    physics_dt = metadata.get("physics_dt")
    control_decimation = metadata.get("control_decimation")
    if physics_dt is not None:
        physics_overrides["physics_dt"] = float(physics_dt)
    if control_decimation is not None:
        physics_overrides["control_decimation"] = int(control_decimation)
    return physics_overrides or None


def _load_cached_collected_metadata(episode_dir: Path) -> dict[str, Any] | None:
    episode_id = episode_dir.name
    dataset_name = episode_dir.parent.name
    candidate_paths = [
        Path("/tmp/metadata") / dataset_name / "collected.json",
        episode_dir.parent.parent.parent / dataset_name / "collected.json",
    ]
    for path in candidate_paths:
        payload = _load_json(path)
        if payload is None:
            continue
        entry = payload.get(episode_id)
        if isinstance(entry, dict):
            return entry
    return None


def _load_dataset_metadata(episode_dir: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    cached = _load_cached_collected_metadata(episode_dir)
    if cached is not None:
        metadata.update(cached)
    episode_metadata = _load_json(episode_dir / "episode_metadata.json")
    if episode_metadata is not None:
        metadata.update(episode_metadata)
    return metadata


def _load_dataset_initial_qpos(
    episode_dir: Path,
    *,
    metadata: dict[str, Any],
) -> np.ndarray | None:
    file_name = metadata.get("initial_qpos_file")
    if isinstance(file_name, str) and file_name:
        path = episode_dir / file_name
    else:
        path = episode_dir / "initial_qpos.npy"
    if not path.exists():
        return None
    qpos = np.load(path)
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.ndim != 1 or len(qpos) == 0:
        raise ValueError(f"Expected initial qpos shape (nq,), got {qpos.shape} from {path}")
    return qpos


def _make_uniform_timestamps(num_steps: int, fps: float) -> np.ndarray:
    if num_steps <= 0:
        return np.empty((0,), dtype=np.float64)
    if fps <= 0:
        raise ValueError("fps must be positive")
    return np.arange(num_steps, dtype=np.float64) / float(fps)


def _load_recorded_actions(episode_dir: Path) -> np.ndarray:
    for file_name in ("actions_full.npy", "actions.npy"):
        path = episode_dir / file_name
        if not path.exists():
            continue
        actions = np.asarray(np.load(path), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise ValueError(f"Expected {file_name} shape (T, 14), got {actions.shape}")
        return actions
    raise FileNotFoundError(f"Expected actions_full.npy or actions.npy in {episode_dir}")


def _load_recorded_qpos(episode_dir: Path) -> np.ndarray:
    path = episode_dir / "qpos.npy"
    qpos = np.asarray(np.load(path))
    if qpos.ndim != 2 or len(qpos) == 0:
        raise ValueError(f"Expected qpos.npy shape (T, nq), got {qpos.shape}")
    return qpos


def _load_recorded_state_spec(metadata: dict[str, Any]) -> int | None:
    value = metadata.get("mj_state_spec")
    if value is None:
        return None
    return int(value)


def _load_optional_recorded_state(
    episode_dir: Path,
    file_name: str,
    *,
    dtype: np.dtype | type | None = None,
) -> np.ndarray | None:
    path = episode_dir / file_name
    if not path.exists():
        return None
    arr = np.asarray(np.load(path))
    if dtype is not None:
        return np.asarray(arr, dtype=dtype)
    return arr


def _load_recorded_timestamps(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    ts = np.asarray(np.load(path), dtype=np.float64)
    if ts.ndim != 1:
        raise ValueError(f"Expected 1D timestamps in {path}, got {ts.shape}")
    return ts


def _maybe_prepend_initial_frame(
    frames: np.ndarray | None,
    initial_frame: np.ndarray | None,
    *,
    num_actions: int,
) -> np.ndarray | None:
    if frames is None:
        return None
    if len(frames) == num_actions + 1:
        return frames
    if len(frames) != num_actions:
        raise ValueError(
            f"Recorded state/actions length mismatch: state={len(frames)} vs actions={num_actions}. "
            "Expected state arrays to contain one initial frame plus one post-step frame per action."
        )
    if initial_frame is None:
        return frames
    return np.concatenate([initial_frame[None, ...], frames], axis=0)


def _load_recorded_episode_context(
    episode_dir: Path,
    *,
    load_recorded_cameras: bool,
) -> EpisodeContext:
    del load_recorded_cameras

    print(f"Loading recorded episode: {episode_dir}")
    metadata = _load_json(episode_dir / "config.json") or {}
    actions = _load_recorded_actions(episode_dir)
    qpos = _load_recorded_qpos(episode_dir)
    integration_state = _load_optional_recorded_state(episode_dir, "integration_state.npy")
    state_spec = _load_recorded_state_spec(metadata)
    ctrl = _load_optional_recorded_state(episode_dir, "ctrl.npy")
    qvel = _load_optional_recorded_state(episode_dir, "qvel.npy")
    act = _load_optional_recorded_state(episode_dir, "act.npy")
    mocap_pos = _load_optional_recorded_state(episode_dir, "mocap_pos.npy")
    mocap_quat = _load_optional_recorded_state(episode_dir, "mocap_quat.npy")
    initial_qpos_path = episode_dir / "initial_qpos.npy"
    initial_qpos = (
        np.asarray(np.load(initial_qpos_path))
        if initial_qpos_path.exists()
        else qpos[0].copy()
    )
    initial_qvel = _load_optional_recorded_state(episode_dir, "initial_qvel.npy")
    initial_act = _load_optional_recorded_state(episode_dir, "initial_act.npy")
    initial_mocap_pos = _load_optional_recorded_state(episode_dir, "initial_mocap_pos.npy")
    initial_mocap_quat = _load_optional_recorded_state(episode_dir, "initial_mocap_quat.npy")

    qpos_ts = _load_recorded_timestamps(episode_dir / "qpos_timestamps.npy")
    action_ts = _load_recorded_timestamps(episode_dir / "action_timestamps.npy")
    control_rate = float(metadata.get("control_rate") or 30.0)

    if qpos_ts is None:
        qpos_ts = _make_uniform_timestamps(len(qpos), control_rate)
    if action_ts is None:
        if len(qpos_ts) == len(actions) + 1:
            action_ts = qpos_ts[1:].copy()
        else:
            action_ts = _make_uniform_timestamps(len(actions), control_rate)

    if len(qpos) == len(actions):
        qpos = np.concatenate([initial_qpos[None, :], qpos], axis=0)
        dt = (1.0 / control_rate) if len(qpos_ts) == len(actions) else 0.0
        qpos_ts = np.concatenate(
            [np.array([float(qpos_ts[0] - dt)], dtype=np.float64), qpos_ts],
            axis=0,
        )

    qvel = _maybe_prepend_initial_frame(qvel, initial_qvel, num_actions=len(actions))
    integration_state = _maybe_prepend_initial_frame(
        integration_state,
        None,
        num_actions=len(actions),
    )
    ctrl = _maybe_prepend_initial_frame(ctrl, None, num_actions=len(actions))
    act = _maybe_prepend_initial_frame(act, initial_act, num_actions=len(actions))
    mocap_pos = _maybe_prepend_initial_frame(
        mocap_pos,
        initial_mocap_pos,
        num_actions=len(actions),
    )
    mocap_quat = _maybe_prepend_initial_frame(
        mocap_quat,
        initial_mocap_quat,
        num_actions=len(actions),
    )

    if len(qpos) != len(actions) + 1:
        raise ValueError(
            f"Recorded qpos/actions length mismatch: qpos={len(qpos)} vs actions={len(actions)}. "
            "Expected qpos to contain the initial frame plus one post-step frame per action."
        )
    if len(qpos_ts) != len(qpos):
        raise ValueError(
            f"Recorded qpos timestamps length mismatch: {len(qpos_ts)} vs {len(qpos)}"
        )
    if len(action_ts) != len(actions):
        raise ValueError(
            f"Recorded action timestamps length mismatch: {len(action_ts)} vs {len(actions)}"
        )
    if integration_state is not None and state_spec is None:
        raise ValueError(
            f"Recorded integration_state.npy exists in {episode_dir}, but config.json is missing mj_state_spec"
        )

    duration = qpos_ts[-1] - qpos_ts[0] if len(qpos_ts) > 1 else 0.0
    hz = (len(action_ts) - 1) / duration if duration > 0 and len(action_ts) > 1 else control_rate
    print(f"  actions: {len(actions)} frames, {duration:.1f}s at ~{hz:.0f} Hz")
    print(f"  qpos: {len(qpos)} frames (dim={qpos.shape[1]})")

    streams = EpisodeStreams(
        episode_dir=episode_dir,
        actions_left=actions[:, :7],
        ts_left=action_ts,
        actions_right=actions[:, 7:],
        ts_right=action_ts,
        camera_frames={},
        camera_ts={},
    )
    scene = str(metadata.get("scene") or "hybrid")
    task = str(metadata.get("task") or metadata.get("task_name") or "empty")
    instruction_raw = metadata.get("prompt") or metadata.get("instruction")
    instruction = str(instruction_raw).strip() if instruction_raw else None
    qpos = _maybe_upgrade_exact_qpos_for_task(task, qpos)
    initial_qpos = _maybe_upgrade_exact_qpos_for_task(task, initial_qpos)
    return EpisodeContext(
        streams=streams,
        episode_format="recorded",
        scene=scene,
        task=task,
        instruction=instruction,
        rand_state=None,
        raw_sim_states=qpos,
        raw_sim_timestamps=qpos_ts,
        raw_sim_integration_states=integration_state,
        raw_sim_state_spec=state_spec,
        raw_sim_ctrls=ctrl,
        initial_scene_qpos=initial_qpos,
        initial_scene_integration_state=(
            integration_state[0].copy() if integration_state is not None and len(integration_state) > 0 else None
        ),
        raw_sim_qvels=qvel,
        raw_sim_acts=act,
        raw_sim_mocap_pos=mocap_pos,
        raw_sim_mocap_quat=mocap_quat,
        initial_scene_qvel=initial_qvel,
        initial_scene_act=initial_act,
        initial_scene_mocap_pos=initial_mocap_pos,
        initial_scene_mocap_quat=initial_mocap_quat,
        replay_actions=actions,
        replay_ctrls=(ctrl[1:] if ctrl is not None and len(ctrl) > 1 else None),
        replay_timestamps=qpos_ts,
        replay_state_kind="qpos",
        replay_state_alignment="initial",
        physics_overrides=_extract_recorded_physics_overrides(metadata),
    )


def _load_dataset_recorded_cameras(
    episode_dir: Path,
    *,
    metadata: dict[str, Any],
    grid_ts: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    import imageio.v3 as iio

    camera_frames: dict[str, np.ndarray] = {}
    camera_ts: dict[str, np.ndarray] = {}
    fps = float(metadata.get("fps") or 30.0)
    videos = metadata.get("videos") if isinstance(metadata.get("videos"), dict) else {}
    cameras = metadata.get("cameras") if isinstance(metadata.get("cameras"), list) else []

    for cam_name in cameras:
        video_name = videos.get(cam_name, f"{cam_name}_camera-images-rgb.mp4")
        video_path = episode_dir / str(video_name)
        if not video_path.exists():
            continue
        frames = iio.imread(str(video_path), plugin="pyav")
        ts_arr = grid_ts[: len(frames)] if len(grid_ts) >= len(frames) else _make_uniform_timestamps(len(frames), fps)
        camera_frames[str(cam_name)] = frames
        camera_ts[str(cam_name)] = np.asarray(ts_arr, dtype=np.float64)
        h, w = frames.shape[1], frames.shape[2]
        duration = camera_ts[str(cam_name)][-1] - camera_ts[str(cam_name)][0] if len(frames) > 1 else 0.0
        print(f"  camera '{cam_name}': {len(frames)} frames ({w}x{h}), {duration:.1f}s")

    if camera_frames:
        return camera_frames, camera_ts

    for fallback_name, label in [
        ("combined_camera-images-rgb.mp4", "combined"),
        ("horizontal.mp4", "horizontal"),
    ]:
        video_path = episode_dir / fallback_name
        if not video_path.exists():
            continue
        frames = iio.imread(str(video_path), plugin="pyav")
        ts_arr = grid_ts[: len(frames)] if len(grid_ts) >= len(frames) else _make_uniform_timestamps(len(frames), fps)
        camera_frames[label] = frames
        camera_ts[label] = np.asarray(ts_arr, dtype=np.float64)
        h, w = frames.shape[1], frames.shape[2]
        duration = camera_ts[label][-1] - camera_ts[label][0] if len(frames) > 1 else 0.0
        print(f"  camera '{label}': {len(frames)} frames ({w}x{h}), {duration:.1f}s")
        break

    return camera_frames, camera_ts


def _load_direct_protobuf_positions(mcap_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load timestamped `position` arrays from a protobuf MCAP file."""
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    positions: list[list[float]] = []
    timestamps: list[float] = []
    with open(mcap_file, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _schema, _channel, _message, decoded in reader.iter_decoded_messages():
            if not hasattr(decoded, "position") or not hasattr(decoded, "timestamp"):
                continue
            positions.append(list(decoded.position))
            timestamps.append(decoded.timestamp.seconds + decoded.timestamp.nanos / 1e9)
    if not positions:
        raise ValueError(f"No protobuf position data found in {mcap_file}")
    return np.array(positions, dtype=np.float64), np.array(timestamps, dtype=np.float64)


def _load_direct_topic_positions(
    mcap_file: Path,
    *,
    topic: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load timestamped `position` arrays from one specific protobuf topic."""
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    positions: list[list[float]] = []
    timestamps: list[float] = []
    with open(mcap_file, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _schema, channel, _message, decoded in reader.iter_decoded_messages():
            if channel.topic != topic:
                continue
            if not hasattr(decoded, "position") or not hasattr(decoded, "timestamp"):
                continue
            positions.append(list(decoded.position))
            timestamps.append(decoded.timestamp.seconds + decoded.timestamp.nanos / 1e9)
    if not positions:
        return None
    return np.array(positions, dtype=np.float64), np.array(timestamps, dtype=np.float64)


def _load_direct_actions(episode_dir: Path, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Load 7D actions from explicit command-state or action-side gripper topics."""
    state_file = episode_dir / f"{side}.mcap"
    command_state = None
    if state_file.exists():
        command_state = _load_direct_topic_positions(
            state_file,
            topic=f"/{side}-command-state",
        )
    if command_state is not None:
        command_pos, command_ts = command_state
        if command_pos.shape[1] != 7:
            raise ValueError(
                f"Expected 7D /{side}-command-state in {state_file}, got shape {command_pos.shape}"
            )
        return command_pos, command_ts

    action_file = episode_dir / f"action-{side}.mcap"
    action_robot = _load_direct_topic_positions(
        action_file,
        topic=f"/action-{side}-robot-state",
    )
    action_gripper = _load_direct_topic_positions(
        action_file,
        topic=f"/action-{side}-gripper-state",
    )

    if action_robot is not None:
        action_pos, action_ts = action_robot
    else:
        action_pos, action_ts = _load_direct_protobuf_positions(action_file)

    if action_pos.shape[1] != 6:
        raise ValueError(f"Expected 6 arm joints in action-{side}.mcap, got shape {action_pos.shape}")

    if action_gripper is not None:
        grip_pos, grip_ts = action_gripper
        if grip_pos.shape[1] != 1:
            raise ValueError(
                f"Expected 1D action gripper in {action_file}, got shape {grip_pos.shape}"
            )
        grip = sample_hold_align(grip_pos, grip_ts, action_ts)
        return np.concatenate([action_pos, grip], axis=1), action_ts

    raise ValueError(
        f"Direct episode is missing explicit action gripper stream "
        f"/action-{side}-gripper-state in {action_file}; refusing to use observed "
        "gripper state or zero-filled gripper actions."
    )


def _load_direct_recorded_cameras(episode_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load recorded MP4 frames and timestamps directly from the episode directory."""
    import imageio.v3 as iio

    camera_frames: dict[str, np.ndarray] = {}
    camera_ts: dict[str, np.ndarray] = {}
    path_map = {
        "top": ("top_camera-images-rgb.mp4", "top_camera-timestamp.npy"),
        "left": ("left_camera-images-rgb.mp4", "left_camera-timestamp.npy"),
        "right": ("right_camera-images-rgb.mp4", "right_camera-timestamp.npy"),
    }
    for cam_name, (video_name, ts_name) in path_map.items():
        video_path = episode_dir / video_name
        ts_path = episode_dir / ts_name
        if not video_path.exists() or not ts_path.exists():
            continue
        frames = iio.imread(str(video_path), plugin="pyav")
        ts_arr = np.load(ts_path)
        camera_frames[cam_name] = frames
        camera_ts[cam_name] = ts_arr
        h, w = frames.shape[1], frames.shape[2]
        print(f"  camera '{cam_name}': {len(frames)} frames ({w}x{h}), {ts_arr[-1] - ts_arr[0]:.1f}s")
    return camera_frames, camera_ts


def _load_episode_streams_direct(episode_dir: Path, *, load_recorded_cameras: bool) -> EpisodeStreams:
    """Load an episode directly from the directory without yam_sdk."""
    print(f"Loading episode directly: {episode_dir}")

    actions_left, ts_left = _load_direct_actions(episode_dir, "left")
    hz_left = (len(ts_left) - 1) / (ts_left[-1] - ts_left[0]) if len(ts_left) > 1 else 0.0
    print(f"  action-left: {len(actions_left)} frames, {ts_left[-1] - ts_left[0]:.1f}s at ~{hz_left:.0f} Hz")

    actions_right, ts_right = _load_direct_actions(episode_dir, "right")
    hz_right = (len(ts_right) - 1) / (ts_right[-1] - ts_right[0]) if len(ts_right) > 1 else 0.0
    print(f"  action-right: {len(actions_right)} frames, {ts_right[-1] - ts_right[0]:.1f}s at ~{hz_right:.0f} Hz")

    camera_frames: dict[str, np.ndarray] = {}
    camera_ts: dict[str, np.ndarray] = {}
    if load_recorded_cameras:
        camera_frames, camera_ts = _load_direct_recorded_cameras(episode_dir)

    return EpisodeStreams(
        episode_dir=episode_dir,
        actions_left=actions_left,
        ts_left=ts_left,
        actions_right=actions_right,
        ts_right=ts_right,
        camera_frames=camera_frames,
        camera_ts=camera_ts,
    )


def _load_delivered_episode_streams(
    episode_dir: Path,
    *,
    load_recorded_cameras: bool,
) -> tuple[EpisodeStreams, str]:
    """Load a delivered BAIR-format episode from output.mcap + sim_state.mcap."""
    import io

    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    print(f"Loading delivered episode: {episode_dir}")

    left_leader_pos: list[list[float]] = []
    left_leader_ts: list[float] = []
    right_leader_pos: list[list[float]] = []
    right_leader_ts: list[float] = []
    left_eef_pos: list[list[float]] = []
    left_eef_ts: list[float] = []
    right_eef_pos: list[list[float]] = []
    right_eef_ts: list[float] = []
    cam_data: dict[str, list[bytes]] = {}
    cam_ts_raw: dict[str, list[float]] = {}
    instruction = ""

    cam_topic_to_name = {
        "/top-camera/image-raw": "top",
        "/left-wrist-camera/image-raw": "left",
        "/right-wrist-camera/image-raw": "right",
    }

    with open(episode_dir / "output.mcap", "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _schema, channel, _message, decoded in reader.iter_decoded_messages():
            topic = channel.topic
            if topic == "/instruction":
                instruction = decoded.data
                continue

            ts = getattr(decoded, "timestamp", None)
            if ts is None:
                continue
            timestamp = ts.seconds + ts.nanos / 1e9

            if topic == "/left-arm-leader":
                left_leader_ts.append(timestamp)
                left_leader_pos.append(list(decoded.position))
            elif topic == "/right-arm-leader":
                right_leader_ts.append(timestamp)
                right_leader_pos.append(list(decoded.position))
            elif topic == "/left-eef-leader":
                left_eef_ts.append(timestamp)
                left_eef_pos.append(list(decoded.position))
            elif topic == "/right-eef-leader":
                right_eef_ts.append(timestamp)
                right_eef_pos.append(list(decoded.position))
            elif load_recorded_cameras and topic in cam_topic_to_name:
                cam_name = cam_topic_to_name[topic]
                cam_data.setdefault(cam_name, []).append(bytes(decoded.data))
                cam_ts_raw.setdefault(cam_name, []).append(timestamp)

    if not left_leader_pos or not right_leader_pos:
        raise ValueError(f"Delivered episode is missing leader action streams in {episode_dir}")
    if not left_eef_pos or not right_eef_pos:
        raise ValueError(
            f"Delivered episode is missing leader gripper streams (/left-eef-leader, /right-eef-leader) in {episode_dir}"
        )

    left_leader = np.array(left_leader_pos, dtype=np.float64)
    right_leader = np.array(right_leader_pos, dtype=np.float64)
    ts_left_leader = np.array(left_leader_ts, dtype=np.float64)
    ts_right_leader = np.array(right_leader_ts, dtype=np.float64)
    left_eef = np.array(left_eef_pos, dtype=np.float64)
    right_eef = np.array(right_eef_pos, dtype=np.float64)
    ts_left_eef = np.array(left_eef_ts, dtype=np.float64)
    ts_right_eef = np.array(right_eef_ts, dtype=np.float64)
    left_gripper = sample_hold_align(left_eef[:, :1], ts_left_eef, ts_left_leader)
    right_gripper = sample_hold_align(right_eef[:, :1], ts_right_eef, ts_right_leader)
    actions_left = np.concatenate([left_leader, left_gripper], axis=1)
    actions_right = np.concatenate([right_leader, right_gripper], axis=1)

    duration_left = ts_left_leader[-1] - ts_left_leader[0]
    duration_right = ts_right_leader[-1] - ts_right_leader[0]
    hz_left = (len(ts_left_leader) - 1) / duration_left if duration_left > 0 else 0.0
    hz_right = (len(ts_right_leader) - 1) / duration_right if duration_right > 0 else 0.0
    print(f"  Instruction: {instruction!r}")
    print(f"  action-left: {len(actions_left)} frames, {duration_left:.1f}s at ~{hz_left:.0f} Hz")
    print(f"  action-right: {len(actions_right)} frames, {duration_right:.1f}s at ~{hz_right:.0f} Hz")

    camera_frames: dict[str, np.ndarray] = {}
    camera_ts: dict[str, np.ndarray] = {}
    if load_recorded_cameras:
        import av

        for cam_name, packets in cam_data.items():
            try:
                container = av.open(io.BytesIO(b"".join(packets)), format="h264")
                frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
                ts_arr = np.array(cam_ts_raw[cam_name], dtype=np.float64)
                n = min(len(frames), len(ts_arr))
                if n == 0:
                    continue
                camera_frames[cam_name] = np.array(frames[:n])
                camera_ts[cam_name] = ts_arr[:n]
                h, w = camera_frames[cam_name][0].shape[:2]
                print(f"  camera '{cam_name}': {n} frames ({w}x{h}), {ts_arr[n - 1] - ts_arr[0]:.1f}s")
            except Exception as exc:
                print(f"  Warning: could not decode camera '{cam_name}': {exc}")

    streams = EpisodeStreams(
        episode_dir=episode_dir,
        actions_left=actions_left,
        ts_left=ts_left_leader,
        actions_right=actions_right,
        ts_right=ts_right_leader,
        camera_frames=camera_frames,
        camera_ts=camera_ts,
    )
    return streams, instruction


def load_episode_streams(episode_dir: Path, *, load_recorded_cameras: bool = True) -> EpisodeStreams:
    """Load action streams and optionally recorded camera frames for a supported layout."""
    episode_format = detect_episode_format(episode_dir)
    if episode_format == "delivered":
        streams, _instruction = _load_delivered_episode_streams(
            episode_dir,
            load_recorded_cameras=load_recorded_cameras,
        )
        return streams
    if episode_format == "recorded":
        return _load_recorded_episode_context(
            episode_dir,
            load_recorded_cameras=load_recorded_cameras,
        ).streams
    if episode_format == "dataset":
        return _load_dataset_episode_context(
            episode_dir,
            load_recorded_cameras=load_recorded_cameras,
        ).streams

    try:
        import json as _json
        from collections import defaultdict

        from mcap.decoder import DecoderFactory as _McapDecoderFactory
        from mcap.reader import make_reader as _make_reader
        from mcap_protobuf.decoder import DecoderFactory as _ProtobufDecoderFactory
        from mcap_ros2.decoder import DecoderFactory as _Ros2DecoderFactory
        from yam_sdk.data.schema.keys import DataKeys, MCAP_TOPIC_FIELD_TO_KEY
        from yam_sdk.data.trajectory import ArmSide, ArmTrajectory, Trajectory, load_trajectory
        import numpy as _np
    except ImportError:
        return _load_episode_streams_direct(episode_dir, load_recorded_cameras=load_recorded_cameras)

    class _JsonDecoderFactory(_McapDecoderFactory):
        def decoder_for(self, message_encoding, schema):
            if message_encoding == "json":
                return lambda data: _json.loads(data)
            return None

    def _patched_load_mcap_file(self, file):
        data = defaultdict(list)
        data_timestamps = defaultdict(list)
        with open(file, "rb") as f:
            reader = _make_reader(
                f,
                decoder_factories=[
                    _ProtobufDecoderFactory(),
                    _Ros2DecoderFactory(),
                    _JsonDecoderFactory(),
                ],
            )
            for _schema, channel, _message, proto_msg in reader.iter_decoded_messages():
                topic = channel.topic
                if topic not in MCAP_TOPIC_FIELD_TO_KEY:
                    continue
                field_mappings = MCAP_TOPIC_FIELD_TO_KEY[topic]
                for field_name, data_key in field_mappings.items():
                    if not hasattr(proto_msg, field_name):
                        continue
                    data[data_key].append(getattr(proto_msg, field_name))
                    data_timestamps[data_key].append(
                        proto_msg.timestamp.seconds + proto_msg.timestamp.nanos / 1e9
                    )
        for data_key, timestamps in data_timestamps.items():
            if len(timestamps) > 1 and _np.any(_np.diff(timestamps) < 0):
                raise ValueError(f"Timestamp is not monotonically increasing for topic {data_key}")
        return data, data_timestamps

    original_load_mcap_file = Trajectory._load_mcap_file
    original_get_joint_pos_obs = ArmTrajectory.get_joint_pos_obs

    def _patched_get_joint_pos_obs(self, arm_side):
        key = f"{arm_side.value}-joint_pos"
        if key not in self._trajectory_data:
            return self._trajectory_data[f"action-{arm_side.value}-pos"][:, : self._n_dof_arm]
        return original_get_joint_pos_obs(self, arm_side)

    print(f"Loading episode: {episode_dir}")
    Trajectory._load_mcap_file = _patched_load_mcap_file
    ArmTrajectory.get_joint_pos_obs = _patched_get_joint_pos_obs
    try:
        traj = load_trajectory(episode_dir)
    finally:
        Trajectory._load_mcap_file = original_load_mcap_file
        ArmTrajectory.get_joint_pos_obs = original_get_joint_pos_obs

    actions_left = None
    ts_left = None
    actions_right = None
    ts_right = None
    for side_label, side in [("left", ArmSide.LEFT), ("right", ArmSide.RIGHT)]:
        joints = traj.get_joint_pos_action(side)
        gripper = traj.get_gripper_pos_action(side)
        pos = np.concatenate([joints, gripper], axis=1)
        ts = traj.get_data_key_timestamp(
            DataKeys.ACTION.JOINT.POS.LEFT if side == ArmSide.LEFT else DataKeys.ACTION.JOINT.POS.RIGHT
        )
        hz = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 else 0.0
        print(f"  action-{side_label}: {len(pos)} frames, {ts[-1] - ts[0]:.1f}s at ~{hz:.0f} Hz")
        if side == ArmSide.LEFT:
            actions_left, ts_left = pos, ts
        else:
            actions_right, ts_right = pos, ts

    if actions_left is None or actions_right is None or ts_left is None or ts_right is None:
        raise ValueError("Episode is missing one or both action streams")

    camera_frames: dict[str, np.ndarray] = {}
    camera_ts: dict[str, np.ndarray] = {}
    if load_recorded_cameras:
        from yam_sdk.data.trajectory import CameraPerspective

        for perspective, ts_key in [
            (CameraPerspective.TOP, DataKeys.CAMERA.TIMESTAMP.TOP),
            (CameraPerspective.LEFT, DataKeys.CAMERA.TIMESTAMP.LEFT),
            (CameraPerspective.RIGHT, DataKeys.CAMERA.TIMESTAMP.RIGHT),
        ]:
            cam_name = perspective.value
            try:
                mp4_file = traj.get_video_path(perspective)
                ts_arr = traj.get_data_key_timestamp(ts_key)
                import imageio.v3 as iio

                frames = iio.imread(str(mp4_file), plugin="pyav")
                camera_frames[cam_name] = frames
                camera_ts[cam_name] = ts_arr
                h, w = frames.shape[1], frames.shape[2]
                print(f"  camera '{cam_name}': {len(frames)} frames ({w}x{h}), {ts_arr[-1] - ts_arr[0]:.1f}s")
            except FileNotFoundError:
                continue
            except Exception as exc:
                print(f"  Warning: could not load camera '{cam_name}': {exc}")

    return EpisodeStreams(
        episode_dir=episode_dir,
        actions_left=actions_left,
        ts_left=ts_left,
        actions_right=actions_right,
        ts_right=ts_right,
        camera_frames=camera_frames,
        camera_ts=camera_ts,
    )


def _resolve_dataset_task(metadata: dict[str, Any], episode_dir: Path) -> tuple[str, str, str | None]:
    raw_instruction = metadata.get("instruction")
    instruction = str(raw_instruction).strip() if raw_instruction else None
    raw_task_name = metadata.get("task_name") or metadata.get("source_delivery") or metadata.get("task")
    raw_task = metadata.get("task")
    scene = str(metadata.get("scene")).strip() if metadata.get("scene") else None
    task = str(raw_task).strip() if raw_task else None

    spec = (
        maybe_get_task_spec(instruction)
        or maybe_get_task_spec(str(raw_task_name) if raw_task_name else None)
        or maybe_get_task_spec(task)
    )
    if spec is not None:
        if instruction is None:
            instruction = spec.prompt
        if not scene:
            scene = spec.scene
        if not task or task.startswith("sim_"):
            task = spec.env_task

    if not scene:
        scene = "hybrid"
    if not task:
        if raw_task_name:
            print(
                f"  Warning: could not resolve dataset env task from {raw_task_name!r}; "
                "falling back to 'empty'"
            )
        task = "empty"
    if instruction is None and raw_task_name:
        instruction = str(raw_task_name).replace("_", " ")
    return scene, task, instruction


def _load_dataset_episode_context(
    episode_dir: Path,
    *,
    load_recorded_cameras: bool,
) -> EpisodeContext:
    print(f"Loading dataset episode: {episode_dir}")
    metadata = _load_dataset_metadata(episode_dir)
    states, actions = _load_dataset_states_actions(episode_dir)
    initial_qpos = _load_dataset_initial_qpos(episode_dir, metadata=metadata)
    fps = float(metadata.get("fps") or 30.0)
    grid_ts = _make_uniform_timestamps(len(actions), fps)

    duration = grid_ts[-1] - grid_ts[0] if len(grid_ts) > 1 else 0.0
    hz = (len(grid_ts) - 1) / duration if duration > 0 else fps
    print(f"  states/actions: {len(actions)} frames, {duration:.1f}s at ~{hz:.0f} Hz")

    camera_frames: dict[str, np.ndarray] = {}
    camera_ts: dict[str, np.ndarray] = {}
    if load_recorded_cameras:
        camera_frames, camera_ts = _load_dataset_recorded_cameras(
            episode_dir,
            metadata=metadata,
            grid_ts=grid_ts,
        )

    streams = EpisodeStreams(
        episode_dir=episode_dir,
        actions_left=actions[:, :7],
        ts_left=grid_ts,
        actions_right=actions[:, 7:],
        ts_right=grid_ts,
        camera_frames=camera_frames,
        camera_ts=camera_ts,
    )
    scene, task, instruction = _resolve_dataset_task(metadata, episode_dir)
    return EpisodeContext(
        streams=streams,
        episode_format="dataset",
        scene=scene,
        task=task,
        instruction=instruction,
        rand_state=None,
        raw_sim_states=states,
        raw_sim_timestamps=grid_ts,
        initial_scene_qpos=initial_qpos,
        replay_actions=actions,
        replay_timestamps=grid_ts,
        replay_state_kind="policy_state",
    )


def read_sim_config(episode_dir: Path) -> dict:
    """Read scene/task from the robot node config in session_meta.json."""
    import json

    meta_file = episode_dir / "session_meta.json"
    if not meta_file.exists():
        return {}
    with open(meta_file) as f:
        meta = json.load(f)
    for node in meta.get("nodes", []):
        if node.get("role") == "robot" and "config" in node:
            cfg = node["config"]
            if "task" in cfg or "scene" in cfg:
                return cfg
    return {}


def read_sim_physics_overrides(episode_dir: Path) -> dict[str, Any] | None:
    """Read physics overrides for a raw episode from session_meta.json when available."""
    import json

    meta_file = episode_dir / "session_meta.json"
    if not meta_file.exists():
        return None

    with open(meta_file) as f:
        meta = json.load(f)

    robot_name = None
    physics_dt = None
    control_decimation = None
    control_dt = None

    for node in meta.get("nodes", []):
        if node.get("role") != "robot":
            continue
        robot_name = node.get("name")
        cfg = node.get("config", {})
        if physics_dt is None and cfg.get("physics_dt") is not None:
            physics_dt = float(cfg["physics_dt"])
        if control_decimation is None and cfg.get("control_decimation") is not None:
            control_decimation = int(cfg["control_decimation"])
        if control_dt is None and cfg.get("control_dt") is not None:
            control_dt = float(cfg["control_dt"])
        break

    if robot_name is not None:
        node_meta = meta.get("node_metadata", {}).get(robot_name, {})
        if isinstance(node_meta, dict):
            if physics_dt is None and node_meta.get("physics_dt") is not None:
                physics_dt = float(node_meta["physics_dt"])
            if control_decimation is None and node_meta.get("control_decimation") is not None:
                control_decimation = int(node_meta["control_decimation"])
            if control_dt is None and node_meta.get("control_dt") is not None:
                control_dt = float(node_meta["control_dt"])

    if control_decimation is None and physics_dt is not None and control_dt is not None:
        control_decimation = int(round(control_dt / physics_dt))

    physics_overrides: dict[str, Any] = {}
    if physics_dt is not None:
        physics_overrides["physics_dt"] = physics_dt
    if control_decimation is not None:
        physics_overrides["control_decimation"] = control_decimation
    return physics_overrides or None


def _build_raw_step_aligned_replay_actions(
    streams: EpisodeStreams,
    raw_sim_timestamps: np.ndarray,
) -> np.ndarray:
    """Align raw controller messages onto the recorded sim-step timeline."""
    left = sample_hold_align(streams.actions_left, streams.ts_left, raw_sim_timestamps)
    right = sample_hold_align(streams.actions_right, streams.ts_right, raw_sim_timestamps)
    return np.concatenate(
        [
            left.astype(np.float32, copy=False),
            right.astype(np.float32, copy=False),
        ],
        axis=1,
    )


def load_sim_states(episode_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load the raw /sim_state/qpos timeline from sim_state.mcap."""
    import json

    from mcap.reader import make_reader

    mcap_file = episode_dir / "sim_state.mcap"
    if not mcap_file.exists():
        return None

    qposes: list[list[float]] = []
    timestamps: list[float] = []
    with open(mcap_file, "rb") as f:
        reader = make_reader(f)
        for _schema, channel, message in reader.iter_messages(topics=["/sim_state/qpos"]):
            payload = json.loads(message.data)
            qposes.append(payload["qpos"])
            timestamps.append(payload["timestamp"])

    if not qposes:
        return None
    return np.array(qposes, dtype=np.float64), np.array(timestamps, dtype=np.float64)


def load_randomization(episode_dir: Path):
    """Load randomization state from randomization.json if present."""
    import json

    from yam_sim.randomization import RandomizationState

    rand_file = episode_dir / "randomization.json"
    if not rand_file.exists():
        return None
    with open(rand_file) as f:
        return RandomizationState.from_dict(json.load(f))


def resolve_delivered_task(instruction: str, episode_dir: Path) -> tuple[str, str]:
    """Resolve yam-sim scene/task from a delivered episode instruction."""
    spec = maybe_get_task_spec(instruction) or maybe_get_task_spec(episode_dir.parent.name)
    if spec is None:
        key = instruction.strip().replace(" ", "_")
        print(
            f"  Warning: unknown delivered task '{key or episode_dir.parent.name}', "
            "falling back to 'empty'"
        )
        return "hybrid", "empty"
    return spec.scene, spec.env_task


def load_episode_context(episode_dir: Path, *, load_recorded_cameras: bool = True) -> EpisodeContext:
    """Load a full replayable episode context."""
    episode_format = detect_episode_format(episode_dir)
    if episode_format == "dataset":
        return _load_dataset_episode_context(
            episode_dir,
            load_recorded_cameras=load_recorded_cameras,
        )
    if episode_format == "recorded":
        return _load_recorded_episode_context(
            episode_dir,
            load_recorded_cameras=load_recorded_cameras,
        )

    instruction: str | None = None

    if episode_format == "delivered":
        streams, instruction = _load_delivered_episode_streams(
            episode_dir,
            load_recorded_cameras=load_recorded_cameras,
        )
        scene, task = resolve_delivered_task(instruction, episode_dir)
        physics_overrides = None
    else:
        streams = load_episode_streams(episode_dir, load_recorded_cameras=load_recorded_cameras)
        sim_cfg = read_sim_config(episode_dir)
        scene = sim_cfg.get("scene", "hybrid")
        task = sim_cfg.get("task", "empty")
        spec = maybe_get_task_spec(task)
        if spec is not None:
            scene = spec.scene
            instruction = spec.prompt
            if task.startswith("sim_"):
                task = spec.env_task
        physics_overrides = (
            read_sim_physics_overrides(episode_dir)
            if episode_format == "raw"
            else _extract_recorded_physics_overrides(sim_cfg)
        )

    raw = load_sim_states(episode_dir)
    raw_states, raw_timestamps = (raw if raw is not None else (None, None))
    raw_states = _maybe_upgrade_exact_qpos_for_task(task, raw_states)
    raw_integration_states = None
    raw_integration_timestamps = None
    raw_state_spec = None
    scene_xml_string = None
    scene_source = None
    if episode_format == "delivered":
        rand_state = load_randomization(episode_dir)
        scene_xml_string = load_scene_assembled_xml(episode_dir)
        if scene_xml_string is not None:
            scene_source = "scene_assembled.xml"
        else:
            if rand_state is None:
                raise ValueError(
                    f"Delivered episode {episode_dir} is missing scene_assembled.xml and "
                    "randomization.json; refusing to render a default scene."
                )
            scene_source = "randomization.json"
        (
            raw_integration_states,
            raw_integration_timestamps,
            raw_integration_wallclock_timestamps,
            raw_state_spec,
        ) = load_integration_states(episode_dir)
    else:
        rand_state = load_randomization(episode_dir)
        raw_integration_wallclock_timestamps = None
    replay_actions = None
    replay_timestamps = None
    replay_state_alignment = "initial"
    if episode_format == "raw" and raw_states is not None and raw_timestamps is not None and len(raw_states) > 0:
        replay_timestamps = np.asarray(raw_timestamps, dtype=np.float64)
        raw_replay_actions = _build_raw_step_aligned_replay_actions(streams, replay_timestamps)
        initial_scene_qpos = raw_states[0].astype(np.float32, copy=False)
        replay_actions = raw_replay_actions[1:] if len(raw_replay_actions) > 1 else raw_replay_actions[:0]
    else:
        initial_scene_qpos = (
            raw_states[0].astype(np.float32, copy=False)
            if raw_states is not None and len(raw_states) > 0
            else None
        )
    return EpisodeContext(
        streams=streams,
        episode_format=episode_format,
        scene=scene,
        task=task,
        instruction=instruction,
        rand_state=rand_state,
        raw_sim_states=raw_states,
        raw_sim_timestamps=raw_timestamps,
        scene_xml_string=scene_xml_string,
        scene_source=scene_source,
        raw_sim_integration_states=raw_integration_states,
        raw_sim_integration_timestamps=raw_integration_timestamps,
        raw_sim_integration_wallclock_timestamps=raw_integration_wallclock_timestamps,
        raw_sim_state_spec=raw_state_spec,
        initial_scene_qpos=initial_scene_qpos,
        initial_scene_integration_state=(
            raw_integration_states[0].copy()
            if raw_integration_states is not None and len(raw_integration_states) > 0
            else None
        ),
        replay_actions=replay_actions,
        replay_timestamps=replay_timestamps,
        replay_state_alignment=replay_state_alignment,
        physics_overrides=physics_overrides,
    )
