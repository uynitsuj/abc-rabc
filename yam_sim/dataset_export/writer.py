"""Filesystem writers for exported dataset episodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from yam_sim.dataset_export.types import EpisodeArtifacts, ExportConfig, ExportTrajectory
from yam_sim.dataset_export.video_io import probe_video_frame_count


def camera_video_name(camera_name: str) -> str:
    return f"{camera_name}_camera-images-rgb.mp4"


def combined_video_name() -> str:
    return "combined_camera-images-rgb.mp4"


def initial_qpos_name() -> str:
    return "initial_qpos.npy"


def write_states_actions(
    output_dir: Path,
    *,
    states: np.ndarray,
    actions: np.ndarray,
) -> tuple[Path, Path]:
    """Write concatenated states/actions to .npy and .bin."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(states) != len(actions):
        raise ValueError(f"states/actions length mismatch: {len(states)} vs {len(actions)}")
    states_actions = np.concatenate([states, actions], axis=1)
    states_actions = np.asarray(states_actions, dtype=np.float64, order="C")

    npy_path = output_dir / "states_actions.npy"
    bin_path = output_dir / "states_actions.bin"
    np.save(npy_path, states_actions, allow_pickle=False)
    with open(bin_path, "wb") as f:
        f.write(states_actions.tobytes())
    return npy_path, bin_path


def write_initial_qpos(
    output_dir: Path,
    *,
    initial_qpos: np.ndarray,
) -> Path:
    """Write the full initial MuJoCo qpos for later action replay."""
    output_dir.mkdir(parents=True, exist_ok=True)
    qpos = np.asarray(initial_qpos, dtype=np.float32)
    path = output_dir / initial_qpos_name()
    np.save(path, qpos, allow_pickle=False)
    return path


def write_episode_metadata(
    output_dir: Path,
    trajectory: ExportTrajectory,
    *,
    config: ExportConfig,
    video_paths: dict[str, Path],
) -> Path:
    """Write episode_metadata.json with task and export provenance."""
    payload: dict[str, Any] = {
        "episode_id": trajectory.episode_id,
        "task_name": trajectory.task_name,
        "instruction": trajectory.instruction,
        "scene": trajectory.scene,
        "task": trajectory.task,
        "source_delivery": trajectory.source_delivery,
        "source_episode_dir": str(trajectory.source_episode_dir),
        "scene_source": trajectory.scene_source,
        "state_source": trajectory.state_source,
        "num_steps": int(len(trajectory.timestamps)),
        "fps": config.fps,
        "image_width": config.image_width,
        "image_height": config.image_height,
        "render_backend": config.render_backend,
        "cameras": list(trajectory.camera_names),
        "videos": {name: path.name for name, path in sorted(video_paths.items())},
        "initial_qpos_file": initial_qpos_name(),
        "initial_qpos_dim": int(len(trajectory.initial_qpos)),
    }
    metadata_path = output_dir / "episode_metadata.json"
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n")
    return metadata_path


def validate_exported_episode(
    output_dir: Path,
    *,
    expected_steps: int,
    camera_names: tuple[str, ...],
) -> None:
    """Fail closed if an exported episode is incomplete or desynchronized."""
    required = [
        "states_actions.npy",
        "states_actions.bin",
        initial_qpos_name(),
        "episode_metadata.json",
        combined_video_name(),
    ]
    required.extend(camera_video_name(name) for name in camera_names)
    required.extend(
        f"{Path(name).stem}_frame_mappings.json" for name in required if name.endswith(".mp4")
    )

    missing = [name for name in required if not (output_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Episode output missing required files: {missing}")

    states_actions = np.load(output_dir / "states_actions.npy")
    if states_actions.shape[0] != expected_steps:
        raise RuntimeError(
            f"states_actions.npy length mismatch: {states_actions.shape[0]} vs {expected_steps}"
        )

    for video_name in [camera_video_name(name) for name in camera_names] + [combined_video_name()]:
        frame_count = probe_video_frame_count(output_dir / video_name)
        if frame_count != expected_steps:
            raise RuntimeError(
                f"Video length mismatch for {video_name}: {frame_count} vs {expected_steps}"
            )


def build_episode_artifacts(
    output_dir: Path,
    *,
    video_paths: dict[str, Path],
) -> EpisodeArtifacts:
    """Construct the artifact record returned by pipeline functions."""
    return EpisodeArtifacts(
        episode_dir=output_dir,
        states_actions_path=output_dir / "states_actions.npy",
        states_actions_bin_path=output_dir / "states_actions.bin",
        initial_qpos_path=output_dir / initial_qpos_name(),
        episode_metadata_path=output_dir / "episode_metadata.json",
        video_paths=video_paths,
    )
