"""High-level dataset export orchestration."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable

from yam_sim.dataset_export.metadata import (
    build_collected_entry,
    finalize_dataset_metadata,
    write_json,
)
from yam_sim.dataset_export.render import render_episode_videos
from yam_sim.dataset_export.subprocess_export import run_export_episode_subprocess
from yam_sim.dataset_export.s3_utils import sync_local_dir_to_s3
from yam_sim.dataset_export.trajectory import build_export_trajectory
from yam_sim.dataset_export.types import EpisodeArtifacts, ExportConfig, ExportTrajectory
from yam_sim.dataset_export.writer import (
    build_episode_artifacts,
    write_initial_qpos,
    validate_exported_episode,
    write_episode_metadata,
    write_states_actions,
)
from yam_sim.rendering.replay.episode import load_episode_context
from yam_sim.rendering.replay.runtime import create_replay_env


def find_episode_dirs(input_root: Path) -> list[Path]:
    """Discover delivered episode directories and deduplicate nested copies."""
    input_root = Path(input_root)
    candidates = sorted({path.parent for path in input_root.rglob("output.mcap")})
    deduped: dict[str, Path] = {}
    for episode_dir in candidates:
        current = deduped.get(episode_dir.name)
        if current is None or len(episode_dir.parts) < len(current.parts):
            deduped[episode_dir.name] = episode_dir
    return [deduped[name] for name in sorted(deduped)]


def export_episode(
    episode_dir: Path,
    output_root: Path,
    *,
    config: ExportConfig,
    source_delivery: str | None = None,
) -> tuple[ExportTrajectory, EpisodeArtifacts, dict]:
    """Export one delivered episode into the trainable dataset layout."""
    episode_dir = Path(episode_dir)
    output_root = Path(output_root)
    context = load_episode_context(episode_dir, load_recorded_cameras=False)
    render_cameras = config.render_backend == "mujoco"
    env = create_replay_env(
        context,
        render_cameras=render_cameras,
        camera_width=config.image_width,
        camera_height=config.image_height,
    )
    try:
        trajectory = build_export_trajectory(
            context,
            env,
            fps=config.fps,
            source_delivery=source_delivery,
        )
        episode_output_dir = output_root / "data" / trajectory.episode_id
        if episode_output_dir.exists():
            shutil.rmtree(episode_output_dir)
        episode_output_dir.mkdir(parents=True, exist_ok=True)

        video_paths = render_episode_videos(
            trajectory,
            env,
            episode_output_dir,
            config=config,
        )
        write_states_actions(
            episode_output_dir,
            states=trajectory.states,
            actions=trajectory.actions,
        )
        write_initial_qpos(
            episode_output_dir,
            initial_qpos=trajectory.initial_qpos,
        )
        write_episode_metadata(
            episode_output_dir,
            trajectory,
            config=config,
            video_paths=video_paths,
        )
        validate_exported_episode(
            episode_output_dir,
            expected_steps=len(trajectory.timestamps),
            camera_names=trajectory.camera_names,
        )
        artifacts = build_episode_artifacts(
            episode_output_dir,
            video_paths=video_paths,
        )
        collected_entry = build_collected_entry(trajectory, config=config)
        return trajectory, artifacts, collected_entry
    finally:
        env.close()


def export_episode_with_backend_lifecycle(
    episode_dir: Path,
    output_root: Path,
    *,
    config: ExportConfig,
    source_delivery: str | None = None,
) -> tuple[EpisodeArtifacts, dict]:
    """Export one episode, isolating backend state when required."""
    if config.render_backend == "madrona":
        return run_export_episode_subprocess(
            episode_dir,
            output_root,
            config=config,
            source_delivery=source_delivery,
        )

    _trajectory, artifacts, metadata = export_episode(
        episode_dir,
        output_root,
        config=config,
        source_delivery=source_delivery,
    )
    return artifacts, metadata


def export_dataset(
    input_root: Path,
    output_root: Path,
    *,
    config: ExportConfig,
    max_episodes: int | None = None,
) -> dict[str, dict]:
    """Export all discovered delivered episodes and write collected.json."""
    output_root = Path(output_root)
    episode_dirs = find_episode_dirs(Path(input_root))
    if max_episodes is not None:
        episode_dirs = episode_dirs[:max_episodes]

    collected: dict[str, dict] = {}
    camera_profiles: dict[str, dict] = {}
    for episode_dir in episode_dirs:
        _artifacts, metadata = export_episode_with_backend_lifecycle(
            episode_dir,
            output_root,
            config=config,
        )
        profile = metadata.pop("_camera_profile", None)
        if profile is not None:
            camera_profiles[metadata["camera_profile_id"]] = profile
        collected[metadata["episode_id"]] = metadata

    metadata_dir = output_root / "metadata"
    write_json(metadata_dir / "collected.json", collected)
    if camera_profiles:
        write_json(metadata_dir / "camera_profiles.json", camera_profiles)
    return collected


def upload_dataset(
    dataset_root: Path,
    s3_prefix: str,
    *,
    aws_profile: str | None = None,
    aws_region: str | None = None,
) -> None:
    """Upload local data/ and metadata/ trees to S3."""
    dataset_root = Path(dataset_root)
    for local_name in ("data", "metadata"):
        local_path = dataset_root / local_name
        if not local_path.exists():
            continue
        remote_path = f"{s3_prefix.rstrip('/')}/{local_name}/"
        sync_local_dir_to_s3(
            local_path,
            remote_path,
            aws_profile=aws_profile,
            aws_region=aws_region,
        )


def load_collected(dataset_root: Path) -> dict[str, dict]:
    """Load collected.json from a dataset root."""
    collected_path = Path(dataset_root) / "metadata" / "collected.json"
    return json.loads(collected_path.read_text())


__all__ = [
    "export_dataset",
    "export_episode",
    "export_episode_with_backend_lifecycle",
    "find_episode_dirs",
    "finalize_dataset_metadata",
    "load_collected",
    "upload_dataset",
]
