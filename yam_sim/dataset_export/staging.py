"""Local scratch staging helpers for remote episode export."""

from __future__ import annotations

import shutil
from pathlib import Path

from yam_sim.dataset_export.s3_source import S3EpisodeSource
from yam_sim.dataset_export.s3_utils import copy_s3_dir_to_local


def stage_episode_locally(
    source: S3EpisodeSource,
    scratch_root: Path,
    *,
    aws_profile: str | None = None,
    aws_region: str | None = None,
) -> Path:
    """Download one remote episode into a local scratch directory."""
    scratch_root = Path(scratch_root)
    stage_dir = scratch_root / source.relative_episode_prefix
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    return copy_s3_dir_to_local(
        source.prefix_uri,
        stage_dir,
        aws_profile=aws_profile,
        aws_region=aws_region,
    )


def cleanup_local_tree(path: Path, *, stop_at: Path | None = None) -> None:
    """Remove a local directory tree and prune empty parents up to stop_at."""
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    if stop_at is None:
        return

    stop_at = Path(stop_at).resolve()
    parent = path.parent.resolve()
    while parent != stop_at and stop_at in parent.parents:
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent
