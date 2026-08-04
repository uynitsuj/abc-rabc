"""Subprocess-backed episode export helpers.

Madrona's BatchRenderer cannot be recreated safely in the same long-lived
Python process. For that backend, export episodes in isolated subprocesses.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from yam_sim.dataset_export.types import EpisodeArtifacts, ExportConfig


def run_export_episode_subprocess(
    episode_dir: Path,
    output_root: Path,
    *,
    config: ExportConfig,
    source_delivery: str | None = None,
) -> tuple[EpisodeArtifacts, dict]:
    """Export one episode in a fresh subprocess and return its metadata."""
    episode_dir = Path(episode_dir)
    output_root = Path(output_root)
    result_dir = output_root / ".export_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{episode_dir.name}.json"
    if result_path.exists():
        result_path.unlink()

    cmd = [
        sys.executable,
        "-m",
        "yam_sim.dataset_export.cli",
        "export-episode",
        "--episode-dir",
        str(episode_dir),
        "--output-root",
        str(output_root),
        "--result-json",
        str(result_path),
        "--batch-name",
        config.batch_name,
        "--render-backend",
        config.render_backend,
        "--img-width",
        str(config.image_width),
        "--img-height",
        str(config.image_height),
        "--fps",
        str(config.fps),
        "--sim-batch-size",
        str(config.sim_batch_size),
        "--state-pad-size",
        str(config.state_pad_size),
        "--action-pad-size",
        str(config.action_pad_size),
    ]
    if config.gpu_id is not None:
        cmd.extend(["--gpu-id", str(config.gpu_id)])
    if source_delivery:
        cmd.extend(["--source-delivery", source_delivery])

    subprocess.run(cmd, check=True)

    payload = json.loads(result_path.read_text())
    metadata = payload["metadata"]
    artifacts = EpisodeArtifacts(
        episode_dir=Path(payload["episode_dir"]),
        states_actions_path=Path(payload["states_actions_path"]),
        states_actions_bin_path=Path(payload["states_actions_bin_path"]),
        initial_qpos_path=Path(payload["initial_qpos_path"]),
        episode_metadata_path=Path(payload["episode_metadata_path"]),
        video_paths={key: Path(value) for key, value in payload["video_paths"].items()},
    )
    result_path.unlink(missing_ok=True)
    return artifacts, metadata
