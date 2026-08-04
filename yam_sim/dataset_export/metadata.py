"""Dataset metadata helpers for export and upload."""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from yam_sim.dataset_export.types import ExportConfig, ExportTrajectory


def episode_id_from_dir(episode_dir: Path) -> str:
    """Normalize an episode directory name to the exported episode id."""
    name = episode_dir.name
    return name if name.startswith("episode_") else f"episode_{name}"


def normalize_task_name(instruction: str | None, fallback: str) -> str:
    """Convert free-form task text into the xdof dataset key format."""
    value = (instruction or fallback or "unknown").strip().lower()
    chars: list[str] = []
    last_was_sep = False
    for char in value:
        if char.isalnum():
            chars.append(char)
            last_was_sep = False
        elif not last_was_sep:
            chars.append("_")
            last_was_sep = True
    return "".join(chars).strip("_") or "unknown"


def data_date_from_timestamps(timestamps: Any) -> str:
    """Derive the yam-style data_date token from Unix timestamps if available."""
    if timestamps is None or len(timestamps) == 0:
        return "unknown"
    ts0 = float(timestamps[0])
    if ts0 < 1e9:
        return "unknown"
    dt = datetime.fromtimestamp(ts0, tz=timezone.utc)
    return dt.strftime("%b_%d").lower()


def make_camera_profile_id(profile: dict[str, Any]) -> str:
    """Generate a stable camera profile id for collected metadata."""
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]
    cameras = profile.get("cameras", [])
    resolution = profile.get("resolution", {})
    width = resolution.get("width", "unknown")
    height = resolution.get("height", "unknown")
    backend = profile.get("backend", "sim")
    return f"{backend}_{len(cameras)}cam_{width}x{height}_{digest}"


def build_camera_profile(
    *,
    camera_names: tuple[str, ...],
    config: ExportConfig,
) -> dict[str, Any]:
    """Build the stable camera profile payload stored in collected metadata."""
    return {
        "backend": config.render_backend,
        "fps": config.fps,
        "resolution": {
            "width": config.image_width,
            "height": config.image_height,
        },
        "cameras": list(camera_names),
    }


def build_collected_entry(
    trajectory: ExportTrajectory,
    *,
    config: ExportConfig,
    operator_id: str = "",
) -> dict[str, Any]:
    """Build the collected.json metadata entry for one exported episode."""
    profile = build_camera_profile(camera_names=trajectory.camera_names, config=config)
    return {
        "batch_name": config.batch_name,
        "source_delivery": trajectory.source_delivery,
        "scene_source": trajectory.scene_source,
        "state_source": trajectory.state_source,
        "task_name": trajectory.task_name,
        "episode_id": trajectory.episode_id,
        "index_name": "collected",
        "data_date": trajectory.data_date,
        "size": int(trajectory.states.shape[0]),
        "operator_id": operator_id,
        "cameras": list(trajectory.camera_names),
        "camera_profile_id": make_camera_profile_id(profile),
        "_camera_profile": profile,
    }


class RunningStats:
    """Streaming statistics helper for norm-stats generation."""

    def __init__(self) -> None:
        self._count = 0
        self._mean: np.ndarray | None = None
        self._mean_of_squares: np.ndarray | None = None
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None
        self._histograms: list[np.ndarray] | None = None
        self._bin_edges: list[np.ndarray] | None = None
        self._num_quantile_bins = 5000

    def update(self, batch: np.ndarray) -> None:
        batch = batch.reshape(-1, batch.shape[-1])
        num_elements, vector_length = batch.shape
        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [
                np.zeros(self._num_quantile_bins, dtype=np.float64)
                for _ in range(vector_length)
            ]
            self._bin_edges = [
                np.linspace(
                    self._min[i] - 1e-10,
                    self._max[i] + 1e-10,
                    self._num_quantile_bins + 1,
                )
                for i in range(vector_length)
            ]
        else:
            if self._mean is None or self._mean_of_squares is None or self._min is None or self._max is None:
                raise RuntimeError("RunningStats internal state is inconsistent")
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)
            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements
        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (
            num_elements / self._count
        )
        self._update_histograms(batch)

    def _adjust_histograms(self) -> None:
        if self._histograms is None or self._bin_edges is None or self._min is None or self._max is None:
            raise RuntimeError("RunningStats histogram state is not initialized")
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])
            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        if self._histograms is None or self._bin_edges is None:
            raise RuntimeError("RunningStats histogram state is not initialized")
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles: list[float]) -> list[np.ndarray]:
        if self._histograms is None or self._bin_edges is None:
            raise RuntimeError("RunningStats histogram state is not initialized")
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                idx = min(idx, len(edges) - 1)
                q_values.append(edges[idx])
            results.append(np.array(q_values))
        return results

    def get_statistics(self) -> dict[str, list[float]]:
        if self._count < 2 or self._mean is None or self._mean_of_squares is None:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")
        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0.0, variance))
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        return {
            "mean": self._mean.tolist(),
            "std": stddev.tolist(),
            "q01": q01.tolist(),
            "q99": q99.tolist(),
        }


def pad_stats(stats_dict: dict[str, dict[str, list[float]]], state_pad_size: int, action_pad_size: int) -> dict:
    """Pad or truncate stats arrays to the training pipeline's expected sizes."""
    padded_stats: dict[str, dict[str, list[float]]] = {}
    for key, stats in stats_dict.items():
        pad_size = state_pad_size if key == "state" else action_pad_size
        padded_stat: dict[str, list[float]] = {}
        for stat_name, stat_array in stats.items():
            arr = np.asarray(stat_array, dtype=np.float64)
            if len(arr) < pad_size:
                arr = np.pad(arr, (0, pad_size - len(arr)), mode="constant", constant_values=0.0)
            elif len(arr) > pad_size:
                arr = arr[:pad_size]
            padded_stat[stat_name] = arr.tolist()
        padded_stats[key] = padded_stat
    return padded_stats


def compute_norm_stats(
    states_actions_paths: dict[str, Path],
    *,
    n_states: int = 14,
    n_actions: int = 14,
    state_pad_size: int = 32,
    action_pad_size: int = 32,
) -> dict[str, dict[str, list[float]]]:
    """Compute padded normalization stats from local states_actions.npy files."""
    state_stats = RunningStats()
    action_stats = RunningStats()
    for path in states_actions_paths.values():
        data = np.load(path)
        if data.ndim != 2 or data.shape[1] < n_states + n_actions:
            raise ValueError(f"Invalid states_actions.npy shape {data.shape} in {path}")
        state_stats.update(data[:, :n_states].astype(np.float64, copy=False))
        action_stats.update(data[:, n_states : n_states + n_actions].astype(np.float64, copy=False))
    return pad_stats(
        {
            "state": state_stats.get_statistics(),
            "actions": action_stats.get_statistics(),
        },
        state_pad_size=state_pad_size,
        action_pad_size=action_pad_size,
    )


def split_episode_ids(
    episode_metadata: dict[str, dict[str, Any]],
    *,
    val_ratio: float = 0.05,
    seed: int = 123,
) -> tuple[list[str], list[str]]:
    """Produce stable train/val episode id lists."""
    episode_ids = sorted(episode_metadata)
    if not episode_ids:
        raise RuntimeError("No episodes available for train/val split")
    rng = random.Random(seed)
    shuffled = list(episode_ids)
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio))
    val_ids = sorted(shuffled[:val_count])
    train_ids = sorted(shuffled[val_count:])
    return train_ids, val_ids


def write_json(path: Path, payload: Any) -> Path:
    """Write JSON with a trailing newline for diff-friendly outputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def finalize_dataset_metadata(
    dataset_root: Path,
    *,
    collected: dict[str, dict[str, Any]] | None = None,
    n_states: int = 14,
    n_actions: int = 14,
    state_pad_size: int = 32,
    action_pad_size: int = 32,
    val_ratio: float = 0.05,
    seed: int = 123,
) -> dict[str, Path]:
    """Generate collected/norm_stats/train-val metadata under dataset_root/metadata."""
    dataset_root = Path(dataset_root)
    metadata_dir = dataset_root / "metadata"
    data_dir = dataset_root / "data"
    if collected is None:
        collected_path = metadata_dir / "collected.json"
        if not collected_path.exists():
            raise FileNotFoundError(f"Missing collected metadata at {collected_path}")
        collected = json.loads(collected_path.read_text())

    states_actions_paths = {
        episode_id: data_dir / episode_id / "states_actions.npy"
        for episode_id in sorted(collected)
    }
    missing = [episode_id for episode_id, path in states_actions_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing states_actions.npy for episodes: {missing[:10]}")

    norm_stats = compute_norm_stats(
        states_actions_paths,
        n_states=n_states,
        n_actions=n_actions,
        state_pad_size=state_pad_size,
        action_pad_size=action_pad_size,
    )
    train_ids, val_ids = split_episode_ids(collected, val_ratio=val_ratio, seed=seed)

    outputs = {
        "collected": write_json(metadata_dir / "collected.json", collected),
        "norm_stats": write_json(metadata_dir / "norm_stats.json", norm_stats),
        "train_episodes": write_json(metadata_dir / "train_episodes.json", train_ids),
        "val_episodes": write_json(metadata_dir / "val_episodes.json", val_ids),
    }
    return outputs
