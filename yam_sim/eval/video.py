"""Per-episode video writing for the eval harness.

When evaluating a batch of ``num_worlds`` worlds in parallel, each world is an
independent episode. :class:`EpisodeVideoWriters` opens one writer per world and
appends that world's camera row each step, so frames are streamed to disk rather
than buffered (a full batch of episodes would otherwise hold gigabytes of frames).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class EpisodeVideoWriters:
    """Stream one video file per world for a single batch of episodes.

    Falls back to per-episode PNG frame directories when ``imageio`` is missing,
    mirroring the fallback in ``run_policy``.
    """

    def __init__(self, out_paths: list[Path], fps: int) -> None:
        self._out_paths = [Path(p) for p in out_paths]
        self._fps = fps
        self._writers: list[object] | None = None
        self._frame_dirs: list[Path] | None = None
        self._frame_counts: list[int] | None = None

        for path in self._out_paths:
            path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import imageio

            self._writers = [
                imageio.get_writer(str(path), fps=fps, macro_block_size=1)
                for path in self._out_paths
            ]
        except ImportError:
            self._frame_dirs = [path.with_suffix("") for path in self._out_paths]
            for frame_dir in self._frame_dirs:
                frame_dir.mkdir(parents=True, exist_ok=True)
            self._frame_counts = [0] * len(self._out_paths)

    def append(self, per_world_frames: np.ndarray) -> None:
        """Append one frame per world. ``per_world_frames`` is ``(num_worlds, H, W, 3)``."""
        frames = np.asarray(per_world_frames)
        if frames.shape[0] != len(self._out_paths):
            raise ValueError(
                f"Expected {len(self._out_paths)} world frames, got {frames.shape[0]}"
            )
        if self._writers is not None:
            for writer, frame in zip(self._writers, frames):
                writer.append_data(frame)  # type: ignore[attr-defined]
        else:
            from PIL import Image

            assert self._frame_dirs is not None and self._frame_counts is not None
            for idx, frame in enumerate(frames):
                count = self._frame_counts[idx]
                Image.fromarray(frame).save(
                    self._frame_dirs[idx] / f"frame_{count:05d}.png"
                )
                self._frame_counts[idx] = count + 1

    def close(self) -> None:
        if self._writers is not None:
            for writer in self._writers:
                writer.close()  # type: ignore[attr-defined]
            self._writers = None
