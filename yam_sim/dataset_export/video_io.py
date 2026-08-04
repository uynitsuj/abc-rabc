"""Video writing and probing helpers for dataset export."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

import numpy as np


class RawRGBVideoWriter:
    """Stream raw RGB frames into ffmpeg without materializing a second copy."""

    def __init__(
        self,
        output_path: Path,
        *,
        fps: float,
        width: int,
        height: int,
        codec: str = "libx264",
        crf: int = 18,
        preset: str = "fast",
    ) -> None:
        self.output_path = Path(output_path)
        self.fps = fps
        self.width = width
        self.height = height
        self.codec = codec
        self.crf = crf
        self.preset = preset
        self._proc: subprocess.Popen | None = None

    def __enter__(self) -> "RawRGBVideoWriter":
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            self.codec,
            "-preset",
            self.preset,
            "-crf",
            str(self.crf),
            "-bf",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "1",
            str(self.output_path),
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
        )
        return self

    def write_batch(self, frames: np.ndarray) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("Video writer is not open")
        frames = np.asarray(frames)
        if frames.ndim != 4 or frames.shape[1:] != (self.height, self.width, 3):
            raise ValueError(
                f"Expected frames of shape (B, {self.height}, {self.width}, 3), got {frames.shape}"
            )
        if frames.dtype != np.uint8:
            frames = frames.astype(np.uint8, copy=False)
        self._proc.stdin.write(np.ascontiguousarray(frames).tobytes(order="C"))

    def write_frame(self, frame: np.ndarray) -> None:
        self.write_batch(np.asarray(frame)[None, ...])

    def close(self) -> None:
        if self._proc is None:
            return
        stderr = b""
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
            stderr = self._proc.stderr.read() if self._proc.stderr is not None else b""
            ret = self._proc.wait()
            if ret != 0:
                raise RuntimeError(
                    f"ffmpeg failed with exit code {ret} for {self.output_path}:\n{stderr.decode(errors='replace')}"
                )
        finally:
            if self._proc.stderr is not None:
                self._proc.stderr.close()
            self._proc = None

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            if self._proc is not None:
                self._proc.kill()
                self._proc.wait()
                self._proc = None
            return
        self.close()


def write_rgb_video(output_path: Path, frames: np.ndarray, *, fps: float) -> Path:
    """Write an in-memory RGB tensor to MP4 and export frame mappings."""
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected RGB frames with shape (T, H, W, 3), got {frames.shape}")
    with RawRGBVideoWriter(
        output_path,
        fps=fps,
        width=int(frames.shape[2]),
        height=int(frames.shape[1]),
    ) as writer:
        writer.write_batch(frames)
    write_frame_mappings(output_path)
    return output_path


def probe_video_frame_count(video_path: Path) -> int:
    """Return the decoded frame count for a video, or -1 if unavailable."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        return -1
    try:
        return int(result.stdout.strip())
    except (TypeError, ValueError):
        return -1


def write_frame_mappings(video_path: Path) -> Path:
    """Write the torchcodec-compatible frame mapping JSON beside an MP4."""
    cmd = f"""
    ffprobe -v error -select_streams v:0 -show_frames -show_streams \
      -show_entries stream=time_base,avg_frame_rate \
      -show_entries frame=best_effort_timestamp,pkt_duration,key_frame \
      -of json {shlex.quote(str(video_path))}
    """
    data = json.loads(subprocess.check_output(cmd, shell=True, text=True))
    frames = []
    for frame in data.get("frames", []):
        pts = frame.get("best_effort_timestamp")
        if pts is None:
            continue
        duration = frame.get("pkt_duration")
        frames.append(
            {
                "pts": int(pts),
                "duration": int(duration) if duration is not None else None,
                "key_frame": int(frame.get("key_frame", 0)),
            }
        )
    for i in range(1, len(frames)):
        if frames[i - 1]["duration"] in (None, 0):
            frames[i - 1]["duration"] = frames[i]["pts"] - frames[i - 1]["pts"]
    if frames and frames[-1]["duration"] in (None, 0):
        frames[-1]["duration"] = frames[-2]["duration"] if len(frames) > 1 else 1

    output = {"frames": frames}
    try:
        stream = next(s for s in data.get("streams", []) if s.get("time_base"))
        output["stream"] = {
            "time_base": stream.get("time_base"),
            "avg_frame_rate": stream.get("avg_frame_rate"),
        }
    except StopIteration:
        pass

    mapping_path = video_path.with_name(video_path.stem + "_frame_mappings.json")
    mapping_path.write_text(json.dumps(output, indent=2) + "\n")
    return mapping_path
