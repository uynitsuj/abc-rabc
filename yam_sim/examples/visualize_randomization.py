"""Render randomized yam-sim task resets and save them to an MP4.

For each selected task:
  - Create an environment in this repo's runtime path.
  - Run N randomized resets when the task has a randomizer, otherwise render 1 static frame.
  - Capture one camera frame after each reset.
  - Burn the task name and randomization index into the frame.

Examples:
    MUJOCO_GL=egl uv run python -m yam_sim.examples.visualize_randomization
    MUJOCO_GL=egl uv run python -m yam_sim.examples.visualize_randomization --tasks sweep pour --rands 8
    MUJOCO_GL=egl uv run python -m yam_sim.examples.visualize_randomization --collection-only --out /tmp/collection_preview.mp4
    MUJOCO_GL=egl uv run python -m yam_sim.examples.visualize_randomization --tasks dishrack --rands 12 --sheet-out /tmp/dishrack_randomization_sheet.png
    MUJOCO_GL=egl uv run python -m yam_sim.examples.visualize_randomization --clean --mocap --flexible-gripper --tasks sweep
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

if "MUJOCO_GL" not in os.environ and sys.platform.startswith("linux"):
    os.environ["MUJOCO_GL"] = "egl"
    os.environ.setdefault("EGL_LOG_LEVEL", "fatal")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import yam_sim  # noqa: E402
from yam_sim.collection_tasks import maybe_get_data_collection_task  # noqa: E402
from yam_sim.scene_xml import SceneXmlTransformOptions, build_scene_xml  # noqa: E402
from yam_sim.task_registry import (  # noqa: E402
    get_task_scene_xml,
    list_scene_task_names,
    resolve_env_task_name,
)


_SKIP_TASKS: set[str] = set()
_AUTO_CAMERA_PREFERENCE = ("top", "overhead", "left", "right")
_MAX_LABELLED_SCALES = 3


def _format_rack_variant_label(rack_variant: str) -> str:
    match = re.search(r"(\d+)$", rack_variant)
    if match is not None:
        return match.group(1)
    return rack_variant


def _model_camera_names(model: mujoco.MjModel) -> list[str]:
    names: list[str] = []
    for cam_id in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
        if name:
            names.append(name)
    return names


def _select_camera_name(model: mujoco.MjModel, requested: str) -> str:
    available = _model_camera_names(model)
    if not available:
        raise RuntimeError("Scene does not expose any cameras in the MuJoCo model")

    if requested != "auto":
        if requested in available:
            return requested
        print(f"  requested camera '{requested}' not found, falling back to '{available[0]}'")
        return available[0]

    for name in _AUTO_CAMERA_PREFERENCE:
        if name in available:
            return name
    return available[0]


def _render_camera_frame(env, camera_name: str) -> np.ndarray:
    renderer = mujoco.Renderer(
        env.model,
        height=env._camera_height,
        width=env._camera_width,
    )
    try:
        renderer.update_scene(env.data, camera=camera_name)
        return renderer.render().copy()
    finally:
        renderer.close()


def _label_text(task: str, frame_idx: int, total_frames: int, camera_name: str, env) -> str:
    header = f"{task} [{frame_idx + 1}/{total_frames}]"
    extra_lines: list[str] = [f"cam={camera_name}"]
    rand_state = getattr(env, "_last_randomization", None)
    if rand_state is not None:
        extra_lines.append(f"seed={rand_state.seed}")
        scale_states = getattr(rand_state, "scale_states", {})
        if scale_states:
            if len(scale_states) <= _MAX_LABELLED_SCALES:
                formatted = ", ".join(
                    f"{name}={factor:.3f}"
                    for name, factor in sorted(scale_states.items())
                )
                extra_lines.append(f"scale {formatted}")
            else:
                extra_lines.append(f"scales={len(scale_states)} objects")

    if rand_state is not None and getattr(rand_state, "metadata", None):
        metadata = rand_state.metadata
        category = metadata.get("category")
        variant = metadata.get("variant")
        side = metadata.get("side")
        if category and variant:
            line = f"{category}/{variant}"
            if side:
                line += f" side={side}"
            extra_lines.append(line)
        plate_variant = metadata.get("plate_variant")
        rack_variant = metadata.get("dish_rack_variant")
        trash_count = metadata.get("trash_count")
        if plate_variant:
            extra_lines.append(f"plate={plate_variant}")
        if rack_variant:
            extra_lines.append(f"rack={_format_rack_variant_label(rack_variant)}")
        if trash_count is not None:
            extra_lines.append(f"trash_count={int(trash_count)}")
    return "\n".join([header, *extra_lines])


def _add_label(frame_hwc: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(frame_hwc)
    draw = ImageDraw.Draw(img)
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        draw.multiline_text((10 + dx, 10 + dy), text, fill=(0, 0, 0), spacing=2)
    draw.multiline_text((10, 10), text, fill=(255, 255, 255), spacing=2)
    return np.asarray(img)


def _resolve_tasks(task_args: list[str] | None, *, collection_only: bool) -> list[str]:
    if task_args:
        requested = task_args
    elif collection_only:
        requested = []
        seen: set[str] = set()
        from yam_sim.collection_tasks import list_data_collection_tasks

        for group in list_data_collection_tasks():
            if group.env_task not in seen:
                requested.append(group.env_task)
                seen.add(group.env_task)
    else:
        requested = list(list_scene_task_names())

    resolved: list[str] = []
    seen: set[str] = set()
    for task_name in requested:
        collection_group = maybe_get_data_collection_task(task_name)
        if collection_group is not None:
            env_task = collection_group.env_task
        else:
            env_task = resolve_env_task_name(task_name) or task_name
        if env_task in _SKIP_TASKS or env_task in seen:
            continue
        resolved.append(env_task)
        seen.add(env_task)
    return resolved


def _make_env_for_preview(task: str, args: argparse.Namespace):
    transform_options = SceneXmlTransformOptions(
        clean=args.clean,
        mocap=args.mocap,
        flexible_gripper=args.flexible_gripper,
    )
    need_xml_edit = args.clean or args.mocap or args.flexible_gripper

    if task == "inhand_transfer":
        return yam_sim.make_env(
            scene=args.scene,
            task=task,
            render_cameras=False,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            scene_xml_transform_options=transform_options,
        )

    if need_xml_edit:
        scene_path = get_task_scene_xml(task)
        if scene_path is None:
            raise ValueError(f"Unknown task scene: {task}")
        xml, _ = build_scene_xml(scene_path, options=transform_options)
        return yam_sim.make_env(
            scene=args.scene,
            task=task,
            render_cameras=False,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            scene_xml_string=xml,
            scene_xml_transform_options=transform_options,
        )

    return yam_sim.make_env(
        scene=args.scene,
        task=task,
        render_cameras=False,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
    )


def capture_task_frames(task: str, args: argparse.Namespace, *, task_index: int) -> list[np.ndarray]:
    print(f"  [{task}] loading env...", end="", flush=True)
    env = None
    try:
        env = _make_env_for_preview(task, args)
    except Exception as exc:
        print(f" FAILED: {exc}")
        return []

    try:
        has_randomizer = getattr(env, "_task_randomizer", None) is not None
        num_rands = args.rands if has_randomizer else 1
        frames: list[np.ndarray] = []

        for rand_idx in range(num_rands):
            seed = args.seed_offset + task_index * max(args.rands, 1) + rand_idx
            env.reset(seed=seed, randomize=has_randomizer)
            camera_name = _select_camera_name(env.model, args.camera)
            frame_hwc = _render_camera_frame(env, camera_name)
            label = _label_text(task, rand_idx, num_rands, camera_name, env)
            frames.append(_add_label(frame_hwc, label))
            print(".", end="", flush=True)

        print(f" done ({len(frames)} frames)")
        return frames
    except Exception as exc:
        print(f" FAILED during reset/render: {exc}")
        return []
    finally:
        env.close()


def _ensure_even_dims(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    return frame[: height & ~1, : width & ~1]


def write_mp4(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    if not frames:
        raise ValueError("No frames to encode")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [_ensure_even_dims(frame) for frame in frames]

    try:
        import imageio.v3 as iio

        iio.imwrite(str(output_path), np.stack(frames), fps=fps, codec="libx264")
        return
    except ImportError:
        pass

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for idx, frame in enumerate(frames):
            Image.fromarray(frame).save(tmp_path / f"{idx:05d}.png")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(tmp_path / "%05d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ],
            check=True,
        )


def _resize_to_first_frame(frames: list[np.ndarray]) -> list[np.ndarray]:
    if not frames:
        return frames
    height, width = frames[0].shape[:2]
    resized: list[np.ndarray] = []
    for frame in frames:
        if frame.shape[:2] == (height, width):
            resized.append(frame)
            continue
        img = Image.fromarray(frame)
        resized.append(np.asarray(img.resize((width, height), Image.Resampling.BILINEAR)))
    return resized


def build_contact_sheet(frames: list[np.ndarray], *, cols: int) -> np.ndarray:
    if not frames:
        raise ValueError("No frames to tile into a contact sheet")
    cols = max(1, cols)
    resized = _resize_to_first_frame(frames)
    tile_h, tile_w = resized[0].shape[:2]
    rows = (len(resized) + cols - 1) // cols
    sheet = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for idx, frame in enumerate(resized):
        row, col = divmod(idx, cols)
        y0 = row * tile_h
        x0 = col * tile_w
        sheet[y0 : y0 + tile_h, x0 : x0 + tile_w] = frame
    return sheet


def write_contact_sheet(frames: list[np.ndarray], output_path: Path, *, cols: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(build_contact_sheet(frames, cols=cols)).save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render randomized yam-sim task resets to an MP4")
    parser.add_argument("--rands", type=int, default=12, help="Randomized resets per task when a randomizer exists")
    parser.add_argument("--fps", type=int, default=10, help="Output MP4 frame rate")
    parser.add_argument("--out", type=str, default="yam_randomization_preview.mp4", help="Output MP4 path")
    parser.add_argument("--sheet-out", type=str, default=None, help="Optional PNG path for a tiled contact sheet of the captured frames")
    parser.add_argument("--sheet-cols", type=int, default=4, help="Number of columns when writing --sheet-out")
    parser.add_argument("--tasks", nargs="+", default=None, help="Subset of tasks or collection-family names to render")
    parser.add_argument(
        "--collection-only",
        action="store_true",
        help="Render the 10 collection-task env_task families instead of all scene tasks",
    )
    parser.add_argument("--scene", choices=("training", "eval", "hybrid"), default="hybrid")
    parser.add_argument("--camera", type=str, default="auto", help="Camera name to render, or 'auto'")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--clean", action="store_true", help="Apply clean XML transform before rendering")
    parser.add_argument("--mocap", action="store_true", help="Apply mocap XML transform before rendering")
    parser.add_argument(
        "--flexible-gripper",
        action="store_true",
        help="Apply flexible-gripper XML transform before rendering",
    )
    args = parser.parse_args()

    tasks = _resolve_tasks(args.tasks, collection_only=args.collection_only)
    if not tasks:
        print("No tasks selected.")
        sys.exit(1)

    print(f"Rendering {len(tasks)} tasks x {args.rands} resets -> {args.out}")
    all_frames: list[np.ndarray] = []
    for task_index, task in enumerate(tasks):
        all_frames.extend(capture_task_frames(task, args, task_index=task_index))

    if not all_frames:
        print("No frames captured.")
        sys.exit(1)

    output_path = Path(args.out)
    resized_frames = _resize_to_first_frame(all_frames)
    print(f"Writing {len(resized_frames)} frames at {args.fps} fps -> {output_path}")
    write_mp4(resized_frames, output_path, fps=args.fps)
    print(f"Saved preview to {output_path}")
    if args.sheet_out:
        sheet_path = Path(args.sheet_out)
        print(f"Writing contact sheet ({args.sheet_cols} cols) -> {sheet_path}")
        write_contact_sheet(resized_frames, sheet_path, cols=args.sheet_cols)
        print(f"Saved contact sheet to {sheet_path}")


if __name__ == "__main__":
    main()
