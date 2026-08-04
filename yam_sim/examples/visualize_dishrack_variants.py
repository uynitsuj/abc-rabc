"""Render a fixed sweep of all dishrack or plate mesh variants.

Examples:
    MUJOCO_GL=egl uv run python -m yam_sim.examples.visualize_dishrack_variants
    MUJOCO_GL=egl uv run python -m yam_sim.examples.visualize_dishrack_variants --plate-variant plate_0
    MUJOCO_GL=egl uv run python -m yam_sim.examples.visualize_dishrack_variants --sweep plate --rack-variant dish_rack_7
    MUJOCO_GL=egl uv run python -m yam_sim.examples.visualize_dishrack_variants --sheet-out /tmp/dishrack_variants.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys

if "MUJOCO_GL" not in os.environ and sys.platform.startswith("linux"):
    os.environ["MUJOCO_GL"] = "egl"
    os.environ.setdefault("EGL_LOG_LEVEL", "fatal")

import numpy as np

import yam_sim
from yam_sim.examples.visualize_randomization import (
    _add_label,
    _render_camera_frame,
    _resize_to_first_frame,
    _select_camera_name,
    write_contact_sheet,
    write_mp4,
)
from yam_sim.randomization import _dishrack_variant_names
from yam_sim.randomization import _dishrack_canonical_variant_name


def _validate_variant(kind: str, variant_name: str) -> str:
    variant_name = _dishrack_canonical_variant_name(kind, variant_name)
    variants = set(_dishrack_variant_names(kind))
    if variant_name not in variants:
        available = ", ".join(sorted(variants))
        raise ValueError(f"Unknown {kind} variant '{variant_name}'. Available: {available}")
    return variant_name


def _format_variant_label(variant_name: str) -> str:
    match = re.search(r"(\d+)$", variant_name)
    if match is not None:
        return match.group(1)
    return variant_name


def _format_rack_variant_label(rack_variant: str) -> str:
    return _format_variant_label(rack_variant)


def _format_plate_variant_label(plate_variant: str) -> str:
    return _format_variant_label(plate_variant)


def _label_text(
    *,
    frame_idx: int,
    total_frames: int,
    camera_name: str,
    sweep_kind: str,
    rack_variant: str,
    plate_variant: str,
) -> str:
    return "\n".join(
        [
            f"{'dishrack' if sweep_kind == 'dish_rack' else 'plate'} [{frame_idx + 1}/{total_frames}]",
            f"cam={camera_name}",
            f"rack={_format_rack_variant_label(rack_variant)}",
            f"plate={_format_plate_variant_label(plate_variant)}",
        ]
    )


def capture_dishrack_variant_frames(args: argparse.Namespace) -> list[np.ndarray]:
    sweep_kind = args.sweep
    rack_variant = _validate_variant("dish_rack", args.rack_variant)
    plate_variant = _validate_variant("plate", args.plate_variant)
    sweep_variants = _dishrack_variant_names(sweep_kind)

    if sweep_kind == "dish_rack":
        fixed_label = f"plate={plate_variant}"
    else:
        fixed_label = f"rack={_format_rack_variant_label(rack_variant)}"

    print(f"Rendering {len(sweep_variants)} {sweep_kind} variants with {fixed_label}")
    env = yam_sim.make_env(
        scene=args.scene,
        task="dishrack",
        render_cameras=False,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
    )
    try:
        randomizer = getattr(env, "_task_randomizer", None)
        if randomizer is None:
            raise RuntimeError("dishrack env did not expose a task randomizer")

        frames: list[np.ndarray] = []
        for idx, sweep_variant in enumerate(sweep_variants):
            current_rack_variant = sweep_variant if sweep_kind == "dish_rack" else rack_variant
            current_plate_variant = sweep_variant if sweep_kind == "plate" else plate_variant
            randomizer._current_scale_states = {}
            randomizer._reload_variant_scene(current_rack_variant, current_plate_variant, {})
            camera_name = _select_camera_name(env.model, args.camera)
            frame_hwc = _render_camera_frame(env, camera_name)
            label = _label_text(
                frame_idx=idx,
                total_frames=len(sweep_variants),
                camera_name=camera_name,
                sweep_kind=sweep_kind,
                rack_variant=current_rack_variant,
                plate_variant=current_plate_variant,
            )
            frames.append(_add_label(frame_hwc, label))
            print(".", end="", flush=True)

        print(f" done ({len(frames)} frames)")
        return frames
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one frame for each dishrack or plate mesh variant")
    parser.add_argument("--fps", type=int, default=6, help="Output MP4 frame rate")
    parser.add_argument("--out", type=str, default="dishrack_variants.mp4", help="Output MP4 path")
    parser.add_argument(
        "--sheet-out",
        type=str,
        default="dishrack_variants_sheet.png",
        help="Optional PNG path for a tiled contact sheet",
    )
    parser.add_argument("--sheet-cols", type=int, default=5, help="Number of columns for --sheet-out")
    parser.add_argument("--scene", choices=("training", "eval", "hybrid"), default="hybrid")
    parser.add_argument("--camera", type=str, default="auto", help="Camera name to render, or 'auto'")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument(
        "--sweep",
        choices=("dish_rack", "plate"),
        default="dish_rack",
        help="Object family to sweep across all variants",
    )
    parser.add_argument(
        "--plate-variant",
        type=str,
        default="plate_0",
        help="Plate variant to keep fixed while sweeping dishracks, or default plate when validating args",
    )
    parser.add_argument(
        "--rack-variant",
        type=str,
        default="dish_rack_0",
        help="Dishrack variant to keep fixed while sweeping plates",
    )
    args = parser.parse_args()

    frames = capture_dishrack_variant_frames(args)
    if not frames:
        raise RuntimeError("No frames captured")

    resized_frames = _resize_to_first_frame(frames)
    output_path = Path(args.out)
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
