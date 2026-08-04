"""VR teleoperation with Vuer and MuJoCo.

Serves a MuJoCo scene via Vuer's web-based VR viewer. The MuJoCo physics
runs in the browser via WebAssembly, and hand tracking from a VR headset
(Quest, Vision Pro, etc.) controls the robot's mocap bodies.

Usage:
    # Install vuer
    pip install "vuer[all]"

    # Run the VR teleop server
    python -m yam_sim.examples.vuer_teleop --task blocks

    # Open in VR headset browser:
    #   http://<your-ip>:8012

    # Or use the vuer editor:
    #   https://vuer.ai/editor?ws=ws://localhost:8012
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import xml.etree.ElementTree as ET
from asyncio import sleep
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np

try:
    from vuer import Vuer, VuerSession
    from vuer.events import ClientEvent
    from vuer.schemas import (
        Box,
        Group,
        HandActuator,
        Html,
        MuJoCo,
        Octahedron,
        Sphere,
        group,
        span,
    )
except ImportError:
    raise SystemExit(
        "Missing vuer dependencies.\n"
        'Install with: pip install "vuer[all]"'
    )


def collect_asset_paths(xml_path: str) -> list[str]:
    """Parse a MuJoCo XML and return all asset file paths."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    paths = set()
    for asset in root.iter("asset"):
        for child in asset:
            fp = child.attrib.get("file")
            if fp:
                paths.add(fp)
    return sorted(paths)


def get_scene_xml_path(task: str) -> Path:
    """Get the path to a scene XML file for a given task name."""
    models_dir = Path(__file__).parent.parent / "models"
    scene_map = {
        "bottles": "yam_bimanual_scene.xml",
        "marker": "yam_marker_scene.xml",
        "ball_sorting": "yam_ball_sorting_scene.xml",
        "empty": "yam_bimanual_empty.xml",
        "dishrack": "yam_dishwasher_scene.xml",
        "chess": "yam_chess_scene.xml",
        "chess2": "yam_chess2_scene.xml",
        "blocks": "yam_blocks_scene.xml",
        "mug_tree": "yam_mug_tree_scene.xml",
        "mug_flip": "yam_mug_flip_scene.xml",
    }
    filename = scene_map.get(task)
    if filename is None:
        raise ValueError(f"Unknown task '{task}'. Available: {list(scene_map.keys())}")
    return models_dir / filename


def main():
    parser = argparse.ArgumentParser(
        description="VR teleoperation with Vuer and MuJoCo"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="blocks",
        choices=[
            "bottles", "marker", "ball_sorting", "empty", "dishrack",
            "chess", "chess2", "blocks", "mug_tree", "mug_flip",
        ],
        help="Task scene (default: blocks)",
    )
    parser.add_argument(
        "--port", type=int, default=8012, help="Vuer server port"
    )
    parser.add_argument(
        "--actuators",
        type=str,
        default="duo",
        choices=["mono", "duo", "none"],
        help="Hand actuator mode: mono (right only), duo (both hands), none (hands only)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to save recorded trajectories",
    )
    parser.add_argument(
        "--visible-groups",
        type=str,
        default="0,1,2",
        help="Comma-separated visible geom groups",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=50,
        help="MuJoCo simulation FPS in browser",
    )
    args = parser.parse_args()

    visible_groups = [int(x) for x in args.visible_groups.split(",")]

    # Resolve scene XML and assets
    scene_xml = get_scene_xml_path(args.task)
    models_dir = scene_xml.parent
    assets_dir = models_dir / "assets"

    # Vuer needs a .mjcf.xml extension — create a symlink if needed
    mjcf_name = scene_xml.stem + ".mjcf.xml"
    mjcf_path = models_dir / mjcf_name
    if not mjcf_path.exists():
        os.symlink(scene_xml.name, str(mjcf_path))
        print(f"Created symlink: {mjcf_name} -> {scene_xml.name}")

    # Collect asset paths from XML
    asset_files = collect_asset_paths(str(scene_xml))
    print(f"Scene: {scene_xml.name}")
    print(f"Assets: {len(asset_files)} files")

    # Vuer serves static files from a root directory
    # Our XML references assets like "i2rt_yam/assets/model2.stl"
    # with meshdir="assets", so the full path is models/assets/i2rt_yam/...
    # We serve from models_dir so the XML is at /static/{mjcf_name}
    # and assets are at /static/assets/{path}
    static_root = str(models_dir)
    asset_prefix = f"http://localhost:{args.port}/static"
    src_url = f"{asset_prefix}/{mjcf_name}"

    # Build full asset URLs
    asset_urls = [f"{asset_prefix}/assets/{a}" for a in asset_files]

    print(f"Serving from: {static_root}")
    print(f"Scene URL: {src_url}")

    # Data collection setup
    session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.data_dir, args.task, session_stamp)
    frames_dir = os.path.join(run_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Create Vuer app
    app = Vuer(static_root=static_root, port=args.port)

    def _get_mujoco_model():
        """Build the MuJoCo scene component with hand actuators."""
        actuators = []

        if args.actuators in ("mono", "duo"):
            actuators.append(
                HandActuator(
                    key="right-pinch",
                    cond="right-squeeze",
                    value="right:thumb-tip,right:index-finger-tip",
                    offset=0.10,
                    scale=-12,
                    low=0,
                    high=1,
                    ctrlId=-1,
                )
            )
        if args.actuators == "duo":
            actuators.append(
                HandActuator(
                    key="left-pinch",
                    cond="left-squeeze",
                    value="left:thumb-tip,left:index-finger-tip",
                    offset=0.10,
                    scale=-12,
                    low=0,
                    high=1,
                    ctrlId=-2,
                )
            )

        return MuJoCo(
            *actuators,
            key="mujoco-sim",
            src=src_url,
            assets=asset_urls,
            frameKeys="mocap_pos mocap_quat qpos qvel ctrl",
            pause=True,
            useLights=False,
            visible=visible_groups,
            mocapHandleSize=0.05,
            mocapHandleWireframe=True,
            fps=args.fps,
        )

    # State
    is_loaded = False
    demo_counter = 0
    frame_stack: list[dict] = []
    box_state = "#23aaff"

    @app.add_handler("ON_CONTRIB_LOAD")
    async def on_contrib_load(event: ClientEvent, proxy: VuerSession):
        nonlocal is_loaded
        is_loaded = True
        print("VR client loaded MuJoCo contrib module")

    @app.add_handler("ON_MUJOCO_LOAD")
    async def on_mujoco_load(event: ClientEvent, proxy: VuerSession):
        print("MuJoCo scene loaded in browser")

    @app.add_handler("ON_MUJOCO_FRAME")
    async def on_mujoco_frame(event: ClientEvent, proxy: VuerSession):
        nonlocal frame_stack
        frame = event.value["keyFrame"]
        frame_stack.append(frame)

    async def handle_reset(log_trajectory: bool, proxy: VuerSession):
        nonlocal demo_counter, frame_stack, box_state

        if log_trajectory and frame_stack:
            demo_counter += 1
            ep_path = os.path.join(frames_dir, f"ep_{demo_counter:05d}.pkl")
            with open(ep_path, "wb") as f:
                pickle.dump(frame_stack[10:], f)  # clip first 10 frames
            print(f"Saved trajectory ({len(frame_stack)} frames) to {ep_path}")

        proxy.upsert @ _get_mujoco_model()
        box_state = "#FFA500"
        await sleep(0.6)
        box_state = "#54f963"
        frame_stack = []

    @app.add_handler("ON_CLICK")
    async def on_click(event: ClientEvent, proxy: VuerSession):
        key = event.value["key"]
        if key == "reset-button":
            await handle_reset(log_trajectory=True, proxy=proxy)
        elif key == "delete-button":
            await handle_reset(log_trajectory=False, proxy=proxy)
        print(f"Clicked: {key}, demos: {demo_counter}")

    @app.spawn(start=True)
    async def run(proxy: VuerSession):
        nonlocal is_loaded, box_state

        is_loaded = False
        t0 = perf_counter() + 5.0

        while not is_loaded and perf_counter() < t0:
            print("\rWaiting for VR client to load...", end="", flush=True)
            await sleep(1.0)
        print()

        if not is_loaded:
            print("Timed out waiting for client. Inserting MuJoCo scene anyway.")

        print("Inserting MuJoCo scene into VR view")
        proxy.upsert @ _get_mujoco_model()
        await sleep(1.0)

        _prev_box = None
        while True:
            if _prev_box == box_state:
                await sleep(0.016)
                continue

            _prev_box = box_state
            await sleep(0.016)

            # VR UI buttons
            proxy.upsert @ group(
                Html(
                    span("Save & Reset"),
                    key="reset-label",
                    style={"top": 30, "width": 200, "fontSize": 20},
                ),
                Box(
                    args=[0.25, 0.25, 0.25],
                    key="reset-button",
                    material={"color": box_state},
                ),
                key="reset-button",
                position=[-0.4, 1.4, -1],
            )

            proxy.upsert @ group(
                Html(
                    span("Discard & Reset"),
                    key="delete-label",
                    style={"top": 30, "width": 200, "fontSize": 20},
                ),
                Octahedron(
                    args=[0.15, 0],
                    key="delete-button",
                    material={"color": box_state},
                ),
                key="delete-button",
                position=[0.4, 1.4, -1],
            )

            proxy.upsert @ group(
                Html(
                    span(f"Trajectory: {demo_counter}"),
                    key="traj-label",
                    style={"top": 30, "width": 200, "fontSize": 20},
                ),
                key="traj-label",
                position=[0, 1.9, -1],
            )

    print(f"\nVuer VR teleop server running on port {args.port}")
    print(f"Open in VR headset: http://<your-ip>:{args.port}")
    print(f"Or editor: https://vuer.ai/editor?ws=ws://localhost:{args.port}")
    print(f"Data will be saved to: {run_dir}")


if __name__ == "__main__":
    main()
