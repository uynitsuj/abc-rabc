"""VR teleoperation with Vuer MuJoCo WASM + GELLO leaders.

Serves a MuJoCo scene via Vuer's browser-based MuJoCo WASM engine.
GELLO leader joint positions are streamed to the browser as ctrl
targets. Physics and rendering happen in the browser — the server
just relays GELLO commands.

Usage:
    # Terminal 1: Start the VR teleop server
    uv run python -m yam_sim.examples.vuer_gello_teleop --task blocks

    # Terminal 2: Start GELLO leader(s)
    uv run python -m yam_sim.teleop.gello_leader --name right --device /dev/cu.usbserial-XXX --hardware clapd

    # Open in browser: http://localhost:8012
    # Open in VR headset: http://<your-ip>:8012
"""

from __future__ import annotations

import argparse
import os
import time
import xml.etree.ElementTree as ET
from asyncio import sleep
from pathlib import Path

import numpy as np

try:
    from vuer import Vuer, VuerSession
    from vuer.schemas import MuJoCo, Scene, PointerControls, AmbientLight, group
except ImportError:
    raise SystemExit(
        "Missing vuer dependencies.\n"
        'Install with: uv add "vuer[all]==0.0.53" "params-proto==2.13.2"'
    )

try:
    import zmq
except ImportError:
    raise SystemExit("Missing zmq. Install with: uv add pyzmq")

from yam_sim.teleop import communication as comms


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
    parser = argparse.ArgumentParser(description="VR GELLO teleop with Vuer + MuJoCo WASM")
    parser.add_argument("--task", type=str, default="blocks",
                        choices=["bottles", "marker", "ball_sorting", "empty",
                                 "dishrack", "chess", "chess2", "blocks",
                                 "mug_tree", "mug_flip"])
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--left-leader", type=str, default="left")
    parser.add_argument("--right-leader", type=str, default="right")
    parser.add_argument("--control-rate", type=float, default=90.0,
                        help="Rate at which ctrl updates are sent to browser")
    parser.add_argument("--tunnel", type=str, default="local",
                        choices=["local", "cloudflare", "localtunnel", "ngrok"],
                        help="Tunneling: local (default), cloudflare, localtunnel (npm), or ngrok")
    parser.add_argument("--tunnel-url", type=str, default=None,
                        help="Explicit tunnel URL (e.g. https://mydomain.ngrok.app). Skips auto-detection.")
    args = parser.parse_args()

    # Resolve scene
    scene_xml = get_scene_xml_path(args.task)
    models_dir = scene_xml.parent

    # Ensure .mjcf.xml symlink exists
    mjcf_name = scene_xml.stem + ".mjcf.xml"
    mjcf_path = models_dir / mjcf_name
    if not mjcf_path.exists():
        os.symlink(scene_xml.name, str(mjcf_path))

    asset_files = collect_asset_paths(str(scene_xml))

    # Start tunnel if requested
    tunnel_url = args.tunnel_url  # Use explicit URL if provided
    tunnel_proc = None
    if tunnel_url:
        print(f"Using tunnel URL: {tunnel_url}")
    elif args.tunnel != "local":
        import subprocess
        import re as re_mod
        import time as _time

        if args.tunnel == "cloudflare":
            print(f"Starting Cloudflare tunnel on port {args.port}...")
            cmd = ["cloudflared", "tunnel", "--url", f"http://localhost:{args.port}"]
            url_pattern = r'(https://[^\s]+\.trycloudflare\.com)'
        elif args.tunnel == "localtunnel":
            print(f"Starting localtunnel on port {args.port}...")
            cmd = ["lt", "--port", str(args.port)]
            url_pattern = r'(https://[^\s]+\.loca\.lt)'
        elif args.tunnel == "ngrok":
            print(f"Starting ngrok on port {args.port}...")
            cmd = ["ngrok", "http", str(args.port), "--log", "stdout"]
            url_pattern = r'(https://[^\s]+\.ngrok[^\s]*\.app)'

        try:
            tunnel_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            deadline = _time.time() + 15
            while _time.time() < deadline:
                line = tunnel_proc.stdout.readline()
                if not line:
                    _time.sleep(0.1)
                    continue
                match = re_mod.search(url_pattern, line)
                if match:
                    tunnel_url = match.group(1)
                    print(f"Tunnel URL: {tunnel_url}")
                    break

            if not tunnel_url:
                print("Warning: Could not get tunnel URL. Falling back to localhost.")
        except FileNotFoundError:
            tool = {"cloudflare": "cloudflared", "localtunnel": "lt (npm install -g localtunnel)", "ngrok": "ngrok"}[args.tunnel]
            print(f"Error: {tool} not found. Install it first.")
            print("Falling back to localhost.")

    # Build asset URLs based on tunnel or local
    if tunnel_url:
        asset_prefix = f"{tunnel_url}/static"
    else:
        asset_prefix = f"http://localhost:{args.port}/static"

    src_url = f"{asset_prefix}/{mjcf_name}"
    asset_urls = [f"{asset_prefix}/assets/{a}" for a in asset_files]

    print(f"Task: {args.task}")
    print(f"Scene: {scene_xml.name} ({len(asset_files)} assets)")
    print(f"Asset prefix: {asset_prefix}")

    # Get actuator count from the XML to know ctrl size
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    n_actuators = model.nu
    print(f"Actuators: {n_actuators}")

    # Get initial ctrl from keyframe
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    init_ctrl = data.ctrl.copy().tolist()
    print(f"Init ctrl: {init_ctrl}")

    # ZMQ
    zmq_context = zmq.Context()
    left_sub = comms.create_subscriber(zmq_context, f"{args.left_leader}_actions", conflate=1)
    right_sub = comms.create_subscriber(zmq_context, f"{args.right_leader}_actions", conflate=1)
    # Non-blocking ZMQ - poll with zero timeout for minimum latency
    left_sub.setsockopt(zmq.RCVTIMEO, 0)
    right_sub.setsockopt(zmq.RCVTIMEO, 0)

    def poll_leader(sub):
        try:
            msg, _ = comms.subscribe(sub)
            return msg
        except (zmq.Again, Exception):
            return None

    # Vuer app — serve from models dir so XML + assets resolve
    cors_origins = "https://vuer.ai,http://localhost:8012"
    if tunnel_url:
        cors_origins += f",{tunnel_url}"
    app = Vuer(static_root=str(models_dir), port=args.port, cors=cors_origins, host="0.0.0.0")

    # Shared state
    current_ctrl = list(init_ctrl)
    connected = False
    contrib_loaded = False

    from vuer.events import ClientEvent

    @app.add_handler("ON_CONTRIB_LOAD")
    async def on_contrib_load(event: ClientEvent, proxy: VuerSession):
        nonlocal contrib_loaded
        contrib_loaded = True
        print("MuJoCo WASM contrib module loaded in browser")

    @app.spawn(start=True)
    async def run(proxy: VuerSession):
        nonlocal current_ctrl, connected, contrib_loaded

        contrib_loaded = False

        # Wait for the MuJoCo contrib module to load in the browser
        print("Waiting for MuJoCo WASM module to load...")
        for _ in range(100):  # up to 10 seconds
            if contrib_loaded:
                break
            await sleep(0.1)

        if not contrib_loaded:
            print("Warning: MuJoCo module didn't signal load, inserting anyway...")

        await sleep(0.5)

        # Set bare scene with just MuJoCo — no extra lights/grid
        # Rotate scene -90° around Y so the table faces the VR user
        proxy.set @ Scene(
            rawChildren=[
                AmbientLight(intensity=0.8, key="ambient"),
                group(
                    MuJoCo(
                        key="sim",
                        src=src_url,
                        assets=asset_urls,
                        ctrl=current_ctrl,
                        pause=False,
                        useLights=False,
                        visible=[0, 1, 2],
                        fps=60,
                    ),
                    key="scene-root",
                    rotation=[0, 1.5708, 0],
                ),
            ],
            bgChildren=[PointerControls()],
            grid=False,
        )

        await sleep(1.0)
        print(f"Vuer VR server: http://localhost:{args.port}")
        print(f"Waiting for GELLO leaders on '{args.left_leader}' and '{args.right_leader}'...")

        GRIP_SCALE = 0.0475
        dt = 1.0 / args.control_rate

        while True:
            # Poll GELLO — non-blocking, zero latency
            left_pos = poll_leader(left_sub)
            right_pos = poll_leader(right_sub)

            if left_pos is not None or right_pos is not None:
                if not connected:
                    connected = True
                    print("GELLO connected!")

                if left_pos is not None and len(left_pos) >= 7:
                    for i in range(6):
                        current_ctrl[i] = float(left_pos[i])
                    current_ctrl[6] = float(np.clip(left_pos[6] * GRIP_SCALE, 0, GRIP_SCALE))
                if right_pos is not None and len(right_pos) >= 7:
                    for i in range(6):
                        current_ctrl[7 + i] = float(right_pos[i])
                    current_ctrl[13] = float(np.clip(right_pos[6] * GRIP_SCALE, 0, GRIP_SCALE))

            # Always send current ctrl (even if unchanged) to keep stream alive
            if connected:
                proxy.update @ MuJoCo(
                    key="sim",
                    ctrl=current_ctrl,
                )

            await sleep(dt)

    print(f"\n[vuer-gello] Starting on port {args.port}")
    print(f"[vuer-gello] Local: http://localhost:{args.port}")
    if tunnel_url:
        print(f"[vuer-gello] Tunnel: {tunnel_url}")
        print(f"[vuer-gello] VR headset: open {tunnel_url} in headset browser")
    else:
        print(f"[vuer-gello] VR: http://<your-ip>:{args.port}")
        print(f"[vuer-gello] For VR headset access, re-run with --tunnel cloudflare")


if __name__ == "__main__":
    main()
