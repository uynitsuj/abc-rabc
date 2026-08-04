"""Live teleoperation with Viser 3D viewer and MuJoCo sim.

Subscribes to GELLO leader ZMQ topics, steps MuJoCo physics, and renders
the scene live in a Viser browser-based 3D viewport. Combines the sim
follower and viser viewer into a single process.

Usage:
    # Terminal 1: Start the live viser teleop viewer
    python -m yam_sim.examples.viser_teleop --scene hybrid

    # Terminal 2: Start left GELLO leader
    python -m yam_sim.teleop.gello_leader --name left --device /dev/ttyUSB0

    # Terminal 3: Start right GELLO leader
    python -m yam_sim.teleop.gello_leader --name right --device /dev/ttyUSB1

    # Open http://localhost:8080 in browser to see live 3D view

Requirements:
    pip install yam-sim[teleop,viser]
    # For GELLO hardware: pip install dynamixel-sdk
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_LOG_LEVEL", "fatal")

import mujoco
import numpy as np

try:
    import trimesh
    import trimesh.visual
    import trimesh.visual.material
    import viser
    import viser.transforms as vtf
except ImportError:
    raise SystemExit(
        "Missing viser dependencies.\n"
        "Install with: pip install yam-sim[teleop,viser]"
    )

try:
    import zmq
except ImportError:
    raise SystemExit(
        "Missing ZMQ dependency.\n"
        "Install with: pip install yam-sim[teleop]"
    )

from yam_sim.teleop import communication as comms

# Reuse mesh conversion utilities from viser_replay
from yam_sim.examples.viser_replay import (
    _create_primitive_mesh,
    _get_body_name,
    _get_geom_rgba,
    _is_fixed_body,
    _merge_geoms,
    _mujoco_mesh_to_trimesh,
    configure_default_camera,
)
from mujoco import mj_id2name, mjtGeom, mjtObj


class ViserTeleopViewer:
    """Live 3D teleoperation viewer: ZMQ subscriber + MuJoCo sim + Viser."""

    def __init__(
        self,
        env,
        *,
        left_leader: str = "left",
        right_leader: str = "right",
        port: int = 8080,
        control_rate: float = 30.0,
        visible_geom_groups: tuple[int, ...] = (0, 1, 2),
        camera_width: int = 320,
        camera_height: int = 240,
        enable_renderer: bool = False,
        views: str = "mono",
    ):
        self.env = env
        self.model = env.model
        self.data = env.data
        self.left_leader = left_leader
        self.right_leader = right_leader
        self.control_rate = control_rate
        self.visible_geom_groups = visible_geom_groups
        self.camera_width = camera_width
        self.camera_height = camera_height

        # Current sim state
        obs, _ = self.env.reset()
        self._state = obs["state"].copy()
        self._step_count = 0

        # ZMQ setup
        self._zmq_context = zmq.Context()
        self._left_sub = comms.create_subscriber(
            self._zmq_context, f"{left_leader}_actions", conflate=1
        )
        self._right_sub = comms.create_subscriber(
            self._zmq_context, f"{right_leader}_actions", conflate=1
        )

        # MuJoCo offscreen renderer for camera PiP
        self._renderer = None
        if enable_renderer:
            try:
                self._renderer = mujoco.Renderer(
                    self.model, height=camera_height, width=camera_width
                )
                print(f"  MuJoCo renderer: {camera_height}x{camera_width}")
            except Exception as e:
                print(f"  Warning: Camera rendering unavailable ({e})")

        # Viser server (primary — front/free view)
        self.server = viser.ViserServer(port=port)
        configure_default_camera(self.server)
        print(f"Viser server (front): http://localhost:{port}")

        # Extra view servers based on --views mode
        self._extra_servers: list[tuple[viser.ViserServer, dict[int, viser.MeshHandle]]] = []
        if views == "stereo":
            extra_views = [
                ("overhead", port + 1, (0.55, 0.0, 2.0), (0.55, 0.0, 0.75), (1.0, 0.0, 0.0)),
            ]
        elif views == "tri":
            extra_views = [
                ("left", port + 1, (0.5, 0.55, 1.05), (0.55, 0.0, 0.8), (0.0, 0.0, 1.0)),
                ("right", port + 2, (0.5, -0.55, 1.05), (0.55, 0.0, 0.8), (0.0, 0.0, 1.0)),
            ]
        else:
            extra_views = []

        if extra_views:
            for view_name, view_port, cam_pos, cam_target, cam_up in extra_views:
                srv = viser.ViserServer(port=view_port)
                @srv.on_client_connect
                def _set_cam(client: viser.ClientHandle, _p=cam_pos, _t=cam_target, _u=cam_up) -> None:
                    client.camera.position = _p
                    client.camera.look_at = _t
                    client.camera.up_direction = _u
                handles: dict[int, viser.MeshHandle] = {}
                self._extra_servers.append((srv, handles))
                print(f"Viser server ({view_name}): http://localhost:{view_port}")

        # Scene state
        self._mesh_handles: dict[int, viser.MeshHandle] = {}
        self._collision_handles: dict[int, viser.MeshHandle] = {}
        self._show_collision = False
        self._connected = False
        self._recording = False
        self._recorded_actions: list[np.ndarray] = []

        # Build scene and GUI
        self._build_scene()
        for srv, handles in self._extra_servers:
            dummy_col: dict[int, viser.MeshHandle] = {}
            self._build_scene_on_server(srv, handles, dummy_col)
        self._build_gui()
        self._update_scene()

    def _build_scene_on_server(self, server, mesh_handles, collision_handles):
        """Build the same scene on a secondary viser server."""
        server.scene.configure_environment_map(environment_intensity=0.8)

        body_visual: dict[int, list[int]] = {}
        body_collision: dict[int, list[int]] = {}
        for i in range(self.model.ngeom):
            body_id = self.model.geom_bodyid[i]
            if self._is_visual_geom(i):
                body_visual.setdefault(body_id, []).append(i)
            else:
                body_collision.setdefault(body_id, []).append(i)

        server.scene.add_frame("/fixed_bodies", show_axes=False)
        all_body_ids = set(body_visual.keys()) | set(body_collision.keys())

        for body_id in all_body_ids:
            body_name = _get_body_name(self.model, body_id)
            visual_ids = body_visual.get(body_id, [])

            if _is_fixed_body(self.model, body_id):
                nonplane_ids = [
                    gid for gid in visual_ids
                    if self.model.geom_type[gid] != mjtGeom.mjGEOM_PLANE
                ]
                for gid in visual_ids:
                    if self.model.geom_type[gid] == mjtGeom.mjGEOM_PLANE:
                        geom_name = mj_id2name(self.model, mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"
                        server.scene.add_grid(
                            f"/fixed_bodies/{body_name}/{geom_name}",
                            width=2000.0, height=2000.0, infinite_grid=True,
                            fade_distance=50.0, shadow_opacity=0.2,
                            position=self.model.geom_pos[gid],
                            wxyz=self.model.geom_quat[gid],
                        )

                # Split textured vs plain
                textured_ids = []
                plain_ids = []
                for gid in nonplane_ids:
                    matid = self.model.geom_matid[gid]
                    has_tex = False
                    if matid >= 0 and matid < self.model.nmat:
                        texid = int(self.model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)])
                        if texid < 0:
                            texid = int(self.model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGBA)])
                        has_tex = texid >= 0
                    if has_tex:
                        textured_ids.append(gid)
                    else:
                        plain_ids.append(gid)

                if plain_ids:
                    merged = _merge_geoms(self.model, plain_ids)
                    server.scene.add_mesh_trimesh(
                        f"/fixed_bodies/{body_name}", merged,
                        cast_shadow=False, receive_shadow=0.2,
                        position=self.model.body(body_id).pos,
                        wxyz=self.model.body(body_id).quat, visible=True,
                    )
                for gid in textured_ids:
                    geom_name = mj_id2name(self.model, mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"
                    tex_mesh = _merge_geoms(self.model, [gid])
                    server.scene.add_mesh_trimesh(
                        f"/fixed_bodies/{body_name}/{geom_name}", tex_mesh,
                        cast_shadow=False, receive_shadow=0.2,
                        position=self.model.body(body_id).pos,
                        wxyz=self.model.body(body_id).quat, visible=True,
                    )
            else:
                if visual_ids:
                    merged = _merge_geoms(self.model, visual_ids)
                    handle = server.scene.add_mesh_trimesh(
                        f"/bodies/{body_name}", merged, visible=True,
                    )
                    mesh_handles[body_id] = handle

    # ----- Scene building (reused from viser_replay) -----

    def _is_visual_geom(self, geom_id: int) -> bool:
        return int(self.model.geom_group[geom_id]) in self.visible_geom_groups

    def _build_scene(self):
        self.server.scene.configure_environment_map(environment_intensity=0.8)

        body_visual: dict[int, list[int]] = {}
        body_collision: dict[int, list[int]] = {}
        for i in range(self.model.ngeom):
            body_id = self.model.geom_bodyid[i]
            if self._is_visual_geom(i):
                body_visual.setdefault(body_id, []).append(i)
            else:
                body_collision.setdefault(body_id, []).append(i)

        self.server.scene.add_frame("/fixed_bodies", show_axes=False)

        all_body_ids = set(body_visual.keys()) | set(body_collision.keys())

        for body_id in all_body_ids:
            body_name = _get_body_name(self.model, body_id)
            visual_ids = body_visual.get(body_id, [])
            collision_ids = body_collision.get(body_id, [])

            if _is_fixed_body(self.model, body_id):
                nonplane_ids = []
                for gid in visual_ids:
                    if self.model.geom_type[gid] == mjtGeom.mjGEOM_PLANE:
                        geom_name = mj_id2name(self.model, mjtObj.mjOBJ_GEOM, gid)
                        if not geom_name:
                            geom_name = f"geom_{gid}"
                        self.server.scene.add_grid(
                            f"/fixed_bodies/{body_name}/{geom_name}",
                            width=2000.0,
                            height=2000.0,
                            infinite_grid=True,
                            fade_distance=50.0,
                            shadow_opacity=0.2,
                            position=self.model.geom_pos[gid],
                            wxyz=self.model.geom_quat[gid],
                        )
                    else:
                        nonplane_ids.append(gid)

                # Separate textured geoms from non-textured to preserve textures
                textured_ids = []
                plain_ids = []
                for gid in nonplane_ids:
                    matid = self.model.geom_matid[gid]
                    has_tex = False
                    if matid >= 0 and matid < self.model.nmat:
                        texid = int(self.model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)])
                        if texid < 0:
                            texid = int(self.model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGBA)])
                        has_tex = texid >= 0
                    if has_tex:
                        textured_ids.append(gid)
                    else:
                        plain_ids.append(gid)

                if plain_ids:
                    merged = _merge_geoms(self.model, plain_ids)
                    self.server.scene.add_mesh_trimesh(
                        f"/fixed_bodies/{body_name}",
                        merged,
                        cast_shadow=False,
                        receive_shadow=0.2,
                        position=self.model.body(body_id).pos,
                        wxyz=self.model.body(body_id).quat,
                        visible=True,
                    )
                for gid in textured_ids:
                    geom_name = mj_id2name(self.model, mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"
                    tex_mesh = _merge_geoms(self.model, [gid])
                    self.server.scene.add_mesh_trimesh(
                        f"/fixed_bodies/{body_name}/{geom_name}",
                        tex_mesh,
                        cast_shadow=False,
                        receive_shadow=0.2,
                        position=self.model.body(body_id).pos,
                        wxyz=self.model.body(body_id).quat,
                        visible=True,
                    )
            else:
                if visual_ids:
                    merged = _merge_geoms(self.model, visual_ids)
                    handle = self.server.scene.add_mesh_trimesh(
                        f"/bodies/{body_name}",
                        merged,
                        visible=True,
                    )
                    self._mesh_handles[body_id] = handle

                if collision_ids:
                    merged = _merge_geoms(self.model, collision_ids)
                    handle = self.server.scene.add_mesh_trimesh(
                        f"/bodies/{body_name}_collision",
                        merged,
                        visible=self._show_collision,
                    )
                    self._collision_handles[body_id] = handle

    # ----- GUI -----

    def _build_gui(self):
        with self.server.gui.add_folder("Teleop Status", expand_by_default=True):
            self._status_html = self.server.gui.add_html("")
            self._update_status()

        with self.server.gui.add_folder("Controls", expand_by_default=True):
            reset_btn = self.server.gui.add_button("Reset", icon=viser.Icon.REFRESH)

            @reset_btn.on_click
            def _(_):
                self._reset()

            self._record_btn = self.server.gui.add_button(
                "Start Recording", icon=viser.Icon.RECORD_MAIL
            )

            @self._record_btn.on_click
            def _(_):
                if self._recording:
                    self._stop_recording()
                else:
                    self._start_recording()

        with self.server.gui.add_folder("Display", expand_by_default=False):
            collision_cb = self.server.gui.add_checkbox(
                "Show Collision Geoms", initial_value=False
            )

            @collision_cb.on_update
            def _(_):
                self._show_collision = collision_cb.value
                for handle in self._collision_handles.values():
                    handle.visible = self._show_collision

        # Camera PiP
        if self._renderer:
            with self.server.gui.add_folder("Cameras", expand_by_default=True):
                self._cam_handles: dict[str, viser.GuiImageHandle] = {}
                for cam_name in self.env.camera_names:
                    placeholder = np.zeros(
                        (self.camera_height, self.camera_width, 3), dtype=np.uint8
                    )
                    self._cam_handles[cam_name] = self.server.gui.add_image(
                        placeholder,
                        label=cam_name,
                        format="jpeg",
                        jpeg_quality=85,
                    )

    # ----- Scene update -----

    def _update_scene(self):
        with self.server.atomic():
            for body_id, handle in self._mesh_handles.items():
                pos = self.data.xpos[body_id]
                xmat = self.data.xmat[body_id].reshape(3, 3)
                quat_wxyz = vtf.SO3.from_matrix(xmat).wxyz
                handle.position = pos
                handle.wxyz = quat_wxyz
            for body_id, handle in self._collision_handles.items():
                pos = self.data.xpos[body_id]
                xmat = self.data.xmat[body_id].reshape(3, 3)
                quat_wxyz = vtf.SO3.from_matrix(xmat).wxyz
                handle.position = pos
                handle.wxyz = quat_wxyz
            self.server.flush()
        # Update extra view servers
        for srv, handles in self._extra_servers:
            with srv.atomic():
                for body_id, handle in handles.items():
                    pos = self.data.xpos[body_id]
                    xmat = self.data.xmat[body_id].reshape(3, 3)
                    quat_wxyz = vtf.SO3.from_matrix(xmat).wxyz
                    handle.position = pos
                    handle.wxyz = quat_wxyz
                srv.flush()

    def _render_cameras(self):
        if not self._renderer:
            return
        for cam_name, handle in self._cam_handles.items():
            self._renderer.update_scene(self.data, camera=cam_name)
            img = self._renderer.render()
            handle.image = img

    def _update_status(self):
        status = "Connected" if self._connected else "Waiting for leaders..."
        rec_status = "RECORDING" if self._recording else "idle"
        lines = [
            f"<strong>Status:</strong> {status}",
            f"<strong>Steps:</strong> {self._step_count}",
            f"<strong>Left leader:</strong> {self.left_leader}_actions",
            f"<strong>Right leader:</strong> {self.right_leader}_actions",
            f"<strong>Recording:</strong> {rec_status}",
        ]
        if self._recording:
            lines.append(
                f"<strong>Recorded frames:</strong> {len(self._recorded_actions)}"
            )
        body = "<br/>".join(lines)
        self._status_html.content = f"""
        <div style="font-size: 0.85em; line-height: 1.25; padding: 0 1em 0.5em 1em;">
          {body}
        </div>
        """

    # ----- Recording -----

    def _start_recording(self):
        self._recording = True
        self._recorded_actions = []
        self._record_btn.label = "Stop Recording"
        self._record_btn.icon = viser.Icon.PLAYER_STOP
        print("[teleop] Recording started")

    def _stop_recording(self):
        self._recording = False
        self._record_btn.label = "Start Recording"
        self._record_btn.icon = viser.Icon.RECORD_MAIL
        if self._recorded_actions:
            actions = np.array(self._recorded_actions)
            save_path = f"teleop_recording_{int(time.time())}.npy"
            np.save(save_path, actions)
            print(f"[teleop] Saved {len(actions)} frames to {save_path}")
        else:
            print("[teleop] No frames recorded")

    # ----- Reset -----

    def _reset(self):
        obs, _ = self.env.reset()
        self._state = obs["state"].copy()
        self._step_count = 0
        self._connected = False
        self._update_scene()
        self._render_cameras()
        self._update_status()
        print("[teleop] Reset sim to initial state")

    # ----- ZMQ polling -----

    def _poll_leader(self, sub: zmq.Socket) -> np.ndarray | None:
        if sub.poll(timeout=0):
            msg, extras = comms.subscribe(sub)
            return msg
        return None

    # ----- Main loop -----

    def run(self):
        dt = 1.0 / self.control_rate
        cam_counter = 0

        print(
            f"[teleop] Waiting for leader connections on "
            f"'{self.left_leader}_actions' and '{self.right_leader}_actions'..."
        )
        print("[teleop] Open the Viser URL above in your browser")
        print("[teleop] Ctrl+C to stop")

        # Wait for at least one leader message
        print("[teleop] Blocking until first leader message...")
        while True:
            left_pos = self._poll_leader(self._left_sub)
            right_pos = self._poll_leader(self._right_sub)
            if left_pos is not None or right_pos is not None:
                break
            time.sleep(0.01)

        self._connected = True
        print("[teleop] Leader connected! Starting control loop.")

        # Interpolate from init to first leader position
        if left_pos is not None or right_pos is not None:
            current = self.env.get_init_q()
            combined = current.copy()
            if left_pos is not None:
                combined[:7] = left_pos
            if right_pos is not None:
                combined[7:] = right_pos

            steps = 50
            for i in range(steps + 1):
                alpha = i / steps
                target = (1 - alpha) * current + alpha * combined
                self.env._step_single(target)
            self._state = self.env.get_obs()["state"]
            self._update_scene()
            self._render_cameras()

        try:
            while True:
                t0 = time.time()

                # Poll both leaders
                left_pos = self._poll_leader(self._left_sub)
                right_pos = self._poll_leader(self._right_sub)

                if left_pos is not None or right_pos is not None:
                    action = self._state.copy()
                    if left_pos is not None:
                        action[:7] = left_pos
                    if right_pos is not None:
                        action[7:] = right_pos

                    self.env._step_single(action)
                    obs = self.env.get_obs()
                    self._state = obs["state"].copy()
                    self._step_count += 1

                    if self._recording:
                        self._recorded_actions.append(action.copy())

                    self._update_scene()
                    cam_counter += 1
                    if cam_counter >= 10:
                        self._render_cameras()
                        cam_counter = 0

                    if self._step_count % 30 == 0:
                        self._update_status()

                elapsed = time.time() - t0
                to_sleep = dt - elapsed
                if to_sleep > 0:
                    time.sleep(to_sleep)

        except KeyboardInterrupt:
            print("\n[teleop] Shutting down...")
            if self._recording:
                self._stop_recording()
            if self._renderer:
                self._renderer.close()
            self._left_sub.close()
            self._right_sub.close()
            self._zmq_context.term()
            self.server.stop()
            for srv, _ in self._extra_servers:
                srv.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Live teleoperation with Viser 3D viewer"
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="hybrid",
        choices=["eval", "training", "hybrid"],
        help="Scene variant",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="bottles",
        choices=["bottles", "put_bottles", "marker", "pour", "drawer", "ball_sorting", "empty", "dishrack", "chess", "chess2", "blocks", "mug_tree", "mug_flip"],
        help="Task scene (default: bottles)",
    )
    parser.add_argument(
        "--left-leader",
        type=str,
        default="left",
        help="Left leader ZMQ topic prefix (default: 'left')",
    )
    parser.add_argument(
        "--right-leader",
        type=str,
        default="right",
        help="Right leader ZMQ topic prefix (default: 'right')",
    )
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    parser.add_argument(
        "--control-rate", type=float, default=30.0, help="Sim control rate in Hz"
    )
    parser.add_argument(
        "--renderer",
        action="store_true",
        help="Enable MuJoCo offscreen renderer for camera PiP",
    )
    parser.add_argument(
        "--camera-width", type=int, default=320, help="PiP camera width"
    )
    parser.add_argument(
        "--camera-height", type=int, default=240, help="PiP camera height"
    )
    parser.add_argument(
        "--mujoco-viewer",
        action="store_true",
        help="Use native MuJoCo viewer instead of Viser web viewer",
    )
    parser.add_argument(
        "--views",
        type=str,
        default="mono",
        choices=["mono", "stereo", "tri"],
        help="View mode: mono (single front), stereo (front + overhead), tri (front + left + right)",
    )
    args = parser.parse_args()

    import yam_sim

    env = yam_sim.make_env(scene=args.scene, task=args.task, render_cameras=False)

    print(f"Scene: {args.scene}")
    print(f"Control rate: {args.control_rate} Hz")
    print(
        f"Environment: nbody={env.model.nbody}, ngeom={env.model.ngeom}, "
        f"nmesh={env.model.nmesh}"
    )

    if args.mujoco_viewer:
        _run_mujoco_viewer(env, args)
    else:
        viewer = ViserTeleopViewer(
            env,
            left_leader=args.left_leader,
            right_leader=args.right_leader,
            port=args.port,
            control_rate=args.control_rate,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            enable_renderer=args.renderer,
            views=args.views,
        )
        viewer.run()


def _run_mujoco_viewer(env, args):
    """Run teleop with the native MuJoCo viewer."""
    import mujoco.viewer

    zmq_context = zmq.Context()
    left_sub = comms.create_subscriber(
        zmq_context, f"{args.left_leader}_actions", conflate=1
    )
    right_sub = comms.create_subscriber(
        zmq_context, f"{args.right_leader}_actions", conflate=1
    )

    obs, _ = env.reset()
    state = obs["state"].copy()
    dt = 1.0 / args.control_rate

    def poll(sub):
        if sub.poll(timeout=0):
            msg, _ = comms.subscribe(sub)
            return msg
        return None

    print(f"[teleop] Waiting for leaders on '{args.left_leader}_actions' / '{args.right_leader}_actions'...")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # Wait for first leader message
        while viewer.is_running():
            left_pos = poll(left_sub)
            right_pos = poll(right_sub)
            if left_pos is not None or right_pos is not None:
                break
            time.sleep(0.01)

        if not viewer.is_running():
            return

        print("[teleop] Leader connected! Starting control loop.")

        # Interpolate to first leader position
        current = env.get_init_q()
        combined = current.copy()
        if left_pos is not None:
            combined[:7] = left_pos
        if right_pos is not None:
            combined[7:] = right_pos
        for i in range(51):
            alpha = i / 50
            target = (1 - alpha) * current + alpha * combined
            env._step_single(target)
        state = env.get_obs()["state"]
        viewer.sync()

        # Main loop
        while viewer.is_running():
            t0 = time.time()

            left_pos = poll(left_sub)
            right_pos = poll(right_sub)

            if left_pos is not None or right_pos is not None:
                action = state.copy()
                if left_pos is not None:
                    action[:7] = left_pos
                if right_pos is not None:
                    action[7:] = right_pos

                env._step_single(action)
                state = env.get_obs()["state"].copy()
                viewer.sync()

            elapsed = time.time() - t0
            to_sleep = dt - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)

    left_sub.close()
    right_sub.close()
    zmq_context.term()
    print("[teleop] MuJoCo viewer closed.")


if __name__ == "__main__":
    main()
