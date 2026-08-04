"""Interactive IK teleoperation with Viser 3D viewer and MuJoCo sim.

Provides 3D transform handles for each arm's end effector. Drag the handles
to set target poses; J-PARSE velocity IK solves for joint angles each frame,
and the MuJoCo sim steps physics in real time.

Usage:
    uv run python -m yam_sim.examples.viser_ik_teleop
    uv run python -m yam_sim.examples.viser_ik_teleop --scene eval --port 8080
    uv run python -m yam_sim.examples.viser_ik_teleop --method jparse --renderer

Requirements:
    pip install yam-sim[viser]
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
    import viser
    import viser.transforms as vtf
except ImportError:
    raise SystemExit(
        "Missing viser dependencies.\n"
        "Install with: pip install yam-sim[viser]"
    )

from mujoco import mj_id2name, mjtGeom, mjtObj

from yam_sim.examples.viser_replay import (
    _create_primitive_mesh,
    _get_body_name,
    _get_geom_rgba,
    _is_fixed_body,
    _merge_geoms,
    _mujoco_mesh_to_trimesh,
    configure_default_camera,
)
from yam_sim.ik.mujoco_ik import MuJoCoIKSolver, _rotation_matrix_to_quat_wxyz


# Gripper actuator ctrl range max (from the MJCF actuator ctrlrange).
# 95mm total throw per i2rt spec → 47.5mm per finger.
_GRIPPER_CTRL_MAX = 0.0495


class ViserIKTeleopViewer:
    """Interactive IK teleoperation viewer: Viser handles + J-PARSE IK + MuJoCo."""

    def __init__(
        self,
        env,
        *,
        port: int = 8080,
        control_rate: float = 30.0,
        ik_method: str = "jparse",
        visible_geom_groups: tuple[int, ...] = (0, 1, 2),
        camera_width: int = 320,
        camera_height: int = 240,
        enable_renderer: bool = False,
    ):
        self.env = env
        self.model = env.model
        self.data = env.data
        self.control_rate = control_rate
        self.ik_method = ik_method
        self.visible_geom_groups = visible_geom_groups
        self.camera_width = camera_width
        self.camera_height = camera_height

        # Reset env to initial state
        obs, _ = self.env.reset()
        self._state = obs["state"].copy()
        self._step_count = 0

        # IK solvers for each arm — target the grasp site (tooltip) rather
        # than the wrist body so the gizmo sits at the fingertip frame.
        self._left_ik = MuJoCoIKSolver(
            self.model, self.data,
            arm_joint_names=[f"left_joint{i}" for i in range(1, 7)],
            tcp_body_name="left_link_6",
            tcp_site_name="left_grasp_site",
        )
        self._right_ik = MuJoCoIKSolver(
            self.model, self.data,
            arm_joint_names=[f"right_joint{i}" for i in range(1, 7)],
            tcp_body_name="right_link_6",
            tcp_site_name="right_grasp_site",
        )

        # Home configs for nullspace bias (from init_q, first 6 joints per arm)
        init_q = env.get_init_q()
        self._left_home = init_q[:6].copy()
        self._right_home = init_q[7:13].copy()

        # MuJoCo offscreen renderer for camera PiP
        self._renderer = None
        if enable_renderer:
            try:
                self._renderer = mujoco.Renderer(
                    self.model, height=camera_height, width=camera_width
                )
            except Exception as e:
                print(f"  Warning: Camera rendering unavailable ({e})")

        # Viser server
        self.server = viser.ViserServer(port=port)
        configure_default_camera(self.server)
        print(f"Viser IK teleop: http://localhost:{port}")

        # Scene state
        self._mesh_handles: dict[int, viser.MeshHandle] = {}
        self._collision_handles: dict[int, viser.MeshHandle] = {}
        self._show_collision = False
        self._recording = False
        self._recorded_actions: list[np.ndarray] = []

        # Build scene, GUI, and IK handles
        self._build_scene()
        self._build_gui()
        self._setup_ik_handles()
        self._update_scene()

    # ----- Scene building (reused from viser_teleop) -----

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

                if nonplane_ids:
                    merged = _merge_geoms(self.model, nonplane_ids)
                    self.server.scene.add_mesh_trimesh(
                        f"/fixed_bodies/{body_name}",
                        merged,
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

    # ----- IK transform handles -----

    def _setup_ik_handles(self):
        """Create 3D transform controls at the current TCP positions."""
        # Get current TCP poses
        left_pos, left_wxyz = self._left_ik.get_tcp_pose()
        right_pos, right_wxyz = self._right_ik.get_tcp_pose()

        self._left_target = self.server.scene.add_transform_controls(
            "/ik/left_target",
            scale=0.15,
            depth_test=False,
            position=left_pos,
            wxyz=left_wxyz,
        )
        self._right_target = self.server.scene.add_transform_controls(
            "/ik/right_target",
            scale=0.15,
            depth_test=False,
            position=right_pos,
            wxyz=right_wxyz,
        )

    # ----- GUI -----

    def _build_gui(self):
        with self.server.gui.add_folder("IK Controls", expand_by_default=True):
            self._method_dropdown = self.server.gui.add_dropdown(
                "IK Method",
                options=["jparse", "pinv", "dls"],
                initial_value=self.ik_method,
            )

            @self._method_dropdown.on_update
            def _(_):
                self.ik_method = self._method_dropdown.value

            self._left_gripper = self.server.gui.add_slider(
                "Left Gripper", min=0.0, max=1.0, step=0.01, initial_value=1.0,
            )
            self._right_gripper = self.server.gui.add_slider(
                "Right Gripper", min=0.0, max=1.0, step=0.01, initial_value=1.0,
            )

            self._gizmo_size = self.server.gui.add_slider(
                "Gizmo Size", min=0.05, max=0.4, step=0.01, initial_value=0.15,
            )

            @self._gizmo_size.on_update
            def _(_):
                self._left_target.scale = self._gizmo_size.value
                self._right_target.scale = self._gizmo_size.value

        with self.server.gui.add_folder("IK Status", expand_by_default=True):
            self._status_html = self.server.gui.add_html("")

        with self.server.gui.add_folder("Controls", expand_by_default=True):
            reset_btn = self.server.gui.add_button("Reset", icon=viser.Icon.REFRESH)

            @reset_btn.on_click
            def _(_):
                self._reset()

            snap_btn = self.server.gui.add_button(
                "Snap Handles to EE", icon=viser.Icon.TARGET,
            )

            @snap_btn.on_click
            def _(_):
                self._snap_handles_to_ee()

            self._record_btn = self.server.gui.add_button(
                "Start Recording", icon=viser.Icon.RECORD_MAIL,
            )

            @self._record_btn.on_click
            def _(_):
                if self._recording:
                    self._stop_recording()
                else:
                    self._start_recording()

        with self.server.gui.add_folder("Display", expand_by_default=False):
            collision_cb = self.server.gui.add_checkbox(
                "Show Collision Geoms", initial_value=False,
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
                        (self.camera_height, self.camera_width, 3), dtype=np.uint8,
                    )
                    self._cam_handles[cam_name] = self.server.gui.add_image(
                        placeholder, label=cam_name, format="jpeg", jpeg_quality=85,
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

    def _render_cameras(self):
        if not self._renderer:
            return
        for cam_name, handle in self._cam_handles.items():
            self._renderer.update_scene(self.data, camera=cam_name)
            img = self._renderer.render()
            handle.image = img

    def _update_status(self, left_info: dict, right_info: dict):
        lines = [
            f"<strong>Steps:</strong> {self._step_count}",
            f"<strong>Method:</strong> {self.ik_method}",
            f"<strong>Left pos err:</strong> {left_info['position_error']:.4f}m",
            f"<strong>Left ori err:</strong> {left_info['orientation_error']:.4f}rad",
            f"<strong>Left manip:</strong> {left_info['manipulability']:.4f}",
            f"<strong>Right pos err:</strong> {right_info['position_error']:.4f}m",
            f"<strong>Right ori err:</strong> {right_info['orientation_error']:.4f}rad",
            f"<strong>Right manip:</strong> {right_info['manipulability']:.4f}",
        ]
        if self._recording:
            lines.append(
                f"<strong>Recording:</strong> {len(self._recorded_actions)} frames"
            )
        body = "<br/>".join(lines)
        self._status_html.content = (
            '<div style="font-size: 0.85em; line-height: 1.25; '
            f'padding: 0 1em 0.5em 1em;">{body}</div>'
        )

    # ----- IK step -----

    def _ik_step(self, dt: float) -> tuple[dict, dict]:
        """Run one IK step for both arms, write to qpos, and step physics."""
        # Read target poses from transform handles
        left_target_pos = np.array(self._left_target.position)
        left_target_wxyz = np.array(self._left_target.wxyz)
        right_target_pos = np.array(self._right_target.position)
        right_target_wxyz = np.array(self._right_target.wxyz)

        # Solve IK for each arm
        left_cfg, left_info = self._left_ik.step(
            left_target_pos,
            target_wxyz=left_target_wxyz,
            method=self.ik_method,
            dt=dt,
            home_cfg=self._left_home,
        )
        right_cfg, right_info = self._right_ik.step(
            right_target_pos,
            target_wxyz=right_target_wxyz,
            method=self.ik_method,
            dt=dt,
            home_cfg=self._right_home,
        )

        # Build 14D action: [left_j1..6, left_grip, right_j1..6, right_grip]
        action = np.zeros(14, dtype=np.float32)
        action[:6] = left_cfg
        action[6] = self._left_gripper.value   # policy space [0, 1]
        action[7:13] = right_cfg
        action[13] = self._right_gripper.value  # policy space [0, 1]

        # Step physics — write actuator ctrls directly so we can use the
        # full actuator ctrl range for grippers (0..0.041) instead of the
        # env's default joint-range scaling (0..0.037524).
        ctrl = np.zeros(self.model.nu)
        for i in range(14):
            val = action[i]
            if i in self.env._gripper_set:
                val = val * _GRIPPER_CTRL_MAX  # scale to actuator ctrl range
            ctrl[self.env._ctrl_indices[i]] = val
        self.data.ctrl[:] = ctrl
        for _ in range(self.env._control_decimation):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        if self._recording:
            self._recorded_actions.append(action.copy())

        return left_info, right_info

    # ----- Reset / snap -----

    def _reset(self):
        obs, _ = self.env.reset()
        self._state = obs["state"].copy()
        self._step_count = 0
        self._snap_handles_to_ee()
        self._update_scene()
        if self._renderer:
            self._render_cameras()
        print("[ik-teleop] Reset to initial state")

    def _snap_handles_to_ee(self):
        """Move transform handles to the current FK end-effector positions."""
        left_pos, left_wxyz = self._left_ik.get_tcp_pose()
        right_pos, right_wxyz = self._right_ik.get_tcp_pose()
        self._left_target.position = left_pos
        self._left_target.wxyz = left_wxyz
        self._right_target.position = right_pos
        self._right_target.wxyz = right_wxyz

    # ----- Recording -----

    def _start_recording(self):
        self._recording = True
        self._recorded_actions = []
        self._record_btn.label = "Stop Recording"
        self._record_btn.icon = viser.Icon.PLAYER_STOP
        print("[ik-teleop] Recording started")

    def _stop_recording(self):
        self._recording = False
        self._record_btn.label = "Start Recording"
        self._record_btn.icon = viser.Icon.RECORD_MAIL
        if self._recorded_actions:
            actions = np.array(self._recorded_actions)
            save_path = f"ik_teleop_recording_{int(time.time())}.npy"
            np.save(save_path, actions)
            print(f"[ik-teleop] Saved {len(actions)} frames to {save_path}")
        else:
            print("[ik-teleop] No frames recorded")

    # ----- Main loop -----

    def run(self):
        dt = 1.0 / self.control_rate
        cam_counter = 0
        _empty_info = {
            "position_error": 0.0,
            "orientation_error": 0.0,
            "manipulability": 0.0,
            "max_joint_vel": 0.0,
        }
        left_info = _empty_info
        right_info = _empty_info

        print(f"[ik-teleop] Running at {self.control_rate} Hz, IK method: {self.ik_method}")
        print("[ik-teleop] Drag the 3D handles to move end effectors")
        print("[ik-teleop] Ctrl+C to stop")

        try:
            while True:
                t0 = time.time()

                # Run IK + physics step
                left_info, right_info = self._ik_step(dt)

                # Update scene
                self._update_scene()

                # Camera PiP every 10 steps
                cam_counter += 1
                if cam_counter >= 10:
                    if self._renderer:
                        self._render_cameras()
                    cam_counter = 0

                # Status update every 15 steps
                if self._step_count % 15 == 0:
                    self._update_status(left_info, right_info)

                elapsed = time.time() - t0
                to_sleep = dt - elapsed
                if to_sleep > 0:
                    time.sleep(to_sleep)

        except KeyboardInterrupt:
            print("\n[ik-teleop] Shutting down...")
            if self._recording:
                self._stop_recording()
            if self._renderer:
                self._renderer.close()
            self.server.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive IK teleoperation with Viser 3D viewer"
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="hybrid",
        choices=["eval", "training", "hybrid"],
        help="Scene variant (default: hybrid)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="bottles",
        choices=["bottles", "marker", "ball_sorting", "empty"],
        help="Task scene (default: bottles)",
    )
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    parser.add_argument(
        "--control-rate", type=float, default=30.0, help="Control rate in Hz",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="jparse",
        choices=["jparse", "pinv", "dls"],
        help="IK method (default: jparse)",
    )
    parser.add_argument(
        "--renderer",
        action="store_true",
        help="Enable MuJoCo offscreen renderer for camera PiP",
    )
    parser.add_argument("--camera-width", type=int, default=320)
    parser.add_argument("--camera-height", type=int, default=240)
    args = parser.parse_args()

    from yam_sim.config import get_viser_ik_config
    from yam_sim.env import MuJoCoYAMEnv
    from yam_sim.scene_variants import apply_scene_variant
    from yam_sim.task_registry import get_task_scene_xml, list_scene_task_names

    scene_xml = get_task_scene_xml(args.task)
    if scene_xml is None:
        raise ValueError(
            f"Unknown task '{args.task}'. Available: {list(list_scene_task_names())}"
        )

    config = get_viser_ik_config()
    env = MuJoCoYAMEnv(
        config=config,
        render_cameras=False,
        scene_xml=scene_xml,
    )
    apply_scene_variant(env.model, args.scene)

    print(f"Scene: {args.scene}, Task: {args.task}")
    print(f"IK method: {args.method}, Control rate: {args.control_rate} Hz")
    print(
        f"Environment: nbody={env.model.nbody}, ngeom={env.model.ngeom}, "
        f"nmesh={env.model.nmesh}"
    )

    viewer = ViserIKTeleopViewer(
        env,
        port=args.port,
        control_rate=args.control_rate,
        ik_method=args.method,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        enable_renderer=args.renderer,
    )
    viewer.run()


if __name__ == "__main__":
    main()
