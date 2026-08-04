"""Viser replay UI for yam-sim episodes."""

from __future__ import annotations

import time

import mujoco
import numpy as np

try:
    import viser
    import viser.transforms as vtf
except ImportError:
    raise SystemExit(
        "Missing dependencies for the Viser viewer.\n"
        "Install them with:\n"
        "  pip install yam-sim[viser]\n"
        "  # or with uv:\n"
        "  uv pip install yam-sim[viser]"
    )

from mujoco import mj_id2name, mjtGeom, mjtObj

from yam_sim.rendering.replay.camera_providers import CameraProvider
from yam_sim.rendering.replay.session import ReplaySession
from yam_sim.rendering.viser_scene import configure_default_camera, get_body_name, is_fixed_body, merge_geoms


class ReplayViewer:
    """Interactive Viser 3D viewer for replaying market42 episodes."""

    def __init__(
        self,
        session: ReplaySession,
        camera_provider: CameraProvider,
        *,
        port: int = 8080,
        speed: float = 1.0,
    ):
        self.session = session
        self.env = session.env
        self.model = session.model
        self.data = session.data
        self.camera_provider = camera_provider
        self._speed = speed
        self._paused = True
        self._mesh_handles: dict[int, object] = {}

        self.server = viser.ViserServer(port=port)
        configure_default_camera(self.server)
        print(f"Viser: http://localhost:{port}")

        self._build_scene()
        self._build_gui()
        self._cam_handles: dict[str, object] = {}
        self._build_camera_panels()
        self._refresh_after_reset()

    def _build_camera_panels(self) -> None:
        initial_frames = self.camera_provider.initial_frames()
        if not initial_frames:
            return

        with self.server.gui.add_folder("Cameras"):
            for cam_name in self.camera_provider.camera_names:
                frame = initial_frames.get(cam_name)
                if frame is None:
                    continue
                self._cam_handles[cam_name] = self.server.gui.add_image(
                    frame,
                    label=cam_name,
                    format="jpeg",
                    jpeg_quality=85,
                )

    def _build_scene(self) -> None:
        model = self.model
        visible_groups = (0, 1, 2)
        body_visual: dict[int, list[int]] = {}
        for i in range(model.ngeom):
            if int(model.geom_group[i]) in visible_groups:
                body_visual.setdefault(int(model.geom_bodyid[i]), []).append(i)

        self.server.scene.add_frame("/fixed_bodies", show_axes=False)
        for body_id, visual_ids in body_visual.items():
            body_name = get_body_name(model, body_id)
            if is_fixed_body(model, body_id):
                nonplane_ids = [gid for gid in visual_ids if model.geom_type[gid] != mjtGeom.mjGEOM_PLANE]
                plane_ids = [gid for gid in visual_ids if model.geom_type[gid] == mjtGeom.mjGEOM_PLANE]
                for gid in plane_ids:
                    geom_name = mj_id2name(model, mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"
                    self.server.scene.add_grid(
                        f"/fixed_bodies/{body_name}/{geom_name}",
                        width=2000.0,
                        height=2000.0,
                        position=model.geom_pos[gid],
                        wxyz=model.geom_quat[gid],
                    )
                if nonplane_ids:
                    merged = merge_geoms(model, nonplane_ids)
                    self.server.scene.add_mesh_trimesh(
                        f"/fixed_bodies/{body_name}",
                        merged,
                        position=model.body(body_id).pos,
                        wxyz=model.body(body_id).quat,
                    )
            elif visual_ids:
                merged = merge_geoms(model, visual_ids)
                handle = self.server.scene.add_mesh_trimesh(f"/bodies/{body_name}", merged, visible=True)
                self._mesh_handles[body_id] = handle

    def _build_gui(self) -> None:
        with self.server.gui.add_folder("Info"):
            self._status_html = self.server.gui.add_markdown("")
            self._update_status()

        with self.server.gui.add_folder("Playback"):
            self._play_btn = self.server.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY)

            @self._play_btn.on_click
            def _(_) -> None:
                self._paused = not self._paused
                self._play_btn.label = "Play" if self._paused else "Pause"
                self._play_btn.icon = viser.Icon.PLAYER_PLAY if self._paused else viser.Icon.PLAYER_PAUSE
                self._update_status()

            step_btn = self.server.gui.add_button("Step", icon=viser.Icon.PLAYER_TRACK_NEXT)

            @step_btn.on_click
            def _(_) -> None:
                self._paused = True
                self._play_btn.label = "Play"
                self._play_btn.icon = viser.Icon.PLAYER_PLAY
                self._sim_step()
                self._update_status()

            reset_btn = self.server.gui.add_button("Reset", icon=viser.Icon.REFRESH)

            @reset_btn.on_click
            def _(_) -> None:
                self._paused = True
                self._play_btn.label = "Play"
                self._play_btn.icon = viser.Icon.PLAYER_PLAY
                self._reset()

            self._speed_slider = self.server.gui.add_slider(
                "Speed", min=0.1, max=5.0, step=0.1, initial_value=self._speed
            )

            @self._speed_slider.on_update
            def _(_) -> None:
                self._speed = self._speed_slider.value

            mode_options = [self.session.state_replay_label, "physics (re-step)"]
            initial_mode = mode_options[0] if self.session.mode == "qpos" else mode_options[1]
            self._mode_dropdown = self.server.gui.add_dropdown(
                "Replay mode", options=mode_options, initial_value=initial_mode
            )
            if not self.session.has_state_replay:
                self._mode_dropdown.disabled = True

            @self._mode_dropdown.on_update
            def _(_) -> None:
                self.session.set_mode("physics" if "physics" in self._mode_dropdown.value else "qpos")
                self._refresh_after_reset()

    def _update_scene(self) -> None:
        with self.server.atomic():
            for body_id, handle in self._mesh_handles.items():
                handle.position = self.data.xpos[body_id]
                xmat = self.data.xmat[body_id].reshape(3, 3)
                handle.wxyz = vtf.SO3.from_matrix(xmat).wxyz
            self.server.flush()

    def _update_cameras(self) -> None:
        if not self._cam_handles:
            return
        frames = self.camera_provider.frames_for_step(self.session.current_frame_idx, self.session.current_timestamp)
        for cam_name, handle in self._cam_handles.items():
            frame = frames.get(cam_name)
            if frame is not None:
                handle.image = frame

    def _sim_step(self) -> None:
        if self.session.is_done:
            return
        if not self.session.step():
            return
        self._update_scene()
        self._update_cameras()

    def _refresh_after_reset(self) -> None:
        self.camera_provider.reset()
        self._update_scene()
        self._update_cameras()
        self._update_status()

    def _reset(self) -> None:
        self.session.reset()
        self._refresh_after_reset()

    def _update_status(self) -> None:
        total = self.session.total_steps
        step_idx = self.session.step_idx
        pct = f"{100 * step_idx / total:.0f}%" if total > 0 else "0%"
        done = " DONE" if self.session.is_done else ""
        self._status_html.content = (
            f"**{'Paused' if self._paused else 'Playing'}**{done}  \n"
            f"Step: {step_idx}/{total} ({pct})  \n"
            f"Time: {self.session.elapsed_s:.1f}s / {self.session.duration_s:.1f}s  \n"
            f"Speed: {self._speed:.1f}x"
        )

    def run(self) -> None:
        """Main loop with wall-clock timing against the replay grid."""
        print("Replay viewer running. Open the URL above in a browser.")
        print("Press Ctrl-C to quit.")
        dt = self.session.step_dt
        try:
            while True:
                t0 = time.monotonic()
                if not self._paused and not self.session.is_done:
                    self._sim_step()
                    self._update_status()
                elapsed = time.monotonic() - t0
                to_sleep = (dt / self._speed) - elapsed
                if to_sleep > 0:
                    time.sleep(to_sleep)
        except KeyboardInterrupt:
            print("\nShutting down.")
            self.camera_provider.close()
            self.server.stop()
