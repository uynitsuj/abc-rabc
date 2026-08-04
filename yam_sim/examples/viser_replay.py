"""Interactive 3D Viser web viewer for replaying yam-sim action trajectories.

Loads a MuJoCo scene via yam-sim, converts all geoms to trimesh objects,
and renders them in Viser's browser-based 3D viewport. Replays a .npy action
trajectory with play/pause/step/reset controls, a frame scrubber, and
picture-in-picture camera views (top, left, right).

No policy inference — this is purely a replay visualizer.

Usage:
    # Replay bundled demo episode (no camera PiP, fastest)
    python -m yam_sim.examples.viser_replay

    # Replay with a one-shot camera check (renders PiP once at start)
    python -m yam_sim.examples.viser_replay --render-check

    # Replay with continuous camera PiP updates
    MUJOCO_GL=egl python -m yam_sim.examples.viser_replay --renderer

    # Watch bottle pickups (auto-trimmed to bottle-in-bin events)
    uv run python -m yam_sim.examples.viser_replay \
        --episode-dir yam_sim/examples/episode_6ib_seed0 --highlight

    # Trim to a specific time window (in seconds)
    python -m yam_sim.examples.viser_replay --trim 54 70

    # Replay custom actions file
    python -m yam_sim.examples.viser_replay \
        --actions /path/to/actions.npy --scene hybrid

    # Custom port and camera resolution
    MUJOCO_GL=egl python -m yam_sim.examples.viser_replay \
        --port 8080 --camera-width 640 --camera-height 480 --renderer

Requirements (in addition to yam-sim):
    pip install yam-sim[viser]
    # or with uv:
    uv pip install yam-sim[viser]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Force GPU EGL for headless rendering before importing mujoco
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
        "Missing dependencies for the Viser viewer.\n"
        "Install them with:\n"
        "  pip install yam-sim[viser]\n"
        "  # or with uv:\n"
        "  uv pip install yam-sim[viser]"
    )

from mujoco import mj_id2name, mjtGeom, mjtObj
from PIL import Image

# Bundled episode data
_EXAMPLES_DIR = Path(__file__).parent
_DEFAULT_EPISODE_DIR = _EXAMPLES_DIR / "episode_seed56"


# ---------------------------------------------------------------------------
# MuJoCo geom → trimesh conversion
# ---------------------------------------------------------------------------


def _get_geom_rgba(model: mujoco.MjModel, geom_id: int) -> np.ndarray:
    """Get RGBA color for a geom, checking material first then geom rgba."""
    matid = model.geom_matid[geom_id]
    if matid >= 0:
        return model.mat_rgba[matid].copy()
    rgba = model.geom_rgba[geom_id].copy()
    if np.all(rgba == 0):
        rgba = np.array([0.5, 0.5, 0.5, 1.0])
    return rgba


def _create_primitive_mesh(model: mujoco.MjModel, geom_id: int) -> trimesh.Trimesh:
    """Convert a MuJoCo primitive geom (sphere, box, capsule, etc.) to trimesh."""
    size = model.geom_size[geom_id]
    geom_type = model.geom_type[geom_id]
    rgba = _get_geom_rgba(model, geom_id)
    rgba_uint8 = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)

    if geom_type == mjtGeom.mjGEOM_SPHERE:
        mesh = trimesh.creation.icosphere(radius=size[0], subdivisions=2)
    elif geom_type == mjtGeom.mjGEOM_BOX:
        mesh = trimesh.creation.box(extents=2.0 * size)
    elif geom_type == mjtGeom.mjGEOM_CAPSULE:
        mesh = trimesh.creation.capsule(radius=size[0], height=2.0 * size[1])
    elif geom_type == mjtGeom.mjGEOM_CYLINDER:
        mesh = trimesh.creation.cylinder(radius=size[0], height=2.0 * size[1])
    elif geom_type == mjtGeom.mjGEOM_ELLIPSOID:
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        mesh.apply_scale(size)
    elif geom_type == mjtGeom.mjGEOM_PLANE:
        plane_x = 2.0 * size[0] if size[0] > 0 else 20.0
        plane_y = 2.0 * size[1] if size[1] > 0 else 20.0
        mesh = trimesh.creation.box((plane_x, plane_y, 0.001))
    else:
        raise ValueError(f"Unsupported primitive geom type: {geom_type}")

    # Check if this geom has a textured material
    matid = model.geom_matid[geom_id]
    has_texture = False
    if matid >= 0 and matid < model.nmat:
        texid = int(model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)])
        if texid < 0:
            texid = int(model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGBA)])
        if texid >= 0 and texid < model.ntex:
            has_texture = True
            mat_rgba = model.mat_rgba[matid]
            tex_w = model.tex_width[texid]
            tex_h = model.tex_height[texid]
            tex_nc = model.tex_nchannel[texid]
            tex_adr = model.tex_adr[texid]
            tex_data = model.tex_data[tex_adr : tex_adr + tex_w * tex_h * tex_nc]
            texrepeat = model.mat_texrepeat[matid]

            if tex_nc == 3:
                image = Image.fromarray(
                    np.flipud(tex_data.reshape(tex_h, tex_w, 3).astype(np.uint8)), mode="RGB"
                )
            elif tex_nc == 4:
                image = Image.fromarray(
                    np.flipud(tex_data.reshape(tex_h, tex_w, 4).astype(np.uint8)), mode="RGBA"
                )
            elif tex_nc == 1:
                image = Image.fromarray(
                    np.flipud(tex_data.reshape(tex_h, tex_w).astype(np.uint8)), mode="L"
                )
            else:
                has_texture = False

    if has_texture:
        # Generate UVs: map box face vertices to tiled UV space
        # For boxes, trimesh.creation.box already has UV-friendly vertex layout
        # Use vertex positions normalized to [0, texrepeat] range
        verts = mesh.vertices
        extents = 2.0 * size
        # Normalize to 0-1 based on bounding box, then apply texrepeat
        uv = np.zeros((len(verts), 2))
        # Project UVs from the dominant face normal direction
        # Simple approach: use XY projection scaled by texrepeat
        bb_min = verts.min(axis=0)
        bb_max = verts.max(axis=0)
        bb_range = bb_max - bb_min
        bb_range[bb_range == 0] = 1
        uv[:, 0] = (verts[:, 0] - bb_min[0]) / bb_range[0] * texrepeat[0]
        uv[:, 1] = (verts[:, 1] - bb_min[1]) / bb_range[1] * texrepeat[1]

        material = trimesh.visual.material.PBRMaterial(
            baseColorFactor=mat_rgba,
            baseColorTexture=image,
            metallicFactor=0.0,
            roughnessFactor=1.0,
        )
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    else:
        vertex_colors = np.tile(rgba_uint8, (len(mesh.vertices), 1))
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=vertex_colors)
    return mesh


def _mujoco_mesh_to_trimesh(
    model: mujoco.MjModel, geom_idx: int
) -> trimesh.Trimesh:
    """Convert a MuJoCo mesh geom to trimesh with texture support."""
    mesh_id = model.geom_dataid[geom_idx]

    vert_start = int(model.mesh_vertadr[mesh_id])
    vert_count = int(model.mesh_vertnum[mesh_id])
    face_start = int(model.mesh_faceadr[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])

    vertices = model.mesh_vert[vert_start : vert_start + vert_count]
    faces = model.mesh_face[face_start : face_start + face_count]

    texcoord_adr = model.mesh_texcoordadr[mesh_id]
    texcoord_num = model.mesh_texcoordnum[mesh_id]

    if texcoord_num > 0:
        texcoords = model.mesh_texcoord[texcoord_adr : texcoord_adr + texcoord_num]
        face_texcoord_idx = model.mesh_facetexcoord[
            face_start : face_start + face_count
        ]

        new_vertices = vertices[faces.flatten()]
        new_uvs = texcoords[face_texcoord_idx.flatten()]
        new_faces = np.arange(face_count * 3).reshape(-1, 3)

        mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)

        matid = model.geom_matid[geom_idx]
        if matid >= 0 and matid < model.nmat:
            rgba = model.mat_rgba[matid]
            texid = int(
                model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)]
            )
            if texid < 0:
                texid = int(
                    model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGBA)]
                )

            if texid >= 0 and texid < model.ntex:
                tex_w = model.tex_width[texid]
                tex_h = model.tex_height[texid]
                tex_nc = model.tex_nchannel[texid]
                tex_adr = model.tex_adr[texid]
                tex_size = tex_w * tex_h * tex_nc
                tex_data = model.tex_data[tex_adr : tex_adr + tex_size]

                image = None
                if tex_nc == 1:
                    arr = np.flipud(tex_data.reshape(tex_h, tex_w))
                    image = Image.fromarray(arr.astype(np.uint8), mode="L")
                elif tex_nc == 3:
                    arr = np.flipud(tex_data.reshape(tex_h, tex_w, 3))
                    image = Image.fromarray(arr.astype(np.uint8), mode="RGB")
                elif tex_nc == 4:
                    arr = np.flipud(tex_data.reshape(tex_h, tex_w, 4))
                    image = Image.fromarray(arr.astype(np.uint8), mode="RGBA")

                if image is not None:
                    material = trimesh.visual.material.PBRMaterial(
                        baseColorFactor=rgba,
                        baseColorTexture=image,
                        metallicFactor=0.0,
                        roughnessFactor=1.0,
                    )
                    mesh.visual = trimesh.visual.TextureVisuals(
                        uv=new_uvs, material=material
                    )
                else:
                    rgba_255 = (rgba * 255).astype(np.uint8)
                    mesh.visual = trimesh.visual.ColorVisuals(
                        vertex_colors=np.tile(rgba_255, (len(new_vertices), 1))
                    )
            else:
                rgba_255 = (rgba * 255).astype(np.uint8)
                mesh.visual = trimesh.visual.ColorVisuals(
                    vertex_colors=np.tile(rgba_255, (len(new_vertices), 1))
                )
        else:
            color = np.array([31, 128, 230, 255], dtype=np.uint8)
            mesh.visual = trimesh.visual.ColorVisuals(
                vertex_colors=np.tile(color, (len(new_vertices), 1))
            )
    else:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        rgba = _get_geom_rgba(model, geom_idx)
        rgba_255 = (rgba * 255).astype(np.uint8)
        mesh.visual = trimesh.visual.ColorVisuals(
            vertex_colors=np.tile(rgba_255, (len(mesh.vertices), 1))
        )

    return mesh


def _merge_geoms(model: mujoco.MjModel, geom_ids: list[int]) -> trimesh.Trimesh:
    """Merge multiple geoms into a single trimesh, applying local transforms."""
    meshes = []
    for geom_id in geom_ids:
        geom_type = model.geom_type[geom_id]
        if geom_type == mjtGeom.mjGEOM_MESH:
            mesh = _mujoco_mesh_to_trimesh(model, geom_id)
        else:
            mesh = _create_primitive_mesh(model, geom_id)

        pos = model.geom_pos[geom_id]
        quat = model.geom_quat[geom_id]
        transform = np.eye(4)
        transform[:3, :3] = vtf.SO3(quat).as_matrix()
        transform[:3, 3] = pos
        mesh.apply_transform(transform)
        meshes.append(mesh)

    if len(meshes) == 1:
        return meshes[0]
    return trimesh.util.concatenate(meshes)


def _is_fixed_body(model: mujoco.MjModel, body_id: int) -> bool:
    """Check if a body is fixed (welded to world, not mocap)."""
    is_weld = model.body_weldid[body_id] == 0
    root_id = model.body_rootid[body_id]
    root_is_mocap = model.body_mocapid[root_id] >= 0
    return is_weld and not root_is_mocap


def _get_body_name(model: mujoco.MjModel, body_id: int) -> str:
    """Get body name or fallback to body_{id}."""
    name = mj_id2name(model, mjtObj.mjOBJ_BODY, body_id)
    return name if name else f"body_{body_id}"


# ---------------------------------------------------------------------------
# Default camera
# ---------------------------------------------------------------------------


def configure_default_camera(server: "viser.ViserServer") -> None:
    """Set the initial camera to an operator's-eye view behind the robots.

    Called once after the ViserServer is created.  Each connecting client
    gets a camera placed behind and above the robot arms, looking toward
    the workspace — roughly where a human teleoperator would sit.
    """
    import viser  # noqa: F811 — may already be imported by caller

    @server.on_client_connect
    def _set_camera(client: viser.ClientHandle) -> None:
        client.camera.position = (-0.4, 0.0, 1.3)
        client.camera.look_at = (0.45, 0.0, 0.85)
        client.camera.up_direction = (0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Viser replay viewer
# ---------------------------------------------------------------------------


class ViserReplayViewer:
    """Interactive Viser 3D viewer for replaying yam-sim action trajectories."""

    def __init__(
        self,
        env,
        actions: np.ndarray,
        *,
        port: int = 8080,
        camera_width: int = 320,
        camera_height: int = 240,
        visible_geom_groups: tuple[int, ...] = (0, 1, 2),
        sim_hz: float = 30.0,
        enable_renderer: bool = False,
        render_check: bool = False,
    ):
        self.env = env
        self.model = env.model
        self.data = env.data
        self.actions = actions
        self.visible_geom_groups = visible_geom_groups
        self.sim_hz = sim_hz
        self.camera_width = camera_width
        self.camera_height = camera_height
        self._render_check = render_check

        # MuJoCo offscreen renderer for camera PiP
        self._renderer = None
        if enable_renderer or render_check:
            try:
                self._renderer = mujoco.Renderer(
                    self.model, height=camera_height, width=camera_width
                )
                print(f"  MuJoCo renderer: {camera_height}x{camera_width}")
            except Exception as e:
                print(f"  Warning: Camera rendering unavailable ({e})")

        # Viser server
        self.server = viser.ViserServer(port=port)
        configure_default_camera(self.server)
        print(f"Viser server: http://localhost:{port}")

        # Scene state
        self._mesh_handles: dict[int, viser.MeshHandle] = {}
        self._collision_handles: dict[int, viser.MeshHandle] = {}
        self._paused = True
        self._step_idx = 0
        self._time_multiplier = 1.0
        self._show_collision = False

        # Frame history for scrubbing
        self._history_xpos: list[np.ndarray] = []
        self._history_xmat: list[np.ndarray] = []
        self._history_qpos: list[np.ndarray] = []

        # Snapshot initial state for reset (may be post-fast-forward)
        self._init_qpos = self.data.qpos.copy()
        self._init_qvel = self.data.qvel.copy()

        # Build scene + GUI
        self._build_scene()
        self._build_gui()
        self._update_scene()
        self._render_cameras()
        self._record_frame()

        # In render-check mode, disable further camera updates after initial render
        if self._render_check and self._renderer:
            print("  Render check: initial frame rendered, disabling camera updates")
            self._renderer.close()
            self._renderer = None

    # ----- Scene building -----

    def _is_visual_geom(self, geom_id: int) -> bool:
        return int(self.model.geom_group[geom_id]) in self.visible_geom_groups

    def _build_scene(self):
        """Convert MuJoCo geoms to Viser scene nodes."""
        self.server.scene.configure_environment_map(environment_intensity=0.8)

        # Classify geoms by body
        body_visual: dict[int, list[int]] = {}
        body_collision: dict[int, list[int]] = {}
        for i in range(self.model.ngeom):
            body_id = self.model.geom_bodyid[i]
            if self._is_visual_geom(i):
                body_visual.setdefault(body_id, []).append(i)
            else:
                body_collision.setdefault(body_id, []).append(i)

        # Log geom group stats
        groups: dict[int, int] = {}
        for i in range(self.model.ngeom):
            g = int(self.model.geom_group[i])
            groups[g] = groups.get(g, 0) + 1
        for g in sorted(groups):
            tag = "visual" if g in self.visible_geom_groups else "collision"
            print(f"  Geom group {g}: {groups[g]} geoms ({tag})")

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

    # ----- GUI -----

    def _build_gui(self):
        """Build play/pause, speed, frame scrubber, and camera PiP controls."""
        with self.server.gui.add_folder("Info", expand_by_default=True):
            self._status_html = self.server.gui.add_html("")
            self._update_status()

        with self.server.gui.add_folder("Playback", expand_by_default=True):
            self._pause_btn = self.server.gui.add_button(
                "Play", icon=viser.Icon.PLAYER_PLAY
            )

            @self._pause_btn.on_click
            def _(_):
                self._paused = not self._paused
                self._pause_btn.label = "Play" if self._paused else "Pause"
                self._pause_btn.icon = (
                    viser.Icon.PLAYER_PLAY
                    if self._paused
                    else viser.Icon.PLAYER_PAUSE
                )
                if self._paused:
                    self._render_cameras()
                self._update_status()

            step_btn = self.server.gui.add_button(
                "Step", icon=viser.Icon.PLAYER_TRACK_NEXT
            )

            @step_btn.on_click
            def _(_):
                self._paused = True
                self._pause_btn.label = "Play"
                self._pause_btn.icon = viser.Icon.PLAYER_PLAY
                self._sim_step()
                self._render_cameras()
                self._update_status()

            reset_btn = self.server.gui.add_button("Reset", icon=viser.Icon.REFRESH)

            @reset_btn.on_click
            def _(_):
                self._paused = True
                self._pause_btn.label = "Play"
                self._pause_btn.icon = viser.Icon.PLAYER_PLAY
                self._reset()

            self._speed_slider = self.server.gui.add_slider(
                "Speed", min=0.1, max=5.0, step=0.1, initial_value=1.0
            )

            @self._speed_slider.on_update
            def _(_):
                self._time_multiplier = self._speed_slider.value

        # Frame scrubber
        with self.server.gui.add_folder("Frame Scrubber", expand_by_default=True):
            self._frame_slider = self.server.gui.add_slider(
                "Frame", min=0, max=0, step=1, initial_value=0
            )
            self._scrubbing = False

            @self._frame_slider.on_update
            def _(_):
                if self._scrubbing:
                    return
                idx = int(self._frame_slider.value)
                if 0 <= idx < len(self._history_xpos):
                    self._restore_frame(idx)

        # Display options
        with self.server.gui.add_folder("Display", expand_by_default=False):
            collision_cb = self.server.gui.add_checkbox(
                "Show Collision Geoms", initial_value=False
            )

            @collision_cb.on_update
            def _(_):
                self._show_collision = collision_cb.value
                for handle in self._collision_handles.values():
                    handle.visible = self._show_collision

        # Camera PiP images
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
        """Update dynamic mesh positions/rotations from MjData."""
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
        """Render camera PiP views via MuJoCo offscreen renderer."""
        if not self._renderer:
            return
        for cam_name, handle in self._cam_handles.items():
            self._renderer.update_scene(self.data, camera=cam_name)
            img = self._renderer.render()
            handle.image = img

    # ----- Frame history / scrubbing -----

    def _record_frame(self):
        """Record current state for frame scrubbing."""
        self._history_xpos.append(self.data.xpos.copy())
        self._history_xmat.append(self.data.xmat.copy().reshape(-1, 3, 3))
        self._history_qpos.append(self.data.qpos.copy())

        self._scrubbing = True
        self._frame_slider.max = max(0, len(self._history_xpos) - 1)
        self._frame_slider.value = len(self._history_xpos) - 1
        self._scrubbing = False

    def _restore_frame(self, idx: int):
        """Restore scene to a recorded frame."""
        if idx < 0 or idx >= len(self._history_xpos):
            return
        xpos = self._history_xpos[idx]
        xmat = self._history_xmat[idx]

        with self.server.atomic():
            for body_id, handle in self._mesh_handles.items():
                handle.position = xpos[body_id]
                quat_wxyz = vtf.SO3.from_matrix(xmat[body_id]).wxyz
                handle.wxyz = quat_wxyz
            self.server.flush()

        if idx < len(self._history_qpos):
            self.data.qpos[:] = self._history_qpos[idx]
            mujoco.mj_forward(self.model, self.data)
            self._render_cameras()

    # ----- Sim stepping -----

    def _sim_step(self):
        """Step physics with the next action from the trajectory."""
        if self._step_idx >= len(self.actions):
            return
        action = self.actions[self._step_idx]
        self.env._step_single(action)
        self._step_idx += 1
        self._update_scene()
        self._record_frame()

    def _reset(self):
        """Reset to the state the viewer was created with (post-trim if applicable)."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._init_qpos
        self.data.qvel[:] = self._init_qvel
        mujoco.mj_forward(self.model, self.data)
        self._step_idx = 0
        self._history_xpos.clear()
        self._history_xmat.clear()
        self._history_qpos.clear()
        self._update_scene()
        self._render_cameras()
        self._record_frame()
        self._update_status()

    def _update_status(self):
        status = "Paused" if self._paused else "Playing"
        progress = f"{self._step_idx}/{len(self.actions)}"
        pct = (
            f"{100 * self._step_idx / len(self.actions):.0f}%"
            if len(self.actions) > 0
            else "0%"
        )
        lines = [
            f"<strong>Status:</strong> {status}",
            f"<strong>Step:</strong> {progress} ({pct})",
            f"<strong>Speed:</strong> {self._time_multiplier:.1f}x",
            f"<strong>Frames:</strong> {len(self._history_xpos)}",
        ]
        if self._step_idx >= len(self.actions):
            lines.append("<strong>Replay:</strong> DONE")
        body = "<br/>".join(lines)
        self._status_html.content = f"""
        <div style="font-size: 0.85em; line-height: 1.25; padding: 0 1em 0.5em 1em;">
          {body}
        </div>
        """

    # ----- Main loop -----

    def run(self):
        """Main loop with play/pause logic."""
        dt = 1.0 / self.sim_hz
        cam_counter = 0
        print("Visualizer running. Open the URL above in a browser.")
        try:
            while True:
                t0 = time.time()
                if not self._paused and self._step_idx < len(self.actions):
                    self._sim_step()
                    cam_counter += 1
                    if cam_counter >= 10:
                        self._render_cameras()
                        cam_counter = 0
                    self._update_status()

                elapsed = time.time() - t0
                target_dt = dt / self._time_multiplier
                to_sleep = target_dt - elapsed
                if to_sleep > 0:
                    time.sleep(to_sleep)
        except KeyboardInterrupt:
            print("\nShutting down...")
            if self._renderer:
                self._renderer.close()
            self.server.stop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Interactive 3D Viser replay viewer for yam-sim"
    )
    parser.add_argument(
        "--actions",
        type=str,
        default=None,
        help="Path to actions .npy file (T, 14). Default: bundled demo episode.",
    )
    parser.add_argument(
        "--episode-dir",
        type=str,
        default=None,
        help="Episode directory with actions.npy and config.json.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        choices=["eval", "training", "hybrid"],
        help="Scene variant (default: from config.json or 'hybrid').",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="bottles",
        choices=["bottles", "marker", "pour", "blocks", "drawer"],
        help="Task scene — 'bottles' (default) or 'marker'.",
    )
    parser.add_argument("--port", type=int, default=8080, help="Viser server port.")
    parser.add_argument(
        "--camera-width", type=int, default=320, help="PiP camera width."
    )
    parser.add_argument(
        "--camera-height", type=int, default=240, help="PiP camera height."
    )
    parser.add_argument(
        "--trim",
        type=float,
        nargs=2,
        metavar=("START_S", "END_S"),
        default=None,
        help="Trim replay to a time window in seconds, e.g. --trim 54 70",
    )
    parser.add_argument(
        "--highlight",
        action="store_true",
        help="Auto-trim to 3s before first bottle-in-bin through 3s after last",
    )
    parser.add_argument(
        "--renderer",
        action="store_true",
        default=False,
        help="Enable MuJoCo offscreen renderer for camera PiP views (disabled by default).",
    )
    parser.add_argument(
        "--render-check",
        action="store_true",
        default=False,
        help="Render camera PiP once at start for a visual check, then disable during playback.",
    )
    args = parser.parse_args()

    # Load actions and config
    if args.episode_dir:
        episode_dir = Path(args.episode_dir)
    elif args.actions:
        episode_dir = None
    else:
        episode_dir = _DEFAULT_EPISODE_DIR

    config = {}
    if episode_dir and (episode_dir / "config.json").exists():
        with open(episode_dir / "config.json") as f:
            config = json.load(f)

    if episode_dir:
        actions = np.load(str(episode_dir / "actions.npy")).astype(np.float32)
    else:
        actions = np.load(args.actions).astype(np.float32)

    scene = args.scene or config.get("scene", "hybrid")
    print(f"Actions: {actions.shape} ({actions.shape[0]} timesteps x {actions.shape[1]}D)")
    print(f"Scene: {scene}")

    # Create environment
    import yam_sim
    from yam_sim.scene_variants import apply_scene_variant

    env = yam_sim.make_env(scene=scene, task=args.task, render_cameras=False)
    env.reset()

    # Apply episode config overrides if present
    from yam_sim.examples.replay_demo import apply_bottle_mass, apply_bottle_alpha

    if "bottle_mass" in config:
        apply_bottle_mass(env.model, config["bottle_mass"])
    if "bottle_alpha" in config:
        apply_bottle_alpha(env.model, config["bottle_alpha"])
    if config.get("all_green_bottles"):
        from yam_sim.examples.replay_demo import apply_green_bottles

        apply_green_bottles(env.model)

    # Set initial state — prefer full qpos/qvel files, fall back to init_q
    if episode_dir and (episode_dir / "initial_qpos.npy").exists():
        init_qpos = np.load(str(episode_dir / "initial_qpos.npy"))
        mujoco.mj_resetData(env.model, env.data)
        env.data.qpos[:] = init_qpos
        if (episode_dir / "initial_qvel.npy").exists():
            init_qvel = np.load(str(episode_dir / "initial_qvel.npy"))
            env.data.qvel[:] = init_qvel
        mujoco.mj_forward(env.model, env.data)
        print("  Initial state: loaded from qpos/qvel .npy files")
    elif "init_q" in config and config["init_q"] is not None:
        init_q = np.array(config["init_q"], dtype=np.float32)
        env._set_qpos_from_state(init_q)
        mujoco.mj_forward(env.model, env.data)
        print("  Initial state: loaded from config init_q")

    print(
        f"Environment: nbody={env.model.nbody}, ngeom={env.model.ngeom}, "
        f"nmesh={env.model.nmesh}"
    )

    # Compute control Hz for trim calculations
    control_hz = 1.0 / (env.model.opt.timestep * env._control_decimation)

    # Auto-detect highlight window
    if args.highlight and not args.trim:
        from yam_sim.examples.replay_demo import is_bottle_in_bin

        highlight_cfg = config.get("highlight")
        if highlight_cfg:
            args.trim = [highlight_cfg["trim_start_s"], highlight_cfg["trim_end_s"]]
            print(f"\nHighlight from config: {args.trim[0]:.1f}s - {args.trim[1]:.1f}s")
        else:
            print("\nScanning for bottle-in-bin events...")
            # Find bottles and bin
            bottle_info = {}
            for i in range(1, 7):
                jnt_name = f"bottle_{i}_joint"
                jnt_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
                if jnt_id >= 0:
                    bottle_info[f"bottle_{i}"] = env.model.jnt_qposadr[jnt_id]
            bin_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "bin_container")
            bin_pos = env.data.xpos[bin_body_id].copy() if bin_body_id >= 0 else None

            if bottle_info and bin_pos is not None:
                first_in_step = None
                last_in_step = None
                bottles_in_bin: set[str] = set()
                for t in range(len(actions)):
                    env._step_single(actions[t])
                    for name, addr in bottle_info.items():
                        if is_bottle_in_bin(env.data.qpos[addr : addr + 2], env.data.qpos[addr + 2], bin_pos):
                            bottles_in_bin.add(name)
                            if first_in_step is None:
                                first_in_step = t
                            last_in_step = t
                if first_in_step is not None:
                    pad = 3.0
                    args.trim = [
                        max(0, first_in_step / control_hz - pad),
                        min(len(actions) / control_hz, last_in_step / control_hz + pad),
                    ]
                    print(f"  Bottles in bin: {len(bottles_in_bin)}/{len(bottle_info)} ({', '.join(sorted(bottles_in_bin))})")
                    print(f"  Found events: {first_in_step / control_hz:.1f}s - {last_in_step / control_hz:.1f}s")
                    print(f"  Auto-trim: {args.trim[0]:.1f}s - {args.trim[1]:.1f}s")
                else:
                    print(f"  No bottles entered bin (0/{len(bottle_info)}), showing full episode")

                # Reset for actual replay
                env.reset()
                if episode_dir and (episode_dir / "initial_qpos.npy").exists():
                    init_qpos = np.load(str(episode_dir / "initial_qpos.npy"))
                    mujoco.mj_resetData(env.model, env.data)
                    env.data.qpos[:] = init_qpos
                    if (episode_dir / "initial_qvel.npy").exists():
                        init_qvel = np.load(str(episode_dir / "initial_qvel.npy"))
                        env.data.qvel[:] = init_qvel
                    mujoco.mj_forward(env.model, env.data)
                elif "init_q" in config and config["init_q"] is not None:
                    init_q = np.array(config["init_q"], dtype=np.float32)
                    env._set_qpos_from_state(init_q)
                    mujoco.mj_forward(env.model, env.data)

    # Apply trim: fast-forward through pre-trim actions, slice the rest
    if args.trim:
        trim_start = max(0, int(args.trim[0] * control_hz))
        trim_end = min(len(actions), int(args.trim[1] * control_hz))
        print(f"\nTrimming to {args.trim[0]:.1f}s - {args.trim[1]:.1f}s "
              f"(steps {trim_start}-{trim_end} of {len(actions)})")
        if trim_start > 0:
            print(f"Fast-forwarding through {trim_start} steps...")
            for t in range(trim_start):
                env._step_single(actions[t])
            mujoco.mj_forward(env.model, env.data)
        actions = actions[trim_start:trim_end]
        print(f"Trimmed actions: {actions.shape}")

    viewer = ViserReplayViewer(
        env,
        actions,
        port=args.port,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        enable_renderer=args.renderer,
        render_check=args.render_check,
    )
    viewer.run()


if __name__ == "__main__":
    main()
