"""Shared Viser scene helpers for MuJoCo models."""

from __future__ import annotations

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
        "Missing dependencies for Viser scene rendering.\n"
        "Install them with:\n"
        "  pip install yam-sim[viser]\n"
        "  # or with uv:\n"
        "  uv pip install yam-sim[viser]"
    )

from mujoco import mj_id2name, mjtGeom, mjtObj
from PIL import Image


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
    """Convert a MuJoCo primitive geom to trimesh."""
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
        verts = mesh.vertices
        uv = np.zeros((len(verts), 2))
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


def _mujoco_mesh_to_trimesh(model: mujoco.MjModel, geom_idx: int) -> trimesh.Trimesh:
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
        face_texcoord_idx = model.mesh_facetexcoord[face_start : face_start + face_count]

        new_vertices = vertices[faces.flatten()]
        new_uvs = texcoords[face_texcoord_idx.flatten()]
        new_faces = np.arange(face_count * 3).reshape(-1, 3)

        mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)

        matid = model.geom_matid[geom_idx]
        if matid >= 0 and matid < model.nmat:
            rgba = model.mat_rgba[matid]
            texid = int(model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)])
            if texid < 0:
                texid = int(model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGBA)])

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
                    mesh.visual = trimesh.visual.TextureVisuals(uv=new_uvs, material=material)
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
            mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=np.tile(color, (len(new_vertices), 1)))
    else:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        rgba = _get_geom_rgba(model, geom_idx)
        rgba_255 = (rgba * 255).astype(np.uint8)
        mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=np.tile(rgba_255, (len(mesh.vertices), 1)))

    return mesh


def merge_geoms(model: mujoco.MjModel, geom_ids: list[int]) -> trimesh.Trimesh:
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


def is_fixed_body(model: mujoco.MjModel, body_id: int) -> bool:
    """Check if a body is fixed to the world."""
    is_weld = model.body_weldid[body_id] == 0
    root_id = model.body_rootid[body_id]
    root_is_mocap = model.body_mocapid[root_id] >= 0
    return is_weld and not root_is_mocap


def get_body_name(model: mujoco.MjModel, body_id: int) -> str:
    """Get body name or fallback to body_{id}."""
    name = mj_id2name(model, mjtObj.mjOBJ_BODY, body_id)
    return name if name else f"body_{body_id}"


def configure_default_camera(server: "viser.ViserServer") -> None:
    """Set the initial camera to an operator's-eye view behind the robots."""

    @server.on_client_connect
    def _set_camera(client: viser.ClientHandle) -> None:
        client.camera.position = (-0.4, 0.0, 1.3)
        client.camera.look_at = (0.45, 0.0, 0.85)
        client.camera.up_direction = (0.0, 0.0, 1.0)
