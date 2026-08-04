"""Lightweight VR teleop: server-side MuJoCo + Three.js WebXR streaming.

Runs MuJoCo physics on the server, exports body meshes as GLB at startup,
and streams only body transforms (position + quaternion) to a minimal
Three.js VR app running on the headset. No WASM, no heavy client-side
computation.

Usage:
    # Terminal 1: Start the VR streaming server
    uv run python -m yam_sim.examples.vr_streamer --task blocks

    # Terminal 2: Start GELLO leader(s)
    uv run python -m yam_sim.teleop.gello_leader --name right ...

    # Open in VR headset browser: http://<your-ip>:8012
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import queue
import struct
import threading
import time
from pathlib import Path

os.environ.pop("MUJOCO_GL", None)

import mujoco
import numpy as np
import trimesh
import trimesh.visual

from aiohttp import web

try:
    import zmq
except ImportError:
    raise SystemExit("Missing zmq. Install with: uv add pyzmq")

from yam_sim.debug import TASK_DASHBOARD_HTML, TaskEvalDashboardState
from yam_sim.scene_xml import SceneXmlTransformOptions, build_scene_xml
from yam_sim.teleop import communication as comms
from yam_sim.teleop.episode_recorder import TeleopEpisodeRecorder


# ---------------------------------------------------------------------------
# Mesh export
# ---------------------------------------------------------------------------

_DEFAULT_VISIBLE_GROUPS = (0, 1, 2)
_AGGREGATE_EXPORT_ROOTS = frozenset({"dishrack", "plate", "tin_box"})
_HIDDEN_BODY_EXPORT_Z = -1.0


def _should_export_scene_after_reset(
    *,
    previous_model: mujoco.MjModel,
    current_model: mujoco.MjModel,
    task: str,
    previous_metadata: dict[str, object] | None,
    current_metadata: dict[str, object] | None,
) -> bool:
    if current_model is not previous_model:
        return True
    if task != "chess":
        return False
    previous_tin_active = bool((previous_metadata or {}).get("tin_active", False))
    current_tin_active = bool((current_metadata or {}).get("tin_active", False))
    return previous_tin_active != current_tin_active


def _parse_visible_groups_arg(values: list[str] | None) -> tuple[int, ...]:
    """Parse ``--visible-groups`` into a validated, deduplicated tuple."""
    if not values:
        return _DEFAULT_VISIBLE_GROUPS

    tokens: list[str] = []
    for value in values:
        tokens.extend(part.strip() for part in value.split(",") if part.strip())

    if not tokens:
        return _DEFAULT_VISIBLE_GROUPS

    if len(tokens) == 1 and tokens[0].lower() == "all":
        return tuple(range(int(mujoco.mjNGROUP)))

    groups: list[int] = []
    for token in tokens:
        try:
            group = int(token)
        except ValueError as exc:
            raise ValueError(
                f"invalid visible group '{token}'; expected integers in [0, {mujoco.mjNGROUP - 1}] or 'all'"
            ) from exc
        if not 0 <= group < int(mujoco.mjNGROUP):
            raise ValueError(
                f"invalid visible group '{group}'; expected integers in [0, {mujoco.mjNGROUP - 1}]"
            )
        groups.append(group)

    return tuple(sorted(set(groups)))

def _get_geom_rgba(model, geom_id):
    matid = model.geom_matid[geom_id]
    if matid >= 0:
        return model.mat_rgba[matid].copy()
    rgba = model.geom_rgba[geom_id].copy()
    if rgba.sum() == 0:
        rgba = np.array([0.5, 0.5, 0.5, 1.0])
    return rgba


def _create_primitive_mesh(model, geom_id):
    """Convert a MuJoCo primitive geom to trimesh with texture support."""
    from PIL import Image

    size = model.geom_size[geom_id]
    geom_type = model.geom_type[geom_id]
    rgba = _get_geom_rgba(model, geom_id)
    rgba_uint8 = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)

    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        mesh = trimesh.creation.icosphere(radius=size[0], subdivisions=2)
    elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        mesh = trimesh.creation.box(extents=2.0 * size)
    elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        mesh = trimesh.creation.capsule(radius=size[0], height=2.0 * size[1])
    elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        mesh = trimesh.creation.cylinder(radius=size[0], height=2.0 * size[1])
    elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        mesh.apply_scale(size)
    elif geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
        plane_x = 2.0 * size[0] if size[0] > 0 else 20.0
        plane_y = 2.0 * size[1] if size[1] > 0 else 20.0
        mesh = trimesh.creation.box((plane_x, plane_y, 0.001))
    else:
        return None

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
            tex_data = model.tex_data[tex_adr:tex_adr + tex_w * tex_h * tex_nc]
            texrepeat = model.mat_texrepeat[matid]

            if tex_nc == 3:
                image = Image.fromarray(
                    np.flipud(tex_data.reshape(tex_h, tex_w, 3).astype(np.uint8)), mode="RGB")
            elif tex_nc == 4:
                image = Image.fromarray(
                    np.flipud(tex_data.reshape(tex_h, tex_w, 4).astype(np.uint8)), mode="RGBA")
            elif tex_nc == 1:
                image = Image.fromarray(
                    np.flipud(tex_data.reshape(tex_h, tex_w).astype(np.uint8)), mode="L")
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


def _mujoco_mesh_to_trimesh(model, geom_id):
    """Convert a MuJoCo mesh geom to trimesh with texture support."""
    from PIL import Image

    mesh_id = model.geom_dataid[geom_id]
    vert_start = int(model.mesh_vertadr[mesh_id])
    vert_count = int(model.mesh_vertnum[mesh_id])
    face_start = int(model.mesh_faceadr[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])

    vertices = model.mesh_vert[vert_start:vert_start + vert_count]
    faces = model.mesh_face[face_start:face_start + face_count]

    texcoord_adr = model.mesh_texcoordadr[mesh_id]
    texcoord_num = model.mesh_texcoordnum[mesh_id]

    if texcoord_num > 0:
        texcoords = model.mesh_texcoord[texcoord_adr:texcoord_adr + texcoord_num]
        face_texcoord_idx = model.mesh_facetexcoord[face_start:face_start + face_count]

        # Expand vertices/UVs per-face for independent texcoord indexing
        new_vertices = vertices[faces.flatten()]
        new_uvs = texcoords[face_texcoord_idx.flatten()]
        new_faces = np.arange(face_count * 3).reshape(-1, 3)

        mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)

        matid = model.geom_matid[geom_id]
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
                tex_data = model.tex_data[tex_adr:tex_adr + tex_w * tex_h * tex_nc]

                image = None
                if tex_nc == 1:
                    image = Image.fromarray(np.flipud(tex_data.reshape(tex_h, tex_w)).astype(np.uint8), mode="L")
                elif tex_nc == 3:
                    image = Image.fromarray(np.flipud(tex_data.reshape(tex_h, tex_w, 3)).astype(np.uint8), mode="RGB")
                elif tex_nc == 4:
                    image = Image.fromarray(np.flipud(tex_data.reshape(tex_h, tex_w, 4)).astype(np.uint8), mode="RGBA")

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
                        vertex_colors=np.tile(rgba_255, (len(new_vertices), 1)))
            else:
                rgba_255 = (rgba * 255).astype(np.uint8)
                mesh.visual = trimesh.visual.ColorVisuals(
                    vertex_colors=np.tile(rgba_255, (len(new_vertices), 1)))
        else:
            rgba = _get_geom_rgba(model, geom_id)
            rgba_255 = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
            mesh.visual = trimesh.visual.ColorVisuals(
                vertex_colors=np.tile(rgba_255, (len(new_vertices), 1)))
    else:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        rgba = _get_geom_rgba(model, geom_id)
        rgba_255 = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
        mesh.visual = trimesh.visual.ColorVisuals(
            vertex_colors=np.tile(rgba_255, (len(mesh.vertices), 1)))

    return mesh


def _patch_glb_add_material(glb_path: str):
    """Patch GLB: add default PBR material for primitives that lack one.

    trimesh exports COLOR_0 vertex colors but no material, which causes
    Three.js GLTFLoader to ignore the vertex colors. Textured meshes
    already have materials and are left untouched.
    """
    import json as _json

    with open(glb_path, "rb") as f:
        header = f.read(12)
        chunk0_len = struct.unpack("<I", f.read(4))[0]
        chunk0_type = f.read(4)
        json_bytes = f.read(chunk0_len)
        rest = f.read()  # binary chunk

    gltf = _json.loads(json_bytes)

    # Check if any primitives need a default material
    needs_patch = False
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            if "material" not in prim or prim["material"] is None:
                needs_patch = True
                break

    if not needs_patch:
        return

    # Add a default material at the end of the materials list
    if "materials" not in gltf:
        gltf["materials"] = []
    default_idx = len(gltf["materials"])
    gltf["materials"].append({
        "pbrMetallicRoughness": {
            "baseColorFactor": [1, 1, 1, 1],
            "metallicFactor": 0.1,
            "roughnessFactor": 0.7,
        }
    })

    # Assign default material only to primitives that lack one
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            if "material" not in prim or prim["material"] is None:
                prim["material"] = default_idx

    # Re-encode
    new_json = _json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    while len(new_json) % 4 != 0:
        new_json += b" "

    with open(glb_path, "wb") as f:
        total = 12 + 8 + len(new_json) + len(rest)
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<I", len(new_json)))
        f.write(b"JSON")
        f.write(new_json)
        f.write(rest)


def _body_name(model, body_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id)) or f"body_{int(body_id)}"


def _is_aggregate_export_root_name(name: str) -> bool:
    if name in _AGGREGATE_EXPORT_ROOTS:
        return True
    if not name.startswith("plate_"):
        return False
    return name[len("plate_"):].isdigit()


def _body_export_root_id(model, body_id: int) -> int:
    current = int(body_id)
    while current > 0:
        if _is_aggregate_export_root_name(_body_name(model, current)):
            return current
        parent = int(model.body_parentid[current])
        if parent <= 0 or parent == current:
            break
        current = parent
    return int(body_id)


def _transform_inverse(pos: np.ndarray, rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rot_inv = rot.T
    pos_inv = -rot_inv @ pos
    return pos_inv, rot_inv


def _compose_transform(pos_a: np.ndarray, rot_a: np.ndarray, pos_b: np.ndarray, rot_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return pos_a + rot_a @ pos_b, rot_a @ rot_b


def _geom_relative_transform(data, root_body_id: int, geom_id: int) -> tuple[np.ndarray, np.ndarray]:
    root_pos = np.asarray(data.xpos[root_body_id], dtype=np.float64)
    root_rot = np.asarray(data.xmat[root_body_id], dtype=np.float64).reshape(3, 3)
    root_pos_inv, root_rot_inv = _transform_inverse(root_pos, root_rot)
    geom_pos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    geom_rot = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    return _compose_transform(root_pos_inv, root_rot_inv, geom_pos, geom_rot)


def _round_tuple(values, ndigits: int = 6) -> tuple[float, ...]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return tuple(float(x) for x in np.round(arr, ndigits))


def _geom_signature(model, data, root_body_id: int, geom_id: int) -> tuple:
    rel_pos, rel_rot = _geom_relative_transform(data, root_body_id, geom_id)
    geom_type = int(model.geom_type[geom_id])
    mesh_id = int(model.geom_dataid[geom_id]) if geom_type == mujoco.mjtGeom.mjGEOM_MESH else -1
    mesh_name = ""
    mesh_shape: tuple[int, int] = (0, 0)
    if mesh_id >= 0:
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) or ""
        mesh_shape = (int(model.mesh_vertnum[mesh_id]), int(model.mesh_facenum[mesh_id]))

    matid = int(model.geom_matid[geom_id])
    mat_name = ""
    mat_rgba: tuple[float, ...]
    tex_name = ""
    if matid >= 0:
        mat_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, matid) or f"mat_{matid}"
        mat_rgba = _round_tuple(model.mat_rgba[matid])
        texid = int(model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)])
        if texid < 0:
            texid = int(model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGBA)])
        if texid >= 0:
            tex_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TEXTURE, texid) or f"tex_{texid}"
    else:
        mat_rgba = _round_tuple(model.geom_rgba[geom_id])

    return (
        geom_type,
        mesh_name,
        mesh_shape,
        _round_tuple(model.geom_size[geom_id]),
        _round_tuple(rel_pos),
        _round_tuple(rel_rot),
        mat_name,
        mat_rgba,
        tex_name,
    )


def _sanitize_export_key(body_key: str) -> str:
    chars: list[str] = []
    for ch in body_key:
        if ch.isalnum() or ch in {"-", "_"}:
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars)


def _collect_export_entries(model, data, visible_groups: set[int]) -> list[dict[str, object]]:
    grouped_geoms: dict[int, list[int]] = {}
    for geom_id in range(model.ngeom):
        if int(model.geom_group[geom_id]) not in visible_groups:
            continue
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        root_body_id = _body_export_root_id(model, int(model.geom_bodyid[geom_id]))
        grouped_geoms.setdefault(root_body_id, []).append(geom_id)

    entries: list[dict[str, object]] = []
    for root_body_id in sorted(grouped_geoms, key=lambda body_id: _body_name(model, body_id)):
        if float(data.xpos[root_body_id][2]) < _HIDDEN_BODY_EXPORT_Z:
            continue
        body_key = _body_name(model, root_body_id)
        geom_ids = list(grouped_geoms[root_body_id])
        signatures = sorted(
            (_geom_signature(model, data, root_body_id, geom_id) for geom_id in geom_ids),
            key=repr,
        )
        mesh_key = hashlib.sha1(repr(signatures).encode("utf-8")).hexdigest()[:16]
        is_fixed = (
            model.body_weldid[root_body_id] == 0
            and model.body_mocapid[model.body_rootid[root_body_id]] < 0
        )
        entries.append(
            {
                "body_key": body_key,
                "geom_ids": geom_ids,
                "transform_body_id": int(root_body_id),
                "is_fixed": bool(is_fixed),
                "mesh_key": mesh_key,
            }
        )
    return entries


def _export_entry_glb(model, data, entry: dict[str, object], glb_path: Path) -> None:
    from scipy.spatial.transform import Rotation

    root_body_id = int(entry["transform_body_id"])
    geom_ids = [int(geom_id) for geom_id in entry["geom_ids"]]
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    meshes = []
    for gid in geom_ids:
        geom_type = model.geom_type[gid]
        if geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = _mujoco_mesh_to_trimesh(model, gid)
        else:
            mesh = _create_primitive_mesh(model, gid)
        if mesh is None:
            continue

        rel_pos, rel_rot = _geom_relative_transform(data, root_body_id, gid)
        T = np.eye(4)
        T[:3, :3] = rel_rot
        T[:3, 3] = rel_pos
        mesh.apply_transform(T)

        verts = np.array(mesh.vertices)
        new_verts = np.empty_like(verts)
        new_verts[:, 0] = verts[:, 0]
        new_verts[:, 1] = verts[:, 2]
        new_verts[:, 2] = -verts[:, 1]
        mesh.vertices = new_verts

        meshes.append(mesh)

    if not meshes:
        return

    has_texture = any(isinstance(m.visual, trimesh.visual.TextureVisuals) for m in meshes)
    if has_texture or len(meshes) > 1:
        scene = trimesh.Scene(meshes)
        scene.export(str(glb_path), file_type="glb")
    else:
        meshes[0].export(str(glb_path), file_type="glb")

    _patch_glb_add_material(str(glb_path))


def export_body_glbs(
    model,
    data,
    output_dir: Path,
    visible_groups: set[int] | None = None,
):
    """Export merged GLBs for visible bodies, reusing unchanged meshes by content key."""
    if visible_groups is None:
        visible_groups = set(_DEFAULT_VISIBLE_GROUPS)
    output_dir.mkdir(parents=True, exist_ok=True)

    body_info: dict[str, dict[str, object]] = {}
    changed = 0
    entries = _collect_export_entries(model, data, visible_groups)
    for entry in entries:
        body_key = str(entry["body_key"])
        mesh_key = str(entry["mesh_key"])
        file_name = f"{_sanitize_export_key(body_key)}__{mesh_key}.glb"
        glb_path = output_dir / file_name
        if not glb_path.exists():
            _export_entry_glb(model, data, entry, glb_path)
            changed += 1

        body_info[body_key] = {
            "file": file_name,
            "url": f"/meshes/{file_name}",
            "is_fixed": bool(entry["is_fixed"]),
            "transform_body_id": int(entry["transform_body_id"]),
            "mesh_key": mesh_key,
        }

    return body_info, {"changed": changed, "total": len(body_info)}


# ---------------------------------------------------------------------------
# HTML app
# ---------------------------------------------------------------------------

VR_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>XDoF VR Teleop</title>
<style>
  body { margin: 0; overflow: hidden; background: #1a1a2e; }
  #info { position: absolute; top: 10px; width: 100%; text-align: center;
          color: #fff; font-family: monospace; font-size: 14px; z-index: 1; }
  #asset-info { position: absolute; top: 10px; left: 10px; z-index: 1;
          color: #fff; font-family: monospace; font-size: 13px;
          background: rgba(0, 0, 0, 0.45); padding: 8px 10px; border-radius: 6px;
          white-space: pre; display: none; }
  #vr-controls { position: absolute; bottom: 20px; width: 100%; text-align: center; z-index: 1;
    display: flex; justify-content: center; gap: 10px; }
  #vr-controls button { position: static !important; transform: none !important; }
</style>
</head>
<body>
<div id="info">Connecting...</div>
<div id="asset-info"></div>
<div id="vr-controls"></div>

<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }
}
</script>

<script type="module">
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRButton } from 'three/addons/webxr/VRButton.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);

// Lighting
const ambient = new THREE.AmbientLight(0xffffff, 0.35);
scene.add(ambient);
const dir = new THREE.DirectionalLight(0xffffff, 1.0);
dir.position.set(2, 4, 3);
scene.add(dir);
const dir2 = new THREE.DirectionalLight(0xffffff, 0.5);
dir2.position.set(-2, 3, -1);
scene.add(dir2);

// Ground plane
const groundGeo = new THREE.PlaneGeometry(10, 10);
const groundMat = new THREE.MeshStandardMaterial({ color: 0x2a2a3e, roughness: 0.9 });
const ground = new THREE.Mesh(groundGeo, groundMat);
ground.rotation.x = -Math.PI / 2;
ground.position.y = -0.001;
scene.add(ground);

// Grid
const grid = new THREE.GridHelper(4, 20, 0x444466, 0x333355);
scene.add(grid);

// Camera
const camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.01, 100);
camera.position.set(0.5, 1.5, 1.5);
camera.lookAt(0, 0.8, 0);

// VR rig — positions the user in the scene when entering VR
const vrPos = window.__VR_POS__ || [0, 1.5, 1.0];
const vrTarget = window.__VR_TARGET__ || [0, 0.8, 0];
const vrRig = new THREE.Group();
vrRig.position.set(vrPos[0], vrPos[1], vrPos[2]);
// Rotate rig to face the target (only yaw — lookAt then zero out pitch)
const targetVec = new THREE.Vector3(vrTarget[0], vrPos[1], vrTarget[2]);
vrRig.lookAt(targetVec);
scene.add(vrRig);
vrRig.add(camera);

// Renderer with WebXR
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.xr.enabled = true;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
document.body.appendChild(renderer.domElement);

// VR button
const vrBtn = VRButton.createButton(renderer);
document.getElementById('vr-controls').appendChild(vrBtn);

// AR/Passthrough button — custom handler
const arBtn = document.createElement('button');
arBtn.textContent = 'START AR';
// Match VRButton style after a tick (VRButton applies styles async)
setTimeout(() => {
  const vbStyle = window.getComputedStyle(vrBtn);
  arBtn.style.padding = vbStyle.padding;
  arBtn.style.fontSize = vbStyle.fontSize;
  arBtn.style.fontFamily = vbStyle.fontFamily;
  arBtn.style.border = vbStyle.border;
  arBtn.style.borderRadius = vbStyle.borderRadius;
  arBtn.style.color = vbStyle.color;
  arBtn.style.cursor = 'pointer';
  arBtn.style.background = '#1565c0';
  arBtn.style.opacity = vbStyle.opacity;
}, 100);
arBtn.addEventListener('click', async () => {
  scene.background = null;
  renderer.setClearColor(0x000000, 0);
  ground.visible = false;
  grid.visible = false;
  try {
    const session = await navigator.xr.requestSession('immersive-ar', {
      requiredFeatures: ['local-floor'],
      optionalFeatures: ['hand-tracking'],
    });
    renderer.xr.setSession(session);
    session.addEventListener('end', () => {
      scene.background = new THREE.Color(0x1a1a2e);
      renderer.setClearColor(0x000000, 1);
      ground.visible = true;
      grid.visible = true;
      vrRig.position.y = vrPos[1];
    });
  } catch (e) {
    console.error('AR failed, trying VR with passthrough:', e);
    try {
      const session = await navigator.xr.requestSession('immersive-vr', {
        requiredFeatures: ['local-floor'],
        optionalFeatures: ['hand-tracking', 'passthrough'],
      });
      renderer.xr.setSession(session);
      session.addEventListener('end', () => {
        scene.background = new THREE.Color(0x1a1a2e);
        renderer.setClearColor(0x000000, 1);
        ground.visible = true;
        grid.visible = true;
        vrRig.position.y = vrPos[1];
      });
    } catch (e2) {
      console.error('Passthrough failed:', e2);
      scene.background = new THREE.Color(0x1a1a2e);
      ground.visible = true;
      grid.visible = true;
    }
  }
});
document.getElementById('vr-controls').appendChild(arBtn);


// Orbit controls for non-VR (use vr target as orbit target too)
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(vrTarget[0], vrTarget[1], vrTarget[2]);
controls.update();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Scene rotation: rotate so table faces VR user
const sceneRoot = new THREE.Group();
sceneRoot.rotation.y = Math.PI / 2;
scene.add(sceneRoot);

// Load body info and meshes via GLTFLoader
const info = document.getElementById('info');
const assetInfo = document.getElementById('asset-info');
const loader = new GLTFLoader();
const bodyMeshes = {};
let bodyInfo = {};
let streamBodyMap = {};
let sceneVersion = 0;
let resetRequestPending = false;

function isPutBottlesDebugTask() {
  const task = window.__ASSET_DEBUG_TASK__;
  return task === 'put_bottles' || task === 'water_bottles';
}

function isChessDebugTask() {
  return window.__ASSET_DEBUG_TASK__ === 'chess';
}

function updateAssetInfo(randomization) {
  if (!window.__ASSET_DEBUG__) {
    assetInfo.style.display = 'none';
    return;
  }
  const task = window.__ASSET_DEBUG_TASK__ || 'dishrack';
  if (isPutBottlesDebugTask()) {
    const bottle = randomization?.bottle_variant || '?';
    const bottleCount = randomization?.bottle_count ?? '?';
    const bottleVariants = Array.isArray(randomization?.bottle_variants) && randomization.bottle_variants.length
      ? randomization.bottle_variants.join(', ')
      : bottle;
    const binScale = randomization?.bin_scale ?? '?';
    assetInfo.style.display = 'block';
    assetInfo.textContent =
      `Bottles (${bottleCount}): ${bottleVariants}\nBin scale: ${binScale}\nX: next bottle\nY: previous bottle\nB/Space: reset`;
    return;
  }
  if (isChessDebugTask()) {
    const scenario = randomization?.scenario || '?';
    const colorMode = randomization?.color_mode || '?';
    const tinVariant = randomization?.tin_variant || '?';
    const targetCount = randomization?.target_count ?? '?';
    const scaleCount = randomization?.scale_count ?? 0;
    assetInfo.style.display = 'block';
    assetInfo.textContent =
      `Chess scenario: ${scenario}\nColor mode: ${colorMode}\nTin: ${tinVariant}\nTargets: ${targetCount}\nScaled pieces: ${scaleCount}\nX: next scenario\nY: next color mode\nB/Space: reset`;
    return;
  }
  const plate = randomization?.plate_variant || '?';
  const plateCount = randomization?.plate_count ?? '?';
  const plateVariants = Array.isArray(randomization?.plate_variants) && randomization.plate_variants.length
    ? randomization.plate_variants.join(', ')
    : plate;
  const rack = randomization?.dish_rack_variant || '?';
  assetInfo.style.display = 'block';
  assetInfo.textContent =
    `Plates (${plateCount}): ${plateVariants}\nRack: ${rack}\nX: next plate\nY: next rack\nB/Space: reset`;
}

updateAssetInfo(window.__ASSET_DEBUG_STATE__);

function disposeMaterial(material) {
  if (!material) return;
  if (Array.isArray(material)) {
    for (const item of material) disposeMaterial(item);
    return;
  }
  for (const value of Object.values(material)) {
    if (value && value.isTexture) value.dispose();
  }
  material.dispose?.();
}

function clearSceneMeshes() {
  for (const bodyKey of Object.keys(bodyMeshes)) {
    removeBodyMesh(bodyKey);
  }
  streamBodyMap = {};
}

function removeBodyMesh(bodyKey) {
  const obj = bodyMeshes[bodyKey];
  if (!obj) return;
  sceneRoot.remove(obj);
  obj.traverse((child) => {
    if (!child.isMesh) return;
    child.geometry?.dispose();
    disposeMaterial(child.material);
  });
  delete bodyMeshes[bodyKey];
}

function rebuildStreamBodyMap(nextBodyInfo) {
  streamBodyMap = {};
  for (const [bodyKey, bdata] of Object.entries(nextBodyInfo)) {
    streamBodyMap[String(bdata.transform_body_id)] = bodyKey;
  }
}

function applyMocapInit(mocapInit) {
  if (!mocapInit) return;
  for (const m of mocapInit) {
    lastMocapPos[m.id] = new THREE.Vector3(m.pos[0], m.pos[1], m.pos[2]);
    if (m.quat) {
      lastMocapQuat[m.id] = new THREE.Quaternion(m.quat[0], m.quat[1], m.quat[2], m.quat[3]);
    } else {
      lastMocapQuat[m.id] = new THREE.Quaternion(0, 0, 0, 1);
    }
  }
  _mocapSeeded = true;
}

function requestSceneReset(source, randomization = null) {
  if (!ws || ws.readyState !== WebSocket.OPEN || resetRequestPending) return;
  resetRequestPending = true;
  info.textContent = `Resetting scene (${source})...`;
  const message = { type: 'reset', source };
  if (randomization) {
    message.randomization = randomization;
  }
  ws.send(JSON.stringify(message));
}

async function loadScene(requestedSceneVersion = null) {
  info.textContent = 'Loading scene...';

  const resp = await fetch('/api/bodies');
  const payload = await resp.json();
  const nextBodyInfo = payload.bodies;
  sceneVersion = payload.scene_version;
  if (requestedSceneVersion !== null && requestedSceneVersion !== sceneVersion) {
    console.warn(`Scene version mismatch: requested ${requestedSceneVersion}, got ${sceneVersion}`);
  }
  rebuildStreamBodyMap(nextBodyInfo);

  const nextKeys = new Set(Object.keys(nextBodyInfo));
  for (const bodyKey of Object.keys(bodyMeshes)) {
    if (!nextKeys.has(bodyKey)) {
      removeBodyMesh(bodyKey);
    }
  }

  const total = Object.keys(nextBodyInfo).length;
  let loaded = 0;
  frameCount = 0;

  const orderedKeys = Object.keys(nextBodyInfo).sort();
  for (const bodyKey of orderedKeys) {
    const bdata = nextBodyInfo[bodyKey];
    const prev = bodyInfo[bodyKey];
    const needsReload = !prev || prev.mesh_key !== bdata.mesh_key || prev.url !== bdata.url;
    if (!needsReload) continue;

    removeBodyMesh(bodyKey);
    try {
      const gltf = await new Promise((resolve, reject) => {
        loader.load(bdata.url, resolve, undefined, reject);
      });

      const obj = gltf.scene;
      obj.traverse((child) => {
        if (child.isMesh) {
          child.geometry.computeVertexNormals();

          if (child.material.map) {
            child.material.side = THREE.DoubleSide;
          } else if (child.geometry.attributes.color) {
            child.material = new THREE.MeshStandardMaterial({
              vertexColors: true,
              side: THREE.DoubleSide,
              roughness: 0.5,
              metalness: 0.15,
            });
          } else {
            child.material.side = THREE.DoubleSide;
          }
        }
      });

      sceneRoot.add(obj);
      bodyMeshes[bodyKey] = obj;
      loaded++;
    } catch (e) {
      console.error(`Failed to load body ${bodyKey}:`, e);
    }
    info.textContent = `Loading meshes: ${loaded}/${total}`;
  }

  bodyInfo = nextBodyInfo;

  if (!ws || ws.readyState !== WebSocket.OPEN) {
    info.textContent = `Loaded ${loaded}/${total} mesh updates. Connecting...`;
  } else if (!window.__DEBUG__) {
    info.textContent = '';
  }
}

// WebSocket for transform streaming
let ws = null;
let frameCount = 0;

function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(proto + '//' + location.host + '/ws');
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    info.textContent = 'Connected! Waiting for GELLO...';
  };

  ws.onmessage = async (event) => {
    if (typeof event.data === 'string') {
      let message = null;
      try {
        message = JSON.parse(event.data);
      } catch (err) {
        console.error('Failed to parse control message:', err);
        return;
      }

      if (message.type === 'scene_reload') {
        applyMocapInit(message.mocap_init);
        updateAssetInfo(message.randomization);
        resetRequestPending = false;
        await loadScene(message.scene_version);
        return;
      }
      if (message.type === 'reset_complete') {
        applyMocapInit(message.mocap_init);
        updateAssetInfo(message.randomization);
        resetRequestPending = false;
        if (!window.__DEBUG__) {
          info.textContent = '';
        }
        return;
      }
      if (message.type === 'reset_error') {
        resetRequestPending = false;
        info.textContent = `Reset failed: ${message.error || 'unknown error'}`;
      }
      return;
    }

    // Binary format: [n_bodies (uint16), body_id (uint16), px, py, pz, qx, qy, qz, qw (float32) × n]
    const buf = new Float32Array(event.data);
    const n = buf.length / 8;  // 8 floats per body: id(as float), px, py, pz, qx, qy, qz, qw

    for (let i = 0; i < n; i++) {
      const offset = i * 8;
      const bid = Math.round(buf[offset]).toString();
      const px = buf[offset + 1];
      const py = buf[offset + 2];
      const pz = buf[offset + 3];
      const qx = buf[offset + 4];
      const qy = buf[offset + 5];
      const qz = buf[offset + 6];
      const qw = buf[offset + 7];

      const bodyKey = streamBodyMap[bid];
      const obj = bodyKey ? bodyMeshes[bodyKey] : null;
      if (obj) {
        obj.position.set(px, py, pz);
        obj.quaternion.set(qx, qy, qz, qw);
      }
    }

    frameCount++;
    if (frameCount === 1 && !window.__DEBUG__) {
      info.textContent = '';
    }
    if (window.__DEBUG__ && frameCount % 30 === 0) {
      info.textContent = `Frame: ${frameCount}`;
    }
  };

  ws.onclose = () => {
    resetRequestPending = false;
    info.textContent = 'Disconnected. Reconnecting...';
    setTimeout(connectWS, 1000);
  };
}

// VR controller tracking (for --mocap mode)
const controllerGrip0 = renderer.xr.getControllerGrip(0);
const controllerGrip1 = renderer.xr.getControllerGrip(1);
scene.add(controllerGrip0);
scene.add(controllerGrip1);

function getTriggerValue(handedness) {
  const session = renderer.xr.getSession();
  if (!session) return 0;
  for (const source of session.inputSources) {
    if (source.handedness === handedness && source.gamepad) {
      return source.gamepad.buttons[0] ? source.gamepad.buttons[0].value : 0;
    }
  }
  return 0;
}

function getGripButton(handedness) {
  const session = renderer.xr.getSession();
  if (!session) return false;
  for (const source of session.inputSources) {
    if (source.handedness === handedness && source.gamepad) {
      return source.gamepad.buttons[1] ? source.gamepad.buttons[1].pressed : false;
    }
  }
  return false;
}

function getFaceButton(handedness, buttonIndex) {
  const session = renderer.xr.getSession();
  if (!session) return false;
  for (const source of session.inputSources) {
    if (source.handedness === handedness && source.gamepad) {
      const buttons = source.gamepad.buttons || [];
      if (buttons[buttonIndex]) return !!buttons[buttonIndex].pressed;
    }
  }
  return false;
}

// Mocap state for relative control
const grabState = {
  left: { held: false, ctrlAnchorPos: new THREE.Vector3(), ctrlAnchorQuat: new THREE.Quaternion(),
          mocapAnchorPos: new THREE.Vector3(), mocapAnchorQuat: new THREE.Quaternion() },
  right: { held: false, ctrlAnchorPos: new THREE.Vector3(), ctrlAnchorQuat: new THREE.Quaternion(),
           mocapAnchorPos: new THREE.Vector3(), mocapAnchorQuat: new THREE.Quaternion() },
};
const lastMocapPos = {};
const lastMocapQuat = {};

// Inverse of sceneRoot rotation to transform controller world coords -> scene local coords
const sceneRotInv = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, -Math.PI / 2, 0));

function getLocalGripPose(grip) {
  // Transform grip world position/quaternion into sceneRoot local frame
  const pos = grip.position.clone().applyQuaternion(sceneRotInv);
  const quat = sceneRotInv.clone().multiply(grip.quaternion);
  return { pos, quat };
}

function updateControllerMocap(handedness, grip, mocapId) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const pressed = getGripButton(handedness);
  const state = grabState[handedness];
  const local = getLocalGripPose(grip);

  if (pressed && !state.held) {
    state.held = true;
    state.ctrlAnchorPos.copy(local.pos);
    state.ctrlAnchorQuat.copy(local.quat);
    if (lastMocapPos[mocapId]) {
      state.mocapAnchorPos.copy(lastMocapPos[mocapId]);
      state.mocapAnchorQuat.copy(lastMocapQuat[mocapId]);
    }
    return;
  }
  if (!pressed) { state.held = false; return; }

  const deltaPos = local.pos.clone().sub(state.ctrlAnchorPos);
  const newPos = state.mocapAnchorPos.clone().add(deltaPos);
  const deltaQuat = local.quat.clone().multiply(state.ctrlAnchorQuat.clone().invert());
  const newQuat = deltaQuat.clone().multiply(state.mocapAnchorQuat);

  if (!lastMocapPos[mocapId]) lastMocapPos[mocapId] = new THREE.Vector3();
  if (!lastMocapQuat[mocapId]) lastMocapQuat[mocapId] = new THREE.Quaternion();
  lastMocapPos[mocapId].copy(newPos);
  lastMocapQuat[mocapId].copy(newQuat);

  ws.send(JSON.stringify({
    type: 'mocap', mocap_id: mocapId,
    position: [newPos.x, newPos.y, newPos.z],
    quaternion: [newQuat.x, newQuat.y, newQuat.z, newQuat.w],
  }));
}

// Seed initial mocap poses
let _mocapSeeded = false;
const faceButtonHeld = { rightB: false, leftX: false, leftY: false };

function assetDebugRequest(extra = {}) {
  return {
    randomize_variants: false,
    ...extra,
  };
}

window.addEventListener('keydown', (event) => {
  if (event.repeat) return;
  if (event.code === 'Space') {
    event.preventDefault();
    if (window.__ASSET_DEBUG__) {
      requestSceneReset('keyboard', assetDebugRequest());
    } else {
      requestSceneReset('keyboard');
    }
    return;
  }
  if (!window.__ASSET_DEBUG__) return;
  if (event.code === 'KeyX') {
    event.preventDefault();
    if (isPutBottlesDebugTask()) {
      requestSceneReset('keyboard_x', assetDebugRequest({ cycle_bottle: 1 }));
    } else if (isChessDebugTask()) {
      requestSceneReset('keyboard_x', assetDebugRequest({ cycle_scenario: 1 }));
    } else {
      requestSceneReset('keyboard_x', assetDebugRequest({ cycle_plate: 1 }));
    }
    return;
  }
  if (event.code === 'KeyY') {
    event.preventDefault();
    if (isPutBottlesDebugTask()) {
      requestSceneReset('keyboard_y', assetDebugRequest({ cycle_bottle: -1 }));
    } else if (isChessDebugTask()) {
      requestSceneReset('keyboard_y', assetDebugRequest({ cycle_color_mode: 1 }));
    } else {
      requestSceneReset('keyboard_y', assetDebugRequest({ cycle_dish_rack: 1 }));
    }
  }
});

// --- Joystick-based VR rig movement ---
// Left stick translates XZ (relative to rig facing), right stick X yaws, right
// stick Y moves vertically. Lets operators nudge the headset pose without
// restarting the server.
const moveSpeed = 1.2;   // m/s
const rotSpeed = 0.8;    // rad/s
const deadzone = 0.15;
let prevTime = performance.now();

function joystickDt() {
  const now = performance.now();
  const dt = Math.min((now - prevTime) / 1000, 0.1);  // cap to avoid jumps
  prevTime = now;
  return dt;
}

function applyJoystickMovement(dt) {
  const session = renderer.xr.getSession();
  if (!session) return;

  let leftAxes = [0, 0];
  let rightAxes = [0, 0];

  for (const source of session.inputSources) {
    if (!source.gamepad) continue;
    const axes = source.gamepad.axes;  // [0]=x, [1]=y (some have 4 axes: [2]=x, [3]=y)
    if (source.handedness === 'left') {
      leftAxes = axes.length >= 4 ? [axes[2], axes[3]] : [axes[0], axes[1]];
    } else if (source.handedness === 'right') {
      rightAxes = axes.length >= 4 ? [axes[2], axes[3]] : [axes[0], axes[1]];
    }
  }

  // Apply deadzone
  const lx = Math.abs(leftAxes[0]) > deadzone ? leftAxes[0] : 0;
  const ly = Math.abs(leftAxes[1]) > deadzone ? leftAxes[1] : 0;
  const rx = Math.abs(rightAxes[0]) > deadzone ? rightAxes[0] : 0;
  const ry = Math.abs(rightAxes[1]) > deadzone ? rightAxes[1] : 0;

  if (lx === 0 && ly === 0 && rx === 0 && ry === 0) return;

  // Right stick X: yaw rotation
  if (rx !== 0) {
    vrRig.rotation.y += rx * rotSpeed * dt;
  }

  // Right stick Y: vertical movement
  if (ry !== 0) {
    vrRig.position.y -= ry * moveSpeed * dt;
  }

  // Left stick: XZ movement relative to rig facing direction
  if (lx !== 0 || ly !== 0) {
    const forward = new THREE.Vector3(0, 0, -1);
    forward.applyQuaternion(vrRig.quaternion);
    forward.y = 0;
    forward.normalize();
    const right = new THREE.Vector3(1, 0, 0);
    right.applyQuaternion(vrRig.quaternion);
    right.y = 0;
    right.normalize();

    vrRig.position.addScaledVector(right, lx * moveSpeed * dt);
    vrRig.position.addScaledVector(forward, -ly * moveSpeed * dt);
  }
}

// Render loop
renderer.setAnimationLoop(() => {
  if (!_mocapSeeded && window.__MOCAP_INIT__) {
    applyMocapInit(window.__MOCAP_INIT__);
  }
  const rightBPressed = getFaceButton('right', 5);
  if (rightBPressed && !faceButtonHeld.rightB) {
    if (window.__ASSET_DEBUG__) {
      requestSceneReset('controller_b', assetDebugRequest());
    } else {
      requestSceneReset('controller_b');
    }
  }
  faceButtonHeld.rightB = rightBPressed;
  if (window.__ASSET_DEBUG__) {
    const leftXPressed = getFaceButton('left', 4);
    if (leftXPressed && !faceButtonHeld.leftX) {
      if (isPutBottlesDebugTask()) {
        requestSceneReset('controller_x', assetDebugRequest({ cycle_bottle: 1 }));
      } else if (isChessDebugTask()) {
        requestSceneReset('controller_x', assetDebugRequest({ cycle_scenario: 1 }));
      } else {
        requestSceneReset('controller_x', assetDebugRequest({ cycle_plate: 1 }));
      }
    }
    faceButtonHeld.leftX = leftXPressed;

    const leftYPressed = getFaceButton('left', 5);
    if (leftYPressed && !faceButtonHeld.leftY) {
      if (isPutBottlesDebugTask()) {
        requestSceneReset('controller_y', assetDebugRequest({ cycle_bottle: -1 }));
      } else if (isChessDebugTask()) {
        requestSceneReset('controller_y', assetDebugRequest({ cycle_color_mode: 1 }));
      } else {
        requestSceneReset('controller_y', assetDebugRequest({ cycle_dish_rack: 1 }));
      }
    }
    faceButtonHeld.leftY = leftYPressed;
  }
  if (renderer.xr.isPresenting && window.__MOCAP__) {
    updateControllerMocap('left', controllerGrip0, 0);
    updateControllerMocap('right', controllerGrip1, 1);

    const leftTrigger = getTriggerValue('left');
    const rightTrigger = getTriggerValue('right');
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'trigger', left: leftTrigger, right: rightTrigger }));
    }
  }
  applyJoystickMovement(joystickDt());
  renderer.render(scene, camera);
});

loadScene().then(connectWS);

// Recording indicator — fixed sphere at the right end of the aluminum frame in world space.
// MuJoCo coords (0, -0.65, 0.82) → Three.js world (0.65, 0.82, 0) via Z-up→Y-up +
// sceneRoot.rotation.y = π/2 conversion.
const recDotMat = new THREE.MeshBasicMaterial({ color: 0x333333 });
const recDot = new THREE.Mesh(new THREE.SphereGeometry(0.025, 16, 16), recDotMat);
recDot.position.set(0.65, 0.82, 0.0);
scene.add(recDot);

async function updateRecLight() {
  try {
    const r = await fetch('/api/recording-state');
    if (r.ok) {
      const { is_recording } = await r.json();
      recDotMat.color.setHex(is_recording ? 0xff1744 : 0x00e676);
    }
  } catch (_) {
    console.warn('Failed to fetch recording state:', _);
  }
}
(function poll() {
  updateRecLight().finally(() => setTimeout(poll, 500));
})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Lightweight VR streaming teleop")
    parser.add_argument("--task", type=str, default="blocks",
                        choices=["bottles", "put_bottles", "marker", "ball_sorting", "empty",
                                 "dishrack", "chess", "blocks",
                                 "mug_tree", "mug_flip", "jenga", "building_blocks", "sweep", "drawer", "pour",
                                 "inhand_transfer"])
    parser.add_argument("--xml", type=str, default=None,
                        help="Load an explicit MuJoCo XML file as-is instead of the built-in task XML")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--left-leader", type=str, default="left")
    parser.add_argument("--right-leader", type=str, default="right")
    parser.add_argument("--control-rate", type=float, default=30.0)
    parser.add_argument("--stream-rate", type=float, default=60.0,
                        help="Rate to stream transforms to VR client")
    parser.add_argument("--debug", action="store_true",
                        help="Show frame counter overlay in VR client")
    parser.add_argument(
        "--visible-groups",
        type=str,
        nargs="+",
        default=None,
        metavar="GROUP",
        help=(
            "MuJoCo geom groups to export into the VR scene. "
            "Accepts space- or comma-separated integers in [0, 5], or 'all'. "
            "Default: 0 1 2"
        ),
    )
    parser.add_argument("--vr-pos", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        metavar=("X", "Y", "Z"),
                        help="VR user floor position in Y-up coords (default: 0 0 -1)")
    parser.add_argument("--vr-target", type=float, nargs=3, default=[0.0, 0.75, 0.0],
                        metavar=("X", "Y", "Z"),
                        help="Point the VR user looks at in Y-up coords (default: 0 0.75 0)")
    parser.add_argument("--clean", action="store_true",
                        help="Remove static visual clutter (cage, walls, table, floor visuals). Keeps collision.")
    parser.add_argument("--flexible-gripper", action="store_true",
                        help="Swap the standard YAM finger bodies for the flexible gripper assembly at launch.")
    parser.add_argument("--mocap", action="store_true",
                        help="Control arms with VR controllers via mocap weld constraints instead of GELLO")
    parser.add_argument(
        "--asset-debug",
        action="store_true",
        help=(
            "Asset debug mode for dishrack/put_bottles/chess: keep variants or "
            "scenario settings pinned on reset and map X/Y to cycle debug choices."
        ),
    )
    parser.add_argument(
        "--no-randomize-scales",
        action="store_true",
        help=(
            "Disable per-reset object scale randomization in VR. This avoids "
            "rebuilding/exporting scene meshes on tasks whose scale changes are XML-based."
        ),
    )
    parser.add_argument(
        "--randomize-scales",
        action="store_true",
        help=(
            "Explicitly enable per-reset object scale randomization. This is useful "
            "for tasks such as chess where scales are disabled by default for faster resets."
        ),
    )
    parser.add_argument(
        "--record-dir",
        type=str,
        default=None,
        help="Optional output dir for exact applied actions + qpos recording",
    )
    args = parser.parse_args()
    try:
        visible_groups = set(_parse_visible_groups_arg(args.visible_groups))
    except ValueError as exc:
        parser.error(str(exc))
    if args.asset_debug and args.task not in {"dishrack", "put_bottles", "chess"}:
        parser.error(
            "--asset-debug is currently only supported for "
            "task='dishrack', task='put_bottles', or task='chess'"
        )
    if args.no_randomize_scales and args.randomize_scales:
        parser.error("--randomize-scales and --no-randomize-scales cannot be used together")

    if args.record_dir and args.mocap:
        raise SystemExit("--record-dir is only supported in GELLO mode, not --mocap mode")

    import yam_sim
    from yam_sim.task_registry import get_task_scene_xml

    custom_xml = Path(args.xml).resolve() if args.xml else None
    if custom_xml is not None and not custom_xml.exists():
        raise SystemExit(f"XML file not found: {custom_xml}")

    transform_options = SceneXmlTransformOptions(
        clean=args.clean,
        mocap=args.mocap,
        flexible_gripper=args.flexible_gripper,
        debug=args.debug,
    )
    need_xml_edit = args.clean or args.mocap or args.flexible_gripper
    if args.task == "inhand_transfer":
        if custom_xml is not None:
            raise SystemExit("--xml is not supported for task 'inhand_transfer'")
        if transform_options.flexible_gripper:
            print("Flexible gripper mode: swapped in flexible gripper assembly")
        if transform_options.clean:
            print("Clean mode: removed cage, walls, floor, cameras")
        if transform_options.mocap:
            print("Mocap mode: added mocap bodies + weld constraints")
        env = yam_sim.make_env(
            scene="hybrid",
            task=args.task,
            render_cameras=False,
            scene_xml_transform_options=transform_options,
        )
    elif custom_xml is not None:
        env = yam_sim.make_env(
            scene="hybrid",
            task=args.task,
            render_cameras=False,
            scene_xml=custom_xml,
        )
    elif need_xml_edit:
        scene_path = get_task_scene_xml(args.task)
        if scene_path is None:
            raise SystemExit(f"Unknown task scene: {args.task}")
        xml, applied_edits = build_scene_xml(
            scene_path,
            options=transform_options,
        )
        if "flexible_gripper" in applied_edits:
            print("Flexible gripper mode: swapped in flexible gripper assembly")
        if "clean" in applied_edits:
            print("Clean mode: removed cage, walls, floor, cameras")
        if "mocap" in applied_edits:
            print("Mocap mode: added mocap bodies + weld constraints")
        env = yam_sim.make_env(
            scene="hybrid",
            task=args.task,
            render_cameras=False,
            scene_xml_string=xml,
            scene_xml_transform_options=transform_options,
        )
    else:
        env = yam_sim.make_env(scene="hybrid", task=args.task, render_cameras=False)

    base_randomization_request: dict[str, object] = {}
    if args.no_randomize_scales:
        base_randomization_request["randomize_scales"] = False
    elif args.randomize_scales:
        base_randomization_request["randomize_scales"] = True

    initial_reset_options = None
    if args.asset_debug:
        initial_request = dict(base_randomization_request)
        initial_request.update(
            {
                "randomize_variants": False,
            }
        )
        initial_reset_options = {"randomization": initial_request}
    elif base_randomization_request:
        initial_reset_options = {"randomization": dict(base_randomization_request)}

    # Reset once up front so the live model matches the first randomization.
    obs, _ = env.reset(options=initial_reset_options)
    state = obs["state"].copy()

    print(f"Task: {args.task}")
    print(f"Environment: nbody={env.model.nbody}, ngeom={env.model.ngeom}")

    # Export meshes into a content-addressed cache so resets only add new GLBs.
    import shutil
    mesh_root = Path("/tmp/yam_vr_meshes")
    if mesh_root.exists():
        shutil.rmtree(mesh_root)
    mesh_root.mkdir(parents=True, exist_ok=True)
    scene_state: dict[str, object] = {
        "version": 0,
        "body_info": {},
        "stream_items": [],
    }

    def current_model() -> mujoco.MjModel:
        return env.model

    def current_data() -> mujoco.MjData:
        return env.data

    def export_current_scene_meshes() -> None:
        version = int(scene_state["version"]) + 1
        body_info, stats = export_body_glbs(
            current_model(),
            current_data(),
            mesh_root,
            visible_groups=visible_groups,
        )
        scene_state["version"] = version
        scene_state["body_info"] = body_info
        scene_state["stream_items"] = [
            (body_key, int(info["transform_body_id"]))
            for body_key, info in sorted(body_info.items())
        ]
        fixed_ids = [body_key for body_key, info in body_info.items() if info["is_fixed"]]
        dynamic_ids = [body_key for body_key, info in body_info.items() if not info["is_fixed"]]
        print(
            f"Exported scene meshes v{version}: {stats['changed']} changed / {stats['total']} bodies "
            f"({len(fixed_ids)} fixed, {len(dynamic_ids)} dynamic)"
        )

    export_current_scene_meshes()
    print(f"Visible geom groups: {sorted(visible_groups)}")

    task_spec = getattr(env, "_task_spec", None)
    task_evaluator = getattr(env, "_task_evaluator", None)
    task_dashboard = TaskEvalDashboardState(
        task_name=args.task,
        prompt=(task_spec.prompt if task_spec is not None else env.prompt),
        evaluator_name=(task_spec.evaluator_name if task_spec is not None else None),
        debug_spec=(task_evaluator.debug_spec() if task_evaluator is not None else None),
    )
    task_dashboard.update(step=0, sim_time=float(current_data().time), result=env.evaluate_task())
    recorder: TeleopEpisodeRecorder | None = None
    if args.record_dir:
        recorder = TeleopEpisodeRecorder(
            args.record_dir,
            task=args.task,
            scene="hybrid",
            prompt=(task_spec.prompt if task_spec is not None else env.prompt),
            control_rate=args.control_rate,
            extra_metadata={
                "source": "vr_streamer",
                "left_leader_topic": f"{args.left_leader}_actions",
                "right_leader_topic": f"{args.right_leader}_actions",
                "clean_mode": bool(args.clean),
                "flexible_gripper": bool(args.flexible_gripper),
                "xml_override": (str(custom_xml) if custom_xml is not None else None),
            },
        )
        recorder.start(env, initial_state=state)
        print(f"Recording exact actions + sim state to {args.record_dir}")

    # ZMQ
    zmq_context = zmq.Context()
    left_sub = comms.create_subscriber(zmq_context, f"{args.left_leader}_actions", conflate=1)
    right_sub = comms.create_subscriber(zmq_context, f"{args.right_leader}_actions", conflate=1)
    left_sub.setsockopt(zmq.RCVTIMEO, 0)
    right_sub.setsockopt(zmq.RCVTIMEO, 0)

    def poll_leader(sub):
        try:
            msg, _ = comms.subscribe(sub)
            return msg
        except (zmq.Again, Exception):
            return None

    # Z-up to Y-up conversion for positions and quaternions
    from scipy.spatial.transform import Rotation

    R_conv = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)

    def build_frame_buffer():
        """Pack all body transforms into a flat float32 buffer (Y-up)."""
        model = current_model()
        data = current_data()
        stream_items = list(scene_state["stream_items"])
        buf = np.zeros(len(stream_items) * 8, dtype=np.float32)
        for i, (_, bid) in enumerate(stream_items):
            xpos = data.xpos[bid]
            xmat = data.xmat[bid].reshape(3, 3)
            pos_yup = R_conv @ xpos
            det = np.linalg.det(xmat)
            if abs(det) > 1e-6:
                mat_yup = R_conv @ xmat @ R_conv.T
                q = Rotation.from_matrix(mat_yup).as_quat()  # xyzw
            else:
                q = np.array([0, 0, 0, 1.0])
            offset = i * 8
            buf[offset] = float(bid)
            buf[offset+1:offset+4] = pos_yup
            buf[offset+4:offset+8] = q
        return buf.tobytes()

    # Shared state for physics thread
    GRIP_SCALE = 0.0475
    lock = threading.Lock()
    stop_event = threading.Event()
    connected = [False]
    step_count = [0]
    frame_bytes = [build_frame_buffer()]
    pending_reset_request: list[dict[str, object] | None] = [None]
    pending_client_events: queue.SimpleQueue[dict[str, object]] = queue.SimpleQueue()

    # Mocap control state
    pending_mocap_updates = []
    pending_trigger = [None]
    R_inv = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)  # Y-up to Z-up

    def find_gripper_ids(model: mujoco.MjModel) -> dict[str, tuple[int, float, float]]:
        gripper_ids: dict[str, tuple[int, float, float]] = {}
        if not args.mocap:
            return gripper_ids
        for i in range(model.nu):
            aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
            if "left_gripper" in aname:
                lo, hi = model.actuator_ctrlrange[i]
                gripper_ids["left"] = (i, lo, hi)
            elif "right_gripper" in aname:
                lo, hi = model.actuator_ctrlrange[i]
                gripper_ids["right"] = (i, lo, hi)
        return gripper_ids

    gripper_ids = [find_gripper_ids(current_model())]
    if args.mocap:
        print(f"Mocap gripper IDs: {gripper_ids[0]}")

    def build_mocap_init_payload() -> list[dict[str, object]]:
        model = current_model()
        data = current_data()
        mocap_init: list[dict[str, object]] = []
        for i in range(model.nmocap):
            p = R_conv @ data.mocap_pos[i]
            q_wxyz = data.mocap_quat[i]
            mat_zup = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
            mat_yup = R_conv @ mat_zup @ R_conv.T
            q_yup = Rotation.from_matrix(mat_yup).as_quat()  # xyzw
            mocap_init.append(
                {
                    "id": i,
                    "pos": [float(p[0]), float(p[1]), float(p[2])],
                    "quat": [float(q_yup[0]), float(q_yup[1]), float(q_yup[2]), float(q_yup[3])],
                }
            )
        return mocap_init

    def current_randomization_metadata() -> dict[str, object] | None:
        randomization_state = getattr(env, "_last_randomization", None)
        metadata = getattr(randomization_state, "metadata", None)
        if not isinstance(metadata, dict):
            return None
        scale_states = getattr(randomization_state, "scale_states", {})
        if not isinstance(scale_states, dict):
            scale_states = {}

        plate_variant = metadata.get("plate_variant")
        plate_variants = metadata.get("plate_variants")
        plate_count = metadata.get("plate_count")
        dish_rack_variant = metadata.get("dish_rack_variant")
        trash_count = metadata.get("trash_count")
        bottle_variant = metadata.get("bottle_variant")
        bottle_variants = metadata.get("bottle_variants")
        bottle_count = metadata.get("bottle_count")
        chess_scenario = metadata.get("scenario")
        chess_color_mode = metadata.get("color_mode")
        chess_target_count = metadata.get("target_count")
        chess_tin_variant = metadata.get("tin_variant")
        if (
            plate_variant is None
            and dish_rack_variant is None
            and plate_count is None
            and trash_count is None
            and bottle_variant is None
            and bottle_count is None
            and chess_scenario is None
            and chess_color_mode is None
            and chess_target_count is None
            and chess_tin_variant is None
        ):
            return None
        payload = {
            "plate_variant": str(plate_variant or ""),
            "plate_variants": [str(value) for value in plate_variants] if isinstance(plate_variants, (list, tuple)) else [],
            "plate_count": int(plate_count) if plate_count is not None else None,
            "dish_rack_variant": str(dish_rack_variant or ""),
            "bottle_variant": str(bottle_variant or ""),
            "bottle_variants": [str(value) for value in bottle_variants] if isinstance(bottle_variants, (list, tuple)) else [],
            "bottle_count": int(bottle_count) if bottle_count is not None else None,
            "scenario": str(chess_scenario or ""),
            "color_mode": str(chess_color_mode or ""),
            "tin_variant": str(chess_tin_variant or ""),
            "target_count": int(chess_target_count) if chess_target_count is not None else None,
        }
        bin_scale = scale_states.get("bin_joint")
        if bin_scale is not None:
            payload["bin_scale"] = round(float(bin_scale), 3)
        elif bottle_variant is not None or bottle_count is not None:
            payload["bin_scale"] = 1.0
        if chess_scenario is not None or chess_color_mode is not None:
            payload["scale_count"] = len(scale_states)
        if trash_count is not None:
            payload["trash_count"] = int(trash_count)
        return payload

    def queue_client_event(event_type: str, *, error: str | None = None) -> None:
        event: dict[str, object] = {
            "type": event_type,
            "scene_version": int(scene_state["version"]),
        }
        randomization_metadata = current_randomization_metadata()
        if randomization_metadata is not None:
            event["randomization"] = randomization_metadata
        if args.mocap:
            event["mocap_init"] = build_mocap_init_payload()
        if error is not None:
            event["error"] = error
        pending_client_events.put(event)

    def reset_scene(randomization_request: dict[str, object] | None = None) -> None:
        nonlocal state
        previous_model = current_model()
        previous_randomization = getattr(env, "_last_randomization", None)
        previous_metadata = getattr(previous_randomization, "metadata", None)
        if not isinstance(previous_metadata, dict):
            previous_metadata = None
        options = None
        merged_randomization_request = dict(base_randomization_request)
        if randomization_request:
            merged_randomization_request.update(randomization_request)
        if merged_randomization_request:
            options = {"randomization": merged_randomization_request}
        obs, _ = env.reset(
            seed=int(time.time_ns() % (2**31 - 1)),
            options=options,
        )
        state = obs["state"].copy()
        step_count[0] = 0
        gripper_ids[0] = find_gripper_ids(current_model())
        current_randomization = getattr(env, "_last_randomization", None)
        current_metadata = getattr(current_randomization, "metadata", None)
        if not isinstance(current_metadata, dict):
            current_metadata = None
        if _should_export_scene_after_reset(
            previous_model=previous_model,
            current_model=current_model(),
            task=args.task,
            previous_metadata=previous_metadata,
            current_metadata=current_metadata,
        ):
            export_current_scene_meshes()
            queue_client_event("scene_reload")
        else:
            queue_client_event("reset_complete")
        frame_bytes[0] = build_frame_buffer()
        task_dashboard.update(step=0, sim_time=float(current_data().time), result=env.evaluate_task())

    def physics_loop():
        nonlocal state
        dt = 1.0 / args.control_rate

        while not stop_event.is_set():
            t0 = time.time()

            if pending_reset_request[0] is not None:
                with lock:
                    reset_request = pending_reset_request[0]
                    pending_reset_request[0] = None
                    try:
                        reset_scene(reset_request)
                    except Exception as exc:
                        queue_client_event("reset_error", error=str(exc))
                        print(f"Scene reset failed: {exc}")

            if args.mocap:
                # Apply mocap updates from VR controllers
                with lock:
                    model = current_model()
                    data = current_data()
                    n_substeps = max(1, round(dt / model.opt.timestep))
                    while pending_mocap_updates:
                        update = pending_mocap_updates.pop(0)
                        mid = update.get("mocap_id", -1)
                        if 0 <= mid < model.nmocap:
                            p = update["position"]
                            q = update["quaternion"]
                            # Convert Y-up to Z-up
                            pos_zup = R_inv @ np.array([p[0], p[1], p[2]])
                            from scipy.spatial.transform import Rotation as _Rot
                            mat_yup = _Rot.from_quat([q[0], q[1], q[2], q[3]]).as_matrix()
                            mat_zup = R_inv @ mat_yup @ R_inv.T
                            q_zup = _Rot.from_matrix(mat_zup).as_quat()  # xyzw
                            data.mocap_pos[mid] = pos_zup
                            data.mocap_quat[mid] = [q_zup[3], q_zup[0], q_zup[1], q_zup[2]]  # wxyz

                    # Apply trigger -> gripper
                    trig = pending_trigger[0]
                    if trig:
                        pending_trigger[0] = None
                        for side in ["left", "right"]:
                            info = gripper_ids[0].get(side)
                            if info:
                                aid, lo, hi = info
                                val = 1.0 - trig.get(side, 0)
                                data.ctrl[aid] = lo + val * (hi - lo)

                    # Set arm ctrl to follow current qpos so actuators don't fight the welds
                    for i in range(model.nu):
                        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
                        if "gripper" not in aname:
                            jid = model.actuator_trnid[i][0]
                            data.ctrl[i] = data.qpos[model.jnt_qposadr[jid]]

                    # Step physics
                    for _ in range(n_substeps):
                        mujoco.mj_step(model, data)
                    state = env.get_obs()["state"].copy()
                    task_dashboard.update(
                        step=step_count[0] + 1,
                        sim_time=float(data.time),
                        result=env.evaluate_task(),
                    )
                    frame_bytes[0] = build_frame_buffer()
                step_count[0] += 1
            else:
                # GELLO control
                left_pos = poll_leader(left_sub)
                right_pos = poll_leader(right_sub)

                if left_pos is not None or right_pos is not None:
                    if not connected[0]:
                        connected[0] = True
                        print("GELLO connected!")

                    action = state.copy()
                    if left_pos is not None and len(left_pos) >= 7:
                        action[:7] = left_pos[:7]
                    if right_pos is not None and len(right_pos) >= 7:
                        action[7:14] = right_pos[:7]

                    with lock:
                        data = current_data()
                        env._step_single(action)
                        state = env.get_obs()["state"].copy()
                        if recorder is not None:
                            recorder.record_step(action, env, state=state)
                        task_dashboard.update(
                            step=step_count[0] + 1,
                            sim_time=float(data.time),
                            result=env.evaluate_task(),
                        )
                        frame_bytes[0] = build_frame_buffer()
                    step_count[0] += 1

            elapsed = time.time() - t0
            remaining = dt - elapsed
            if remaining > 0:
                stop_event.wait(remaining)

    # Start physics in background thread
    physics_thread = threading.Thread(target=physics_loop, daemon=True)
    physics_thread.start()

    # --- aiohttp web server ---
    app_web = web.Application()
    ws_clients: list[web.WebSocketResponse] = []

    async def handle_index(request):
        config_script = '<script>'
        if args.debug:
            config_script += 'window.__DEBUG__ = true;'
        config_script += f'window.__VR_POS__ = {args.vr_pos};'
        config_script += f'window.__VR_TARGET__ = {args.vr_target};'
        if args.asset_debug:
            config_script += 'window.__ASSET_DEBUG__ = true;'
            config_script += f'window.__ASSET_DEBUG_TASK__ = {json.dumps(args.task)};'
            config_script += (
                f'window.__ASSET_DEBUG_STATE__ = {json.dumps(current_randomization_metadata())};'
            )
        if args.mocap:
            config_script += 'window.__MOCAP__ = true;'
            config_script += f'window.__MOCAP_INIT__ = {json.dumps(build_mocap_init_payload())};'
        config_script += '</script>'
        html = VR_HTML.replace('</head>', config_script + '</head>')
        return web.Response(text=html, content_type="text/html")

    async def handle_bodies(request):
        return web.json_response({
            "scene_version": int(scene_state["version"]),
            "bodies": dict(scene_state["body_info"]),
        })

    async def handle_task_debug(request):
        return web.Response(text=TASK_DASHBOARD_HTML, content_type="text/html")

    async def handle_task_eval(request):
        history_tail = int(request.query.get("history", "256"))
        return web.json_response(task_dashboard.snapshot(history_tail=history_tail))

    async def handle_mesh(request):
        filename = request.match_info["filename"]
        path = mesh_root / filename
        if not path.exists():
            return web.Response(status=404)
        return web.FileResponse(path, headers={
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "no-cache",
            "Content-Type": "model/gltf-binary",
        })

    async def handle_ws(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        ws_clients.append(ws)
        print(f"VR client connected ({len(ws_clients)} total)")

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    import json
                    try:
                        d = json.loads(msg.data)
                        if d.get("type") == "reset":
                            randomization_request = d.get("randomization")
                            if randomization_request is not None and not isinstance(randomization_request, dict):
                                raise ValueError("reset.randomization must be an object")
                            pending_reset_request[0] = dict(randomization_request or {})
                            print(f"Scene reset requested via {d.get('source', 'client')}")
                        elif d.get("type") == "mocap" and args.mocap:
                            pending_mocap_updates.append(d)
                        elif d.get("type") == "trigger" and args.mocap:
                            pending_trigger[0] = d
                    except Exception:
                        pass
        finally:
            ws_clients.remove(ws)
            print(f"VR client disconnected ({len(ws_clients)} remaining)")

        return ws

    async def stream_loop():
        """Broadcast transforms to all connected WebSocket clients."""
        dt = 1.0 / args.stream_rate
        while True:
            events: list[dict[str, object]] = []
            while True:
                try:
                    events.append(pending_client_events.get_nowait())
                except queue.Empty:
                    break
            if ws_clients:
                for event in events:
                    for ws in list(ws_clients):
                        try:
                            await ws.send_str(json.dumps(event))
                        except Exception:
                            pass
                with lock:
                    data_bytes = frame_bytes[0]
                for ws in list(ws_clients):
                    try:
                        await ws.send_bytes(data_bytes)
                    except Exception:
                        pass
            await asyncio.sleep(dt)

    async def on_startup(app):
        asyncio.create_task(stream_loop())

    app_web.router.add_get("/", handle_index)
    app_web.router.add_get("/debug", handle_task_debug)
    app_web.router.add_get("/api/task-eval", handle_task_eval)
    app_web.router.add_get("/api/bodies", handle_bodies)
    app_web.router.add_get("/meshes/{filename}", handle_mesh)
    app_web.router.add_get("/ws", handle_ws)
    app_web.on_startup.append(on_startup)

    print(f"\nVR Streaming Server: http://0.0.0.0:{args.port}")
    print(f"Task debug dashboard: http://0.0.0.0:{args.port}/debug")
    print(f"VR position: {args.vr_pos}, looking at: {args.vr_target}")
    print(f"Open in VR headset browser to connect")
    print(f"Waiting for GELLO leaders on '{args.left_leader}' and '{args.right_leader}'...")

    try:
        web.run_app(app_web, host="0.0.0.0", port=args.port, print=None)
    finally:
        stop_event.set()
        physics_thread.join(timeout=5.0)
        if recorder is not None:
            saved_dir = recorder.close()
            if saved_dir is not None:
                print(f"Saved recording to {saved_dir}")
        left_sub.close()
        right_sub.close()
        zmq_context.term()
        env.close()


if __name__ == "__main__":
    main()
