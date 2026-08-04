"""Runtime scene configuration for MuJoCo YAM simulation.

Applies visual changes (colors, visibility) to the MuJoCo model at runtime
without needing separate XML files. Call apply_scene_variant() after loading
the model but before rendering.

Usage:
    env = yam_sim.make_env(scene="hybrid")

Scene variants:
    eval     — cage walls + single bottle + blue-grey bin
    training — no walls + orange bucket
    hybrid   — cage walls + orange bucket

All variants enforce: white floor, white table, white walls (when visible).
"""

from __future__ import annotations

import mujoco
import numpy as np


# ── Named colors ─────────────────────────────────────────────────────

WHITE = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
OFF_WHITE = np.array([0.95, 0.93, 0.96, 1.0], dtype=np.float32)
BRIGHT_WHITE = np.array([0.97, 0.97, 0.97, 1.0], dtype=np.float32)
LIGHT_GREY = np.array([0.92, 0.92, 0.92, 1.0], dtype=np.float32)
HIDDEN = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

# Bin colors
DARK_BLUE_GREY = np.array([0.15, 0.25, 0.35, 1.0], dtype=np.float32)
ORANGE_BUCKET = np.array([0.95, 0.55, 0.15, 1.0], dtype=np.float32)
DARK_MESH_BIN = np.array([0.20, 0.20, 0.20, 1.0], dtype=np.float32)

# Bottle colors
GREEN_BOTTLE_BODY = np.array([0.30, 0.70, 0.20, 0.50], dtype=np.float32)
GREEN_BOTTLE_NECK = np.array([0.35, 0.75, 0.25, 0.40], dtype=np.float32)
GREEN_BOTTLE_CAP = np.array([0.20, 0.60, 0.15, 0.90], dtype=np.float32)


# ── Geom name groups ─────────────────────────────────────────────────

WALL_GEOMS = ["back_wall", "left_wall", "right_wall"]
BIN_GEOMS = ["bin_bottom"] + [f"bin_wall_{a}" for a in range(0, 360, 30)]
BOTTLE_GEOMS = ["bottle_body", "bottle_neck", "bottle_cap"]


# ── Scene variant definitions ────────────────────────────────────────

VARIANTS = {
    "eval": {
        "description": "Eval-like: cage walls + blue-grey bin + purple-tinted white background",
        "floor_rgba": OFF_WHITE,
        "table_rgba": BRIGHT_WHITE,
        "wall_rgba": OFF_WHITE,
        "walls_visible": True,
        "bin_rgba": DARK_BLUE_GREY,
        "gate_rgba": np.array([0.85, 0.85, 0.85, 1.0]),
    },
    "training": {
        "description": "Training-like: no walls + orange bucket + purple-tinted white background",
        "floor_rgba": OFF_WHITE,
        "table_rgba": BRIGHT_WHITE,
        "wall_rgba": HIDDEN,
        "walls_visible": False,
        "bin_rgba": ORANGE_BUCKET,
        "gate_rgba": np.array([0.85, 0.85, 0.85, 1.0]),
    },
    "hybrid": {
        "description": "Hybrid: cage walls + orange bucket + purple-tinted white background",
        "floor_rgba": OFF_WHITE,
        "table_rgba": BRIGHT_WHITE,
        "wall_rgba": OFF_WHITE,
        "walls_visible": True,
        "bin_rgba": ORANGE_BUCKET,
        "gate_rgba": np.array([0.85, 0.85, 0.85, 1.0]),
    },
}


# ── Core function ────────────────────────────────────────────────────


def apply_scene_variant(model: mujoco.MjModel, variant_name: str = "eval") -> None:
    """Apply a scene variant by modifying MuJoCo model properties at runtime."""
    if variant_name not in VARIANTS:
        available = ", ".join(VARIANTS.keys())
        raise ValueError(f"Unknown variant '{variant_name}'. Choose from: {available}")

    v = VARIANTS[variant_name]
    print(f"Applying scene variant: {variant_name} — {v['description']}")

    def _set_geom_rgba(name: str, rgba: np.ndarray) -> None:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            model.geom_rgba[gid] = rgba

    # Floor
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id >= 0:
        model.geom_matid[floor_id] = -1
        model.geom_rgba[floor_id] = v["floor_rgba"]

    # Table
    table_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "play_table")
    if table_body_id >= 0:
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] == table_body_id:
                model.geom_rgba[gid] = v["table_rgba"]

    # Walls
    for name in WALL_GEOMS:
        _set_geom_rgba(name, v["wall_rgba"])

    # Bin
    for name in BIN_GEOMS:
        _set_geom_rgba(name, v["bin_rgba"])

    # Gate
    gate_mesh_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_MESH, "base_visual_gate"
    )
    if gate_mesh_id >= 0:
        for gid in range(model.ngeom):
            if model.geom_dataid[gid] == gate_mesh_id:
                model.geom_rgba[gid] = v["gate_rgba"]
                break


def apply_table_color(model: mujoco.MjModel, rgba: tuple[float, ...]) -> None:
    """Override play_table geom colors."""
    table_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "play_table")
    if table_body_id >= 0:
        color = np.array(rgba, dtype=np.float32)
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] == table_body_id:
                model.geom_rgba[gid] = color


def apply_wall_color(model: mujoco.MjModel, rgba: tuple[float, ...]) -> None:
    """Override cage wall geom colors."""
    color = np.array(rgba, dtype=np.float32)
    for name in WALL_GEOMS:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            model.geom_rgba[gid] = color


def apply_bottle_rgba(model: mujoco.MjModel, rgba: tuple[float, ...]) -> None:
    """Override all bottle body geom colors."""
    color = np.array(rgba, dtype=np.float32)
    for i in range(1, 7):
        gname = f"b{i}_body"
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        if gid >= 0:
            model.geom_rgba[gid] = color


def apply_bottle_opacity(model: mujoco.MjModel, alpha: float) -> None:
    """Set alpha channel on all bottle body geoms."""
    for i in range(1, 7):
        gname = f"b{i}_body"
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        if gid >= 0:
            model.geom_rgba[gid][3] = alpha


def apply_bin_position(
    model: mujoco.MjModel, data: "mujoco.MjData", x: float, y: float
) -> None:
    """Move the bin body to (x, y) via its freejoint qpos."""
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bin_joint")
    if jnt_id >= 0:
        addr = model.jnt_qposadr[jnt_id]
        data.qpos[addr] = x
        data.qpos[addr + 1] = y


def list_variants() -> list[str]:
    """Return list of available variant names."""
    return list(VARIANTS.keys())
