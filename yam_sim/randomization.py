"""Object pose randomization for yam-sim scenes.

Provides per-task ``SceneRandomizer`` subclasses that perturb free-jointed
objects around their nominal XML positions.  Collision avoidance uses
two-stage rejection sampling:

1. Fast pairwise XY clearance check (no mj_forward needed).
2. Full MuJoCo contact check after applying the proposed placement.

``RandomizationState`` stores *absolute* positions and quaternions (not
deltas), so replay is a direct blind write into qpos — no dependency on
nominal positions or the sampler implementation.
"""

from __future__ import annotations

import copy
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PerturbRange:
    """Translation and orientation perturbation range for one object.

    For free-jointed objects set ``joint_name`` to the joint name and leave
    ``fixed_body=False`` (default).  For fixed bodies that should be moved by
    writing directly to ``model.body_pos`` / ``model.body_quat``, set
    ``fixed_body=True`` and use the body name in ``joint_name``.

    Deltas are relative to the object's nominal position at the time
    ``randomize()`` is called (i.e. after ``mj_resetData``), so they remain
    valid even if the XML default positions change.
    """

    joint_name: str
    delta_x: tuple[float, float]       # (min, max) metres
    delta_y: tuple[float, float]       # (min, max) metres
    delta_z: tuple[float, float] = field(default=(0.0, 0.0))   # stay on table
    delta_roll: tuple[float, float] = field(default=(0.0, 0.0))
    delta_pitch: tuple[float, float] = field(default=(0.0, 0.0))
    delta_yaw: tuple[float, float] = field(default=(-np.pi, np.pi))
    fixed_body: bool = False  # if True, treat joint_name as a body name


@dataclass
class ScalePerturbRange:
    """Uniform multiplicative object-scale perturbation for one movable object."""

    target_name: str
    scale_factor: tuple[float, float] = field(default=(0.95, 1.05))


@dataclass
class RandomizationState:
    """Serialisable randomization outcome.

    Stores absolute positions and quaternions — fully self-contained for
    replay without re-running the sampler.  The ``seed`` field is retained
    for audit / debugging only; replay always uses ``object_states`` directly.
    """

    seed: int
    object_states: dict[str, dict[str, list[float]]]
    scale_states: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    # e.g. {"bottle_1_joint": {"pos": [x,y,z], "quat": [w,x,y,z]}, ...}

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "object_states": self.object_states,
            "scale_states": self.scale_states,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RandomizationState":
        return cls(
            seed=d["seed"],
            object_states=d["object_states"],
            scale_states=d.get("scale_states", {}),
            metadata=d.get("metadata", {}),
        )


@dataclass(frozen=True)
class DishRackResetRequest:
    """Optional reset controls for the dishrack randomizer."""

    plate_variant: str | None = None
    plate_count: int | None = None
    dish_rack_variant: str | None = None
    cycle_plate: int = 0
    cycle_dish_rack: int = 0
    randomize_variants: bool | None = None
    randomize_scales: bool | None = None

    @classmethod
    def from_value(cls, value: Any | None) -> "DishRackResetRequest":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError(
                "DishRack reset request must be a dict or DishRackResetRequest, "
                f"got {type(value).__name__}"
            )

        def _optional_str(key: str) -> str | None:
            raw = value.get(key)
            if raw is None:
                return None
            return str(raw)

        def _int_value(key: str) -> int:
            raw = value.get(key, 0)
            if raw is None:
                return 0
            return int(raw)

        randomize_variants = value.get("randomize_variants")
        if randomize_variants is not None:
            randomize_variants = bool(randomize_variants)

        randomize_scales = value.get("randomize_scales")
        if randomize_scales is not None:
            randomize_scales = bool(randomize_scales)

        return cls(
            plate_variant=_optional_str("plate_variant"),
            plate_count=value.get("plate_count"),
            dish_rack_variant=_optional_str("dish_rack_variant"),
            cycle_plate=_int_value("cycle_plate"),
            cycle_dish_rack=_int_value("cycle_dish_rack"),
            randomize_variants=randomize_variants,
            randomize_scales=randomize_scales,
        )


@dataclass(frozen=True)
class MugResetRequest:
    """Optional reset controls for mug asset randomization tasks."""

    mug_variant: str | None = None
    mug_variants: tuple[str, ...] | None = None
    mug_count: int | None = None
    cycle_mug: int = 0
    randomize_variants: bool | None = None
    randomize_scales: bool | None = None
    # When False, fixed-body fixtures (e.g. the mug_flip tray) are held at their
    # nominal pose instead of being perturbed. Used by batched eval, where the tray
    # is a model-level fixed body frozen to the construction model: freezing it keeps
    # the per-world tray (which mugs are placed relative to) aligned with the rendered
    # construction tray, so the mugs actually start ON the tray.
    randomize_fixed_bodies: bool | None = None
    # Pin the mug_flip tray to an exact pose (x,y,z, qw,qx,qy,qz) instead of sampling.
    # Batched eval sets this to the construction model's tray pose so per-world mugs are
    # placed relative to the SAME tray the renderer shows (mugs-on-tray), while the tray
    # pose itself still varies per visual group (it comes from the construction reset).
    tray_pose: tuple[float, ...] | None = None
    # When True, skip the mesh-variant scene reload + color randomization entirely and
    # randomize POSES only on the scene's inline mugs (mug_1/2/3 with geoms directly on
    # the body, colored mug_N_color). Used by hang_mug eval so the geometry-based reward
    # -- calibrated on these inline mugs -- attributes tree contacts correctly (variant
    # reload nests geoms under child bodies the evaluator misses).
    use_inline_mugs: bool | None = None

    @classmethod
    def from_value(cls, value: Any | None) -> "MugResetRequest":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError(
                "Mug reset request must be a dict or MugResetRequest, "
                f"got {type(value).__name__}"
            )

        raw_variant = value.get("mug_variant")
        raw_variants = value.get("mug_variants")
        if raw_variants is None:
            mug_variants = None
        elif isinstance(raw_variants, str):
            mug_variants = (raw_variants,)
        else:
            mug_variants = tuple(str(variant) for variant in raw_variants)

        raw_count = value.get("mug_count")
        raw_cycle = value.get("cycle_mug", 0)
        randomize_variants = value.get("randomize_variants")
        if randomize_variants is not None:
            randomize_variants = bool(randomize_variants)

        randomize_scales = value.get("randomize_scales")
        if randomize_scales is not None:
            randomize_scales = bool(randomize_scales)

        randomize_fixed_bodies = value.get("randomize_fixed_bodies")
        if randomize_fixed_bodies is not None:
            randomize_fixed_bodies = bool(randomize_fixed_bodies)

        raw_tray_pose = value.get("tray_pose")
        tray_pose = None if raw_tray_pose is None else tuple(float(v) for v in raw_tray_pose)

        use_inline_mugs = value.get("use_inline_mugs")
        if use_inline_mugs is not None:
            use_inline_mugs = bool(use_inline_mugs)

        return cls(
            mug_variant=None if raw_variant is None else str(raw_variant),
            mug_variants=mug_variants,
            mug_count=None if raw_count is None else int(raw_count),
            cycle_mug=0 if raw_cycle is None else int(raw_cycle),
            randomize_variants=randomize_variants,
            randomize_scales=randomize_scales,
            randomize_fixed_bodies=randomize_fixed_bodies,
            tray_pose=tray_pose,
            use_inline_mugs=use_inline_mugs,
        )


@dataclass(frozen=True)
class WaterBottleResetRequest:
    """Optional reset controls for the RoboCasa water-bottle randomizer."""

    bottle_variant: str | None = None
    bottle_variants: tuple[str, ...] | None = None
    bottle_count: int | None = None
    cycle_bottle: int = 0
    randomize_variants: bool | None = None
    randomize_scales: bool | None = None
    # Where bottles may spawn: None/"anywhere" = full table (default);
    # "opposite_bin" = only the table half on the far side (in y) of the
    # sampled bin position.
    bottle_spawn: str | None = None

    @classmethod
    def from_value(cls, value: Any | None) -> "WaterBottleResetRequest":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError(
                "WaterBottle reset request must be a dict or WaterBottleResetRequest, "
                f"got {type(value).__name__}"
            )

        raw_variant = value.get("bottle_variant")
        raw_variants = value.get("bottle_variants")
        if raw_variants is None:
            bottle_variants = None
        elif isinstance(raw_variants, str):
            bottle_variants = (raw_variants,)
        else:
            bottle_variants = tuple(str(variant) for variant in raw_variants)

        randomize_variants = value.get("randomize_variants")
        if randomize_variants is not None:
            randomize_variants = bool(randomize_variants)

        randomize_scales = value.get("randomize_scales")
        if randomize_scales is not None:
            randomize_scales = bool(randomize_scales)

        raw_count = value.get("bottle_count")
        raw_cycle = value.get("cycle_bottle", 0)
        raw_spawn = value.get("bottle_spawn")
        if raw_spawn is not None:
            raw_spawn = str(raw_spawn)
            if raw_spawn not in ("anywhere", "opposite_bin"):
                raise ValueError(
                    f"bottle_spawn must be 'anywhere' or 'opposite_bin', got {raw_spawn!r}"
                )
        return cls(
            bottle_variant=None if raw_variant is None else str(raw_variant),
            bottle_variants=bottle_variants,
            bottle_count=None if raw_count is None else int(raw_count),
            cycle_bottle=0 if raw_cycle is None else int(raw_cycle),
            randomize_variants=randomize_variants,
            randomize_scales=randomize_scales,
            bottle_spawn=raw_spawn,
        )


@dataclass(frozen=True)
class ChessResetRequest:
    """Optional reset controls for the chess setup randomizer."""

    scenario: str | None = None
    target_count: int | None = None
    color_mode: str | None = None
    tin_variant: str | None = None
    cycle_scenario: int = 0
    cycle_color_mode: int = 0
    cycle_tin: int = 0
    randomize_variants: bool | None = None
    randomize_scales: bool | None = None

    @classmethod
    def from_value(cls, value: Any | None) -> "ChessResetRequest":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError(
                "Chess reset request must be a dict or ChessResetRequest, "
                f"got {type(value).__name__}"
            )

        scenario = value.get("scenario")
        color_mode = value.get("color_mode")
        tin_variant = value.get("tin_variant")
        target_count = value.get("target_count")
        raw_cycle_scenario = value.get("cycle_scenario", 0)
        raw_cycle_color_mode = value.get("cycle_color_mode", 0)
        raw_cycle_tin = value.get("cycle_tin", 0)
        randomize_variants = value.get("randomize_variants")
        if randomize_variants is not None:
            randomize_variants = bool(randomize_variants)
        randomize_scales = value.get("randomize_scales")
        if randomize_scales is not None:
            randomize_scales = bool(randomize_scales)

        return cls(
            scenario=None if scenario is None else str(scenario),
            target_count=None if target_count is None else int(target_count),
            color_mode=None if color_mode is None else str(color_mode),
            tin_variant=None if tin_variant is None else str(tin_variant),
            cycle_scenario=0 if raw_cycle_scenario is None else int(raw_cycle_scenario),
            cycle_color_mode=0 if raw_cycle_color_mode is None else int(raw_cycle_color_mode),
            cycle_tin=0 if raw_cycle_tin is None else int(raw_cycle_tin),
            randomize_variants=randomize_variants,
            randomize_scales=randomize_scales,
        )


@dataclass(frozen=True)
class SweepResetRequest:
    """Optional reset controls for the sweep randomizer."""

    trash_count: int | None = None
    randomize_scales: bool | None = None

    @classmethod
    def from_value(cls, value: Any | None) -> "SweepResetRequest":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError(
                "Sweep reset request must be a dict or SweepResetRequest, "
                f"got {type(value).__name__}"
            )
        raw_trash_count = value.get("trash_count")
        randomize_scales = value.get("randomize_scales")
        if randomize_scales is not None:
            randomize_scales = bool(randomize_scales)
        return cls(
            trash_count=None if raw_trash_count is None else int(raw_trash_count),
            randomize_scales=randomize_scales,
        )


class _SweepPlacementFailure(RuntimeError):
    """Internal signal that one sweep placement attempt should be retried."""


class _WaterBottlePlacementFailure(RuntimeError):
    """Internal signal that one water-bottle placement attempt should be retried."""


class _ChessPlacementFailure(RuntimeError):
    """Internal signal that one chess placement attempt should be retried."""


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------


def _quat_from_yaw(yaw: float) -> np.ndarray:
    """wxyz quaternion for a rotation of ``yaw`` radians about world Z."""
    return _quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), yaw)


def _quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """wxyz quaternion for a rotation of ``angle`` radians about ``axis``."""
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12 or abs(angle) < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = axis / norm
    half_angle = angle / 2.0
    sin_half = np.sin(half_angle)
    return np.array(
        [np.cos(half_angle), axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half],
        dtype=np.float64,
    )


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two wxyz quaternions: result = q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _yaw_from_quat(q: np.ndarray) -> float:
    """Return the world-Z yaw for a wxyz quaternion."""
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _sample_orientation_delta(
    perturbation: PerturbRange,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a quaternion delta from roll/pitch/yaw ranges."""
    q_roll = _quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), rng.uniform(*perturbation.delta_roll))
    q_pitch = _quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), rng.uniform(*perturbation.delta_pitch))
    q_yaw = _quat_from_yaw(rng.uniform(*perturbation.delta_yaw))
    return _quat_mul(_quat_mul(q_yaw, q_pitch), q_roll)


def _parse_float_list(value: str) -> list[float]:
    return [float(part) for part in value.split()]


def _format_float_list(values: list[float]) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _scale_numeric_attr(elem: _ET.Element, attr: str, factor: float) -> None:
    value = elem.get(attr)
    if not value:
        return
    elem.set(attr, _format_float_list([part * factor for part in _parse_float_list(value)]))


def _scaled_mesh_attr(scale_value: str | None, factor: float) -> str:
    if not scale_value:
        parts = [1.0, 1.0, 1.0]
    else:
        parts = _parse_float_list(scale_value)
        if len(parts) == 1:
            parts = parts * 3
    return _format_float_list([part * factor for part in parts])


def _safe_scale_suffix(name: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")
    return safe or "scaled"


def _find_body_for_scale_target(root: _ET.Element, target_name: str) -> _ET.Element | None:
    for body in root.iter("body"):
        for child in body:
            if child.tag in {"joint", "freejoint"} and child.get("name") == target_name:
                return body
    for body in root.iter("body"):
        if body.get("name") == target_name:
            return body
    return None


def _clone_scaled_mesh(
    *,
    asset_elem: _ET.Element,
    mesh_assets: dict[str, _ET.Element],
    mesh_name: str,
    factor: float,
    target_name: str,
    cache: dict[str, str],
) -> str:
    if mesh_name in cache:
        return cache[mesh_name]

    mesh_elem = mesh_assets.get(mesh_name)
    if mesh_elem is None:
        return mesh_name

    new_name = f"{mesh_name}__scaled__{_safe_scale_suffix(target_name)}"
    suffix = 1
    while new_name in mesh_assets:
        suffix += 1
        new_name = f"{mesh_name}__scaled__{_safe_scale_suffix(target_name)}_{suffix}"

    cloned = copy.deepcopy(mesh_elem)
    cloned.set("name", new_name)
    cloned.set("scale", _scaled_mesh_attr(mesh_elem.get("scale"), factor))
    asset_elem.append(cloned)
    mesh_assets[new_name] = cloned
    cache[mesh_name] = new_name
    return new_name


def _scale_body_subtree(
    *,
    body: _ET.Element,
    factor: float,
    target_name: str,
    asset_elem: _ET.Element,
    mesh_assets: dict[str, _ET.Element],
) -> None:
    mesh_cache: dict[str, str] = {}
    for elem in body.iter():
        if elem is not body and elem.tag == "body":
            _scale_numeric_attr(elem, "pos", factor)
            continue

        if elem.tag == "geom":
            _scale_numeric_attr(elem, "pos", factor)
            _scale_numeric_attr(elem, "size", factor)
            _scale_numeric_attr(elem, "fromto", factor)
            mesh_name = elem.get("mesh")
            if mesh_name:
                elem.set(
                    "mesh",
                    _clone_scaled_mesh(
                        asset_elem=asset_elem,
                        mesh_assets=mesh_assets,
                        mesh_name=mesh_name,
                        factor=factor,
                        target_name=target_name,
                        cache=mesh_cache,
                    ),
                )
            continue

        if elem.tag == "site":
            _scale_numeric_attr(elem, "pos", factor)
            _scale_numeric_attr(elem, "size", factor)
            continue

        if elem.tag == "inertial":
            _scale_numeric_attr(elem, "pos", factor)


def _apply_object_scales_to_scene_xml(xml: str, scale_states: dict[str, float]) -> str:
    if not scale_states:
        return xml

    root = _ET.fromstring(xml)
    asset_elem = root.find("asset")
    if asset_elem is None:
        return xml
    mesh_assets = {
        mesh.get("name", ""): mesh
        for mesh in asset_elem.findall("mesh")
        if mesh.get("name")
    }

    for target_name, factor in scale_states.items():
        if abs(factor - 1.0) < 1e-9:
            continue
        body = _find_body_for_scale_target(root, target_name)
        if body is None:
            logger.warning("Scale target '%s' not found in scene XML — skipping", target_name)
            continue
        _scale_body_subtree(
            body=body,
            factor=factor,
            target_name=target_name,
            asset_elem=asset_elem,
            mesh_assets=mesh_assets,
        )

    return _ET.tostring(root, encoding="unicode")


def _resolve_scene_xml_paths(xml: str, base_dir: _Path | None) -> str:
    if base_dir is None:
        return xml

    root = _ET.fromstring(xml)
    compiler = root.find("compiler")
    if compiler is not None:
        for attr in ("meshdir", "texturedir"):
            value = compiler.get(attr)
            if value and not _Path(value).is_absolute():
                compiler.set(attr, str((base_dir / value).resolve()))
    return _ET.tostring(root, encoding="unicode")


def _body_subtree_xy_keepout_discs(
    model: Any,
    data: Any,
    root_body_id: int,
) -> list[tuple[np.ndarray, float]]:
    """Return XY geom discs relative to the root body origin.

    Each disc is represented as ``(offset_xy, radius)`` using the geom centre
    relative to the root body plus MuJoCo's bounding-sphere radius. For sweep
    randomization this provides a cheap footprint proxy that is far more
    accurate than a single body-origin clearance.
    """
    if root_body_id < 0:
        return []

    root_xy = np.asarray(data.xpos[root_body_id][:2], dtype=np.float64)
    discs: list[tuple[np.ndarray, float]] = []
    for geom_id in range(model.ngeom):
        geom_body_id = int(model.geom_bodyid[geom_id])
        if int(model.body_rootid[geom_body_id]) != root_body_id:
            continue
        geom_xy = np.asarray(data.geom_xpos[geom_id][:2], dtype=np.float64)
        discs.append((geom_xy - root_xy, float(model.geom_rbound[geom_id])))
    return discs


# ---------------------------------------------------------------------------
# Base randomizer
# ---------------------------------------------------------------------------


class SceneRandomizer:
    """Base class for scene object randomization with two-stage rejection sampling.

    Subclasses only need to define ``perturbations`` (and optionally override
    ``min_clearance_m``, ``max_tries``, or ``table_bounds``).  The base class
    handles reading nominal positions, sampling, collision checking, and state
    serialisation.

    ``table_bounds`` is checked on every sample as Stage 0 (before the cheap
    pairwise distance check).  It is expressed as absolute world XY coordinates:
    ``(x_min, x_max, y_min, y_max)``.  Objects whose absolute position falls
    outside these bounds are immediately rejected, so delta ranges can be set
    generously without risk of objects falling off the edge.

    The default matches the standard sim table:
      centre (0.6, 0), half-extents (0.2975, 0.65) minus a 0.06 m edge margin.
    """

    perturbations: list[PerturbRange] = []
    max_tries: int = 200
    min_clearance_m: float = 0.03  # minimum XY centre-to-centre distance
    size_perturbations: list[ScalePerturbRange] = []
    # Absolute XY workspace limits — overrideable per task if the table differs.
    table_bounds: tuple[float, float, float, float] = (0.36, 0.82, -0.55, 0.55)

    def __init__(self) -> None:
        # Cache fixed-body nominal positions on first read so that subsequent
        # randomizations don't drift (mj_resetData restores data.qpos but not
        # model.body_pos, so we must remember the original XML values ourselves).
        self._fixed_body_nominals: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
        self._env_ref = None
        self._scene_variant = "hybrid"
        self._base_scene_xml_string: str | None = None
        self._base_scene_xml_dir: _Path | None = None
        self._base_scene_xml_transformed = False
        self._scene_xml_transform_options = None
        self._current_scale_states: dict[str, float] = {}

    def clone(self) -> "SceneRandomizer":
        return type(self)()

    def bind_env(self, env: Any, *, scene_variant: str = "hybrid") -> None:
        self._env_ref = env
        self._scene_variant = scene_variant
        self._scene_xml_transform_options = getattr(env, "_scene_xml_transform_options", None)
        if getattr(env, "_scene_xml_string", None):
            self._base_scene_xml_string = env._scene_xml_string
            self._base_scene_xml_dir = getattr(env, "_scene_xml", None)
            if self._base_scene_xml_dir is not None:
                self._base_scene_xml_dir = _Path(self._base_scene_xml_dir).parent
            self._base_scene_xml_transformed = self._scene_xml_transform_options is not None
        else:
            self._base_scene_xml_string = env._scene_xml.read_text()
            self._base_scene_xml_dir = _Path(env._scene_xml).parent
            self._base_scene_xml_transformed = False

    def randomize(
        self,
        model: Any,
        data: Any,
        seed: int | None = None,
        request: Any | None = None,
    ) -> RandomizationState:
        """Sample a collision-free placement, apply it to data, and return state."""
        rng = np.random.default_rng(seed)
        scale_states = self._sample_scale_states(rng)
        self._current_scale_states = dict(scale_states)
        if scale_states:
            self._reload_scene_for_scale_states(scale_states)
            if self._env_ref is not None:
                model = self._env_ref.model
                data = self._env_ref.data

        if not self.perturbations:
            return RandomizationState(seed=seed or 0, object_states={}, scale_states=scale_states)

        self._before_sampling(model, data)

        # Nominal positions come from the current qpos (caller should have
        # called mj_resetData + mj_forward before invoking randomize).
        nominals = self._read_nominals(model, data)

        last_states: dict[str, dict] = {}
        for attempt in range(self.max_tries):
            states = self._sample_once(nominals, rng)
            last_states = states

            # Stage 0: table bounds — free, no geometry queries needed.
            if not self._bounds_ok(states):
                continue

            # Stage 1: fast pairwise XY distance (no mj_forward cost).
            if not self._pairwise_ok(states):
                continue

            # Stage 2: full MuJoCo contact check.
            self._apply_states(model, data, states)
            mujoco.mj_forward(model, data)
            if not self._contacts_ok(model, data):
                continue

            return RandomizationState(
                seed=seed or 0,
                object_states=states,
                scale_states=scale_states,
            )

        logger.warning(
            "%s: no collision-free placement found after %d tries — using last sample",
            type(self).__name__,
            self.max_tries,
        )
        self._apply_states(model, data, last_states)
        mujoco.mj_forward(model, data)
        return RandomizationState(
            seed=seed or 0,
            object_states=last_states,
            scale_states=scale_states,
        )

    def apply(self, model: Any, data: Any, state: RandomizationState) -> None:
        """Restore a previously saved state without sampling (for replay)."""
        self._current_scale_states = dict(state.scale_states)
        if state.scale_states:
            self._reload_scene_for_scale_states(state.scale_states)
            if self._env_ref is not None:
                model = self._env_ref.model
                data = self._env_ref.data
        self._apply_states(model, data, state.object_states)
        mujoco.mj_forward(model, data)

    # -- internals -----------------------------------------------------------

    def _get_size_perturbations(self) -> list[ScalePerturbRange]:
        if self.size_perturbations:
            return self.size_perturbations
        return [
            ScalePerturbRange(p.joint_name)
            for p in self.perturbations
            if not p.fixed_body
        ]

    def _sample_scale_states(self, rng: np.random.Generator) -> dict[str, float]:
        if self._env_ref is None:
            return {}
        return {
            target.target_name: float(rng.uniform(*target.scale_factor))
            for target in self._get_size_perturbations()
        }

    def _before_sampling(self, model: Any, data: Any) -> None:
        """Hook for subclasses that need model/data-derived metadata."""
        return None

    def _scene_xml_for_scale_states(self, scale_states: dict[str, float]) -> str | None:
        if self._base_scene_xml_string is None:
            return None
        xml = self._base_scene_xml_string
        if scale_states:
            xml = _apply_object_scales_to_scene_xml(xml, scale_states)
        return _resolve_scene_xml_paths(xml, self._base_scene_xml_dir)

    def _reload_scene_for_scale_states(self, scale_states: dict[str, float]) -> None:
        if not scale_states:
            return
        if self._env_ref is None:
            logger.warning(
                "%s: scale replay requested without a bound env — skipping scale restoration",
                type(self).__name__,
            )
            return

        xml = self._scene_xml_for_scale_states(scale_states)
        if xml is None:
            return

        from yam_sim.scene_variants import apply_scene_variant

        preserved_arm_state = self._env_ref._get_reset_arm_state()
        self._env_ref.reload_from_xml(xml)
        apply_scene_variant(self._env_ref.model, self._scene_variant)
        mujoco.mj_resetData(self._env_ref.model, self._env_ref.data)
        self._env_ref._set_qpos_from_state(preserved_arm_state)
        mujoco.mj_forward(self._env_ref.model, self._env_ref.data)

    def _read_nominals(
        self, model: Any, data: Any
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Read current pos and quat for each perturbation.

        Free-jointed objects read from ``data.qpos``; fixed bodies read from
        a cache populated on the first call (the XML default values).  We must
        cache because ``_apply_states`` writes to ``model.body_pos`` and
        ``mj_resetData`` does not restore it, which would cause the nominal to
        drift on every reset if we re-read from the model each time.
        """
        # Populate fixed-body cache on first call (before any writes).
        if self._fixed_body_nominals is None:
            self._fixed_body_nominals = {}
            for p in self.perturbations:
                if p.fixed_body:
                    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, p.joint_name)
                    if body_id >= 0:
                        self._fixed_body_nominals[p.joint_name] = (
                            model.body_pos[body_id].copy(),
                            model.body_quat[body_id].copy(),
                        )

        nominals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for p in self.perturbations:
            if p.fixed_body:
                if p.joint_name not in self._fixed_body_nominals:
                    logger.warning("Body '%s' not found in model — skipping", p.joint_name)
                    continue
                pos, quat = self._fixed_body_nominals[p.joint_name]
                nominals[p.joint_name] = (pos.copy(), quat.copy())
            else:
                jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, p.joint_name)
                if jnt_id < 0:
                    logger.warning("Joint '%s' not found in model — skipping", p.joint_name)
                    continue
                adr = int(model.jnt_qposadr[jnt_id])
                nominals[p.joint_name] = (
                    data.qpos[adr: adr + 3].copy(),
                    data.qpos[adr + 3: adr + 7].copy(),
                )
        return nominals

    def _sample_once(
        self,
        nominals: dict[str, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
    ) -> dict[str, dict[str, list[float]]]:
        """Draw one set of absolute object placements from nominal + delta.

        The delta ranges are intersected with ``table_bounds`` before sampling
        so every draw is guaranteed to land on the table.  This avoids the
        exponential failure rate of rejection sampling when large delta ranges
        are combined with a bounded workspace.
        """
        x_min, x_max, y_min, y_max = self.table_bounds
        states: dict[str, dict[str, list[float]]] = {}
        for p in self.perturbations:
            if p.joint_name not in nominals:
                continue
            nom_pos, nom_quat = nominals[p.joint_name]

            # Clamp the effective delta range so the absolute position stays
            # within table_bounds regardless of how large the delta is set.
            eff_dx = (
                max(p.delta_x[0], x_min - nom_pos[0]),
                min(p.delta_x[1], x_max - nom_pos[0]),
            )
            eff_dy = (
                max(p.delta_y[0], y_min - nom_pos[1]),
                min(p.delta_y[1], y_max - nom_pos[1]),
            )
            # If nominal is outside bounds (shouldn't happen), sample at 0.
            if eff_dx[0] > eff_dx[1]:
                eff_dx = (0.0, 0.0)
            if eff_dy[0] > eff_dy[1]:
                eff_dy = (0.0, 0.0)

            # Draw the deltas unconditionally so the RNG stream is identical whether or
            # not fixed bodies are frozen (only the final pose differs).
            delta = np.array([
                rng.uniform(*eff_dx),
                rng.uniform(*eff_dy),
                rng.uniform(*p.delta_z),
            ])
            ori_delta = _sample_orientation_delta(p, rng)
            if p.fixed_body and getattr(self, "_freeze_fixed_bodies", False):
                # Freeze fixtures (e.g. mug_tree) to nominal: a fixed body lives in the
                # shared model.body_pos, so in a batched warp model only one pose can be
                # realized -- freezing keeps every world's fixture consistent with the
                # objects placed relative to it.
                new_pos, new_quat = nom_pos.copy(), nom_quat.copy()
            else:
                new_pos = nom_pos + delta
                new_quat = _quat_mul(ori_delta, nom_quat)

            states[p.joint_name] = {
                "pos": new_pos.tolist(),
                "quat": new_quat.tolist(),
            }
        return states

    def _apply_states(
        self,
        model: Any,
        data: Any,
        states: dict[str, dict[str, list[float]]],
    ) -> None:
        """Write absolute pos/quat for each object.

        Tries joint lookup first (free-jointed objects → ``data.qpos``).
        Falls back to body lookup (fixed bodies → ``model.body_pos/quat``).
        """
        for name, s in states.items():
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jnt_id >= 0:
                adr = int(model.jnt_qposadr[jnt_id])
                data.qpos[adr: adr + 3] = s["pos"]
                data.qpos[adr + 3: adr + 7] = s["quat"]
            else:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                if body_id >= 0:
                    mocap_id = int(model.body_mocapid[body_id])
                    if mocap_id >= 0:
                        # Mocap body (e.g. mug_flip tray): pose lives in per-world
                        # data.mocap_pos, so the batched env can carry a distinct pose
                        # per world. (model.body_pos is model-level/shared across worlds.)
                        data.mocap_pos[mocap_id] = s["pos"]
                        data.mocap_quat[mocap_id] = s["quat"]
                    else:
                        # Static fixed body — write directly into model geometry.
                        model.body_pos[body_id] = s["pos"]
                        model.body_quat[body_id] = s["quat"]

    def _bounds_ok(self, states: dict[str, dict[str, list[float]]]) -> bool:
        """Return False if any object's absolute XY position is outside table_bounds."""
        x_min, x_max, y_min, y_max = self.table_bounds
        for s in states.values():
            x, y = s["pos"][0], s["pos"][1]
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                return False
        return True

    def _pairwise_ok(self, states: dict[str, dict[str, list[float]]]) -> bool:
        """Return False if any two objects are closer than min_clearance_m in XY."""
        if self.min_clearance_m <= 0.0 or len(states) < 2:
            return True
        positions = [np.array(s["pos"][:2]) for s in states.values()]
        n = len(positions)
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(positions[i] - positions[j]) < self.min_clearance_m:
                    return False
        return True

    def _contacts_ok(self, model: Any, data: Any) -> bool:
        """Return False if any two randomized object bodies are in contact."""
        obj_root_ids: set[int] = set()
        for p in self.perturbations:
            if p.fixed_body:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, p.joint_name)
                if body_id >= 0:
                    obj_root_ids.add(int(model.body_rootid[body_id]))
            else:
                jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, p.joint_name)
                if jnt_id >= 0:
                    body_id = int(model.jnt_bodyid[jnt_id])
                    obj_root_ids.add(int(model.body_rootid[body_id]))

        for c in range(data.ncon):
            contact = data.contact[c]
            b1 = int(model.geom_bodyid[contact.geom1])
            b2 = int(model.geom_bodyid[contact.geom2])
            r1 = int(model.body_rootid[b1])
            r2 = int(model.body_rootid[b2])
            if r1 != r2 and r1 in obj_root_ids and r2 in obj_root_ids:
                return False
        return True


# ---------------------------------------------------------------------------
# Color randomization helpers
# ---------------------------------------------------------------------------

# Preset mug color palette (RGBA, alpha=1.0).  Includes the original green so
# the default appearance is part of the distribution.
_MUG_COLOR_PALETTE: list[tuple[float, float, float, float]] = [
    (0.172, 0.780, 0.435, 1.0),  # original green
    (0.850, 0.325, 0.098, 1.0),  # red-orange
    (0.929, 0.694, 0.125, 1.0),  # amber
    (0.494, 0.184, 0.557, 1.0),  # purple
    (0.301, 0.745, 0.933, 1.0),  # sky blue
    (0.635, 0.078, 0.184, 1.0),  # dark red
    (0.047, 0.482, 0.863, 1.0),  # blue
    (0.960, 0.960, 0.960, 1.0),  # near-white
    (0.173, 0.173, 0.173, 1.0),  # near-black
]

_TRAY_COLOR_PALETTE: list[tuple[float, float, float, float]] = [
    (0.000, 0.188, 1.000, 1.0),  # original blue
    (0.050, 0.580, 0.420, 1.0),  # teal
    (0.950, 0.420, 0.120, 1.0),  # orange
    (0.620, 0.220, 0.780, 1.0),  # purple
    (0.930, 0.820, 0.160, 1.0),  # yellow
    (0.780, 0.120, 0.180, 1.0),  # red
    (0.120, 0.140, 0.160, 1.0),  # charcoal
    (0.880, 0.900, 0.880, 1.0),  # light gray
]

_WATER_BOTTLE_BIN_COLOR_PALETTE: list[tuple[float, float, float, float]] = [
    (0.100, 0.100, 0.250, 1.0),  # original navy
    (0.950, 0.420, 0.120, 1.0),  # orange
    (0.080, 0.500, 0.360, 1.0),  # green
    (0.610, 0.200, 0.760, 1.0),  # purple
    (0.820, 0.130, 0.180, 1.0),  # red
    (0.160, 0.170, 0.180, 1.0),  # charcoal
    (0.870, 0.890, 0.860, 1.0),  # light gray
]

# Probability of applying a random color (vs. keeping the XML default) each episode.
_COLOR_RANDOMIZE_PROB = 1.0


def _apply_mat_color(
    model: Any, mat_name: str, rgba: tuple[float, float, float, float]
) -> None:
    """Set a MuJoCo material's RGBA at runtime."""
    mat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_name)
    if mat_id >= 0:
        model.mat_rgba[mat_id] = rgba
    else:
        logger.warning("Material '%s' not found in model — skipping color randomization", mat_name)


# ---------------------------------------------------------------------------
# Per-task randomizers
# ---------------------------------------------------------------------------


class BottlesRandomizer(SceneRandomizer):
    min_clearance_m = 0.08
    perturbations = [
        # Bottles placed near edges so range covers a large area of the table.
        # table_bounds check ensures nothing falls off regardless of delta size.
        PerturbRange("bottle_1_joint", delta_x=(-1.0, 1.0), delta_y=(-1.0, 1.0)),
        PerturbRange("bottle_2_joint", delta_x=(-1.0, 1.0), delta_y=(-1.0, 1.0)),
        PerturbRange("bottle_3_joint", delta_x=(-1.0, 1.0), delta_y=(-1.0, 1.0)),
        PerturbRange("bottle_4_joint", delta_x=(-1.0, 1.0), delta_y=(-1.0, 1.0)),
        # Bin: moderate range to keep it reachable; small yaw (asymmetric shape).
        PerturbRange("bin_joint", delta_x=(-0.05, 0.03), delta_y=(-0.2, 0.2),
                     delta_yaw=(-0.5, 0.5)),
    ]


class WaterBottleRandomizer(SceneRandomizer):
    """Randomizer for the RoboCasa mesh water-bottle scene."""

    min_bottle_count = 2
    max_bottle_count = 6
    # Floor for an EXPLICITLY requested bottle_count (reset options /
    # force_object_count). The seed-driven random draw still uses
    # [min_bottle_count, max_bottle_count], so default behavior is unchanged.
    forced_min_bottle_count = 1
    # "opposite_bin" spawn: bottles keep this much clearance (m) past the table
    # centerline on the far side (in y) of the sampled bin position.
    opposite_bin_y_buffer_m = 0.15
    bottle_scale_factor = (0.90, 1.10)
    bin_scale_factor = (1.00, 1.40)
    table_edge_bounds: tuple[float, float, float, float] = (
        0.3025,
        0.8975,
        -0.65,
        0.65,
    )
    table_margin_m = 0.015
    bottle_margin_m = 0.010
    bin_margin_m = 0.020
    max_tries = 300
    _scene_model_cache_size = 12
    _bottle_sample_tries = 128
    _bin_footprint_half_xy = (0.105, 0.105)
    _bin_body_to_bottom_z = 0.084
    _bin_perturbation = PerturbRange(
        "bin_joint",
        delta_x=(-0.08, 0.08),
        delta_y=(-0.25, 0.25),
        delta_yaw=(-0.75, 0.75),
    )
    _bottle_perturbations = [
        PerturbRange(
            f"bottle_{index}_joint",
            delta_x=(-1.0, 1.0),
            delta_y=(-1.0, 1.0),
            delta_yaw=(-np.pi, np.pi),
        )
        for index in range(1, 7)
    ]
    perturbations = [*_bottle_perturbations[:min_bottle_count], _bin_perturbation]

    def prepare_env(self) -> None:
        self._compiled_scene_model_cache: OrderedDict[tuple[Any, ...], mujoco.MjModel] = OrderedDict()
        self._current_bottle_variants = [_WATER_BOTTLE_DEFAULT_VARIANT] * self.min_bottle_count
        self._current_active_bottle_count = self.min_bottle_count
        self._set_active_bottle_count(self.min_bottle_count)
        self._reload_bottle_variant_scene(self._current_bottle_variants, {})

    def _get_size_perturbations(self) -> list[ScalePerturbRange]:
        active_count = int(getattr(self, "_current_active_bottle_count", self.min_bottle_count))
        return [
            *[
                ScalePerturbRange(perturbation.joint_name, scale_factor=self.bottle_scale_factor)
                for perturbation in self._bottle_perturbations[:active_count]
            ],
            ScalePerturbRange(self._bin_perturbation.joint_name, scale_factor=self.bin_scale_factor),
        ]

    def randomize(
        self,
        model: Any,
        data: Any,
        seed: int | None = None,
        request: Any | None = None,
    ) -> RandomizationState:
        rng = np.random.default_rng(seed)
        reset_request = WaterBottleResetRequest.from_value(request)
        self._current_bottle_spawn = reset_request.bottle_spawn
        bottle_variants = self._resolve_bottle_variants(rng, reset_request)
        self._set_active_bottle_count(len(bottle_variants))
        should_randomize_scales = (
            True if reset_request.randomize_scales is None else bool(reset_request.randomize_scales)
        )
        scale_states = self._sample_scale_states(rng) if should_randomize_scales else {}
        self._current_scale_states = dict(scale_states)
        self._reload_bottle_variant_scene(bottle_variants, scale_states)

        if self._env_ref is not None:
            model = self._env_ref.model
            data = self._env_ref.data

        metadata = {
            "bottle_variant": bottle_variants[0],
            "bottle_variants": list(bottle_variants),
            "bottle_count": len(bottle_variants),
        }
        state = self._randomize_pose_with_rng(
            model=model,
            data=data,
            seed=seed,
            rng=rng,
            scale_states=scale_states,
            metadata=metadata,
        )
        self._sample_and_apply_bin_color(model, state, rng)
        return state

    def apply(self, model: Any, data: Any, state: RandomizationState) -> None:
        bottle_variants = self._bottle_variants_from_state(state)
        self._set_active_bottle_count(len(bottle_variants))
        self._current_scale_states = dict(state.scale_states)
        self._reload_bottle_variant_scene(bottle_variants, state.scale_states)
        if self._env_ref is not None:
            model = self._env_ref.model
            data = self._env_ref.data
        self._apply_states(model, data, state.object_states)
        mujoco.mj_forward(model, data)
        self._apply_bin_color(model, state.metadata.get("bin_color"))

    def _resolve_bottle_variants(
        self,
        rng: np.random.Generator,
        request: WaterBottleResetRequest,
    ) -> list[str]:
        variants = _water_bottle_variant_names()
        active_count = self._resolve_bottle_count(rng, request)
        randomize_variants = request.randomize_variants
        if randomize_variants is None:
            randomize_variants = not any(
                (
                    request.bottle_variant is not None,
                    request.bottle_variants is not None,
                    request.cycle_bottle != 0,
                )
            )

        if request.bottle_variants is not None:
            requested = [_water_bottle_canonical_variant_name(variant) for variant in request.bottle_variants]
            invalid = [variant for variant in requested if variant not in variants]
            if invalid:
                raise ValueError(
                    f"Unknown water bottle variants {invalid}. Available: {', '.join(variants)}"
                )
            if request.bottle_count is not None and int(request.bottle_count) != len(requested):
                raise ValueError(
                    f"bottle_count={request.bottle_count} does not match "
                    f"{len(requested)} bottle_variants"
            )
            self._validate_bottle_count(len(requested))
            return requested

        if request.bottle_variant is not None or request.cycle_bottle != 0:
            if request.bottle_variant is not None:
                explicit_variant = _water_bottle_canonical_variant_name(request.bottle_variant)
                if explicit_variant not in variants:
                    raise ValueError(
                        f"Unknown water bottle variant {explicit_variant!r}. Available: {', '.join(variants)}"
                    )
            else:
                current_variants = list(
                    getattr(
                        self,
                        "_current_bottle_variants",
                        [_WATER_BOTTLE_DEFAULT_VARIANT],
                    )
                )
                current_variant = current_variants[0] if current_variants else _WATER_BOTTLE_DEFAULT_VARIANT
                if current_variant not in variants:
                    current_variant = variants[0]
                explicit_variant = variants[(variants.index(current_variant) + request.cycle_bottle) % len(variants)]
            return [explicit_variant] * active_count

        if not randomize_variants:
            current_variants = list(
                getattr(
                    self,
                    "_current_bottle_variants",
                    [_WATER_BOTTLE_DEFAULT_VARIANT] * active_count,
                )
            )
            if not current_variants:
                current_variants = [_WATER_BOTTLE_DEFAULT_VARIANT]
            while len(current_variants) < active_count:
                current_variants.append(current_variants[-1])
            return current_variants[:active_count]

        return [
            variants[int(rng.integers(0, len(variants)))]
            for _ in range(active_count)
        ]

    def _resolve_bottle_count(
        self,
        rng: np.random.Generator,
        request: WaterBottleResetRequest,
    ) -> int:
        if request.bottle_variants is not None:
            count = len(request.bottle_variants)
        elif request.bottle_count is not None:
            count = int(request.bottle_count)
        elif (
            request.bottle_variant is not None
            or request.cycle_bottle != 0
            or request.randomize_variants is False
        ):
            count = int(getattr(self, "_current_active_bottle_count", self.min_bottle_count))
        else:
            count = int(rng.integers(self.min_bottle_count, self.max_bottle_count + 1))
        self._validate_bottle_count(count)
        return count

    def _validate_bottle_count(self, count: int) -> None:
        if not self.forced_min_bottle_count <= count <= self.max_bottle_count:
            raise ValueError(
                f"{type(self).__name__} bottle_count must be in "
                f"[{self.forced_min_bottle_count}, {self.max_bottle_count}], got {count}"
            )

    def _set_active_bottle_count(self, bottle_count: int) -> None:
        self._validate_bottle_count(bottle_count)
        self._current_active_bottle_count = bottle_count
        self.perturbations = [
            *self._bottle_perturbations[:bottle_count],
            self._bin_perturbation,
        ]

    def _bottle_variants_from_state(self, state: RandomizationState) -> list[str]:
        raw_variants = state.metadata.get("bottle_variants")
        if isinstance(raw_variants, (list, tuple)) and raw_variants:
            variants = [_water_bottle_canonical_variant_name(str(variant)) for variant in raw_variants]
        else:
            count = int(
                state.metadata.get(
                    "bottle_count",
                    sum(1 for name in state.object_states if name.startswith("bottle_")),
                )
            )
            variant = _water_bottle_canonical_variant_name(
                str(state.metadata.get("bottle_variant", _WATER_BOTTLE_DEFAULT_VARIANT))
            )
            variants = [variant] * count
        self._validate_bottle_count(len(variants))
        return variants

    def _sample_and_apply_bin_color(
        self,
        model: Any,
        state: RandomizationState,
        rng: np.random.Generator,
    ) -> None:
        bin_color = list(
            _WATER_BOTTLE_BIN_COLOR_PALETTE[
                int(rng.integers(len(_WATER_BOTTLE_BIN_COLOR_PALETTE)))
            ]
        )
        state.metadata["bin_color"] = bin_color
        self._apply_bin_color(model, bin_color)

    def _apply_bin_color(self, model: Any, raw_rgba: Any) -> None:
        if raw_rgba is None:
            return
        rgba = tuple(float(value) for value in raw_rgba)
        if len(rgba) != 4:
            logger.warning("Invalid put_bottles bin color: %r", raw_rgba)
            return
        _apply_mat_color(model, "water_bottle_garbage_can_mat", rgba)

    def _scene_model_cache_key(
        self,
        bottle_variants: list[str],
        scale_states: dict[str, float],
    ) -> tuple[Any, ...]:
        scale_key = tuple(
            sorted((str(name), round(float(value), 8)) for name, value in scale_states.items())
        )
        return (tuple(bottle_variants), scale_key)

    def _try_reload_cached_scene_model(self, cache_key: tuple[Any, ...]) -> bool:
        if self._env_ref is None:
            return False
        cache = getattr(self, "_compiled_scene_model_cache", None)
        if not cache:
            return False
        cached_model = cache.get(cache_key)
        if cached_model is None:
            return False
        cache.move_to_end(cache_key)
        self._env_ref.reload_from_model(cached_model)
        return True

    def _store_compiled_scene_model(self, cache_key: tuple[Any, ...]) -> None:
        if self._env_ref is None:
            return
        cache = getattr(self, "_compiled_scene_model_cache", None)
        if cache is None:
            self._compiled_scene_model_cache = OrderedDict()
            cache = self._compiled_scene_model_cache
        cache[cache_key] = copy.deepcopy(self._env_ref.model)
        cache.move_to_end(cache_key)
        while len(cache) > self._scene_model_cache_size:
            cache.popitem(last=False)

    def _reload_bottle_variant_scene(
        self,
        bottle_variants: list[str],
        scale_states: dict[str, float],
    ) -> None:
        if self._env_ref is None:
            return

        bottle_variants = [_water_bottle_canonical_variant_name(variant) for variant in bottle_variants]
        preserved_arm_state = self._env_ref._get_reset_arm_state()
        scene_cache_key = self._scene_model_cache_key(bottle_variants, scale_states)
        if not self._try_reload_cached_scene_model(scene_cache_key):
            base_scene_xml = self._base_scene_xml_string
            base_scene_dir = self._base_scene_xml_dir
            base_scene_transformed = self._base_scene_xml_transformed
            if (
                base_scene_xml is None
                or "<!-- TASK_ASSETS_PLACEHOLDER -->" not in base_scene_xml
                or "<!-- TASK_BODY_PLACEHOLDER -->" not in base_scene_xml
            ):
                base_scene_xml = _WATER_BOTTLE_BASE_SCENE_XML.read_text()
                base_scene_dir = _WATER_BOTTLE_BASE_SCENE_XML.parent
                base_scene_transformed = False

            xml = _build_water_bottle_scene_xml(
                bottle_variants=bottle_variants,
                scale_states=scale_states,
                base_scene_xml=base_scene_xml,
                base_scene_dir=base_scene_dir,
            )
            if self._scene_xml_transform_options is not None and not base_scene_transformed:
                from yam_sim.scene_xml import transform_scene_xml

                xml, _ = transform_scene_xml(xml, options=self._scene_xml_transform_options)

            self._env_ref.reload_from_xml(xml)
            self._store_compiled_scene_model(scene_cache_key)

        from yam_sim.scene_variants import apply_scene_variant

        self._current_bottle_variants = list(bottle_variants)
        self._current_active_bottle_count = len(bottle_variants)
        apply_scene_variant(self._env_ref.model, self._scene_variant)
        mujoco.mj_resetData(self._env_ref.model, self._env_ref.data)
        self._env_ref._set_qpos_from_state(preserved_arm_state)
        mujoco.mj_forward(self._env_ref.model, self._env_ref.data)
        self._fixed_body_nominals = None

    def _randomize_pose_with_rng(
        self,
        *,
        model: Any,
        data: Any,
        seed: int | None,
        rng: np.random.Generator,
        scale_states: dict[str, float],
        metadata: dict[str, Any],
    ) -> RandomizationState:
        self._before_sampling(model, data)
        nominals = self._read_nominals(model, data)
        last_states: dict[str, dict[str, list[float]]] = {}
        for _attempt in range(self.max_tries):
            try:
                states = self._sample_once(nominals, rng)
            except _WaterBottlePlacementFailure:
                continue
            last_states = states
            if not self._bounds_ok(states):
                continue
            if not self._pairwise_ok(states):
                continue
            self._apply_states(model, data, states)
            mujoco.mj_forward(model, data)
            if not self._contacts_ok(model, data):
                continue
            return RandomizationState(
                seed=seed or 0,
                object_states=states,
                scale_states=scale_states,
                metadata=dict(metadata),
            )

        logger.warning(
            "%s: no collision-free placement found after %d tries — using last sample",
            type(self).__name__,
            self.max_tries,
        )
        if not last_states:
            last_states = self._sample_once(nominals, rng)
        self._apply_states(model, data, last_states)
        mujoco.mj_forward(model, data)
        return RandomizationState(
            seed=seed or 0,
            object_states=last_states,
            scale_states=scale_states,
            metadata=dict(metadata),
        )

    @staticmethod
    def _rotated_half_extents(half_x: float, half_y: float, yaw: float) -> tuple[float, float]:
        c = abs(float(np.cos(yaw)))
        s = abs(float(np.sin(yaw)))
        return c * half_x + s * half_y, s * half_x + c * half_y

    @staticmethod
    def _obb_axes(yaw: float) -> tuple[np.ndarray, np.ndarray]:
        c = float(np.cos(yaw))
        s = float(np.sin(yaw))
        return (
            np.array([c, s], dtype=np.float64),
            np.array([-s, c], dtype=np.float64),
        )

    @classmethod
    def _obb_overlap(
        cls,
        center_a: np.ndarray,
        half_a: tuple[float, float],
        yaw_a: float,
        center_b: np.ndarray,
        half_b: tuple[float, float],
        yaw_b: float,
        margin: float,
    ) -> bool:
        axis_ax, axis_ay = cls._obb_axes(yaw_a)
        axis_bx, axis_by = cls._obb_axes(yaw_b)
        axes = (axis_ax, axis_ay, axis_bx, axis_by)
        delta = center_b - center_a
        half_a = (half_a[0] + margin, half_a[1] + margin)
        half_b = (half_b[0] + margin, half_b[1] + margin)

        for axis in axes:
            distance = abs(float(np.dot(delta, axis)))
            radius_a = (
                half_a[0] * abs(float(np.dot(axis_ax, axis)))
                + half_a[1] * abs(float(np.dot(axis_ay, axis)))
            )
            radius_b = (
                half_b[0] * abs(float(np.dot(axis_bx, axis)))
                + half_b[1] * abs(float(np.dot(axis_by, axis)))
            )
            if distance > radius_a + radius_b:
                return False
        return True

    def _bottle_footprint(
        self,
        index: int,
        state: dict[str, list[float]],
    ) -> tuple[np.ndarray, tuple[float, float], float]:
        bottle_variants = list(getattr(self, "_current_bottle_variants", []))
        variant_name = (
            bottle_variants[index]
            if index < len(bottle_variants)
            else _WATER_BOTTLE_DEFAULT_VARIANT
        )
        scale_factor = float(
            getattr(self, "_current_scale_states", {}).get(
                self._bottle_perturbations[index].joint_name,
                1.0,
            )
        )
        half_length, half_radius, _vertical_radius = _water_bottle_flat_compiled_metadata(variant_name)
        yaw = _water_bottle_flat_yaw_from_quat(np.asarray(state["quat"], dtype=np.float64))
        return (
            _water_bottle_flat_center_from_pose(
                pos=state["pos"],
                quat=state["quat"],
                variant_name=variant_name,
                scale_factor=scale_factor,
            ),
            (half_length * scale_factor, half_radius * scale_factor),
            yaw,
        )

    def _bin_footprint(
        self,
        state: dict[str, list[float]],
    ) -> tuple[np.ndarray, tuple[float, float], float]:
        scale_factor = float(
            getattr(self, "_current_scale_states", {}).get(
                self._bin_perturbation.joint_name,
                1.0,
            )
        )
        half_x, half_y = self._bin_footprint_half_xy
        return (
            np.asarray(state["pos"][:2], dtype=np.float64),
            (half_x * scale_factor, half_y * scale_factor),
            _yaw_from_quat(np.asarray(state["quat"], dtype=np.float64)),
        )

    def _bounds_ok(self, states: dict[str, dict[str, list[float]]]) -> bool:
        x_min, x_max, y_min, y_max = self.table_edge_bounds
        for index, perturbation in enumerate(self._bottle_perturbations[:self._current_active_bottle_count]):
            state = states.get(perturbation.joint_name)
            if state is None:
                return False
            center, half, yaw = self._bottle_footprint(index, state)
            extent_x, extent_y = self._rotated_half_extents(half[0], half[1], yaw)
            if center[0] - extent_x < x_min + self.table_margin_m:
                return False
            if center[0] + extent_x > x_max - self.table_margin_m:
                return False
            if center[1] - extent_y < y_min + self.table_margin_m:
                return False
            if center[1] + extent_y > y_max - self.table_margin_m:
                return False

        bin_state = states.get(self._bin_perturbation.joint_name)
        if bin_state is None:
            return False
        center, half, yaw = self._bin_footprint(bin_state)
        extent_x, extent_y = self._rotated_half_extents(half[0], half[1], yaw)
        if center[0] - extent_x < x_min + self.table_margin_m:
            return False
        if center[0] + extent_x > x_max - self.table_margin_m:
            return False
        if center[1] - extent_y < y_min + self.table_margin_m:
            return False
        if center[1] + extent_y > y_max - self.table_margin_m:
            return False
        return True

    def _pairwise_ok(self, states: dict[str, dict[str, list[float]]]) -> bool:
        entries: list[tuple[np.ndarray, tuple[float, float], float]] = []
        for index, perturbation in enumerate(self._bottle_perturbations[:self._current_active_bottle_count]):
            state = states.get(perturbation.joint_name)
            if state is None:
                return False
            entries.append(self._bottle_footprint(index, state))

        for i in range(len(entries)):
            center_i, half_i, yaw_i = entries[i]
            for j in range(i + 1, len(entries)):
                center_j, half_j, yaw_j = entries[j]
                if self._obb_overlap(
                    center_i,
                    half_i,
                    yaw_i,
                    center_j,
                    half_j,
                    yaw_j,
                    self.bottle_margin_m,
                ):
                    return False

        bin_state = states.get(self._bin_perturbation.joint_name)
        if bin_state is None:
            return False
        bin_center, bin_half, bin_yaw = self._bin_footprint(bin_state)
        for bottle_center, bottle_half, bottle_yaw in entries:
            if self._obb_overlap(
                bottle_center,
                bottle_half,
                bottle_yaw,
                bin_center,
                bin_half,
                bin_yaw,
                self.bin_margin_m,
            ):
                return False
        return True

    def _contacts_ok(self, model: Any, data: Any) -> bool:
        return super()._contacts_ok(model, data)

    def _sample_once(
        self,
        nominals: dict[str, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
    ) -> dict[str, dict[str, list[float]]]:
        states: dict[str, dict[str, list[float]]] = {}
        entries: list[tuple[np.ndarray, tuple[float, float], float]] = []
        x_min, x_max, y_min, y_max = self.table_edge_bounds

        bin_state = self._sample_bin_state(nominals, rng)
        states[self._bin_perturbation.joint_name] = bin_state
        bin_entry = self._bin_footprint(bin_state)
        opposite_bin = getattr(self, "_current_bottle_spawn", None) == "opposite_bin"
        bin_y = float(bin_state["pos"][1])

        for index, perturbation in enumerate(self._bottle_perturbations[:self._current_active_bottle_count]):
            variant_name = (
                self._current_bottle_variants[index]
                if index < len(self._current_bottle_variants)
                else _WATER_BOTTLE_DEFAULT_VARIANT
            )
            scale_factor = float(self._current_scale_states.get(perturbation.joint_name, 1.0))
            half_length, half_radius, vertical_radius = _water_bottle_flat_compiled_metadata(variant_name)
            half = (half_length * scale_factor, half_radius * scale_factor)
            z = (
                _WATER_BOTTLE_TABLE_Z
                + vertical_radius * scale_factor
                + _WATER_BOTTLE_FLAT_SPAWN_CLEARANCE_M
            )

            for _candidate_attempt in range(self._bottle_sample_tries):
                yaw = float(rng.uniform(*perturbation.delta_yaw))
                extent_x, extent_y = self._rotated_half_extents(half[0], half[1], yaw)
                x_lo = x_min + self.table_margin_m + extent_x
                x_hi = x_max - self.table_margin_m - extent_x
                y_lo = y_min + self.table_margin_m + extent_y
                y_hi = y_max - self.table_margin_m - extent_y
                if opposite_bin:
                    if bin_y >= 0.0:
                        y_hi = min(y_hi, -self.opposite_bin_y_buffer_m - extent_y)
                    else:
                        y_lo = max(y_lo, self.opposite_bin_y_buffer_m + extent_y)
                if x_lo > x_hi or y_lo > y_hi:
                    break

                center = np.array(
                    [
                        float(rng.uniform(x_lo, x_hi)),
                        float(rng.uniform(y_lo, y_hi)),
                    ],
                    dtype=np.float64,
                )
                if any(
                    self._obb_overlap(
                        center,
                        half,
                        yaw,
                        prev_center,
                        prev_half,
                        prev_yaw,
                        self.bottle_margin_m,
                    )
                    for prev_center, prev_half, prev_yaw in entries
                ):
                    continue
                if self._obb_overlap(
                    center,
                    half,
                    yaw,
                    bin_entry[0],
                    bin_entry[1],
                    bin_entry[2],
                    self.bin_margin_m,
                ):
                    continue

                anchor_offset = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64) * half[0]
                pos_xy = center - anchor_offset
                quat = _water_bottle_flat_quat(yaw)
                states[perturbation.joint_name] = {
                    "pos": [float(pos_xy[0]), float(pos_xy[1]), float(z)],
                    "quat": quat.tolist(),
                }
                entries.append((center, half, yaw))
                break
            else:
                raise _WaterBottlePlacementFailure()

            if perturbation.joint_name not in states:
                raise _WaterBottlePlacementFailure()

        return states

    def _sample_bin_state(
        self,
        nominals: dict[str, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
    ) -> dict[str, list[float]]:
        joint_name = self._bin_perturbation.joint_name
        if joint_name not in nominals:
            raise _WaterBottlePlacementFailure()

        nominal_pos, nominal_quat = nominals[joint_name]
        scale_factor = float(self._current_scale_states.get(joint_name, 1.0))
        half_x = self._bin_footprint_half_xy[0] * scale_factor
        half_y = self._bin_footprint_half_xy[1] * scale_factor
        x_min, x_max, y_min, y_max = self.table_edge_bounds
        x_lo = max(
            nominal_pos[0] + self._bin_perturbation.delta_x[0],
            x_min + self.table_margin_m + half_x,
        )
        x_hi = min(
            nominal_pos[0] + self._bin_perturbation.delta_x[1],
            x_max - self.table_margin_m - half_x,
        )
        y_lo = max(
            nominal_pos[1] + self._bin_perturbation.delta_y[0],
            y_min + self.table_margin_m + half_y,
        )
        y_hi = min(
            nominal_pos[1] + self._bin_perturbation.delta_y[1],
            y_max - self.table_margin_m - half_y,
        )
        if x_lo > x_hi or y_lo > y_hi:
            raise _WaterBottlePlacementFailure()

        yaw = float(rng.uniform(*self._bin_perturbation.delta_yaw))
        quat = _quat_mul(_quat_from_yaw(yaw), nominal_quat)
        z = _WATER_BOTTLE_TABLE_Z + self._bin_body_to_bottom_z * scale_factor + 0.001
        return {
            "pos": [
                float(rng.uniform(x_lo, x_hi)),
                float(rng.uniform(y_lo, y_hi)),
                float(z),
            ],
            "quat": quat.tolist(),
        }


class MarkerRandomizer(SceneRandomizer):
    min_clearance_m = 0.03
    perturbations = [
        PerturbRange("marker_joint", delta_x=(-0.10, 0.10), delta_y=(-0.14, 0.14)),
        PerturbRange("cap_joint",    delta_x=(-0.08, 0.08), delta_y=(-0.08, 0.08)),
    ]


class DishRackRandomizer(SceneRandomizer):
    min_clearance_m = 0.1
    rack_plate_margin_m = 0.02
    plate_plate_margin_m = 0.02
    rack_table_margin_m = 0.02
    plate_table_margin_m = 0.02
    # Physical tabletop edges, before the center-position margin in table_bounds.
    rack_table_edge_bounds: tuple[float, float, float, float] = (
        0.3025,
        0.8975,
        -0.65,
        0.65,
    )
    max_plate_count = 4
    _scene_model_cache_size = 16
    _plate_sample_tries = 64
    perturbations = [
        PerturbRange("dishrack",    delta_x=(-0.10, 0.09), delta_y=(-0.15, 0.15), delta_yaw=(-0.25, 0.25)),
        PerturbRange("plate_joint", delta_x=(-0.35, 0.25), delta_y=(-0.35, 0.35)),
    ]

    def prepare_env(self) -> None:
        self._current_scale_states = {}
        self._current_plate_variants = [_DISHRACK_DEFAULT_VARIANTS["plate"]]
        self._current_variant_names = {
            "dish_rack": _DISHRACK_DEFAULT_VARIANTS["dish_rack"],
            "plate": _DISHRACK_DEFAULT_VARIANTS["plate"],
        }
        self._current_plate_collision_radii: list[float] = [0.0]
        self._current_rack_half_extents_xy: tuple[float, float] = (0.0, 0.0)
        self._compiled_scene_model_cache: OrderedDict[tuple[Any, ...], mujoco.MjModel] = OrderedDict()
        self._set_active_plate_count(1)
        self._refresh_collision_metadata()
        self._reload_variant_scene(
            _DISHRACK_DEFAULT_VARIANTS["dish_rack"],
            [_DISHRACK_DEFAULT_VARIANTS["plate"]],
            {},
        )

    def randomize(
        self,
        model: Any,
        data: Any,
        seed: int | None = None,
        request: Any | None = None,
    ) -> RandomizationState:
        rng = np.random.default_rng(seed)
        reset_request = DishRackResetRequest.from_value(request)
        dish_rack_variant, plate_variants = self._resolve_variant_selection(rng, reset_request)
        self._set_active_plate_count(len(plate_variants))
        should_randomize_scales = (
            True if reset_request.randomize_scales is None else bool(reset_request.randomize_scales)
        )
        scale_states = self._sample_scale_states(rng) if should_randomize_scales else {}
        self._current_scale_states = dict(scale_states)
        self._reload_variant_scene(dish_rack_variant, plate_variants, scale_states)

        if self._env_ref is not None:
            model = self._env_ref.model
            data = self._env_ref.data

        nominals = self._read_nominals(model, data)
        last_states: dict[str, dict[str, list[float]]] = {}
        for _ in range(self.max_tries):
            states = self._sample_once(nominals, rng)
            last_states = states
            if not self._bounds_ok(states):
                continue
            if not self._pairwise_ok(states):
                continue
            self._apply_states(model, data, states)
            mujoco.mj_forward(model, data)
            if not self._contacts_ok(model, data):
                continue
            return RandomizationState(
                seed=seed or 0,
                object_states=states,
                scale_states=scale_states,
                metadata={
                    "dish_rack_variant": dish_rack_variant,
                    "plate_variant": plate_variants[0],
                    "plate_variants": list(plate_variants),
                    "plate_count": len(plate_variants),
                },
            )

        logger.warning(
            "%s: no collision-free placement found after %d tries — using last sample",
            type(self).__name__,
            self.max_tries,
        )
        self._apply_states(model, data, last_states)
        mujoco.mj_forward(model, data)
        return RandomizationState(
            seed=seed or 0,
            object_states=last_states,
            scale_states=scale_states,
            metadata={
                "dish_rack_variant": dish_rack_variant,
                "plate_variant": plate_variants[0],
                "plate_variants": list(plate_variants),
                "plate_count": len(plate_variants),
            },
        )

    def apply(self, model: Any, data: Any, state: RandomizationState) -> None:
        dish_rack_variant = _dishrack_canonical_variant_name(
            "dish_rack",
            str(state.metadata.get("dish_rack_variant", _DISHRACK_DEFAULT_VARIANTS["dish_rack"])),
        )
        raw_plate_variants = state.metadata.get("plate_variants")
        if isinstance(raw_plate_variants, (list, tuple)) and raw_plate_variants:
            plate_variants = [_dishrack_canonical_variant_name("plate", str(value)) for value in raw_plate_variants]
        else:
            plate_variants = [
                _dishrack_canonical_variant_name(
                    "plate",
                    str(state.metadata.get("plate_variant", _DISHRACK_DEFAULT_VARIANTS["plate"])),
                )
            ]
        self._current_scale_states = dict(state.scale_states)
        self._current_plate_variants = list(plate_variants)
        self._current_variant_names = {
            "dish_rack": dish_rack_variant,
            "plate": plate_variants[0],
        }
        self._set_active_plate_count(len(plate_variants))
        self._reload_variant_scene(dish_rack_variant, plate_variants, state.scale_states)
        if self._env_ref is not None:
            model = self._env_ref.model
            data = self._env_ref.data
        self._apply_states(model, data, state.object_states)
        mujoco.mj_forward(model, data)

    def _refresh_collision_metadata(self) -> None:
        plate_radii: list[float] = []
        for index, plate_variant in enumerate(
            getattr(self, "_current_plate_variants", [_DISHRACK_DEFAULT_VARIANTS["plate"]])
        ):
            plate_half_x, plate_half_y, _, _, _ = _dishrack_compiled_metadata("plate", plate_variant)
            plate_scale = float(
                self._current_scale_states.get(
                    _dishrack_plate_joint_name(index),
                    self._current_scale_states.get("plate_joint", 1.0),
                )
            )
            plate_radii.append(max(plate_half_x, plate_half_y) * plate_scale)

        rack_variant = getattr(self, "_current_variant_names", {}).get(
            "dish_rack", _DISHRACK_DEFAULT_VARIANTS["dish_rack"]
        )
        rack_half_x, rack_half_y, _, _, _ = _dishrack_compiled_metadata("dish_rack", rack_variant)
        rack_scale = float(self._current_scale_states.get("dishrack", 1.0))

        self._current_plate_collision_radii = plate_radii
        self._current_rack_half_extents_xy = (
            rack_half_x * rack_scale,
            rack_half_y * rack_scale,
        )

    def _scene_model_cache_key(
        self,
        dish_rack_variant: str,
        plate_variants: list[str],
        scale_states: dict[str, float],
    ) -> tuple[Any, ...]:
        scale_key = tuple(
            sorted((str(name), round(float(value), 8)) for name, value in scale_states.items())
        )
        return (dish_rack_variant, tuple(plate_variants), scale_key)

    def _try_reload_cached_scene_model(self, cache_key: tuple[Any, ...]) -> bool:
        if self._env_ref is None:
            return False
        cache = getattr(self, "_compiled_scene_model_cache", None)
        if not cache:
            return False
        cached_model = cache.get(cache_key)
        if cached_model is None:
            return False
        cache.move_to_end(cache_key)
        self._env_ref.reload_from_model(cached_model)
        return True

    def _store_compiled_scene_model(self, cache_key: tuple[Any, ...]) -> None:
        if self._env_ref is None:
            return
        cache = getattr(self, "_compiled_scene_model_cache", None)
        if cache is None:
            self._compiled_scene_model_cache = OrderedDict()
            cache = self._compiled_scene_model_cache
        cache[cache_key] = copy.deepcopy(self._env_ref.model)
        cache.move_to_end(cache_key)
        while len(cache) > self._scene_model_cache_size:
            cache.popitem(last=False)

    def _resolve_variant_selection(
        self,
        rng: np.random.Generator,
        request: DishRackResetRequest,
    ) -> tuple[str, list[str]]:
        randomize_variants = request.randomize_variants
        if randomize_variants is None:
            randomize_variants = not any(
                (
                    request.plate_variant is not None,
                    request.dish_rack_variant is not None,
                    request.cycle_plate != 0,
                    request.cycle_dish_rack != 0,
                )
            )

        plate_count = self._resolve_plate_count(
            rng=rng,
            request=request,
            randomize_variants=randomize_variants,
        )
        dish_rack_variant = self._resolve_variant_name(
            kind="dish_rack",
            explicit_variant=request.dish_rack_variant,
            cycle_step=request.cycle_dish_rack,
            randomize_variants=randomize_variants,
            rng=rng,
        )
        repeated_variant = self._resolve_variant_name(
            kind="plate",
            explicit_variant=request.plate_variant,
            cycle_step=request.cycle_plate,
            randomize_variants=randomize_variants,
            rng=rng,
        )
        if request.plate_variant is not None or request.cycle_plate != 0 or not randomize_variants:
            plate_variants = [repeated_variant] * plate_count
        else:
            plate_variants = [
                _dishrack_sample_variant_name("plate", rng)
                for _ in range(plate_count)
            ]
        return dish_rack_variant, plate_variants

    def _resolve_plate_count(
        self,
        *,
        rng: np.random.Generator,
        request: DishRackResetRequest,
        randomize_variants: bool,
    ) -> int:
        if request.plate_count is not None:
            plate_count = int(request.plate_count)
        elif request.plate_variant is not None or request.cycle_plate != 0 or not randomize_variants:
            plate_count = len(getattr(self, "_current_plate_variants", [_DISHRACK_DEFAULT_VARIANTS["plate"]]))
        else:
            plate_count = int(rng.integers(1, self.max_plate_count + 1))

        if not 1 <= plate_count <= self.max_plate_count:
            raise ValueError(
                f"DishRack plate_count must be in [1, {self.max_plate_count}], got {plate_count}"
            )
        return plate_count

    def _set_active_plate_count(self, plate_count: int) -> None:
        if not 1 <= plate_count <= self.max_plate_count:
            raise ValueError(
                f"DishRack plate_count must be in [1, {self.max_plate_count}], got {plate_count}"
            )
        self.perturbations = [
            PerturbRange(
                "dishrack",
                delta_x=(-0.10, 0.09),
                delta_y=(-0.15, 0.15),
                delta_yaw=(-0.25, 0.25),
            ),
            *[
                PerturbRange(
                    _dishrack_plate_joint_name(index),
                    delta_x=(-0.35, 0.25),
                    delta_y=(-0.35, 0.35),
                )
                for index in range(plate_count)
            ],
        ]

    def _resolve_variant_name(
        self,
        *,
        kind: str,
        explicit_variant: str | None,
        cycle_step: int,
        randomize_variants: bool,
        rng: np.random.Generator,
    ) -> str:
        variants = _dishrack_variant_names(kind)
        current_variant = getattr(self, "_current_variant_names", {}).get(kind, _DISHRACK_DEFAULT_VARIANTS[kind])
        if current_variant not in variants:
            current_variant = variants[0]

        if explicit_variant is not None:
            explicit_variant = _dishrack_canonical_variant_name(kind, explicit_variant)
            if explicit_variant not in variants:
                raise ValueError(
                    f"Unknown {kind} variant {explicit_variant!r}. Available: {', '.join(variants)}"
                )
            return explicit_variant

        if cycle_step:
            current_index = variants.index(current_variant)
            return variants[(current_index + cycle_step) % len(variants)]

        if not randomize_variants:
            return current_variant

        return _dishrack_sample_variant_name(kind, rng)

    def _sample_once(
        self,
        nominals: dict[str, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
    ) -> dict[str, dict[str, list[float]]]:
        x_min, x_max, y_min, y_max = self.table_bounds
        perturbations_by_name = {p.joint_name: p for p in self.perturbations}
        states: dict[str, dict[str, list[float]]] = {}

        def sample_state(p: PerturbRange) -> dict[str, list[float]]:
            nom_pos, nom_quat = nominals[p.joint_name]
            eff_dx = (
                max(p.delta_x[0], x_min - nom_pos[0]),
                min(p.delta_x[1], x_max - nom_pos[0]),
            )
            eff_dy = (
                max(p.delta_y[0], y_min - nom_pos[1]),
                min(p.delta_y[1], y_max - nom_pos[1]),
            )
            if eff_dx[0] > eff_dx[1]:
                eff_dx = (0.0, 0.0)
            if eff_dy[0] > eff_dy[1]:
                eff_dy = (0.0, 0.0)

            new_pos = nom_pos + np.array([
                rng.uniform(*eff_dx),
                rng.uniform(*eff_dy),
                rng.uniform(*p.delta_z),
            ])
            new_quat = _quat_mul(_sample_orientation_delta(p, rng), nom_quat)
            return {
                "pos": new_pos.tolist(),
                "quat": new_quat.tolist(),
            }

        rack_perturb = perturbations_by_name.get("dishrack")
        if rack_perturb is not None and "dishrack" in nominals:
            states["dishrack"] = sample_state(rack_perturb)

        for index, _plate_variant in enumerate(
            getattr(self, "_current_plate_variants", [_DISHRACK_DEFAULT_VARIANTS["plate"]])
        ):
            joint_name = _dishrack_plate_joint_name(index)
            perturb = perturbations_by_name.get(joint_name)
            if perturb is None or joint_name not in nominals:
                continue

            last_candidate: dict[str, list[float]] | None = None
            for _ in range(self._plate_sample_tries):
                candidate = sample_state(perturb)
                trial_states = dict(states)
                trial_states[joint_name] = candidate
                last_candidate = candidate
                if self._pairwise_ok(trial_states):
                    break
            if last_candidate is not None:
                states[joint_name] = last_candidate

        # 50% chance: reflect the scene about the XZ plane (negate all Y coords)
        # and rotate the rack 180 about Z so it faces the opposite direction.
        if rng.integers(2) == 1:
            q_180 = _quat_from_yaw(np.pi)
            for key, s in states.items():
                s["pos"][1] = -s["pos"][1]
                if key == "dishrack":
                    s["quat"] = _quat_mul(q_180, np.array(s["quat"])).tolist()
        return states

    def _bounds_ok(self, states: dict[str, dict[str, list[float]]]) -> bool:
        if not super()._bounds_ok(states):
            return False

        x_min, x_max, y_min, y_max = self.rack_table_edge_bounds
        plate_radii = getattr(self, "_current_plate_collision_radii", [])
        for index, _plate_variant in enumerate(
            getattr(self, "_current_plate_variants", [_DISHRACK_DEFAULT_VARIANTS["plate"]])
        ):
            joint_name = _dishrack_plate_joint_name(index)
            plate_state = states.get(joint_name)
            if plate_state is None:
                continue
            if index < len(plate_radii):
                plate_radius = plate_radii[index]
            else:
                plate_half_x, plate_half_y, _, _, _ = _dishrack_compiled_metadata("plate", _plate_variant)
                plate_scale = float(
                    self._current_scale_states.get(joint_name, self._current_scale_states.get("plate_joint", 1.0))
                )
                plate_radius = max(plate_half_x, plate_half_y) * plate_scale

            margin = plate_radius + self.plate_table_margin_m
            x, y = float(plate_state["pos"][0]), float(plate_state["pos"][1])
            if not (x_min + margin <= x <= x_max - margin and y_min + margin <= y <= y_max - margin):
                return False

        rack_state = states.get("dishrack")
        if rack_state is None:
            return True

        rack_half_x, rack_half_y = getattr(self, "_current_rack_half_extents_xy", (0.0, 0.0))
        rack_yaw = _yaw_from_quat(np.asarray(rack_state["quat"], dtype=np.float64))
        c = abs(float(np.cos(rack_yaw)))
        s = abs(float(np.sin(rack_yaw)))
        world_half_x = c * rack_half_x + s * rack_half_y
        world_half_y = s * rack_half_x + c * rack_half_y

        margin = self.rack_table_margin_m
        x, y = float(rack_state["pos"][0]), float(rack_state["pos"][1])
        return (
            x_min + world_half_x + margin <= x <= x_max - world_half_x - margin
            and y_min + world_half_y + margin <= y <= y_max - world_half_y - margin
        )

    def _pairwise_ok(self, states: dict[str, dict[str, list[float]]]) -> bool:
        rack_state = states.get("dishrack")
        plate_entries: list[tuple[np.ndarray, float]] = []
        plate_radii = getattr(self, "_current_plate_collision_radii", [])
        for index, _plate_variant in enumerate(
            getattr(self, "_current_plate_variants", [_DISHRACK_DEFAULT_VARIANTS["plate"]])
        ):
            joint_name = _dishrack_plate_joint_name(index)
            plate_state = states.get(joint_name)
            if plate_state is None:
                continue
            if index < len(plate_radii):
                plate_radius = plate_radii[index]
            else:
                plate_half_x, plate_half_y, _, _, _ = _dishrack_compiled_metadata("plate", _plate_variant)
                plate_scale = float(
                    self._current_scale_states.get(joint_name, self._current_scale_states.get("plate_joint", 1.0))
                )
                plate_radius = max(plate_half_x, plate_half_y) * plate_scale
            plate_entries.append(
                (
                    np.asarray(plate_state["pos"][:2], dtype=np.float64),
                    plate_radius,
                )
            )

        for i in range(len(plate_entries)):
            pos_i, radius_i = plate_entries[i]
            for j in range(i + 1, len(plate_entries)):
                pos_j, radius_j = plate_entries[j]
                min_dist = radius_i + radius_j + self.plate_plate_margin_m
                if np.linalg.norm(pos_i - pos_j) < min_dist:
                    return False

        if rack_state is None:
            return True

        rack_half_x, rack_half_y = getattr(self, "_current_rack_half_extents_xy", (0.0, 0.0))
        rack_pos_xy = np.asarray(rack_state["pos"][:2], dtype=np.float64)
        rack_yaw = _yaw_from_quat(np.asarray(rack_state["quat"], dtype=np.float64))
        c = float(np.cos(rack_yaw))
        s = float(np.sin(rack_yaw))
        for plate_pos_xy, plate_radius in plate_entries:
            rel_xy = plate_pos_xy - rack_pos_xy
            rack_local_xy = np.array(
                [c * rel_xy[0] + s * rel_xy[1], -s * rel_xy[0] + c * rel_xy[1]],
                dtype=np.float64,
            )
            exclude_x = rack_half_x + plate_radius + self.rack_plate_margin_m
            exclude_y = rack_half_y + plate_radius + self.rack_plate_margin_m
            if abs(rack_local_xy[0]) < exclude_x and abs(rack_local_xy[1]) < exclude_y:
                return False
        return True

    def _reload_variant_scene(
        self,
        dish_rack_variant: str,
        plate_variants: str | list[str] | tuple[str, ...],
        scale_states: dict[str, float],
    ) -> None:
        if self._env_ref is None:
            raise RuntimeError("DishRackRandomizer requires a bound env before scene reload")

        dish_rack_variant = _dishrack_canonical_variant_name("dish_rack", dish_rack_variant)
        preserved_arm_state = self._env_ref._get_reset_arm_state()
        normalized_plate_variants = _dishrack_normalize_plate_variants(plate_variants)
        self._set_active_plate_count(len(normalized_plate_variants))
        scene_cache_key = self._scene_model_cache_key(
            dish_rack_variant,
            normalized_plate_variants,
            scale_states,
        )

        base_scene_xml = self._base_scene_xml_string
        base_scene_dir = self._base_scene_xml_dir
        base_scene_transformed = self._base_scene_xml_transformed
        if not self._try_reload_cached_scene_model(scene_cache_key):
            if (
                base_scene_xml is None
                or "<!-- TASK_ASSETS_PLACEHOLDER -->" not in base_scene_xml
                or "<!-- TASK_BODY_PLACEHOLDER -->" not in base_scene_xml
            ):
                base_scene_xml = _DISHRACK_BASE_SCENE_XML.read_text()
                base_scene_dir = _DISHRACK_BASE_SCENE_XML.parent
                base_scene_transformed = False

            xml = _build_dishrack_scene_xml(
                dish_rack_variant=dish_rack_variant,
                plate_variants=normalized_plate_variants,
                scale_states=scale_states,
                base_scene_xml=base_scene_xml,
                base_scene_dir=base_scene_dir,
            )
            if self._scene_xml_transform_options is not None and not base_scene_transformed:
                from yam_sim.scene_xml import transform_scene_xml

                xml, _ = transform_scene_xml(xml, options=self._scene_xml_transform_options)

            self._env_ref.reload_from_xml(xml)
            self._store_compiled_scene_model(scene_cache_key)

        from yam_sim.scene_variants import apply_scene_variant

        self._current_plate_variants = list(normalized_plate_variants)
        self._current_variant_names = {
            "dish_rack": dish_rack_variant,
            "plate": normalized_plate_variants[0],
        }
        self._refresh_collision_metadata()
        apply_scene_variant(self._env_ref.model, self._scene_variant)
        mujoco.mj_resetData(self._env_ref.model, self._env_ref.data)
        self._env_ref._set_qpos_from_state(preserved_arm_state)
        mujoco.mj_forward(self._env_ref.model, self._env_ref.data)
        self._fixed_body_nominals = None


class BlocksRandomizer(SceneRandomizer):
    """Per-block perturbation for the 26-letter blocks scene.

    Blocks are arranged in a 4-row grid with 55 mm spacing; perturbations
    scatter blocks across the top half of the table (±80 mm x/y, ±0.25 rad yaw).
    """
    min_clearance_m = 0.01
    perturbations = [
        PerturbRange(f"block_{letter}_jnt",
                     delta_x=(-0.08, 0.08),
                     delta_y=(-0.45, 0.45),
                     delta_yaw=(-0.25, 0.25))
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ]


class MugVariantRandomizer(SceneRandomizer):
    """Base randomizer for mug tasks that swap mug mesh variants on reset."""

    mug_task_name: str = ""
    mug_body_names: tuple[str, ...] = ("mug_1", "mug_2")
    min_mug_count = 2
    max_mug_count = 2
    randomize_mug_count = False
    independent_mug_variants = False
    mug_scale_factor = (0.90, 1.10)
    _scene_model_cache_size = 16

    def prepare_env(self) -> None:
        self._current_mug_variants = [_MUG_DEFAULT_VARIANT] * self.max_mug_count
        self._current_active_mug_count = self.max_mug_count
        self._compiled_scene_model_cache: OrderedDict[tuple[Any, ...], mujoco.MjModel] = OrderedDict()
        self._set_active_mug_count(self._current_active_mug_count)

    def _get_size_perturbations(self) -> list[ScalePerturbRange]:
        active_count = int(getattr(self, "_current_active_mug_count", self.max_mug_count))
        return [
            ScalePerturbRange(f"{body_name}_jnt", scale_factor=self.mug_scale_factor)
            for body_name in self.mug_body_names[:active_count]
        ]

    def randomize(
        self,
        model: Any,
        data: Any,
        seed: int | None = None,
        request: Any | None = None,
    ) -> RandomizationState:
        rng = np.random.default_rng(seed)
        reset_request = MugResetRequest.from_value(request)
        mug_variants = self._resolve_mug_variants(rng, reset_request)
        self._set_active_mug_count(len(mug_variants))
        should_randomize_scales = (
            True if reset_request.randomize_scales is None else bool(reset_request.randomize_scales)
        )
        scale_states = self._sample_scale_states(rng) if should_randomize_scales else {}
        self._current_scale_states = dict(scale_states)
        self._reload_mug_variant_scene(mug_variants, scale_states)

        if self._env_ref is not None:
            model = self._env_ref.model
            data = self._env_ref.data

        metadata = {
            "mug_variant": mug_variants[0],
            "mug_variants": list(mug_variants),
            "mug_count": len(mug_variants),
        }

        state = self._randomize_pose_with_rng(
            model=model,
            data=data,
            seed=seed,
            rng=rng,
            scale_states=scale_states,
            metadata=metadata,
        )
        mug_colors = self._sample_plain_mug_colors(rng, mug_variants)
        if mug_colors:
            state.metadata["mug_colors"] = mug_colors
            self._apply_plain_mug_colors(model, mug_colors, mug_variants)
        return state

    def apply(self, model: Any, data: Any, state: RandomizationState) -> None:
        if state.metadata.get("use_inline_mugs"):
            # Inline state: no variant reload / color -- just re-apply the inline mug poses.
            self._set_active_mug_count(int(state.metadata.get("mug_count", self.max_mug_count)))
            if self._env_ref is not None:
                model = self._env_ref.model
                data = self._env_ref.data
            self._apply_states(model, data, state.object_states)
            mujoco.mj_forward(model, data)
            return
        mug_variants = self._mug_variants_from_state(state)
        self._set_active_mug_count(len(mug_variants))
        self._current_scale_states = dict(state.scale_states)
        self._reload_mug_variant_scene(mug_variants, state.scale_states)
        if self._env_ref is not None:
            model = self._env_ref.model
            data = self._env_ref.data
        self._apply_states(model, data, state.object_states)
        raw_colors = state.metadata.get("mug_colors")
        if isinstance(raw_colors, dict):
            self._apply_plain_mug_colors(model, raw_colors, mug_variants)
        mujoco.mj_forward(model, data)

    def _randomize_pose_with_rng(
        self,
        *,
        model: Any,
        data: Any,
        seed: int | None,
        rng: np.random.Generator,
        scale_states: dict[str, float],
        metadata: dict[str, Any],
    ) -> RandomizationState:
        if not self.perturbations:
            return RandomizationState(
                seed=seed or 0,
                object_states={},
                scale_states=scale_states,
                metadata=dict(metadata),
            )

        self._before_sampling(model, data)
        nominals = self._read_nominals(model, data)
        last_states: dict[str, dict[str, list[float]]] = {}
        for _attempt in range(self.max_tries):
            states = self._sample_once(nominals, rng)
            last_states = states
            if not self._bounds_ok(states):
                continue
            if not self._pairwise_ok(states):
                continue
            self._apply_states(model, data, states)
            mujoco.mj_forward(model, data)
            if not self._contacts_ok(model, data):
                continue
            return RandomizationState(
                seed=seed or 0,
                object_states=states,
                scale_states=scale_states,
                metadata=dict(metadata),
            )

        logger.warning(
            "%s: no collision-free placement found after %d tries — using last sample",
            type(self).__name__,
            self.max_tries,
        )
        self._apply_states(model, data, last_states)
        mujoco.mj_forward(model, data)
        return RandomizationState(
            seed=seed or 0,
            object_states=last_states,
            scale_states=scale_states,
            metadata=dict(metadata),
        )

    def _resolve_mug_variants(
        self,
        rng: np.random.Generator,
        request: MugResetRequest,
    ) -> list[str]:
        if self._env_ref is None:
            return [_MUG_DEFAULT_VARIANT] * self.max_mug_count

        variants = _mug_variant_names(self.mug_task_name)
        randomize_variants = request.randomize_variants
        if randomize_variants is None:
            randomize_variants = not any(
                (
                    request.mug_variant is not None,
                    request.mug_variants is not None,
                    request.cycle_mug != 0,
                )
            )

        active_count = self._resolve_mug_count(
            rng=rng,
            request=request,
            randomize_variants=randomize_variants,
        )

        if request.mug_variants is not None:
            requested = [_mug_canonical_variant_name(variant) for variant in request.mug_variants]
            invalid = [variant for variant in requested if variant not in variants]
            if invalid:
                raise ValueError(
                    f"Unknown mug variants {invalid}. Available: {', '.join(variants)}"
                )
            if request.mug_count is not None and int(request.mug_count) != len(requested):
                raise ValueError(
                    f"mug_count={request.mug_count} does not match {len(requested)} mug_variants"
                )
            self._validate_mug_count(len(requested))
            return requested

        current_variants = list(
            getattr(self, "_current_mug_variants", [_MUG_DEFAULT_VARIANT] * self.max_mug_count)
        )
        if not current_variants:
            current_variants = [_MUG_DEFAULT_VARIANT] * self.max_mug_count
        current_variant = current_variants[0]
        if current_variant not in variants:
            current_variant = variants[0]

        if request.mug_variant is not None:
            explicit_variant = _mug_canonical_variant_name(request.mug_variant)
            if explicit_variant not in variants:
                raise ValueError(
                    f"Unknown mug variant {explicit_variant!r}. Available: {', '.join(variants)}"
                )
            return [explicit_variant] * active_count

        if request.cycle_mug:
            current_index = variants.index(current_variant)
            return [variants[(current_index + request.cycle_mug) % len(variants)]] * active_count

        if not randomize_variants:
            if len(current_variants) >= active_count:
                return current_variants[:active_count]
            return current_variants + [current_variants[-1]] * (active_count - len(current_variants))

        if self.independent_mug_variants:
            return [_mug_sample_variant_name(self.mug_task_name, rng) for _ in range(active_count)]
        return [_mug_sample_variant_name(self.mug_task_name, rng)] * active_count

    def _resolve_mug_count(
        self,
        *,
        rng: np.random.Generator,
        request: MugResetRequest,
        randomize_variants: bool,
    ) -> int:
        if request.mug_variants is not None:
            return len(request.mug_variants)
        if request.mug_count is not None:
            count = int(request.mug_count)
        elif not self.randomize_mug_count:
            count = int(getattr(self, "_current_active_mug_count", self.max_mug_count))
        elif request.mug_variant is not None or request.cycle_mug != 0 or not randomize_variants:
            count = int(getattr(self, "_current_active_mug_count", self.max_mug_count))
        else:
            count = int(rng.integers(self.min_mug_count, self.max_mug_count + 1))
        self._validate_mug_count(count)
        return count

    def _validate_mug_count(self, count: int) -> None:
        if not self.min_mug_count <= count <= self.max_mug_count:
            raise ValueError(
                f"{type(self).__name__} mug_count must be in "
                f"[{self.min_mug_count}, {self.max_mug_count}], got {count}"
            )

    def _set_active_mug_count(self, mug_count: int) -> None:
        self._validate_mug_count(mug_count)
        self._current_active_mug_count = mug_count

    def _mug_variants_from_state(self, state: RandomizationState) -> list[str]:
        raw_variants = state.metadata.get("mug_variants")
        if isinstance(raw_variants, (list, tuple)) and raw_variants:
            variants = [_mug_canonical_variant_name(str(variant)) for variant in raw_variants]
        else:
            inferred_count = int(
                state.metadata.get(
                    "mug_count",
                    self._infer_mug_count_from_object_states(state.object_states),
                )
            )
            variant = _mug_canonical_variant_name(
                str(state.metadata.get("mug_variant", _MUG_DEFAULT_VARIANT))
            )
            variants = [variant] * inferred_count
        self._validate_mug_count(len(variants))
        return variants

    def _infer_mug_count_from_object_states(
        self,
        object_states: dict[str, dict[str, list[float]]],
    ) -> int:
        count = 0
        for body_name in self.mug_body_names:
            joint_name = f"{body_name}_jnt"
            if joint_name in object_states:
                count += 1
        return count or int(getattr(self, "_current_active_mug_count", self.max_mug_count))

    def _scene_model_cache_key(
        self,
        mug_variants: list[str],
        scale_states: dict[str, float],
    ) -> tuple[Any, ...]:
        scale_key = tuple(
            sorted((str(name), round(float(value), 8)) for name, value in scale_states.items())
        )
        return (tuple(mug_variants), scale_key)

    def _try_reload_cached_scene_model(self, cache_key: tuple[Any, ...]) -> bool:
        if self._env_ref is None:
            return False
        cache = getattr(self, "_compiled_scene_model_cache", None)
        if not cache:
            return False
        cached_model = cache.get(cache_key)
        if cached_model is None:
            return False
        cache.move_to_end(cache_key)
        self._env_ref.reload_from_model(cached_model)
        return True

    def _store_compiled_scene_model(self, cache_key: tuple[Any, ...]) -> None:
        if self._env_ref is None:
            return
        cache = getattr(self, "_compiled_scene_model_cache", None)
        if cache is None:
            self._compiled_scene_model_cache = OrderedDict()
            cache = self._compiled_scene_model_cache
        cache[cache_key] = copy.deepcopy(self._env_ref.model)
        cache.move_to_end(cache_key)
        while len(cache) > self._scene_model_cache_size:
            cache.popitem(last=False)

    def _reload_mug_variant_scene(
        self,
        mug_variants: list[str],
        scale_states: dict[str, float],
    ) -> None:
        if self._env_ref is None:
            return

        mug_variants = [_mug_canonical_variant_name(variant) for variant in mug_variants]
        preserved_arm_state = self._env_ref._get_reset_arm_state()
        scene_cache_key = self._scene_model_cache_key(mug_variants, scale_states)
        if not self._try_reload_cached_scene_model(scene_cache_key):
            base_scene_xml = self._base_scene_xml_string
            base_scene_dir = self._base_scene_xml_dir
            if base_scene_xml is None:
                base_scene_path = _MUG_BASE_SCENE_XMLS[self.mug_task_name]
                base_scene_xml = base_scene_path.read_text()
                base_scene_dir = base_scene_path.parent

            xml = _build_mug_scene_xml(
                task_name=self.mug_task_name,
                mug_variant=mug_variants[0],
                mug_variants=mug_variants,
                scale_states=scale_states,
                base_scene_xml=base_scene_xml,
                base_scene_dir=base_scene_dir,
            )
            if self._scene_xml_transform_options is not None and not self._base_scene_xml_transformed:
                from yam_sim.scene_xml import transform_scene_xml

                xml, _ = transform_scene_xml(xml, options=self._scene_xml_transform_options)

            self._env_ref.reload_from_xml(xml)
            self._store_compiled_scene_model(scene_cache_key)

        from yam_sim.scene_variants import apply_scene_variant

        self._current_mug_variants = list(mug_variants)
        self._current_active_mug_count = len(mug_variants)
        apply_scene_variant(self._env_ref.model, self._scene_variant)
        mujoco.mj_resetData(self._env_ref.model, self._env_ref.data)
        self._env_ref._set_qpos_from_state(preserved_arm_state)
        mujoco.mj_forward(self._env_ref.model, self._env_ref.data)
        self._fixed_body_nominals = None

    def _sample_plain_mug_colors(
        self,
        rng: np.random.Generator,
        mug_variants: list[str],
    ) -> dict[str, list[float]]:
        if rng.random() >= _COLOR_RANDOMIZE_PROB:
            return {}
        return {
            body_name: list(_MUG_COLOR_PALETTE[int(rng.integers(len(_MUG_COLOR_PALETTE)))])
            for body_name, mug_variant in zip(self.mug_body_names, mug_variants)
            if mug_variant == _MUG_DEFAULT_VARIANT
        }

    def _apply_plain_mug_colors(
        self,
        model: Any,
        mug_colors: dict[str, Any],
        mug_variants: list[str],
    ) -> None:
        for instance_index, (body_name, mug_variant) in enumerate(zip(self.mug_body_names, mug_variants)):
            if mug_variant != _MUG_DEFAULT_VARIANT:
                continue
            raw_rgba = mug_colors.get(body_name)
            if raw_rgba is None:
                continue
            rgba = tuple(float(value) for value in raw_rgba)
            if len(rgba) != 4:
                logger.warning("Invalid mug color for %s: %r", body_name, raw_rgba)
                continue
            candidate_names = (
                *_mug_plain_color_material_names(self.mug_task_name, instance_index),
                f"{body_name}_color",
            )
            applied = False
            for mat_name in candidate_names:
                mat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_name)
                if mat_id >= 0:
                    model.mat_rgba[mat_id] = rgba
                    applied = True
            if not applied:
                logger.warning("No colorable material found for %s", body_name)


class MugTreeRandomizer(MugVariantRandomizer):
    mug_task_name = "mug_tree"
    mug_body_names = ("mug_1", "mug_2", "mug_3")
    min_mug_count = 1
    max_mug_count = 3
    randomize_mug_count = True
    independent_mug_variants = True
    min_clearance_m = 0.12
    _tree_perturbation = PerturbRange(
        "mug_tree",
        delta_x=(-0.10, 0.10),
        delta_y=(-0.20, 0.20),
        delta_yaw=(-0.25, 0.25),
        fixed_body=True,
    )
    _mug_perturbations = [
        PerturbRange("mug_1_jnt", delta_x=(-0.10, 0.10), delta_y=(-0.3, 0.3)),
        PerturbRange("mug_2_jnt", delta_x=(-0.10, 0.10), delta_y=(-0.3, 0.3)),
        PerturbRange("mug_3_jnt", delta_x=(-0.10, 0.10), delta_y=(-0.3, 0.3)),
    ]
    perturbations = [
        _tree_perturbation,
        *_mug_perturbations,
    ]

    def randomize(
        self,
        model: Any,
        data: Any,
        seed: int | None = None,
        request: Any | None = None,
    ) -> RandomizationState:
        rng = np.random.default_rng(seed)
        reset_request = MugResetRequest.from_value(request)
        # Freeze fixed-body fixtures (e.g. the mug_flip tray) to nominal when requested
        # (batched eval), so the per-world fixture that objects are placed relative to
        # matches the rendered/warp fixture (the frozen construction model). Default:
        # perturb as usual. Only mug_flip's _sample_once consumes this.
        self._freeze_fixed_bodies = reset_request.randomize_fixed_bodies is False
        self._pinned_tray_pose = reset_request.tray_pose

        if reset_request.use_inline_mugs:
            # Inline mode: keep the scene's inline mugs (geoms directly on mug_1/2/3,
            # colored mug_N_color) -- skip the variant scene reload AND color
            # randomization, randomize POSES only. The geometry-based reward is
            # calibrated on these inline mugs; a variant reload nests geoms under child
            # bodies the evaluator can't attribute tree contacts to.
            count = (
                int(reset_request.mug_count)
                if reset_request.mug_count
                else self.max_mug_count
            )
            self._set_active_mug_count(count)
            self._freeze_fixed_bodies = True  # keep the fixture consistent across the batch
            if self._env_ref is not None:
                model = self._env_ref.model
                data = self._env_ref.data
            return self._randomize_pose_with_rng(
                model=model,
                data=data,
                seed=seed,
                rng=rng,
                scale_states={},
                metadata={
                    "mug_variant": _MUG_DEFAULT_VARIANT,
                    "mug_variants": [_MUG_DEFAULT_VARIANT] * count,
                    "mug_count": count,
                    "use_inline_mugs": True,
                },
            )

        requested_mug_variants = self._resolve_mug_variants(rng, reset_request)
        requested_count = len(requested_mug_variants)
        should_randomize_scales = (
            True if reset_request.randomize_scales is None else bool(reset_request.randomize_scales)
        )

        last_scale_states: dict[str, float] = {}
        for candidate_count in range(requested_count, self.min_mug_count - 1, -1):
            mug_variants = requested_mug_variants[:candidate_count]
            self._set_active_mug_count(candidate_count)
            scale_states = self._sample_scale_states(rng) if should_randomize_scales else {}
            last_scale_states = scale_states
            self._current_scale_states = dict(scale_states)
            self._reload_mug_variant_scene(mug_variants, scale_states)

            if self._env_ref is not None:
                model = self._env_ref.model
                data = self._env_ref.data

            metadata: dict[str, Any] = {
                "mug_variant": mug_variants[0],
                "mug_variants": list(mug_variants),
                "mug_count": len(mug_variants),
            }
            if candidate_count != requested_count:
                metadata["requested_mug_count"] = requested_count
                metadata["mug_count_reduced"] = True

            state = self._try_randomize_pose_with_rng(
                model=model,
                data=data,
                seed=seed,
                rng=rng,
                scale_states=scale_states,
                metadata=metadata,
            )
            if state is None:
                continue

            mug_colors = self._sample_plain_mug_colors(rng, mug_variants)
            if mug_colors:
                state.metadata["mug_colors"] = mug_colors
                self._apply_plain_mug_colors(model, mug_colors, mug_variants)
            return state

        mug_variants = requested_mug_variants[: self.min_mug_count]
        self._set_active_mug_count(len(mug_variants))
        self._current_scale_states = dict(last_scale_states)
        self._reload_mug_variant_scene(mug_variants, last_scale_states)
        if self._env_ref is not None:
            model = self._env_ref.model
            data = self._env_ref.data
        logger.warning(
            "%s: no collision-free mug placement found after reducing to %d mug; using a single-mug sample",
            type(self).__name__,
            self.min_mug_count,
        )
        state = self._randomize_pose_with_rng(
            model=model,
            data=data,
            seed=seed,
            rng=rng,
            scale_states=last_scale_states,
            metadata={
                "mug_variant": mug_variants[0],
                "mug_variants": list(mug_variants),
                "mug_count": len(mug_variants),
                "requested_mug_count": requested_count,
                "mug_count_reduced": True,
            },
        )
        mug_colors = self._sample_plain_mug_colors(rng, mug_variants)
        if mug_colors:
            state.metadata["mug_colors"] = mug_colors
            self._apply_plain_mug_colors(model, mug_colors, mug_variants)
        return state

    def _set_active_mug_count(self, mug_count: int) -> None:
        self._validate_mug_count(mug_count)
        self._current_active_mug_count = mug_count
        self.perturbations = [
            self._tree_perturbation,
            *self._mug_perturbations[:mug_count],
        ]

    def _try_randomize_pose_with_rng(
        self,
        *,
        model: Any,
        data: Any,
        seed: int | None,
        rng: np.random.Generator,
        scale_states: dict[str, float],
        metadata: dict[str, Any],
    ) -> RandomizationState | None:
        self._before_sampling(model, data)
        nominals = self._read_nominals(model, data)
        for _attempt in range(self.max_tries):
            states = self._sample_once(nominals, rng)
            if not self._bounds_ok(states):
                continue
            if not self._pairwise_ok(states):
                continue
            self._apply_states(model, data, states)
            mujoco.mj_forward(model, data)
            if not self._contacts_ok(model, data):
                continue
            return RandomizationState(
                seed=seed or 0,
                object_states=states,
                scale_states=scale_states,
                metadata=dict(metadata),
            )
        return None


class MugFlipRandomizer(MugVariantRandomizer):
    mug_task_name = "mug_flip"
    mug_body_names = ("mug_1", "mug_2", "mug_3", "mug_4")
    min_mug_count = 1
    max_mug_count = 4
    randomize_mug_count = True
    independent_mug_variants = True
    # Mugs start upside-down (quat=[0,1,0,0]); yaw rotation is still world-Z.
    # Tray is a fixed body; mugs are sampled relative to the tray's new position
    # so they stay on the tray after randomization.
    min_clearance_m = 0.03
    _tray_perturbation = PerturbRange("tray", delta_x=(-0.10, 0.05), delta_y=(-0.3, 0.3),
                                      delta_yaw=(-0.25, 0.25), fixed_body=True)
    _mug_slot_perturbations = [
        # Mug deltas are tray-local jitter around count-specific slots.
        PerturbRange("mug_1_jnt", delta_x=(-0.012, 0.012), delta_y=(-0.012, 0.012)),
        PerturbRange("mug_2_jnt", delta_x=(-0.012, 0.012), delta_y=(-0.012, 0.012)),
        PerturbRange("mug_3_jnt", delta_x=(-0.012, 0.012), delta_y=(-0.012, 0.012)),
        PerturbRange("mug_4_jnt", delta_x=(-0.012, 0.012), delta_y=(-0.012, 0.012)),
    ]
    _mug_slot_centers_by_count: dict[int, tuple[tuple[float, float], ...]] = {
        1: ((0.0, 0.0),),
        2: ((-0.064, 0.044), (0.064, -0.044)),
        3: ((-0.068, 0.044), (0.068, 0.044), (0.0, -0.050)),
        4: ((-0.068, 0.044), (-0.068, -0.044), (0.068, 0.044), (0.068, -0.044)),
    }
    perturbations = [
        _tray_perturbation,
        *_mug_slot_perturbations,
    ]

    def randomize(
        self,
        model: Any,
        data: Any,
        seed: int | None = None,
        request: Any | None = None,
    ) -> RandomizationState:
        rng = np.random.default_rng(seed)
        reset_request = MugResetRequest.from_value(request)
        # Freeze fixed-body fixtures (e.g. the mug_flip tray) to nominal when requested
        # (batched eval), so the per-world fixture that objects are placed relative to
        # matches the rendered/warp fixture (the frozen construction model). Default:
        # perturb as usual. Only mug_flip's _sample_once consumes this.
        self._freeze_fixed_bodies = reset_request.randomize_fixed_bodies is False
        self._pinned_tray_pose = reset_request.tray_pose

        if reset_request.use_inline_mugs:
            # Inline mode: keep the scene's inline mugs (geoms directly on mug_1/2/3,
            # colored mug_N_color) -- skip the variant scene reload AND color
            # randomization, randomize POSES only. The geometry-based reward is
            # calibrated on these inline mugs; a variant reload nests geoms under child
            # bodies the evaluator can't attribute tree contacts to.
            count = (
                int(reset_request.mug_count)
                if reset_request.mug_count
                else self.max_mug_count
            )
            self._set_active_mug_count(count)
            self._freeze_fixed_bodies = True  # keep the fixture consistent across the batch
            if self._env_ref is not None:
                model = self._env_ref.model
                data = self._env_ref.data
            return self._randomize_pose_with_rng(
                model=model,
                data=data,
                seed=seed,
                rng=rng,
                scale_states={},
                metadata={
                    "mug_variant": _MUG_DEFAULT_VARIANT,
                    "mug_variants": [_MUG_DEFAULT_VARIANT] * count,
                    "mug_count": count,
                    "use_inline_mugs": True,
                },
            )

        requested_mug_variants = self._resolve_mug_variants(rng, reset_request)
        requested_count = len(requested_mug_variants)
        should_randomize_scales = (
            True if reset_request.randomize_scales is None else bool(reset_request.randomize_scales)
        )

        last_scale_states: dict[str, float] = {}
        for candidate_count in range(requested_count, self.min_mug_count - 1, -1):
            mug_variants = requested_mug_variants[:candidate_count]
            self._set_active_mug_count(candidate_count)
            scale_states = self._sample_scale_states(rng) if should_randomize_scales else {}
            last_scale_states = scale_states
            self._current_scale_states = dict(scale_states)
            self._reload_mug_variant_scene(mug_variants, scale_states)

            if self._env_ref is not None:
                model = self._env_ref.model
                data = self._env_ref.data

            metadata: dict[str, Any] = {
                "mug_variant": mug_variants[0],
                "mug_variants": list(mug_variants),
                "mug_count": len(mug_variants),
            }
            if candidate_count != requested_count:
                metadata["requested_mug_count"] = requested_count
                metadata["mug_count_reduced"] = True

            state = self._try_randomize_pose_with_rng(
                model=model,
                data=data,
                seed=seed,
                rng=rng,
                scale_states=scale_states,
                metadata=metadata,
            )
            if state is None:
                continue

            if candidate_count != requested_count:
                logger.info(
                    "%s: reduced mug_count from %d to %d because selected mug assets did not fit",
                    type(self).__name__,
                    requested_count,
                    candidate_count,
                )
            mug_colors = self._sample_plain_mug_colors(rng, mug_variants)
            if mug_colors:
                state.metadata["mug_colors"] = mug_colors
                self._apply_plain_mug_colors(model, mug_colors, mug_variants)
            self._sample_and_apply_tray_color(model, state, rng)
            return state

        mug_variants = requested_mug_variants[: self.min_mug_count]
        self._set_active_mug_count(len(mug_variants))
        self._current_scale_states = dict(last_scale_states)
        self._reload_mug_variant_scene(mug_variants, last_scale_states)
        if self._env_ref is not None:
            model = self._env_ref.model
            data = self._env_ref.data
        logger.warning(
            "%s: no collision-free mug placement found after reducing to %d mug; using a single-mug sample",
            type(self).__name__,
            self.min_mug_count,
        )
        state = self._randomize_pose_with_rng(
            model=model,
            data=data,
            seed=seed,
            rng=rng,
            scale_states=last_scale_states,
            metadata={
                "mug_variant": mug_variants[0],
                "mug_variants": list(mug_variants),
                "mug_count": len(mug_variants),
                "requested_mug_count": requested_count,
                "mug_count_reduced": True,
            },
        )
        mug_colors = self._sample_plain_mug_colors(rng, mug_variants)
        if mug_colors:
            state.metadata["mug_colors"] = mug_colors
            self._apply_plain_mug_colors(model, mug_colors, mug_variants)
        self._sample_and_apply_tray_color(model, state, rng)
        return state

    def apply(self, model: Any, data: Any, state: RandomizationState) -> None:
        super().apply(model, data, state)
        target_model = self._env_ref.model if self._env_ref is not None else model
        tray_color = state.metadata.get("tray_color")
        if tray_color is not None:
            self._apply_tray_color(target_model, tray_color)

    def _sample_and_apply_tray_color(
        self,
        model: Any,
        state: RandomizationState,
        rng: np.random.Generator,
    ) -> None:
        tray_color = list(_TRAY_COLOR_PALETTE[int(rng.integers(len(_TRAY_COLOR_PALETTE)))])
        state.metadata["tray_color"] = tray_color
        self._apply_tray_color(model, tray_color)

    def _apply_tray_color(self, model: Any, raw_rgba: Any) -> None:
        rgba = tuple(float(value) for value in raw_rgba)
        if len(rgba) != 4:
            logger.warning("Invalid mug_flip tray color: %r", raw_rgba)
            return
        _apply_mat_color(model, "tray_blue", rgba)

    def _set_active_mug_count(self, mug_count: int) -> None:
        self._validate_mug_count(mug_count)
        self._current_active_mug_count = mug_count
        self.perturbations = [
            self._tray_perturbation,
            *self._mug_slot_perturbations[:mug_count],
        ]

    def _get_size_perturbations(self) -> list[ScalePerturbRange]:
        return [
            ScalePerturbRange(perturbation.joint_name, scale_factor=self.mug_scale_factor)
            for perturbation in self._mug_slot_perturbations[: int(getattr(self, "_current_active_mug_count", 2))]
        ]

    def _try_randomize_pose_with_rng(
        self,
        *,
        model: Any,
        data: Any,
        seed: int | None,
        rng: np.random.Generator,
        scale_states: dict[str, float],
        metadata: dict[str, Any],
    ) -> RandomizationState | None:
        self._before_sampling(model, data)
        nominals = self._read_nominals(model, data)
        for _attempt in range(self.max_tries):
            states = self._sample_once(nominals, rng)
            if not self._bounds_ok(states):
                continue
            if not self._pairwise_ok(states):
                continue
            self._apply_states(model, data, states)
            mujoco.mj_forward(model, data)
            if not self._contacts_ok(model, data):
                continue
            return RandomizationState(
                seed=seed or 0,
                object_states=states,
                scale_states=scale_states,
                metadata=dict(metadata),
            )
        return None

    @staticmethod
    def _obb_axes(yaw: float) -> tuple[np.ndarray, np.ndarray]:
        c = float(np.cos(yaw))
        s = float(np.sin(yaw))
        return (
            np.array([c, s], dtype=np.float64),
            np.array([-s, c], dtype=np.float64),
        )

    @classmethod
    def _obb_overlap(
        cls,
        center_a: np.ndarray,
        half_a: tuple[float, float],
        yaw_a: float,
        center_b: np.ndarray,
        half_b: tuple[float, float],
        yaw_b: float,
    ) -> bool:
        axis_ax, axis_ay = cls._obb_axes(yaw_a)
        axis_bx, axis_by = cls._obb_axes(yaw_b)
        axes = (axis_ax, axis_ay, axis_bx, axis_by)
        delta = center_b - center_a
        half_a = (half_a[0] + _MUG_FLIP_MUG_MARGIN_M, half_a[1] + _MUG_FLIP_MUG_MARGIN_M)
        half_b = (half_b[0] + _MUG_FLIP_MUG_MARGIN_M, half_b[1] + _MUG_FLIP_MUG_MARGIN_M)

        for axis in axes:
            distance = abs(float(np.dot(delta, axis)))
            radius_a = (
                half_a[0] * abs(float(np.dot(axis_ax, axis)))
                + half_a[1] * abs(float(np.dot(axis_ay, axis)))
            )
            radius_b = (
                half_b[0] * abs(float(np.dot(axis_bx, axis)))
                + half_b[1] * abs(float(np.dot(axis_by, axis)))
            )
            if distance > radius_a + radius_b:
                return False
        return True

    def _pairwise_ok(self, states: dict[str, dict[str, list[float]]]) -> bool:
        tray_state = states.get("tray")
        if tray_state is None:
            return False

        tray_pos = np.asarray(tray_state["pos"], dtype=np.float64)
        tray_yaw = _yaw_from_quat(np.asarray(tray_state["quat"], dtype=np.float64))
        c = float(np.cos(-tray_yaw))
        s = float(np.sin(-tray_yaw))

        entries: list[tuple[np.ndarray, tuple[float, float], float]] = []
        mug_variants = list(getattr(self, "_current_mug_variants", []))
        scale_states = getattr(self, "_current_scale_states", {})
        for instance_index, p in enumerate(self.perturbations):
            if p.joint_name == "tray":
                continue
            state = states.get(p.joint_name)
            if state is None:
                continue
            rel_xy = np.asarray(state["pos"][:2], dtype=np.float64) - tray_pos[:2]
            local_center = np.array(
                [c * rel_xy[0] - s * rel_xy[1], s * rel_xy[0] + c * rel_xy[1]],
                dtype=np.float64,
            )
            variant_index = instance_index - 1
            variant_name = (
                mug_variants[variant_index]
                if variant_index < len(mug_variants)
                else _MUG_DEFAULT_VARIANT
            )
            scale_factor = float(scale_states.get(p.joint_name, 1.0))
            half_x, half_y, _min_z, _max_z = _mug_flip_compiled_metadata(variant_name)
            local_yaw = _yaw_from_quat(np.asarray(state["quat"], dtype=np.float64)) - tray_yaw
            entries.append(
                (
                    local_center,
                    (half_x * scale_factor, half_y * scale_factor),
                    local_yaw,
                )
            )

        for i in range(len(entries)):
            center_i, half_i, yaw_i = entries[i]
            for j in range(i + 1, len(entries)):
                center_j, half_j, yaw_j = entries[j]
                if self._obb_overlap(center_i, half_i, yaw_i, center_j, half_j, yaw_j):
                    return False
        return True

    def _contacts_ok(self, model: Any, data: Any) -> bool:
        mug_root_ids: set[int] = set()
        for p in self.perturbations:
            if p.joint_name == "tray":
                continue
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, p.joint_name)
            if jnt_id >= 0:
                body_id = int(model.jnt_bodyid[jnt_id])
                mug_root_ids.add(int(model.body_rootid[body_id]))

        tray_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tray")
        for c in range(data.ncon):
            contact = data.contact[c]
            b1 = int(model.geom_bodyid[contact.geom1])
            b2 = int(model.geom_bodyid[contact.geom2])
            r1 = int(model.body_rootid[b1])
            r2 = int(model.body_rootid[b2])
            if r1 != r2 and r1 in mug_root_ids and r2 in mug_root_ids:
                return False
            if tray_body_id >= 0:
                mug_tray_contact = (
                    (r1 in mug_root_ids and _mujoco_body_in_subtree(model, b2, tray_body_id))
                    or (r2 in mug_root_ids and _mujoco_body_in_subtree(model, b1, tray_body_id))
                )
                if mug_tray_contact:
                    normal_z = abs(float(np.asarray(contact.frame, dtype=np.float64).reshape(-1)[2]))
                    if normal_z < 0.5:
                        return False
        return True

    @staticmethod
    def _rotated_mug_half_extents(half_x: float, half_y: float, yaw: float) -> tuple[float, float]:
        c = abs(float(np.cos(yaw)))
        s = abs(float(np.sin(yaw)))
        return c * half_x + s * half_y, s * half_x + c * half_y

    @staticmethod
    def _clamp_tray_local_xy(
        local_xy: np.ndarray,
        extent_x: float,
        extent_y: float,
    ) -> np.ndarray:
        tray_half_x, tray_half_y = _MUG_FLIP_TRAY_INNER_HALF_XY
        x_limit = max(0.0, tray_half_x - extent_x - _MUG_FLIP_TRAY_MARGIN_M)
        y_limit = max(0.0, tray_half_y - extent_y - _MUG_FLIP_TRAY_MARGIN_M)
        return np.array(
            [
                float(np.clip(local_xy[0], -x_limit, x_limit)),
                float(np.clip(local_xy[1], -y_limit, y_limit)),
            ],
            dtype=np.float64,
        )

    def _sample_once(
        self,
        nominals: dict[str, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
    ) -> dict[str, dict[str, list[float]]]:
        x_min, x_max, y_min, y_max = self.table_bounds
        states: dict[str, dict[str, list[float]]] = {}

        # 1. Sample the tray (fixed body) using the standard clamped-delta logic.
        tray_p = next(p for p in self.perturbations if p.joint_name == "tray")
        tray_nom_pos, tray_nom_quat = nominals["tray"]
        eff_dx = (max(tray_p.delta_x[0], x_min - tray_nom_pos[0]),
                  min(tray_p.delta_x[1], x_max - tray_nom_pos[0]))
        eff_dy = (max(tray_p.delta_y[0], y_min - tray_nom_pos[1]),
                  min(tray_p.delta_y[1], y_max - tray_nom_pos[1]))
        # Always draw the tray deltas so the downstream per-mug rng stream is unchanged,
        # but when fixtures are frozen (batched eval) hold the tray at its nominal pose
        # so mugs are placed relative to the same tray the warp/renderer shows.
        tray_delta = np.array([
            rng.uniform(*eff_dx),
            rng.uniform(*eff_dy),
            rng.uniform(*tray_p.delta_z),
        ])
        q_yaw = _quat_from_yaw(rng.uniform(*tray_p.delta_yaw))
        pinned = getattr(self, "_pinned_tray_pose", None)
        if pinned is not None:
            # Batched eval: place the tray at the exact construction pose (T_g) so mugs
            # are sampled relative to the same tray the renderer/warp shows. T_g varies
            # per visual group (it came from that group's construction reset).
            tray_new_pos = np.asarray(pinned[:3], dtype=np.float64)
            tray_new_quat = np.asarray(pinned[3:7], dtype=np.float64)
        elif getattr(self, "_freeze_fixed_bodies", False):
            tray_new_pos = tray_nom_pos.copy()
            tray_new_quat = tray_nom_quat.copy()
        else:
            tray_new_pos = tray_nom_pos + tray_delta
            tray_new_quat = _quat_mul(q_yaw, tray_nom_quat)
        states["tray"] = {"pos": tray_new_pos.tolist(), "quat": tray_new_quat.tolist()}

        # 2. Sample each mug relative to the tray's new position.
        # Offsets and per-mug deltas are tray-local, then rotated by the
        # sampled tray yaw so every active mug stays in the tray footprint.
        tray_nom_yaw = _yaw_from_quat(tray_nom_quat)
        tray_new_yaw = _yaw_from_quat(tray_new_quat)
        c_new, s_new = float(np.cos(tray_new_yaw)), float(np.sin(tray_new_yaw))

        def to_world_xy(local_xy: np.ndarray) -> np.ndarray:
            return np.array(
                [c_new * local_xy[0] - s_new * local_xy[1], s_new * local_xy[0] + c_new * local_xy[1]],
                dtype=np.float64,
            )

        mug_perturbations = [p for p in self.perturbations if p.joint_name != "tray"]
        mug_variants = list(
            getattr(self, "_current_mug_variants", [_MUG_DEFAULT_VARIANT] * len(mug_perturbations))
        )
        scale_states = getattr(self, "_current_scale_states", {})
        tray_floor_z = tray_new_pos[2] + _MUG_FLIP_TRAY_FLOOR_Z_OFFSET

        for instance_index, p in enumerate(mug_perturbations):
            if p.joint_name not in nominals:
                continue
            _mug_nom_pos, mug_nom_quat = nominals[p.joint_name]
            local_yaw = rng.uniform(*p.delta_yaw)
            variant_name = (
                mug_variants[instance_index]
                if instance_index < len(mug_variants)
                else _MUG_DEFAULT_VARIANT
            )
            scale_factor = float(scale_states.get(p.joint_name, 1.0))
            half_x, half_y, min_z, _max_z = _mug_flip_compiled_metadata(variant_name)
            half_x *= scale_factor
            half_y *= scale_factor
            min_z *= scale_factor
            extent_x, extent_y = self._rotated_mug_half_extents(half_x, half_y, local_yaw)

            slot_centers = self._mug_slot_centers_by_count[len(mug_perturbations)]
            local_xy = np.asarray(slot_centers[instance_index], dtype=np.float64) + np.array(
                [rng.uniform(*p.delta_x), rng.uniform(*p.delta_y)],
                dtype=np.float64,
            )
            local_xy = self._clamp_tray_local_xy(local_xy, extent_x, extent_y)
            world_xy = to_world_xy(local_xy)
            mug_new_z = tray_floor_z + _MUG_FLIP_SPAWN_CLEARANCE_M - min_z + rng.uniform(*p.delta_z)
            mug_new_pos = np.array(
                [tray_new_pos[0] + world_xy[0], tray_new_pos[1] + world_xy[1], mug_new_z],
                dtype=np.float64,
            )
            q_yaw_mug = _quat_from_yaw((tray_new_yaw - tray_nom_yaw) + local_yaw)
            states[p.joint_name] = {
                "pos": mug_new_pos.tolist(),
                "quat": _quat_mul(q_yaw_mug, mug_nom_quat).tolist(),
            }

        return states


class PourRandomizer(SceneRandomizer):
    """Randomize mug, cup, and beads-inside-mug for the pour/screw scene.

    Beads translate with the mug (same dx/dy) so they stay inside it after reset.
    Mug and cup are placed independently; contacts checked between them only.
    """

    min_clearance_m = 0.08
    # Only mug and cup in perturbations — used for pairwise/contact checks.
    # Beads are injected by _sample_once below.
    perturbations = [
        PerturbRange("mug_1_jnt",  delta_x=(-0.12, 0.12), delta_y=(-0.15, 0.15)),
        PerturbRange("cup_1_jnt", delta_x=(-0.10, 0.10), delta_y=(-0.20, 0.20)),
    ]
    _bead_joints = [f"bead_{i}_jnt" for i in range(1, 11)]
    size_perturbations = [
        ScalePerturbRange("mug_1_jnt"),
        ScalePerturbRange("cup_1_jnt"),
        *[ScalePerturbRange(f"bead_{i}_jnt") for i in range(1, 11)],
    ]

    def _read_nominals(self, model: Any, data: Any) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        nominals = super()._read_nominals(model, data)
        for bead_name in self._bead_joints:
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, bead_name)
            if jnt_id >= 0:
                adr = int(model.jnt_qposadr[jnt_id])
                nominals[bead_name] = (
                    data.qpos[adr: adr + 3].copy(),
                    data.qpos[adr + 3: adr + 7].copy(),
                )
        return nominals

    def _sample_once(
        self,
        nominals: dict[str, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
    ) -> dict[str, dict[str, list[float]]]:
        x_min, x_max, y_min, y_max = self.table_bounds
        states: dict[str, dict[str, list[float]]] = {}

        # Sample mug with standard clamped-delta logic.
        mug_p = next(p for p in self.perturbations if p.joint_name == "mug_1_jnt")
        mug_nom_pos, mug_nom_quat = nominals["mug_1_jnt"]
        eff_dx = (max(mug_p.delta_x[0], x_min - mug_nom_pos[0]),
                  min(mug_p.delta_x[1], x_max - mug_nom_pos[0]))
        eff_dy = (max(mug_p.delta_y[0], y_min - mug_nom_pos[1]),
                  min(mug_p.delta_y[1], y_max - mug_nom_pos[1]))
        mug_dx = rng.uniform(*eff_dx)
        mug_dy = rng.uniform(*eff_dy)
        mug_new_pos = mug_nom_pos + np.array([mug_dx, mug_dy, 0.0])
        q_yaw = _quat_from_yaw(rng.uniform(*mug_p.delta_yaw))
        states["mug_1_jnt"] = {
            "pos": mug_new_pos.tolist(),
            "quat": _quat_mul(q_yaw, mug_nom_quat).tolist(),
        }

        # Move beads by the same (dx, dy) as the mug so they stay inside it.
        for bead_name in self._bead_joints:
            if bead_name not in nominals:
                continue
            bead_nom_pos, bead_nom_quat = nominals[bead_name]
            states[bead_name] = {
                "pos": (bead_nom_pos + np.array([mug_dx, mug_dy, 0.0])).tolist(),
                "quat": bead_nom_quat.tolist(),
            }

        # Sample cup independently.
        cup_p = next(p for p in self.perturbations if p.joint_name == "cup_1_jnt")
        cup_nom_pos, cup_nom_quat = nominals["cup_1_jnt"]
        eff_dx = (max(cup_p.delta_x[0], x_min - cup_nom_pos[0]),
                  min(cup_p.delta_x[1], x_max - cup_nom_pos[0]))
        eff_dy = (max(cup_p.delta_y[0], y_min - cup_nom_pos[1]),
                  min(cup_p.delta_y[1], y_max - cup_nom_pos[1]))
        cup_new_pos = cup_nom_pos + np.array([
            rng.uniform(*eff_dx), rng.uniform(*eff_dy), 0.0,
        ])
        q_yaw_cup = _quat_from_yaw(rng.uniform(*cup_p.delta_yaw))
        states["cup_1_jnt"] = {
            "pos": cup_new_pos.tolist(),
            "quat": _quat_mul(q_yaw_cup, cup_nom_quat).tolist(),
        }

        return states

    def randomize(
        self,
        model: Any,
        data: Any,
        seed: int | None = None,
        request: Any | None = None,
    ) -> RandomizationState:
        state = super().randomize(model, data, seed, request=request)
        rng = np.random.default_rng(seed)
        target_model = self._env_ref.model if self._env_ref is not None else model
        if rng.random() < _COLOR_RANDOMIZE_PROB:
            for mat_name in ("mug_1_color", "cup_body_color"):
                _apply_mat_color(target_model, mat_name, _MUG_COLOR_PALETTE[rng.integers(len(_MUG_COLOR_PALETTE))])
        return state


class DrawerRandomizer(SceneRandomizer):
    """Randomize the drawer frame position/yaw and marker positions.

    The drawer frame (drawer_body) is a fixed body moved via model.body_pos.
    Markers are free-jointed and sampled independently around the table, then
    shifted by the same (dx, dy) as the drawer so they remain nearby after reset.
    """

    min_clearance_m = 0.04
    perturbations = [
        PerturbRange("drawer_body", delta_x=(-0.08, 0.0), delta_y=(-0.10, 0.10),
                     delta_yaw=(-0.3, 0.3), fixed_body=True),
        *[
            PerturbRange(f"marker_{i}_joint", delta_x=(-0.08, 0.08), delta_y=(-0.10, 0.10))
            for i in range(1, 6)
        ],
    ]

    def _sample_once(
        self,
        nominals: dict[str, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
    ) -> dict[str, dict[str, list[float]]]:
        states = super()._sample_once(nominals, rng)

        # Shift markers by the same (dx, dy) as the drawer so they stay near it.
        if "drawer_body" in states:
            drawer_nom_pos = nominals["drawer_body"][0]
            drawer_new_pos = np.array(states["drawer_body"]["pos"])
            dx = drawer_new_pos[0] - drawer_nom_pos[0]
            dy = drawer_new_pos[1] - drawer_nom_pos[1]
            for i in range(1, 6):
                key = f"marker_{i}_joint"
                if key in states:
                    states[key]["pos"][0] += dx
                    states[key]["pos"][1] += dy

        return states


class BallSortingRandomizer(SceneRandomizer):
    min_clearance_m = 0.06
    perturbations = [
        # Toy box: fixed body, randomized via model.body_pos (no physics).
        # Cylinders are rejection-sampled against it via the MuJoCo contact check.
        PerturbRange("ball-sorting-toy", delta_x=(-0.18, 0.08), delta_y=(-0.15, 0.15),
                     delta_yaw=(-0.3, 0.3), fixed_body=True),
        PerturbRange("cylinder-1", delta_x=(-0.5, 0.5), delta_y=(-0.5, 0.5)),
        PerturbRange("cylinder-2", delta_x=(-0.5, 0.5), delta_y=(-0.5, 0.5)),
        PerturbRange("cylinder-3", delta_x=(-0.5, 0.5), delta_y=(-0.5, 0.5)),
    ]


class SweepRandomizer(SceneRandomizer):
    max_tries = 400
    min_clearance_m = 0.015
    min_trash_count = 2
    default_trash_count = 3
    max_trash_count = 4
    _all_trash_joints = [f"trash_{i}_jnt" for i in range(1, 8)]
    _tool_joints = ("brush_jnt", "bin_joint", "dustpan_jnt")
    _trash_cluster_shift_x = (-0.03, 0.07)
    _trash_cluster_shift_y = (-0.18, 0.18)
    _cluster_shift_sample_tries = 24
    _tool_sample_tries = 32
    _trash_sample_tries = 48
    _trash_clearance_m = 0.04
    _trash_tool_clearance_m = 0.065
    _brush_trash_clearance_m = 0.09
    _tool_tool_clearance_m = 0.08
    _dustpan_keepout_margin_m = 0.006
    _brush_keepout_margin_m = 0.01

    def __init__(self) -> None:
        super().__init__()
        self._tool_keepout_discs: dict[str, list[tuple[np.ndarray, float]]] = {}
        self._tool_nominal_yaws: dict[str, float] = {}
        self._trash_joints: list[str] = []
        self._set_active_trash_count(self.default_trash_count)

    def prepare_env(self) -> None:
        self._set_active_trash_count(self.default_trash_count)

    def randomize(
        self,
        model: Any,
        data: Any,
        seed: int | None = None,
        request: Any | None = None,
    ) -> RandomizationState:
        reset_request = SweepResetRequest.from_value(request)
        rng = np.random.default_rng(seed)
        trash_count = self._resolve_trash_count(rng, reset_request)
        self._set_active_trash_count(trash_count)
        should_randomize_scales = (
            True if reset_request.randomize_scales is None else bool(reset_request.randomize_scales)
        )
        scale_states = self._sample_scale_states(rng) if should_randomize_scales else {}
        self._current_scale_states = dict(scale_states)
        if scale_states:
            self._reload_scene_for_scale_states(scale_states)
            if self._env_ref is not None:
                model = self._env_ref.model
                data = self._env_ref.data

        self._before_sampling(model, data)
        nominals = self._read_nominals(model, data)
        last_states: dict[str, dict[str, list[float]]] = {
            joint_name: {
                "pos": nom_pos.tolist(),
                "quat": nom_quat.tolist(),
            }
            for joint_name, (nom_pos, nom_quat) in nominals.items()
            if joint_name in {p.joint_name for p in self.perturbations}
        }

        for _ in range(self.max_tries):
            try:
                states = self._sample_once(nominals, rng)
            except _SweepPlacementFailure:
                continue
            last_states = states
            if not self._bounds_ok(states):
                continue
            if not self._pairwise_ok(states):
                continue
            self._apply_states(model, data, states)
            mujoco.mj_forward(model, data)
            if not self._contacts_ok(model, data):
                continue
            return self._finalize_randomization_state(
                model,
                data,
                seed=seed or 0,
                object_states=states,
                scale_states=scale_states,
                trash_count=trash_count,
            )

        logger.warning(
            "%s: no collision-free placement found after %d tries — using last sample",
            type(self).__name__,
            self.max_tries,
        )
        self._apply_states(model, data, last_states)
        mujoco.mj_forward(model, data)
        return self._finalize_randomization_state(
            model,
            data,
            seed=seed or 0,
            object_states=last_states,
            scale_states=scale_states,
            trash_count=trash_count,
        )

    def apply(self, model: Any, data: Any, state: RandomizationState) -> None:
        raw_trash_count = state.metadata.get("trash_count")
        if raw_trash_count is None:
            trash_count = sum(
                1
                for joint_name in self._all_trash_joints
                if joint_name in state.object_states
                and float(state.object_states[joint_name]["pos"][2]) >= 0.0
            )
            trash_count = min(max(trash_count, self.min_trash_count), self.max_trash_count)
        else:
            trash_count = int(raw_trash_count)
        self._set_active_trash_count(trash_count)

        object_states = dict(state.object_states)
        object_states.update(
            {
                joint_name: parked_state
                for joint_name, parked_state in self._inactive_trash_states().items()
                if joint_name not in object_states
            }
        )
        full_state = RandomizationState(
            seed=state.seed,
            object_states=object_states,
            scale_states=dict(state.scale_states),
            metadata=dict(state.metadata),
        )
        super().apply(model, data, full_state)

    def _resolve_trash_count(
        self,
        rng: np.random.Generator,
        request: SweepResetRequest,
    ) -> int:
        if request.trash_count is None:
            trash_count = int(rng.integers(self.min_trash_count, self.max_trash_count + 1))
        else:
            trash_count = int(request.trash_count)
        if not self.min_trash_count <= trash_count <= self.max_trash_count:
            raise ValueError(
                f"Sweep trash_count must be in [{self.min_trash_count}, {self.max_trash_count}], "
                f"got {trash_count}"
            )
        return trash_count

    def _set_active_trash_count(self, trash_count: int) -> None:
        if not self.min_trash_count <= trash_count <= self.max_trash_count:
            raise ValueError(
                f"Sweep trash_count must be in [{self.min_trash_count}, {self.max_trash_count}], "
                f"got {trash_count}"
            )

        self._trash_joints = list(self._all_trash_joints[:trash_count])
        self.perturbations = [
            PerturbRange("brush_jnt", delta_x=(-0.08, 0.10), delta_y=(-0.14, 0.12), delta_yaw=(-0.6, 0.6)),
            PerturbRange("bin_joint", delta_x=(-0.08, 0.05), delta_y=(-0.20, 0.20), delta_yaw=(-0.5, 0.5)),
            PerturbRange("dustpan_jnt", delta_x=(-0.08, 0.04), delta_y=(-0.12, 0.16), delta_yaw=(-0.6, 0.6)),
            *[
                PerturbRange(
                    joint_name,
                    delta_x=(-0.015, 0.015),
                    delta_y=(-0.015, 0.015),
                    delta_roll=(-np.pi, np.pi),
                    delta_pitch=(-np.pi, np.pi),
                    delta_yaw=(-np.pi, np.pi),
                )
                for joint_name in self._trash_joints
            ],
        ]
        self.size_perturbations = [
            ScalePerturbRange("brush_jnt"),
            ScalePerturbRange("bin_joint"),
            ScalePerturbRange("dustpan_jnt"),
            *[
                ScalePerturbRange(joint_name, scale_factor=(0.85, 1.15))
                for joint_name in self._trash_joints
            ],
        ]

    def _inactive_trash_joints(self) -> list[str]:
        active = set(self._trash_joints)
        return [joint_name for joint_name in self._all_trash_joints if joint_name not in active]

    def _inactive_trash_states(self) -> dict[str, dict[str, list[float]]]:
        # Park unused scrap slots RESTING ON the (infinite) table plane, well off
        # to the side. The old z=-1 parking spot sat ~2 m below the plane, which
        # generates plane<->scrap contacts with meter-scale penetration on every
        # reset: MuJoCo shrugs these off, but mjwarp's solver NaNs the whole
        # batch from them (whether a world survived depended on which contacts
        # the too-small nconmax happened to drop).
        park_z = 0.02
        if self._env_ref is not None:
            import mujoco as _mujoco

            m = self._env_ref.model
            gid = _mujoco.mj_name2id(m, _mujoco.mjtObj.mjOBJ_GEOM, "table_plane")
            if gid >= 0:
                park_z = float(m.geom_pos[gid, 2]) + 0.02
        states: dict[str, dict[str, list[float]]] = {}
        for index, joint_name in enumerate(self._inactive_trash_joints()):
            states[joint_name] = {
                "pos": [-1.5 - 0.1 * index, 0.0, park_z],
                "quat": [1.0, 0.0, 0.0, 0.0],
            }
        return states

    def _finalize_randomization_state(
        self,
        model: Any,
        data: Any,
        *,
        seed: int,
        object_states: dict[str, dict[str, list[float]]],
        scale_states: dict[str, float],
        trash_count: int,
    ) -> RandomizationState:
        state = RandomizationState(
            seed=seed,
            object_states=dict(object_states),
            scale_states=dict(scale_states),
            metadata={
                "trash_count": trash_count,
                "trash_joints": list(self._trash_joints),
            },
        )
        inactive_states = self._inactive_trash_states()
        if inactive_states:
            state.object_states.update(inactive_states)
            self._apply_states(model, data, inactive_states)
            mujoco.mj_forward(model, data)
        return state

    def _before_sampling(self, model: Any, data: Any) -> None:
        self._refresh_collision_metadata(model, data)

    def _refresh_collision_metadata(self, model: Any, data: Any) -> None:
        tool_keepout_discs: dict[str, list[tuple[np.ndarray, float]]] = {}
        tool_nominal_yaws: dict[str, float] = {}
        for joint_name in self._tool_joints:
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jnt_id < 0:
                continue
            body_id = int(model.jnt_bodyid[jnt_id])
            tool_keepout_discs[joint_name] = _body_subtree_xy_keepout_discs(model, data, body_id)
            tool_nominal_yaws[joint_name] = _yaw_from_quat(np.asarray(data.xquat[body_id], dtype=np.float64))
        self._tool_keepout_discs = tool_keepout_discs
        self._tool_nominal_yaws = tool_nominal_yaws

    def _clearance_for_pair(self, name_a: str, name_b: str) -> float:
        scale_a = float(self._current_scale_states.get(name_a, 1.0))
        scale_b = float(self._current_scale_states.get(name_b, 1.0))
        scale = max(scale_a, scale_b)
        if name_a in self._trash_joints and name_b in self._trash_joints:
            return self._trash_clearance_m * scale
        if name_a in self._tool_joints and name_b in self._tool_joints:
            return self._tool_tool_clearance_m * scale
        if "brush_jnt" in (name_a, name_b):
            return self._brush_trash_clearance_m * scale
        return self._trash_tool_clearance_m * scale

    @staticmethod
    def _rotate_xy(vec: np.ndarray, yaw: float) -> np.ndarray:
        cos_yaw = float(np.cos(yaw))
        sin_yaw = float(np.sin(yaw))
        return np.array(
            [
                cos_yaw * vec[0] - sin_yaw * vec[1],
                sin_yaw * vec[0] + cos_yaw * vec[1],
            ],
            dtype=np.float64,
        )

    def _tool_keepout_ok(
        self,
        candidate_xy: np.ndarray,
        *,
        states: dict[str, dict[str, list[float]]],
    ) -> bool:
        for tool_name in ("brush_jnt", "dustpan_jnt"):
            tool_state = states.get(tool_name)
            nominal_yaw = self._tool_nominal_yaws.get(tool_name)
            if tool_state is None or nominal_yaw is None:
                continue

            candidate_tool_xy = np.asarray(tool_state["pos"][:2], dtype=np.float64)
            yaw_delta = _yaw_from_quat(np.asarray(tool_state["quat"], dtype=np.float64)) - nominal_yaw
            margin = (
                self._brush_keepout_margin_m
                if tool_name == "brush_jnt"
                else self._dustpan_keepout_margin_m
            )
            for offset_xy, radius in self._tool_keepout_discs.get(tool_name, []):
                geom_xy = candidate_tool_xy + self._rotate_xy(offset_xy, yaw_delta)
                if np.linalg.norm(candidate_xy - geom_xy) < radius + margin:
                    return False
        return True

    def _sample_cluster_shift(
        self,
        *,
        states: dict[str, dict[str, list[float]]],
        nominals: dict[str, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
    ) -> tuple[float, float]:
        x_min, x_max, y_min, y_max = self.table_bounds
        last_shift = (0.0, 0.0)
        for _ in range(self._cluster_shift_sample_tries):
            cluster_dx = float(rng.uniform(*self._trash_cluster_shift_x))
            cluster_dy = float(rng.uniform(*self._trash_cluster_shift_y))
            last_shift = (cluster_dx, cluster_dy)
            placed_xy = {
                name: np.asarray(state["pos"][:2], dtype=np.float64)
                for name, state in states.items()
            }
            ok = True
            for joint_name in self._trash_joints:
                nominal = nominals.get(joint_name)
                if nominal is None:
                    continue
                candidate_xy = nominal[0][:2] + np.array([cluster_dx, cluster_dy], dtype=np.float64)
                if not (x_min <= candidate_xy[0] <= x_max and y_min <= candidate_xy[1] <= y_max):
                    ok = False
                    break
                if not self._tool_keepout_ok(candidate_xy, states=states):
                    ok = False
                    break
                if any(
                    np.linalg.norm(candidate_xy - other_xy) < self._clearance_for_pair(joint_name, other_name)
                    for other_name, other_xy in placed_xy.items()
                ):
                    ok = False
                    break
                placed_xy[joint_name] = candidate_xy
            if ok:
                return cluster_dx, cluster_dy
        return last_shift

    def _sample_once(
        self,
        nominals: dict[str, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
    ) -> dict[str, dict[str, list[float]]]:
        x_min, x_max, y_min, y_max = self.table_bounds
        states: dict[str, dict[str, list[float]]] = {}

        def sample_from_range(
            perturbation: PerturbRange,
            *,
            extra_xy: tuple[float, float] | None = None,
        ) -> dict[str, list[float]]:
            nom_pos, nom_quat = nominals[perturbation.joint_name]
            eff_dx = (
                max(perturbation.delta_x[0], x_min - nom_pos[0]),
                min(perturbation.delta_x[1], x_max - nom_pos[0]),
            )
            eff_dy = (
                max(perturbation.delta_y[0], y_min - nom_pos[1]),
                min(perturbation.delta_y[1], y_max - nom_pos[1]),
            )
            if eff_dx[0] > eff_dx[1]:
                eff_dx = (0.0, 0.0)
            if eff_dy[0] > eff_dy[1]:
                eff_dy = (0.0, 0.0)
            dx = rng.uniform(*eff_dx)
            dy = rng.uniform(*eff_dy)
            if extra_xy is not None:
                dx += extra_xy[0]
                dy += extra_xy[1]
            new_pos = nom_pos + np.array([dx, dy, rng.uniform(*perturbation.delta_z)])
            new_pos[0] = np.clip(new_pos[0], x_min, x_max)
            new_pos[1] = np.clip(new_pos[1], y_min, y_max)
            new_quat = _quat_mul(_sample_orientation_delta(perturbation, rng), nom_quat)
            return {"pos": new_pos.tolist(), "quat": new_quat.tolist()}

        perturb_by_name = {p.joint_name: p for p in self.perturbations}

        placed_xy: dict[str, np.ndarray] = {}

        def sample_clear_placement(
            joint_name: str,
            *,
            extra_xy: tuple[float, float] | None = None,
            tries: int,
        ) -> dict[str, list[float]]:
            perturbation = perturb_by_name[joint_name]
            last_state: dict[str, list[float]] | None = None
            for _ in range(tries):
                candidate = sample_from_range(perturbation, extra_xy=extra_xy)
                candidate_xy = np.asarray(candidate["pos"][:2], dtype=np.float64)
                if joint_name in self._trash_joints and not self._tool_keepout_ok(
                    candidate_xy,
                    states=states,
                ):
                    last_state = candidate
                    continue
                if all(
                    np.linalg.norm(candidate_xy - other_xy) >= self._clearance_for_pair(joint_name, other_name)
                    for other_name, other_xy in placed_xy.items()
                ):
                    return candidate
                last_state = candidate
            raise _SweepPlacementFailure(f"could not place {joint_name}")

        for joint_name in self._tool_joints:
            if joint_name not in nominals:
                continue
            state = sample_clear_placement(joint_name, tries=self._tool_sample_tries)
            states[joint_name] = state
            placed_xy[joint_name] = np.asarray(state["pos"][:2], dtype=np.float64)

        cluster_dx, cluster_dy = self._sample_cluster_shift(
            states=states,
            nominals=nominals,
            rng=rng,
        )
        for joint_name in self._trash_joints:
            if joint_name not in nominals:
                continue
            state = sample_clear_placement(
                extra_xy=(cluster_dx, cluster_dy),
                joint_name=joint_name,
                tries=self._trash_sample_tries,
            )
            states[joint_name] = state
            placed_xy[joint_name] = np.asarray(state["pos"][:2], dtype=np.float64)

        return states

    def _pairwise_ok(self, states: dict[str, dict[str, list[float]]]) -> bool:
        if len(states) < 2:
            return True

        names = list(states.keys())
        positions = {name: np.asarray(state["pos"][:2], dtype=np.float64) for name, state in states.items()}
        for i, name_a in enumerate(names):
            for name_b in names[i + 1:]:
                clearance = self._clearance_for_pair(name_a, name_b)
                if np.linalg.norm(positions[name_a] - positions[name_b]) < clearance:
                    return False

        for trash_name in self._trash_joints:
            trash_state = states.get(trash_name)
            if trash_state is None:
                continue
            if not self._tool_keepout_ok(
                np.asarray(trash_state["pos"][:2], dtype=np.float64),
                states=states,
            ):
                return False
        return True


_CHESS_PIECE_JOINTS = [
    "black_bishop_1_jnt", "black_bishop_2_jnt", "black_king_1_jnt",
    "black_knight_1_jnt", "black_knight_2_jnt",
    "black_pawn_1_jnt", "black_pawn_2_jnt", "black_pawn_3_jnt",
    "black_pawn_4_jnt", "black_pawn_5_jnt", "black_pawn_6_jnt",
    "black_pawn_7_jnt", "black_pawn_8_jnt",
    "black_queen_1_jnt", "black_rook_1_jnt", "black_rook_2_jnt",
    "white_bishop_1_jnt", "white_bishop_2_jnt", "white_king_1_jnt",
    "white_knight_1_jnt", "white_knight_2_jnt",
    "white_pawn_1_jnt", "white_pawn_2_jnt", "white_pawn_3_jnt",
    "white_pawn_4_jnt", "white_pawn_5_jnt", "white_pawn_6_jnt",
    "white_pawn_7_jnt", "white_pawn_8_jnt",
    "white_queen_1_jnt", "white_rook_1_jnt", "white_rook_2_jnt",
]


class LegacyChessScatterRandomizer(SceneRandomizer):
    """Scatter chess pieces off the board to random locations on the table.

    The chessboard itself receives moderate position and orientation
    randomization as a fixed body in the legacy chess2 scene.  Pieces are
    placed anywhere on the table outside the board's *new* footprint.

    Because the board occupies most of the table's X range, the probability
    that all 32 pieces simultaneously land off the board in one random draw
    is near zero.  ``_sample_once`` therefore uses per-piece rejection:
    the board is sampled first, then each piece is individually resampled
    until it falls outside the board footprint.  The outer rejection loop
    in the base class still handles pairwise clearance and MuJoCo contacts.
    """
    min_clearance_m = 0.056
    perturbations = [
        PerturbRange("chessboard", delta_x=(-0.05, 0.05), delta_y=(-0.10, 0.10),
                     delta_yaw=(-0.15, 0.15), fixed_body=True),
        # Pieces: scatter across entire table.
        *[PerturbRange(jnt, delta_x=(-1.0, 1.0), delta_y=(-1.0, 1.0))
          for jnt in _CHESS_PIECE_JOINTS],
    ]

    # Board half-extent (from the checker overlay size in the XML) + margin.
    _board_half: float = 0.224
    _board_margin: float = 0.05
    # Max per-piece rejection attempts before giving up on that piece.
    _per_piece_tries: int = 50

    def _sample_once(
        self,
        nominals: dict[str, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
    ) -> dict[str, dict[str, list[float]]]:
        """Sample board first, then per-piece reject until each is off the board."""
        x_min, x_max, y_min, y_max = self.table_bounds
        half = self._board_half + self._board_margin
        states: dict[str, dict[str, list[float]]] = {}

        # --- 1. Sample the chessboard (fixed body) -------------------------
        board_p = next(
            (p for p in self.perturbations if p.joint_name == "chessboard"), None
        )
        if board_p is not None and "chessboard" in nominals:
            nom_pos, nom_quat = nominals["chessboard"]
            eff_dx = (
                max(board_p.delta_x[0], x_min - nom_pos[0]),
                min(board_p.delta_x[1], x_max - nom_pos[0]),
            )
            eff_dy = (
                max(board_p.delta_y[0], y_min - nom_pos[1]),
                min(board_p.delta_y[1], y_max - nom_pos[1]),
            )
            board_new_pos = nom_pos + np.array([
                rng.uniform(*eff_dx),
                rng.uniform(*eff_dy),
                rng.uniform(*board_p.delta_z),
            ])
            q_yaw = _quat_from_yaw(rng.uniform(*board_p.delta_yaw))
            states["chessboard"] = {
                "pos": board_new_pos.tolist(),
                "quat": _quat_mul(q_yaw, nom_quat).tolist(),
            }
            bcx, bcy = board_new_pos[0], board_new_pos[1]
        else:
            # Fallback: use XML default board centre.
            bcx, bcy = 0.6, 0.0

        # --- 2. Sample each piece, rejecting until off-board AND clear of
        #        all previously placed pieces.
        placed_xy: list[np.ndarray] = []
        clearance = self.min_clearance_m

        for p in self.perturbations:
            if p.joint_name == "chessboard" or p.joint_name not in nominals:
                continue
            nom_pos, nom_quat = nominals[p.joint_name]
            eff_dx = (
                max(p.delta_x[0], x_min - nom_pos[0]),
                min(p.delta_x[1], x_max - nom_pos[0]),
            )
            eff_dy = (
                max(p.delta_y[0], y_min - nom_pos[1]),
                min(p.delta_y[1], y_max - nom_pos[1]),
            )

            for _ in range(self._per_piece_tries):
                new_pos = nom_pos + np.array([
                    rng.uniform(*eff_dx),
                    rng.uniform(*eff_dy),
                    rng.uniform(*p.delta_z),
                ])
                # Must be off the board.
                if abs(new_pos[0] - bcx) < half and abs(new_pos[1] - bcy) < half:
                    continue
                # Must be clear of all previously placed pieces.
                xy = new_pos[:2]
                if all(np.linalg.norm(xy - prev) >= clearance for prev in placed_xy):
                    break
            else:
                # Exhausted tries — keep last sample (outer loop may still reject).
                xy = new_pos[:2]

            placed_xy.append(xy)
            q_yaw = _quat_from_yaw(rng.uniform(*p.delta_yaw))
            states[p.joint_name] = {
                "pos": new_pos.tolist(),
                "quat": _quat_mul(q_yaw, nom_quat).tolist(),
            }

        return states


class ChessRandomizer(SceneRandomizer):
    """Generate partial chess-board setup tasks.

    Non-target pieces stay on their correct board squares. Target pieces are
    either staged on the table, knocked near their target squares, or placed in
    a simple free-jointed tin box.
    """

    scenarios = ("table_setup", "knocked_setup", "tin_setup")
    color_modes = ("white_only", "black_only", "mixed_partial", "both_broad")
    scenario_weights = (0.40, 0.30, 0.30)
    target_count_range = (6, 16)
    min_clearance_m = 0.035
    max_tries = 200
    _scene_model_cache_size = 8
    table_bounds = (0.34, 0.84, -0.65, 0.65)
    _piece_sample_tries = 200
    _board_half = 0.224
    _board_margin = 0.025
    _table_stage_x_range = (0.40, 0.80)
    _table_stage_y_ranges = ((0.34, 0.58), (-0.58, -0.34))
    _table_stage_grid = (4, 4)
    _table_stage_jitter = 0.018
    _table_stage_board_margin = 0.08
    _knocked_center_grid = (4, 5)
    _knocked_center_half_xy = (0.075, 0.145)
    _knocked_center_jitter = 0.010
    _knocked_center_clearance_m = 0.045
    _tin_inner_half_xy = (0.220, 0.095)
    _tin_outer_half_xy = (0.245, 0.120)
    _tin_floor_half_z = 0.006
    _tin_piece_margin = 0.017
    _tin_piece_grid = (6, 3)
    _table_z = 0.75
    _tin_joint = "tin_box_joint"
    _hidden_tin_pos = (0.6, 0.0, -10.0)
    perturbations = [
        PerturbRange(
            "chessboard",
            delta_x=(-0.05, 0.05),
            delta_y=(-0.10, 0.10),
            delta_yaw=(-0.15, 0.15),
        ),
        PerturbRange(
            _tin_joint,
            delta_x=(-0.04, 0.04),
            delta_y=(-0.005, 0.005),
            delta_yaw=(-0.25, 0.25),
        ),
        *[
            PerturbRange(jnt, delta_x=(-1.0, 1.0), delta_y=(-1.0, 1.0))
            for jnt in _CHESS_PIECE_JOINTS
        ],
    ]

    def prepare_env(self) -> None:
        self._compiled_scene_model_cache: OrderedDict[tuple[Any, ...], mujoco.MjModel] = OrderedDict()
        self._current_scene_model_cache_key: tuple[Any, ...] | None = None
        self._set_current_tin_variant(_CHESS_TIN_DEFAULT_VARIANT)
        self._reload_chess_tin_scene(_CHESS_TIN_DEFAULT_VARIANT, {})

    @staticmethod
    def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
        q = np.asarray(quat, dtype=np.float64)
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)

    @staticmethod
    def _normalize_quat(quat: np.ndarray) -> np.ndarray:
        q = np.asarray(quat, dtype=np.float64)
        norm = float(np.linalg.norm(q))
        if norm <= 0.0:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return q / norm

    @staticmethod
    def _piece_color(joint_name: str) -> str:
        return "white" if joint_name.startswith("white_") else "black"

    @staticmethod
    def _rotate_xy(vec: np.ndarray, yaw: float) -> np.ndarray:
        c = float(np.cos(yaw))
        s = float(np.sin(yaw))
        return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]], dtype=np.float64)

    def _get_size_perturbations(self) -> list[ScalePerturbRange]:
        return [ScalePerturbRange(joint_name) for joint_name in _CHESS_PIECE_JOINTS]

    def randomize(
        self,
        model: Any,
        data: Any,
        seed: int | None = None,
        request: Any | None = None,
    ) -> RandomizationState:
        rng = np.random.default_rng(seed)
        reset_request = ChessResetRequest.from_value(request)
        randomize_variants = (
            True if reset_request.randomize_variants is None else bool(reset_request.randomize_variants)
        )
        scenario = self._resolve_scenario(
            rng,
            reset_request.scenario,
            cycle_step=reset_request.cycle_scenario,
            randomize_variants=randomize_variants,
        )
        color_mode = self._resolve_color_mode(
            rng,
            reset_request.color_mode,
            cycle_step=reset_request.cycle_color_mode,
            randomize_variants=randomize_variants,
        )
        target_count = self._resolve_target_count(
            rng,
            reset_request.target_count,
            randomize_variants=randomize_variants,
        )
        tin_variant = self._resolve_tin_variant(
            rng,
            reset_request,
            randomize_variants=randomize_variants,
        )
        randomize_scales = (
            False if reset_request.randomize_scales is None else bool(reset_request.randomize_scales)
        )
        scale_states = self._sample_scale_states(rng) if randomize_scales else {}
        self._current_scale_states = dict(scale_states)
        self._reload_chess_tin_scene(tin_variant, scale_states)
        if self._env_ref is not None:
            model = self._env_ref.model
            data = self._env_ref.data

        self._before_sampling(model, data)
        nominals = self._read_nominals(model, data)
        last_states: dict[str, dict[str, list[float]]] = {}
        last_metadata: dict[str, Any] = {}
        for _attempt in range(self.max_tries):
            try:
                states, metadata = self._sample_setup_once(
                    nominals,
                    rng,
                    scenario=scenario,
                    color_mode=color_mode,
                    target_count=target_count,
                    tin_variant=tin_variant,
                )
            except _ChessPlacementFailure:
                continue
            last_states = states
            last_metadata = metadata
            if not self._bounds_ok(states):
                continue
            if not self._pairwise_ok(states):
                continue
            self._apply_states(model, data, states)
            self._set_tin_collision_enabled(model, bool(metadata.get("tin_active", False)))
            mujoco.mj_forward(model, data)
            if scenario != "knocked_setup" and not self._contacts_ok(model, data):
                continue
            self._remember_selection(metadata)
            return RandomizationState(
                seed=seed or 0,
                object_states=states,
                scale_states=scale_states,
                metadata=metadata,
            )

        logger.warning(
            "%s: no collision-free placement found after %d tries — using last sample",
            type(self).__name__,
            self.max_tries,
        )
        self._apply_states(model, data, last_states)
        self._set_tin_collision_enabled(model, bool(last_metadata.get("tin_active", False)))
        mujoco.mj_forward(model, data)
        self._remember_selection(last_metadata)
        return RandomizationState(
            seed=seed or 0,
            object_states=last_states,
            scale_states=scale_states,
            metadata=last_metadata,
        )

    def apply(self, model: Any, data: Any, state: RandomizationState) -> None:
        self._current_scale_states = dict(state.scale_states)
        raw_tin_variant = state.metadata.get(
            "tin_variant",
            getattr(self, "_current_chess_tin_variant", _CHESS_TIN_DEFAULT_VARIANT),
        )
        tin_variant = _chess_tin_canonical_variant_name(str(raw_tin_variant))
        self._reload_chess_tin_scene(tin_variant, state.scale_states)
        if self._env_ref is not None:
            model = self._env_ref.model
            data = self._env_ref.data
        self._apply_states(model, data, state.object_states)
        self._set_tin_collision_enabled(model, bool(state.metadata.get("tin_active", False)))
        mujoco.mj_forward(model, data)
        self._remember_selection(state.metadata)

    def _remember_selection(self, metadata: dict[str, Any]) -> None:
        scenario = metadata.get("scenario")
        color_mode = metadata.get("color_mode")
        target_count = metadata.get("target_count")
        tin_variant = metadata.get("tin_variant")
        if scenario in self.scenarios:
            self._current_chess_scenario = str(scenario)
        if color_mode in self.color_modes:
            self._current_chess_color_mode = str(color_mode)
        if target_count is not None:
            self._current_chess_target_count = int(target_count)
        if tin_variant is not None:
            self._set_current_tin_variant(str(tin_variant))

    def _set_current_tin_variant(self, tin_variant: str) -> str:
        tin_variant = _chess_tin_canonical_variant_name(tin_variant)
        self._current_chess_tin_variant = tin_variant
        self._active_tin_scale = _chess_tin_variant_scale(tin_variant)
        self._active_tin_outer_half_xy = _chess_tin_scaled_outer_half_xy(tin_variant)
        self._active_tin_inner_half_xy = _chess_tin_scaled_inner_half_xy(tin_variant)
        return tin_variant

    def _current_tin_outer_half_xy(self) -> tuple[float, float]:
        return tuple(getattr(self, "_active_tin_outer_half_xy", self._tin_outer_half_xy))

    def _current_tin_inner_half_xy(self) -> tuple[float, float]:
        return tuple(getattr(self, "_active_tin_inner_half_xy", self._tin_inner_half_xy))

    def _set_tin_collision_enabled(self, model: Any, enabled: bool) -> None:
        contype = 1 if enabled else 0
        conaffinity = 1 if enabled else 0
        for geom_name in (
            "tin_box_floor",
            "tin_box_wall_y_pos",
            "tin_box_wall_y_neg",
            "tin_box_wall_x_pos",
            "tin_box_wall_x_neg",
        ):
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id >= 0:
                model.geom_contype[geom_id] = contype
                model.geom_conaffinity[geom_id] = conaffinity

    def _sample_setup_once(
        self,
        nominals: dict[str, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
        *,
        scenario: str,
        color_mode: str,
        target_count: int,
        tin_variant: str,
    ) -> tuple[dict[str, dict[str, list[float]]], dict[str, Any]]:
        target_joints = self._sample_target_joints(rng, color_mode, target_count)

        board_state = self._sample_board_state(nominals, rng)
        target_poses = self._target_board_poses(nominals, board_state)
        tin_active = scenario == "tin_setup"
        tin_state = (
            self._sample_tin_state(nominals, rng, board_state)
            if tin_active
            else self._hidden_tin_state(nominals)
        )
        states: dict[str, dict[str, list[float]]] = {
            "chessboard": board_state,
            self._tin_joint: tin_state,
        }

        target_set = set(target_joints)
        for joint_name in _CHESS_PIECE_JOINTS:
            states[joint_name] = copy.deepcopy(target_poses[joint_name])

        if scenario == "table_setup":
            self._place_table_targets(states, target_poses, target_joints, board_state, rng)
        elif scenario == "knocked_setup":
            self._place_knocked_targets(states, target_poses, target_joints, board_state, rng)
        elif scenario == "tin_setup":
            self._place_tin_targets(states, target_poses, target_joints, tin_state, rng)
        else:
            raise ValueError(f"Unsupported chess scenario {scenario!r}")

        metadata = {
            "scenario": scenario,
            "target_count": len(target_joints),
            "target_pieces": list(target_joints),
            "target_joints": list(target_joints),
            "target_colors": sorted({self._piece_color(joint) for joint in target_joints}),
            "color_mode": color_mode,
            "target_poses": copy.deepcopy({joint: target_poses[joint] for joint in target_joints}),
            "board_target_poses": copy.deepcopy(target_poses),
            "tin_active": tin_active,
            "tin_variant": tin_variant,
        }
        if tin_active:
            metadata["tin_pose"] = copy.deepcopy(tin_state)
            metadata["tin_inner_half_xy"] = list(self._current_tin_inner_half_xy())
            metadata["tin_outer_half_xy"] = list(self._current_tin_outer_half_xy())
            metadata["tin_scale"] = float(getattr(self, "_active_tin_scale", 1.0))
            metadata["tin_floor_z"] = float(tin_state["pos"][2] + self._tin_floor_half_z)
        metadata["non_target_pieces"] = [
            joint_name for joint_name in _CHESS_PIECE_JOINTS if joint_name not in target_set
        ]
        return states, metadata

    def _resolve_scenario(
        self,
        rng: np.random.Generator,
        requested: str | None,
        *,
        cycle_step: int,
        randomize_variants: bool,
    ) -> str:
        current = str(getattr(self, "_current_chess_scenario", self.scenarios[0]))
        if current not in self.scenarios:
            current = self.scenarios[0]
        if requested is None:
            if cycle_step:
                current_index = self.scenarios.index(current)
                return self.scenarios[(current_index + cycle_step) % len(self.scenarios)]
            if not randomize_variants:
                return current
            return str(rng.choice(self.scenarios, p=np.asarray(self.scenario_weights, dtype=np.float64)))
        requested = requested.strip().lower()
        aliases = {
            "table": "table_setup",
            "table_setup": "table_setup",
            "knocked": "knocked_setup",
            "knocked_setup": "knocked_setup",
            "tin": "tin_setup",
            "tin_setup": "tin_setup",
        }
        try:
            return aliases[requested]
        except KeyError as exc:
            raise ValueError(f"Unsupported chess scenario {requested!r}") from exc

    def _resolve_color_mode(
        self,
        rng: np.random.Generator,
        requested: str | None,
        *,
        cycle_step: int,
        randomize_variants: bool,
    ) -> str:
        current = str(getattr(self, "_current_chess_color_mode", self.color_modes[0]))
        if current not in self.color_modes:
            current = self.color_modes[0]
        if requested is None:
            if cycle_step:
                current_index = self.color_modes.index(current)
                return self.color_modes[(current_index + cycle_step) % len(self.color_modes)]
            if not randomize_variants:
                return current
            return str(rng.choice(self.color_modes))
        requested = requested.strip().lower()
        aliases = {
            "white": "white_only",
            "white_only": "white_only",
            "black": "black_only",
            "black_only": "black_only",
            "mixed": "mixed_partial",
            "mixed_partial": "mixed_partial",
            "both": "both_broad",
            "both_broad": "both_broad",
        }
        try:
            return aliases[requested]
        except KeyError as exc:
            raise ValueError(f"Unsupported chess color_mode {requested!r}") from exc

    def _resolve_target_count(
        self,
        rng: np.random.Generator,
        requested: int | None,
        *,
        randomize_variants: bool,
    ) -> int:
        lo, hi = self.target_count_range
        if requested is not None:
            count = int(requested)
        elif not randomize_variants:
            count = int(getattr(self, "_current_chess_target_count", lo))
        else:
            count = int(rng.integers(lo, hi + 1))
        if not lo <= count <= hi:
            raise ValueError(f"Chess target_count must be in [{lo}, {hi}], got {count}")
        return count

    def _resolve_tin_variant(
        self,
        rng: np.random.Generator,
        request: ChessResetRequest,
        *,
        randomize_variants: bool,
    ) -> str:
        variants = _chess_tin_variant_names()
        current = _chess_tin_canonical_variant_name(
            str(getattr(self, "_current_chess_tin_variant", _CHESS_TIN_DEFAULT_VARIANT))
        )
        if current not in variants:
            current = variants[0]

        if request.tin_variant is not None:
            requested = _chess_tin_canonical_variant_name(request.tin_variant)
            if requested not in variants:
                raise ValueError(
                    f"Unknown chess tin variant {requested!r}. Available: {', '.join(variants)}"
                )
            return requested

        if request.cycle_tin:
            return variants[(variants.index(current) + request.cycle_tin) % len(variants)]

        if not randomize_variants:
            return current

        return variants[int(rng.integers(0, len(variants)))]

    def _scene_model_cache_key(
        self,
        tin_variant: str,
        scale_states: dict[str, float],
    ) -> tuple[Any, ...]:
        scale_key = tuple(
            sorted((str(name), round(float(value), 8)) for name, value in scale_states.items())
        )
        return (tin_variant, scale_key)

    def _try_reload_cached_scene_model(self, cache_key: tuple[Any, ...]) -> bool:
        if self._env_ref is None:
            return False
        cache = getattr(self, "_compiled_scene_model_cache", None)
        if not cache:
            return False
        cached_model = cache.get(cache_key)
        if cached_model is None:
            return False
        cache.move_to_end(cache_key)
        self._env_ref.reload_from_model(cached_model)
        self._current_scene_model_cache_key = cache_key
        return True

    def _store_compiled_scene_model(self, cache_key: tuple[Any, ...]) -> None:
        if self._env_ref is None:
            return
        cache = getattr(self, "_compiled_scene_model_cache", None)
        if cache is None:
            self._compiled_scene_model_cache = OrderedDict()
            cache = self._compiled_scene_model_cache
        cache[cache_key] = copy.deepcopy(self._env_ref.model)
        cache.move_to_end(cache_key)
        while len(cache) > self._scene_model_cache_size:
            cache.popitem(last=False)

    def _reload_chess_tin_scene(
        self,
        tin_variant: str,
        scale_states: dict[str, float],
    ) -> None:
        tin_variant = self._set_current_tin_variant(tin_variant)
        if self._env_ref is None:
            return

        preserved_arm_state = self._env_ref._get_reset_arm_state()
        scene_cache_key = self._scene_model_cache_key(tin_variant, scale_states)
        scene_reloaded = False
        if getattr(self, "_current_scene_model_cache_key", None) == scene_cache_key:
            pass
        elif self._try_reload_cached_scene_model(scene_cache_key):
            scene_reloaded = True
        else:
            base_scene_xml = self._base_scene_xml_string
            base_scene_dir = self._base_scene_xml_dir
            base_scene_transformed = self._base_scene_xml_transformed
            if base_scene_xml is None:
                base_scene_xml = _CHESS_BASE_SCENE_XML.read_text()
                base_scene_dir = _CHESS_BASE_SCENE_XML.parent
                base_scene_transformed = False

            xml = _build_chess_tin_scene_xml(
                tin_variant=tin_variant,
                scale_states=scale_states,
                base_scene_xml=base_scene_xml,
                base_scene_dir=base_scene_dir,
            )
            if self._scene_xml_transform_options is not None and not base_scene_transformed:
                from yam_sim.scene_xml import transform_scene_xml

                xml, _ = transform_scene_xml(xml, options=self._scene_xml_transform_options)

            self._env_ref.reload_from_xml(xml)
            self._store_compiled_scene_model(scene_cache_key)
            self._current_scene_model_cache_key = scene_cache_key
            scene_reloaded = True

        from yam_sim.scene_variants import apply_scene_variant

        if scene_reloaded:
            apply_scene_variant(self._env_ref.model, self._scene_variant)
            mujoco.mj_resetData(self._env_ref.model, self._env_ref.data)
            self._env_ref._set_qpos_from_state(preserved_arm_state)
            mujoco.mj_forward(self._env_ref.model, self._env_ref.data)
            self._fixed_body_nominals = None

    def _hidden_tin_state(
        self,
        nominals: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, list[float]]:
        _nom_pos, nom_quat = nominals[self._tin_joint]
        return {
            "pos": [float(value) for value in self._hidden_tin_pos],
            "quat": self._normalize_quat(nom_quat).tolist(),
        }

    def _sample_target_joints(
        self,
        rng: np.random.Generator,
        color_mode: str,
        target_count: int,
    ) -> list[str]:
        white = [joint for joint in _CHESS_PIECE_JOINTS if joint.startswith("white_")]
        black = [joint for joint in _CHESS_PIECE_JOINTS if joint.startswith("black_")]
        if color_mode == "white_only":
            return list(rng.choice(white, size=min(target_count, len(white)), replace=False))
        if color_mode == "black_only":
            return list(rng.choice(black, size=min(target_count, len(black)), replace=False))

        if color_mode == "both_broad":
            white_count = target_count // 2
            black_count = target_count - white_count
            if bool(rng.integers(0, 2)):
                white_count, black_count = black_count, white_count
        else:
            white_count = int(rng.integers(1, target_count))
            black_count = target_count - white_count

        selected = [
            *rng.choice(white, size=min(white_count, len(white)), replace=False).tolist(),
            *rng.choice(black, size=min(black_count, len(black)), replace=False).tolist(),
        ]
        rng.shuffle(selected)
        return selected

    def _sample_board_state(
        self,
        nominals: dict[str, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
    ) -> dict[str, list[float]]:
        board_p = next(p for p in self.perturbations if p.joint_name == "chessboard")
        nom_pos, nom_quat = nominals["chessboard"]
        x_min, x_max, y_min, y_max = self.table_bounds
        eff_dx = (
            max(board_p.delta_x[0], x_min - nom_pos[0]),
            min(board_p.delta_x[1], x_max - nom_pos[0]),
        )
        eff_dy = (
            max(board_p.delta_y[0], y_min - nom_pos[1]),
            min(board_p.delta_y[1], y_max - nom_pos[1]),
        )
        board_pos = nom_pos + np.array(
            [
                rng.uniform(*eff_dx),
                rng.uniform(*eff_dy),
                rng.uniform(*board_p.delta_z),
            ],
            dtype=np.float64,
        )
        board_quat = self._normalize_quat(
            _quat_mul(_quat_from_yaw(rng.uniform(*board_p.delta_yaw)), nom_quat)
        )
        return {"pos": board_pos.tolist(), "quat": board_quat.tolist()}

    def _sample_tin_state(
        self,
        nominals: dict[str, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
        board_state: dict[str, list[float]],
    ) -> dict[str, list[float]]:
        tin_p = next(p for p in self.perturbations if p.joint_name == self._tin_joint)
        nom_pos, nom_quat = nominals[self._tin_joint]
        x_min, x_max, y_min, y_max = (0.34, 0.84, -0.65, 0.65)
        half_x, half_y = self._current_tin_outer_half_xy()
        for _ in range(100):
            x = float(rng.uniform(
                max(nom_pos[0] + tin_p.delta_x[0], x_min + half_x),
                min(nom_pos[0] + tin_p.delta_x[1], x_max - half_x),
            ))
            y = float(rng.uniform(
                max(nom_pos[1] + tin_p.delta_y[0], y_min + half_y),
                min(nom_pos[1] + tin_p.delta_y[1], y_max - half_y),
            ))
            yaw = float(rng.uniform(*tin_p.delta_yaw))
            candidate = {
                "pos": [x, y, float(self._table_z + self._tin_floor_half_z + 0.001)],
                "quat": self._normalize_quat(_quat_mul(_quat_from_yaw(yaw), nom_quat)).tolist(),
            }
            if not self._tin_overlaps_board(candidate, board_state):
                return candidate
        return candidate

    def _target_board_poses(
        self,
        nominals: dict[str, tuple[np.ndarray, np.ndarray]],
        board_state: dict[str, list[float]],
    ) -> dict[str, dict[str, list[float]]]:
        board_nom_pos, board_nom_quat = nominals["chessboard"]
        board_new_pos = np.asarray(board_state["pos"], dtype=np.float64)
        board_new_quat = np.asarray(board_state["quat"], dtype=np.float64)
        board_nom_inv = self._quat_conjugate(board_nom_quat)
        poses: dict[str, dict[str, list[float]]] = {}
        for joint_name in _CHESS_PIECE_JOINTS:
            piece_nom_pos, piece_nom_quat = nominals[joint_name]
            local_pos = _quat_rotate_vector(board_nom_inv, piece_nom_pos - board_nom_pos)
            local_quat = _quat_mul(board_nom_inv, piece_nom_quat)
            pos = board_new_pos + _quat_rotate_vector(board_new_quat, local_pos)
            quat = self._normalize_quat(_quat_mul(board_new_quat, local_quat))
            poses[joint_name] = {"pos": pos.tolist(), "quat": quat.tolist()}
        return poses

    def _place_table_targets(
        self,
        states: dict[str, dict[str, list[float]]],
        target_poses: dict[str, dict[str, list[float]]],
        target_joints: list[str],
        board_state: dict[str, list[float]],
        rng: np.random.Generator,
    ) -> None:
        stage_slots: list[np.ndarray] = []
        for side_index in rng.permutation(len(self._table_stage_y_ranges)):
            stage_slots.extend(self._table_stage_slots(side_index, rng))
        rng.shuffle(stage_slots)

        placed_xy: list[np.ndarray] = []
        for joint_name in target_joints:
            target_pose = target_poses[joint_name]
            placed = False
            for slot_index, slot in enumerate(list(stage_slots)):
                pos = np.array(
                    [
                        slot[0] + rng.uniform(-self._table_stage_jitter, self._table_stage_jitter),
                        slot[1] + rng.uniform(-self._table_stage_jitter, self._table_stage_jitter),
                        target_pose["pos"][2],
                    ],
                    dtype=np.float64,
                )
                if not self._xy_inside_table(pos[:2], margin=0.02):
                    continue
                if self._inside_board_footprint(
                    pos[:2],
                    board_state,
                    margin=self._table_stage_board_margin,
                ):
                    continue
                if any(np.linalg.norm(pos[:2] - prev) < self.min_clearance_m for prev in placed_xy):
                    continue
                yaw = float(rng.uniform(-np.pi, np.pi))
                states[joint_name] = {
                    "pos": pos.tolist(),
                    "quat": self._normalize_quat(
                        _quat_mul(_quat_from_yaw(yaw), np.asarray(target_pose["quat"], dtype=np.float64))
                    ).tolist(),
                }
                placed_xy.append(pos[:2])
                placed = True
                del stage_slots[slot_index]
                break
            if not placed:
                raise _ChessPlacementFailure()

    def _table_stage_slots(self, side_index: int, rng: np.random.Generator) -> list[np.ndarray]:
        x_count, y_count = self._table_stage_grid
        x_min, x_max = self._table_stage_x_range
        y_min, y_max = self._table_stage_y_ranges[side_index]
        slots = [
            np.array([x, y], dtype=np.float64)
            for x in np.linspace(x_min, x_max, x_count)
            for y in np.linspace(y_min, y_max, y_count)
        ]
        rng.shuffle(slots)
        return slots

    def _xy_inside_table(self, xy: np.ndarray, *, margin: float = 0.0) -> bool:
        x_min, x_max, y_min, y_max = self.table_bounds
        return (
            x_min + margin <= float(xy[0]) <= x_max - margin
            and y_min + margin <= float(xy[1]) <= y_max - margin
        )

    def _place_knocked_targets(
        self,
        states: dict[str, dict[str, list[float]]],
        target_poses: dict[str, dict[str, list[float]]],
        target_joints: list[str],
        board_state: dict[str, list[float]],
        rng: np.random.Generator,
    ) -> None:
        cells = self._knocked_center_cells()
        rng.shuffle(cells)
        board_center_xy = np.mean(
            [np.asarray(pose["pos"][:2], dtype=np.float64) for pose in target_poses.values()],
            axis=0,
        )
        board_yaw = _yaw_from_quat(np.asarray(board_state["quat"], dtype=np.float64))
        target_set = set(target_joints)
        occupied_xy = [
            np.asarray(states[joint_name]["pos"][:2], dtype=np.float64)
            for joint_name in _CHESS_PIECE_JOINTS
            if joint_name not in target_set
        ]
        placed_xy: list[np.ndarray] = []
        for joint_name in target_joints:
            target_pose = target_poses[joint_name]
            placed = False
            for cell_index, cell in enumerate(list(cells)):
                local_xy = np.asarray(cell, dtype=np.float64) + rng.uniform(
                    -self._knocked_center_jitter,
                    self._knocked_center_jitter,
                    size=2,
                )
                world_xy = board_center_xy + self._rotate_xy(local_xy, board_yaw)
                if any(
                    np.linalg.norm(world_xy - prev) < self._knocked_center_clearance_m
                    for prev in (*occupied_xy, *placed_xy)
                ):
                    continue
                pos = np.array(
                    [
                        world_xy[0],
                        world_xy[1],
                        float(target_pose["pos"][2]) + rng.uniform(0.026, 0.040),
                    ],
                    dtype=np.float64,
                )
                tilt_axis = np.array(
                    [np.cos(rng.uniform(-np.pi, np.pi)), np.sin(rng.uniform(-np.pi, np.pi)), 0.0],
                    dtype=np.float64,
                )
                q_tilt = _quat_from_axis_angle(tilt_axis, float(rng.uniform(0.85, 1.25)))
                q_yaw = _quat_from_yaw(float(rng.uniform(-np.pi, np.pi)))
                states[joint_name] = {
                    "pos": pos.tolist(),
                    "quat": self._normalize_quat(
                        _quat_mul(_quat_mul(q_yaw, q_tilt), np.asarray(target_pose["quat"], dtype=np.float64))
                    ).tolist(),
                }
                placed_xy.append(world_xy)
                del cells[cell_index]
                placed = True
                break
            if not placed:
                raise _ChessPlacementFailure()

    def _knocked_center_cells(self) -> list[np.ndarray]:
        x_count, y_count = self._knocked_center_grid
        x_half, y_half = self._knocked_center_half_xy
        return [
            np.array([x, y], dtype=np.float64)
            for x in np.linspace(-x_half, x_half, x_count)
            for y in np.linspace(-y_half, y_half, y_count)
        ]

    def _place_tin_targets(
        self,
        states: dict[str, dict[str, list[float]]],
        target_poses: dict[str, dict[str, list[float]]],
        target_joints: list[str],
        tin_state: dict[str, list[float]],
        rng: np.random.Generator,
    ) -> None:
        cells = self._tin_cells()
        rng.shuffle(cells)
        tin_pos = np.asarray(tin_state["pos"], dtype=np.float64)
        tin_quat = np.asarray(tin_state["quat"], dtype=np.float64)
        floor_top_z = tin_pos[2] + self._tin_floor_half_z
        for joint_name, cell in zip(target_joints, cells):
            target_pose = target_poses[joint_name]
            piece_nom_z_offset = max(float(target_pose["pos"][2]) - self._table_z, 0.040)
            jitter = rng.uniform(-0.006, 0.006, size=2)
            local_xy = np.asarray(cell, dtype=np.float64) + jitter
            world_xy = tin_pos[:2] + self._rotate_xy(local_xy, _yaw_from_quat(tin_quat))
            pos = np.array([world_xy[0], world_xy[1], floor_top_z + piece_nom_z_offset], dtype=np.float64)
            q_yaw = _quat_from_yaw(float(rng.uniform(-np.pi, np.pi)))
            q_tilt = _quat_from_axis_angle(
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
                float(rng.uniform(-0.35, 0.35)),
            )
            states[joint_name] = {
                "pos": pos.tolist(),
                "quat": self._normalize_quat(
                    _quat_mul(_quat_mul(q_yaw, q_tilt), np.asarray(target_pose["quat"], dtype=np.float64))
                ).tolist(),
            }

    def _tin_cells(self) -> list[np.ndarray]:
        tin_inner_half_xy = self._current_tin_inner_half_xy()
        x_half = tin_inner_half_xy[0] - self._tin_piece_margin
        y_half = tin_inner_half_xy[1] - self._tin_piece_margin
        x_count, y_count = self._tin_piece_grid
        return [
            np.array([x, y], dtype=np.float64)
            for x in np.linspace(-x_half, x_half, x_count)
            for y in np.linspace(-y_half, y_half, y_count)
        ]

    def _inside_board_footprint(
        self,
        xy: np.ndarray,
        board_state: dict[str, list[float]],
        *,
        margin: float,
    ) -> bool:
        center = np.asarray(board_state["pos"][:2], dtype=np.float64)
        yaw = _yaw_from_quat(np.asarray(board_state["quat"], dtype=np.float64))
        local = self._rotate_xy(np.asarray(xy, dtype=np.float64) - center, -yaw)
        half = self._board_half + margin
        return abs(float(local[0])) < half and abs(float(local[1])) < half

    def _inside_tin_footprint(
        self,
        xy: np.ndarray,
        tin_state: dict[str, list[float]],
        *,
        margin: float,
    ) -> bool:
        center = np.asarray(tin_state["pos"][:2], dtype=np.float64)
        yaw = _yaw_from_quat(np.asarray(tin_state["quat"], dtype=np.float64))
        local = self._rotate_xy(np.asarray(xy, dtype=np.float64) - center, -yaw)
        tin_outer_half_xy = self._current_tin_outer_half_xy()
        return (
            abs(float(local[0])) < tin_outer_half_xy[0] + margin
            and abs(float(local[1])) < tin_outer_half_xy[1] + margin
        )

    def _tin_overlaps_board(
        self,
        tin_state: dict[str, list[float]],
        board_state: dict[str, list[float]],
    ) -> bool:
        tin_center = np.asarray(tin_state["pos"][:2], dtype=np.float64)
        board_center = np.asarray(board_state["pos"][:2], dtype=np.float64)
        board_yaw = _yaw_from_quat(np.asarray(board_state["quat"], dtype=np.float64))
        local = self._rotate_xy(tin_center - board_center, -board_yaw)
        tin_outer_half_xy = self._current_tin_outer_half_xy()
        return (
            abs(float(local[0])) < self._board_half + tin_outer_half_xy[0] + 0.02
            and abs(float(local[1])) < self._board_half + tin_outer_half_xy[1] + 0.02
        )

    def _pairwise_ok(self, states: dict[str, dict[str, list[float]]]) -> bool:
        # Nominal chess squares are intentionally close, so the generic
        # centre-distance check would reject valid board setups.
        return True

    def _contacts_ok(self, model: Any, data: Any) -> bool:
        piece_roots: set[int] = set()
        for joint_name in _CHESS_PIECE_JOINTS:
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jnt_id >= 0:
                piece_roots.add(int(model.body_rootid[int(model.jnt_bodyid[jnt_id])]))

        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            root_1 = int(model.body_rootid[int(model.geom_bodyid[contact.geom1])])
            root_2 = int(model.body_rootid[int(model.geom_bodyid[contact.geom2])])
            if root_1 != root_2 and root_1 in piece_roots and root_2 in piece_roots:
                return False
        return True


# chess2 keeps the original scatter-off-board randomization behavior.
Chess2Randomizer = LegacyChessScatterRandomizer


# ---------------------------------------------------------------------------
# InHand Transfer randomizer — reloads the MuJoCo model on every reset
# ---------------------------------------------------------------------------

import xml.etree.ElementTree as _ET
from functools import lru_cache as _lru_cache
from pathlib import Path as _Path

_MODELS_DIR = _Path(__file__).parent / "models"
_CHESS_BASE_SCENE_XML = _MODELS_DIR / "yam_chess_scene.xml"
_CHESS_TIN_ASSET_ROOT = _MODELS_DIR / "assets" / "task_chess" / "tin_box"
_CHESS_TIN_DEFAULT_VARIANT = "tin_2"
_CHESS_TIN_TARGET_FOOTPRINT_MAX = 0.357
_CHESS_TIN_VISUAL_MARGIN_M = 0.004
_CHESS_TIN_WALL_THICKNESS_M = 0.022
_DISHRACK_BASE_SCENE_XML = _MODELS_DIR / "yam_dishrack_base.xml"
_DISHRACK_TASK_ASSET_ROOT = _MODELS_DIR / "assets" / "task_dishrack"
_DISHRACK_VARIANT_ROOTS: dict[str, _Path] = {
    "plate": _DISHRACK_TASK_ASSET_ROOT / "plate",
    "dish_rack": _DISHRACK_TASK_ASSET_ROOT / "dish_rack",
}
_DISHRACK_DEFAULT_VARIANTS: dict[str, str] = {
    "dish_rack": "dish_rack_0",
    "plate": "plate_0",
}
_DISHRACK_VARIANT_ALIASES: dict[str, dict[str, str]] = {
    "dish_rack": {
        "current": "dish_rack_0",
        "DishRack026": "dish_rack_1",
        "DishRack027": "dish_rack_2",
        "DishRack028": "dish_rack_3",
        "DishRack030": "dish_rack_4",
        "DishRack038": "dish_rack_5",
        "DishRack039": "dish_rack_6",
        "DishRack040": "dish_rack_7",
        "DishRack041": "dish_rack_8",
        "DishRack043": "dish_rack_9",
        "DishRack044": "dish_rack_10",
        "DishRack047": "dish_rack_11",
        "DishRack050": "dish_rack_12",
    },
    "plate": {
        "current": "plate_0",
    },
}
_DISHRACK_RACK_WRAPPER_POSITION: tuple[float, float, float] = (0.62, -0.24, 0.75)
_DISHRACK_PLATE_WRAPPER_XY: tuple[tuple[float, float], ...] = (
    (0.50, 0.20),
    (0.72, 0.20),
    (0.50, 0.44),
    (0.72, 0.44),
)
_DISHRACK_PLATE_WRAPPER_Z: float = 0.75
_BASE_SCENE_XML = _MODELS_DIR / "yam_inhand_transfer_base.xml"
_LIGHTWHEEL_BASE = _MODELS_DIR / "assets_robocasa" / "objects_lightwheel" / "lightwheel"
_OBJAVERSE_BASE = _MODELS_DIR / "assets_robocasa" / "objaverse" / "objaverse"
_MUG_DEFAULT_VARIANT = "mug_0"
_MUG_BASE_SCENE_XMLS: dict[str, _Path] = {
    "mug_flip": _MODELS_DIR / "yam_mug_flip_scene.xml",
    "mug_tree": _MODELS_DIR / "yam_mug_tree_scene.xml",
}
_MUG_TASK_ASSET_ROOTS: dict[str, _Path] = {
    "mug_flip": _MODELS_DIR / "assets" / "task_mug_flip" / "mug",
    "mug_tree": _MODELS_DIR / "assets" / "task_mug_tree" / "mug",
}
_MUG_TASK_BODY_NAMES: dict[str, tuple[str, ...]] = {
    "mug_flip": ("mug_1", "mug_2", "mug_3", "mug_4"),
    "mug_tree": ("mug_1", "mug_2", "mug_3"),
}
_WATER_BOTTLE_BASE_SCENE_XML = _MODELS_DIR / "yam_put_bottles_scene.xml"
_WATER_BOTTLE_TASK_ASSET_ROOT = _MODELS_DIR / "assets" / "task_water_bottles" / "bottle"
_WATER_BOTTLE_DEFAULT_VARIANT = "bottle_0"
_WATER_BOTTLE_MASS_KG = 0.05
_WATER_BOTTLE_TABLE_Z = 0.75
_WATER_BOTTLE_FLAT_SPAWN_CLEARANCE_M = 0.002
_WATER_BOTTLE_WRAPPER_XY: tuple[tuple[float, float], ...] = (
    (0.46, -0.38),
    (0.62, -0.38),
    (0.78, -0.38),
    (0.46, 0.38),
    (0.62, 0.38),
    (0.78, 0.38),
)
_WATER_BOTTLE_WRAPPER_Z = 0.754
_MUG_FLIP_TRAY_INNER_HALF_XY = (0.166, 0.112)
_MUG_FLIP_TRAY_MARGIN_M = 0.004
_MUG_FLIP_TRAY_FLOOR_Z_OFFSET = 0.008
_MUG_FLIP_SPAWN_CLEARANCE_M = 0.004
_MUG_FLIP_MUG_MARGIN_M = 0.004
_DISHRACK_ROBOCASA_VARIANT_DIRS: dict[str, dict[str, _Path]] = {
    "dish_rack": {
        "dish_rack_10": _LIGHTWHEEL_BASE / "dish_rack" / "DishRack044",
    },
    "plate": {
        "plate_19": _OBJAVERSE_BASE / "plate" / "plate_19",
        "plate_20": _OBJAVERSE_BASE / "plate" / "plate_20",
    },
}

_OBJAVERSE_CATEGORIES = {"rolling_pin", "water_bottle", "can", "ladle"}
# Approved object categories for the inhand_transfer task.
# Edit this list to add/remove objects from the randomization pool.
INHAND_OBJECT_CATEGORIES: list[str] = [
    # lightwheel pack
    "dish_brush", "whisk", "salt_and_pepper_shaker", "cream_cheese_stick",
    "cheese_grater", "pizza_cutter",
    # objaverse pack
    "rolling_pin", "water_bottle", "can", "ladle",
]
_INHAND_CATEGORIES = INHAND_OBJECT_CATEGORIES  # internal alias
_OBJ_DENSITY = 30.0
_CATEGORY_MESH_SCALE: dict[str, str] = {"wooden_spoon": "1 1 2.5"}
_X_MIN, _X_MAX = 0.42, 0.78
_Y_LEFT_MIN, _Y_LEFT_MAX = 0.10, 0.40
_Y_RIGHT_MIN, _Y_RIGHT_MAX = -0.40, -0.10
_OBJ_Z = 0.82


@_lru_cache(maxsize=None)
def _dishrack_variant_names(kind: str) -> list[str]:
    root = _DISHRACK_VARIANT_ROOTS[kind]
    variants = [
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / "model.xml").exists()
    ]
    prefix = "plate_" if kind == "plate" else "dish_rack_"
    variants.sort(
        key=lambda name: (
            0,
            int(name[len(prefix) :]),
        )
        if name.startswith(prefix) and name[len(prefix) :].isdigit()
        else (1, name)
    )
    return variants


def _dishrack_canonical_variant_name(kind: str, variant_name: str) -> str:
    return _DISHRACK_VARIANT_ALIASES.get(kind, {}).get(variant_name, variant_name)


def _dishrack_variant_dir(kind: str, variant_name: str) -> _Path:
    variant_name = _dishrack_canonical_variant_name(kind, variant_name)
    path = _DISHRACK_VARIANT_ROOTS[kind] / variant_name
    if not (path / "model.xml").exists():
        fallback_path = _DISHRACK_ROBOCASA_VARIANT_DIRS.get(kind, {}).get(variant_name)
        if fallback_path is None or not (fallback_path / "model.xml").exists():
            raise FileNotFoundError(f"Missing {kind} variant model.xml: {path}")
        return fallback_path
    return path


def _dishrack_sample_variant_name(kind: str, rng: np.random.Generator) -> str:
    variants = _dishrack_variant_names(kind)
    return variants[int(rng.integers(0, len(variants)))]


def _dishrack_plate_body_name(index: int) -> str:
    if index == 0:
        return "plate"
    return f"plate_{index}"


def _dishrack_plate_joint_name(index: int) -> str:
    if index == 0:
        return "plate_joint"
    return f"plate_joint_{index}"


def _dishrack_instance_prefix(kind: str, variant_name: str, instance_index: int) -> str:
    return f"{kind}_{instance_index}_{variant_name}"


def _dishrack_normalize_plate_variants(
    plate_variants: str | list[str] | tuple[str, ...],
) -> list[str]:
    if isinstance(plate_variants, str):
        normalized = [_dishrack_canonical_variant_name("plate", plate_variants)]
    else:
        normalized = [_dishrack_canonical_variant_name("plate", str(value)) for value in plate_variants]

    if not 1 <= len(normalized) <= DishRackRandomizer.max_plate_count:
        raise ValueError(
            "DishRack requires between 1 and "
            f"{DishRackRandomizer.max_plate_count} plate variants, got {len(normalized)}"
        )

    available = set(_dishrack_variant_names("plate"))
    available.update(_DISHRACK_ROBOCASA_VARIANT_DIRS.get("plate", {}))
    invalid = [name for name in normalized if name not in available]
    if invalid:
        raise ValueError(
            f"Unknown plate variants {invalid}. Available: {', '.join(sorted(available))}"
        )
    return normalized


def _dishrack_asset_local_name(elem: _ET.Element) -> str:
    name = elem.get("name")
    if name:
        return name
    file_attr = elem.get("file", "")
    stem = _Path(file_attr).stem
    if stem:
        return stem
    raise ValueError(f"Unable to infer asset name for element: {elem.tag}")


def _dishrack_prefixed_name(prefix: str, local_name: str) -> str:
    return f"{prefix}_{local_name.replace('.', '_')}"


def _dishrack_find_object_body(root: _ET.Element) -> _ET.Element:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Variant model.xml is missing <worldbody>")

    for candidate_name in ("object", "model"):
        body = worldbody.find(f".//body[@name='{candidate_name}']")
        if body is not None:
            return body

    for body in worldbody.iter("body"):
        if any(child.tag == "geom" for child in body.iter()):
            return body
    raise ValueError("Variant model.xml does not contain a body with geoms")


def _dishrack_find_bbox_geom(body: _ET.Element) -> _ET.Element | None:
    for geom in body.iter("geom"):
        if geom.get("name") == "reg_bbox":
            return geom
    return None


def _dishrack_bbox_anchor_offset(body: _ET.Element) -> np.ndarray:
    bbox = _dishrack_find_bbox_geom(body)
    if bbox is None:
        return np.zeros(3, dtype=np.float64)
    pos = np.asarray(_parse_float_list(bbox.get("pos", "0 0 0")), dtype=np.float64)
    size = np.asarray(_parse_float_list(bbox.get("size", "0 0 0")), dtype=np.float64)
    return np.array([-pos[0], -pos[1], -(pos[2] - size[2])], dtype=np.float64)


def _dishrack_geom_world_bounds(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    geom_type = int(model.geom_type[geom_id])
    pos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    rot = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)

    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        size = np.asarray(model.geom_size[geom_id][:3], dtype=np.float64)
        half = np.abs(rot) @ size
        return pos - half, pos + half

    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        radius = float(model.geom_size[geom_id][0])
        half = np.full(3, radius, dtype=np.float64)
        return pos - half, pos + half

    if geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        size = np.asarray(model.geom_size[geom_id][:3], dtype=np.float64)
        half = np.abs(rot) @ size
        return pos - half, pos + half

    if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        radius = float(model.geom_size[geom_id][0])
        half_len = float(model.geom_size[geom_id][1])
        half = np.abs(rot) @ np.array([radius, radius, half_len], dtype=np.float64)
        return pos - half, pos + half

    if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        radius = float(model.geom_size[geom_id][0])
        half_len = float(model.geom_size[geom_id][1]) + radius
        half = np.abs(rot) @ np.array([radius, radius, half_len], dtype=np.float64)
        return pos - half, pos + half

    if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            raise ValueError(f"Mesh geom {geom_id} is missing compiled mesh data")
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        verts = np.asarray(model.mesh_vert[start : start + count], dtype=np.float64)
        world = verts @ rot.T + pos
        return world.min(axis=0), world.max(axis=0)

    radius = float(model.geom_rbound[geom_id])
    half = np.full(3, radius, dtype=np.float64)
    return pos - half, pos + half


def _dishrack_compiled_model_bounds(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> tuple[np.ndarray, np.ndarray]:
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name.startswith("reg_"):
            continue
        lower, upper = _dishrack_geom_world_bounds(model, data, geom_id)
        mins.append(lower)
        maxs.append(upper)
    if not mins:
        raise ValueError("Variant model did not contain any non-region geoms")
    return np.vstack(mins).min(axis=0), np.vstack(maxs).max(axis=0)


@_lru_cache(maxsize=None)
def _dishrack_compiled_metadata(
    kind: str,
    variant_name: str,
) -> tuple[float, float, float, float, float]:
    variant_dir = _dishrack_variant_dir(kind, variant_name)
    model = mujoco.MjModel.from_xml_path(str(variant_dir / "model.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    min_xyz, max_xyz = _dishrack_compiled_model_bounds(model, data)
    half_xy = 0.5 * (max_xyz[:2] - min_xyz[:2])
    center_xy = 0.5 * (min_xyz[:2] + max_xyz[:2])
    return (
        float(half_xy[0]),
        float(half_xy[1]),
        -float(center_xy[0]),
        -float(center_xy[1]),
        -float(min_xyz[2]),
    )


@_lru_cache(maxsize=None)
def _dishrack_compiled_xy_half_extents(kind: str, variant_name: str) -> tuple[float, float]:
    half_x, half_y, _, _, _ = _dishrack_compiled_metadata(kind, variant_name)
    return half_x, half_y


@_lru_cache(maxsize=None)
def _dishrack_compiled_anchor_offset(kind: str, variant_name: str) -> tuple[float, float, float]:
    _, _, offset_x, offset_y, offset_z = _dishrack_compiled_metadata(kind, variant_name)
    return offset_x, offset_y, offset_z


def _dishrack_absolutize_file_attr(elem: _ET.Element, variant_dir: _Path) -> None:
    file_attr = elem.get("file")
    if file_attr and not _Path(file_attr).is_absolute():
        elem.set("file", str((variant_dir / file_attr).resolve()))


def _dishrack_remove_dynamic_joints(body: _ET.Element) -> None:
    for parent in body.iter():
        for child in list(parent):
            if child.tag in {"freejoint", "joint"}:
                parent.remove(child)


def _dishrack_prefix_body_names(elem: _ET.Element, prefix: str) -> None:
    for child in elem.iter():
        name = child.get("name")
        if name and child.tag in {"body", "geom", "site"}:
            child.set("name", _dishrack_prefixed_name(prefix, name))


def _dishrack_rewrite_asset_refs(
    body: _ET.Element,
    *,
    mesh_map: dict[str, str],
    material_map: dict[str, str],
) -> None:
    for child in body.iter():
        mesh_name = child.get("mesh")
        if mesh_name and mesh_name in mesh_map:
            child.set("mesh", mesh_map[mesh_name])
        material_name = child.get("material")
        if material_name and material_name in material_map:
            child.set("material", material_map[material_name])


def _dishrack_prepare_imported_geoms(body: _ET.Element) -> None:
    for parent in body.iter():
        for child in list(parent):
            if child.tag != "geom":
                continue
            if child.get("name", "").endswith("_reg_bbox") or child.get("class") == "region":
                parent.remove(child)
                continue

            child.attrib.pop("class", None)


def _dishrack_shift_body(body: _ET.Element, offset: np.ndarray) -> None:
    base_pos = np.zeros(3, dtype=np.float64)
    if body.get("pos"):
        base_pos = np.asarray(_parse_float_list(body.get("pos", "0 0 0")), dtype=np.float64)
    body.set("pos", _format_float_list((base_pos + offset).tolist()))


def _dishrack_serialize_elements(elements: list[_ET.Element], indent: str = "    ") -> str:
    rendered: list[str] = []
    for elem in elements:
        if hasattr(_ET, "indent"):
            _ET.indent(elem, space="  ")
        xml = _ET.tostring(elem, encoding="unicode")
        rendered.append("\n".join(f"{indent}{line}" if line else line for line in xml.splitlines()))
    return "\n".join(rendered)


def _dishrack_wrapper_position(
    kind: str,
    variant_name: str,
    instance_index: int = 0,
) -> tuple[float, float, float]:
    if kind == "dish_rack":
        return _DISHRACK_RACK_WRAPPER_POSITION
    if kind != "plate":
        raise ValueError(f"Unsupported dishrack object kind {kind!r}")
    if not 0 <= instance_index < len(_DISHRACK_PLATE_WRAPPER_XY):
        raise ValueError(
            f"Plate instance index must be in [0, {len(_DISHRACK_PLATE_WRAPPER_XY) - 1}], "
            f"got {instance_index}"
        )
    x, y = _DISHRACK_PLATE_WRAPPER_XY[instance_index]
    return (x, y, _DISHRACK_PLATE_WRAPPER_Z)


def _dishrack_build_object_block(
    *,
    kind: str,
    variant_name: str,
    scale_factor: float,
    object_name: str,
    joint_name: str | None,
    instance_index: int,
) -> tuple[list[_ET.Element], _ET.Element]:
    variant_name = _dishrack_canonical_variant_name(kind, variant_name)
    variant_dir = _dishrack_variant_dir(kind, variant_name)
    root = _ET.parse(str(variant_dir / "model.xml")).getroot()
    object_body = _dishrack_find_object_body(root)
    offset = np.asarray(_dishrack_compiled_anchor_offset(kind, variant_name), dtype=np.float64) * float(scale_factor)

    prefix = _dishrack_instance_prefix(kind, variant_name, instance_index)
    mesh_map: dict[str, str] = {}
    texture_map: dict[str, str] = {}
    material_map: dict[str, str] = {}
    temp_asset = _ET.Element("asset")

    asset_root = root.find("asset")
    if asset_root is not None:
        asset_children = list(asset_root)
        for asset_child in asset_children:
            local_name = _dishrack_asset_local_name(asset_child)
            if asset_child.tag == "mesh":
                mesh_map[local_name] = _dishrack_prefixed_name(prefix, local_name)
            elif asset_child.tag == "texture":
                texture_map[local_name] = _dishrack_prefixed_name(prefix, local_name)
            elif asset_child.tag == "material":
                material_map[local_name] = _dishrack_prefixed_name(prefix, local_name)

        for asset_child in asset_children:
            cloned = copy.deepcopy(asset_child)
            local_name = _dishrack_asset_local_name(asset_child)
            if asset_child.tag == "mesh":
                cloned.set("name", mesh_map[local_name])
                _dishrack_absolutize_file_attr(cloned, variant_dir)
            elif asset_child.tag == "texture":
                cloned.set("name", texture_map[local_name])
                _dishrack_absolutize_file_attr(cloned, variant_dir)
            elif asset_child.tag == "material":
                cloned.set("name", material_map[local_name])
                texture_name = asset_child.get("texture")
                if texture_name and texture_name in texture_map:
                    cloned.set("texture", texture_map[texture_name])
            temp_asset.append(cloned)

    cloned_body = copy.deepcopy(object_body)
    _dishrack_remove_dynamic_joints(cloned_body)
    _dishrack_prefix_body_names(cloned_body, prefix)
    _dishrack_rewrite_asset_refs(
        cloned_body,
        mesh_map=mesh_map,
        material_map=material_map,
    )
    _dishrack_prepare_imported_geoms(cloned_body)
    _dishrack_shift_body(cloned_body, offset)

    if abs(scale_factor - 1.0) > 1e-9:
        mesh_assets = {
            mesh.get("name", ""): mesh
            for mesh in temp_asset.findall("mesh")
            if mesh.get("name")
        }
        _scale_body_subtree(
            body=cloned_body,
            factor=scale_factor,
            target_name=joint_name or object_name,
            asset_elem=temp_asset,
            mesh_assets=mesh_assets,
        )

    wrapper = _ET.Element(
        "body",
        name=object_name,
        pos=_format_float_list(list(_dishrack_wrapper_position(kind, variant_name, instance_index))),
    )
    if joint_name is not None:
        _ET.SubElement(wrapper, "freejoint", name=joint_name)
    wrapper.append(cloned_body)
    return list(temp_asset), wrapper


@_lru_cache(maxsize=None)
def _chess_tin_variant_names() -> list[str]:
    if not (_CHESS_TIN_ASSET_ROOT / _CHESS_TIN_DEFAULT_VARIANT / "body" / "body.xml").exists():
        raise FileNotFoundError(
            "Missing chess tin body.xml: "
            f"{_CHESS_TIN_ASSET_ROOT / _CHESS_TIN_DEFAULT_VARIANT / 'body' / 'body.xml'}"
        )
    return [_CHESS_TIN_DEFAULT_VARIANT]


def _chess_tin_canonical_variant_name(variant_name: str) -> str:
    if variant_name == "current":
        return _CHESS_TIN_DEFAULT_VARIANT
    if variant_name.isdigit():
        return f"tin_{variant_name}"
    if variant_name.startswith("tin") and variant_name[3:].isdigit():
        return f"tin_{variant_name[3:]}"
    return variant_name


def _chess_tin_variant_dir(variant_name: str) -> _Path:
    variant_name = _chess_tin_canonical_variant_name(variant_name)
    path = _CHESS_TIN_ASSET_ROOT / variant_name
    if not (path / "body" / "body.xml").exists():
        raise FileNotFoundError(f"Missing chess tin body.xml: {path / 'body' / 'body.xml'}")
    return path


@_lru_cache(maxsize=None)
def _chess_tin_compiled_metadata(variant_name: str) -> tuple[float, float, float, float, float, float]:
    variant_dir = _chess_tin_variant_dir(variant_name)
    model = mujoco.MjModel.from_xml_path(str(variant_dir / "body" / "body.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    min_xyz, max_xyz = _dishrack_compiled_model_bounds(model, data)
    half_xy = 0.5 * (max_xyz[:2] - min_xyz[:2])
    center_xy = 0.5 * (min_xyz[:2] + max_xyz[:2])
    return (
        float(half_xy[0]),
        float(half_xy[1]),
        float(min_xyz[2]),
        float(max_xyz[2]),
        float(center_xy[0]),
        float(center_xy[1]),
    )


@_lru_cache(maxsize=None)
def _chess_tin_variant_scale(variant_name: str) -> float:
    half_x, half_y, *_ = _chess_tin_compiled_metadata(variant_name)
    max_extent = 2.0 * max(half_x, half_y)
    if max_extent <= 0.0:
        return 1.0
    return max(1.0, _CHESS_TIN_TARGET_FOOTPRINT_MAX / max_extent)


@_lru_cache(maxsize=None)
def _chess_tin_scaled_outer_half_xy(variant_name: str) -> tuple[float, float]:
    half_x, half_y, *_ = _chess_tin_compiled_metadata(variant_name)
    scale = _chess_tin_variant_scale(variant_name)
    return (
        float(half_x * scale + _CHESS_TIN_VISUAL_MARGIN_M),
        float(half_y * scale + _CHESS_TIN_VISUAL_MARGIN_M),
    )


@_lru_cache(maxsize=None)
def _chess_tin_scaled_inner_half_xy(variant_name: str) -> tuple[float, float]:
    outer_x, outer_y = _chess_tin_scaled_outer_half_xy(variant_name)
    wall = _CHESS_TIN_WALL_THICKNESS_M
    return (
        max(0.040, float(outer_x - wall)),
        max(0.040, float(outer_y - wall)),
    )


def _chess_tin_instance_prefix(variant_name: str) -> str:
    return f"chess_tin_{_chess_tin_canonical_variant_name(variant_name)}"


def _chess_tin_prepare_imported_visual_geoms(body: _ET.Element) -> None:
    for parent in body.iter():
        for child in list(parent):
            if child.tag != "geom":
                continue

            mesh_name = child.get("mesh", "")
            is_visual = (
                child.get("class") == "visual"
                or (
                    child.get("contype") == "0"
                    and child.get("conaffinity") == "0"
                    and "collision" not in mesh_name
                )
            )
            if not is_visual:
                parent.remove(child)
                continue

            child.attrib.pop("class", None)
            child.set("type", "mesh")
            if child.get("group") is None:
                child.set("group", "2")
            child.set("contype", "0")
            child.set("conaffinity", "0")
            child.set("density", "0")
            child.attrib.pop("mass", None)


def _chess_tin_visual_anchor_offset(variant_name: str) -> np.ndarray:
    scale = _chess_tin_variant_scale(variant_name)
    _half_x, _half_y, min_z, _max_z, center_x, center_y = _chess_tin_compiled_metadata(variant_name)
    return np.array(
        [
            -center_x * scale,
            -center_y * scale,
            -ChessRandomizer._tin_floor_half_z - min_z * scale,
        ],
        dtype=np.float64,
    )


def _chess_tin_build_visual_body_block(
    *,
    variant_name: str,
) -> tuple[list[_ET.Element], _ET.Element]:
    variant_name = _chess_tin_canonical_variant_name(variant_name)
    variant_dir = _chess_tin_variant_dir(variant_name)
    body_dir = variant_dir / "body"
    root = _ET.parse(str(body_dir / "body.xml")).getroot()
    object_body = _dishrack_find_object_body(root)

    prefix = _chess_tin_instance_prefix(variant_name)
    mesh_map: dict[str, str] = {}
    texture_map: dict[str, str] = {}
    material_map: dict[str, str] = {}
    temp_asset = _ET.Element("asset")

    asset_root = root.find("asset")
    if asset_root is not None:
        asset_children = []
        visual_mesh_names = {
            geom.get("mesh", "")
            for geom in object_body.iter("geom")
            if geom.get("mesh")
            and (
                geom.get("class") == "visual"
                or (geom.get("contype") == "0" and geom.get("conaffinity") == "0")
            )
        }
        for asset_child in list(asset_root):
            local_name = _dishrack_asset_local_name(asset_child)
            if (
                asset_child.tag == "mesh"
                and "collision" in local_name
                and local_name not in visual_mesh_names
            ):
                continue
            asset_children.append(asset_child)
            if asset_child.tag == "mesh":
                mesh_map[local_name] = _dishrack_prefixed_name(prefix, local_name)
            elif asset_child.tag == "texture":
                texture_map[local_name] = _dishrack_prefixed_name(prefix, local_name)
            elif asset_child.tag == "material":
                material_map[local_name] = _dishrack_prefixed_name(prefix, local_name)

        for asset_child in asset_children:
            cloned = copy.deepcopy(asset_child)
            local_name = _dishrack_asset_local_name(asset_child)
            if asset_child.tag == "mesh":
                cloned.set("name", mesh_map[local_name])
                _dishrack_absolutize_file_attr(cloned, body_dir)
            elif asset_child.tag == "texture":
                cloned.set("name", texture_map[local_name])
                _dishrack_absolutize_file_attr(cloned, body_dir)
            elif asset_child.tag == "material":
                cloned.set("name", material_map[local_name])
                texture_name = asset_child.get("texture")
                if texture_name and texture_name in texture_map:
                    cloned.set("texture", texture_map[texture_name])
            temp_asset.append(cloned)

    cloned_body = copy.deepcopy(object_body)
    _dishrack_remove_dynamic_joints(cloned_body)
    _dishrack_prefix_body_names(cloned_body, prefix)
    _dishrack_rewrite_asset_refs(
        cloned_body,
        mesh_map=mesh_map,
        material_map=material_map,
    )
    _chess_tin_prepare_imported_visual_geoms(cloned_body)

    scale_factor = _chess_tin_variant_scale(variant_name)
    if abs(scale_factor - 1.0) > 1e-9:
        mesh_assets = {
            mesh.get("name", ""): mesh
            for mesh in temp_asset.findall("mesh")
            if mesh.get("name")
        }
        _scale_body_subtree(
            body=cloned_body,
            factor=scale_factor,
            target_name=ChessRandomizer._tin_joint,
            asset_elem=temp_asset,
            mesh_assets=mesh_assets,
        )

    _dishrack_shift_body(cloned_body, _chess_tin_visual_anchor_offset(variant_name))
    return list(temp_asset), cloned_body


def _chess_tin_collision_geoms(variant_name: str) -> list[_ET.Element]:
    outer_x, outer_y = _chess_tin_scaled_outer_half_xy(variant_name)
    inner_x, inner_y = _chess_tin_scaled_inner_half_xy(variant_name)
    wall_x = max(0.006, outer_x - inner_x)
    wall_y = max(0.006, outer_y - inner_y)
    _half_x, _half_y, min_z, max_z, _center_x, _center_y = _chess_tin_compiled_metadata(variant_name)
    visual_height = max(0.0, (max_z - min_z) * _chess_tin_variant_scale(variant_name))
    wall_half_z = max(0.035, min(0.095, 0.5 * visual_height))
    floor_half_z = ChessRandomizer._tin_floor_half_z
    wall_z = floor_half_z + wall_half_z

    specs = (
        ("tin_box_floor", [outer_x, outer_y, floor_half_z], [0.0, 0.0, 0.0], 0.060),
        ("tin_box_wall_y_pos", [outer_x, 0.5 * wall_y, wall_half_z], [0.0, inner_y + 0.5 * wall_y, wall_z], 0.015),
        ("tin_box_wall_y_neg", [outer_x, 0.5 * wall_y, wall_half_z], [0.0, -inner_y - 0.5 * wall_y, wall_z], 0.015),
        ("tin_box_wall_x_pos", [0.5 * wall_x, inner_y, wall_half_z], [inner_x + 0.5 * wall_x, 0.0, wall_z], 0.015),
        ("tin_box_wall_x_neg", [0.5 * wall_x, inner_y, wall_half_z], [-inner_x - 0.5 * wall_x, 0.0, wall_z], 0.015),
    )
    geoms: list[_ET.Element] = []
    for name, size, pos, mass in specs:
        geoms.append(
            _ET.Element(
                "geom",
                name=name,
                type="box",
                size=_format_float_list(size),
                pos=_format_float_list(pos),
                material="tin_box_mat",
                mass=f"{mass:.3f}",
                group="3",
                rgba="0 0 0 0",
                friction="2.0 0.1 0.01",
                condim="6",
                solref="0.004 1",
                solimp="0.998 0.998 0.001",
                priority="1",
            )
        )
    return geoms


def _chess_remove_body_by_name(root: _ET.Element, body_name: str) -> tuple[_ET.Element, int] | None:
    for parent in root.iter():
        children = list(parent)
        for index, child in enumerate(children):
            if child.tag == "body" and child.get("name") == body_name:
                parent.remove(child)
                return parent, index
    return None


def _build_chess_tin_scene_xml(
    *,
    tin_variant: str,
    scale_states: dict[str, float],
    base_scene_xml: str | None,
    base_scene_dir: _Path | None,
) -> str:
    tin_variant = _chess_tin_canonical_variant_name(tin_variant)
    base_text = base_scene_xml or _CHESS_BASE_SCENE_XML.read_text()
    base_dir = base_scene_dir or _CHESS_BASE_SCENE_XML.parent
    root = _ET.fromstring(base_text)
    asset_elem = root.find("asset")
    if asset_elem is None:
        raise ValueError("Chess scene XML is missing <asset>")
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Chess scene XML is missing <worldbody>")

    for child in list(asset_elem):
        name = child.get("name", "")
        if name.startswith("chess_tin_"):
            asset_elem.remove(child)

    removed = _chess_remove_body_by_name(root, "tin_box")
    parent, insert_index = removed if removed is not None else (worldbody, len(list(worldbody)))

    tin_assets, visual_body = _chess_tin_build_visual_body_block(variant_name=tin_variant)
    for asset in tin_assets:
        asset_elem.append(asset)

    wrapper = _ET.Element(
        "body",
        name="tin_box",
        pos=_format_float_list([0.6, 0.460, ChessRandomizer._table_z + ChessRandomizer._tin_floor_half_z + 0.001]),
    )
    _ET.SubElement(wrapper, "freejoint", name=ChessRandomizer._tin_joint)
    for geom in _chess_tin_collision_geoms(tin_variant):
        wrapper.append(geom)
    wrapper.append(visual_body)
    parent.insert(insert_index, wrapper)

    xml = _ET.tostring(root, encoding="unicode")
    if scale_states:
        xml = _apply_object_scales_to_scene_xml(xml, scale_states)
    return _resolve_scene_xml_paths(xml, base_dir)


def _build_dishrack_scene_xml(
    *,
    dish_rack_variant: str,
    plate_variant: str | None = None,
    plate_variants: str | list[str] | tuple[str, ...] | None = None,
    scale_states: dict[str, float],
    base_scene_xml: str | None,
    base_scene_dir: _Path | None,
) -> str:
    base_text = base_scene_xml or _DISHRACK_BASE_SCENE_XML.read_text()
    base_dir = base_scene_dir or _DISHRACK_BASE_SCENE_XML.parent
    normalized_plate_variants = _dishrack_normalize_plate_variants(
        plate_variants if plate_variants is not None else (plate_variant or _DISHRACK_DEFAULT_VARIANTS["plate"])
    )
    dish_rack_variant = _dishrack_canonical_variant_name("dish_rack", dish_rack_variant)

    rack_assets, rack_body = _dishrack_build_object_block(
        kind="dish_rack",
        variant_name=dish_rack_variant,
        scale_factor=float(scale_states.get("dishrack", 1.0)),
        object_name="dishrack",
        joint_name="dishrack",
        instance_index=0,
    )
    task_assets = list(rack_assets)
    task_bodies = [rack_body]
    for index, current_plate_variant in enumerate(normalized_plate_variants):
        plate_assets, plate_body = _dishrack_build_object_block(
            kind="plate",
            variant_name=current_plate_variant,
            scale_factor=float(scale_states.get(_dishrack_plate_joint_name(index), scale_states.get("plate_joint", 1.0))),
            object_name=_dishrack_plate_body_name(index),
            joint_name=_dishrack_plate_joint_name(index),
            instance_index=index,
        )
        task_assets.extend(plate_assets)
        task_bodies.append(plate_body)

    xml = base_text.replace("<!-- TASK_DEFAULTS_PLACEHOLDER -->", "")
    xml = xml.replace("<!-- TASK_ASSETS_PLACEHOLDER -->", _dishrack_serialize_elements(task_assets))
    xml = xml.replace("<!-- TASK_BODY_PLACEHOLDER -->", _dishrack_serialize_elements(task_bodies))
    bin_scale = float(scale_states.get("bin_joint", 1.0))
    if abs(bin_scale - 1.0) > 1e-9:
        xml = _apply_object_scales_to_scene_xml(xml, {"bin_joint": bin_scale})
    return _resolve_scene_xml_paths(xml, base_dir)


@_lru_cache(maxsize=None)
def _mug_variant_names(task_name: str) -> list[str]:
    root = _MUG_TASK_ASSET_ROOTS[task_name]
    variants = [
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / "model.xml").exists()
    ]
    variants.sort(
        key=lambda name: (
            0,
            int(name[len("mug_") :]),
        )
        if name.startswith("mug_") and name[len("mug_") :].isdigit()
        else (1, name)
    )
    if not variants:
        raise FileNotFoundError(f"No mug variants found under {root}")
    return variants


def _mug_canonical_variant_name(variant_name: str) -> str:
    if variant_name == "current":
        return _MUG_DEFAULT_VARIANT
    if variant_name.isdigit():
        return f"mug_{variant_name}"
    if variant_name.startswith("mug") and variant_name[3:].isdigit():
        return f"mug_{variant_name[3:]}"
    return variant_name


def _mug_variant_dir(task_name: str, variant_name: str) -> _Path:
    variant_name = _mug_canonical_variant_name(variant_name)
    path = _MUG_TASK_ASSET_ROOTS[task_name] / variant_name
    if not (path / "model.xml").exists():
        raise FileNotFoundError(f"Missing mug variant model.xml: {path}")
    return path


def _mug_sample_variant_name(task_name: str, rng: np.random.Generator) -> str:
    variants = _mug_variant_names(task_name)
    return variants[int(rng.integers(0, len(variants)))]


def _mug_instance_prefix(variant_name: str, instance_index: int) -> str:
    return f"mug_{instance_index}_{variant_name}"


def _mug_prepare_imported_geoms(body: _ET.Element) -> None:
    mass_assigned = False
    for parent in body.iter():
        for child in list(parent):
            if child.tag != "geom":
                continue

            name = child.get("name", "")
            child_class = child.get("class", "")
            mesh_name = child.get("mesh")
            if (
                child_class == "region"
                or name.startswith("reg_")
                or mesh_name is None
            ):
                parent.remove(child)
                continue

            is_visual = child.get("contype") == "0" and child.get("conaffinity") == "0"
            child.attrib.pop("class", None)
            if is_visual:
                child.set("group", "2")
                child.set("contype", "0")
                child.set("conaffinity", "0")
                child.set("density", "0")
                if not mass_assigned:
                    child.set("mass", "0.05")
                    mass_assigned = True
                else:
                    child.attrib.pop("mass", None)
                continue

            child.set("group", "3")
            child.set("rgba", "0 0 0 0")
            child.set("density", "0")
            child.set("friction", "3.0 0.03 0.003")
            child.set("condim", "6")
            child.set("solref", "0.004 1")
            child.set("solimp", "0.998 0.998 0.001")
            child.set("priority", "1")
            child.attrib.pop("mass", None)


def _mug_build_object_block(
    *,
    task_name: str,
    variant_name: str,
    instance_index: int,
    scale_factor: float,
    joint_name: str,
) -> tuple[list[_ET.Element], _ET.Element]:
    variant_name = _mug_canonical_variant_name(variant_name)
    variant_dir = _mug_variant_dir(task_name, variant_name)
    root = _ET.parse(str(variant_dir / "model.xml")).getroot()
    object_body = _dishrack_find_object_body(root)

    prefix = _mug_instance_prefix(variant_name, instance_index)
    mesh_map: dict[str, str] = {}
    texture_map: dict[str, str] = {}
    material_map: dict[str, str] = {}
    temp_asset = _ET.Element("asset")

    asset_root = root.find("asset")
    if asset_root is not None:
        asset_children = list(asset_root)
        for asset_child in asset_children:
            local_name = _dishrack_asset_local_name(asset_child)
            if asset_child.tag == "mesh":
                mesh_map[local_name] = _dishrack_prefixed_name(prefix, local_name)
            elif asset_child.tag == "texture":
                texture_map[local_name] = _dishrack_prefixed_name(prefix, local_name)
            elif asset_child.tag == "material":
                material_map[local_name] = _dishrack_prefixed_name(prefix, local_name)

        for asset_child in asset_children:
            cloned = copy.deepcopy(asset_child)
            local_name = _dishrack_asset_local_name(asset_child)
            if asset_child.tag == "mesh":
                cloned.set("name", mesh_map[local_name])
                _dishrack_absolutize_file_attr(cloned, variant_dir)
            elif asset_child.tag == "texture":
                cloned.set("name", texture_map[local_name])
                _dishrack_absolutize_file_attr(cloned, variant_dir)
            elif asset_child.tag == "material":
                cloned.set("name", material_map[local_name])
                texture_name = asset_child.get("texture")
                if texture_name and texture_name in texture_map:
                    cloned.set("texture", texture_map[texture_name])
            temp_asset.append(cloned)

    cloned_body = copy.deepcopy(object_body)
    _dishrack_remove_dynamic_joints(cloned_body)
    _dishrack_prefix_body_names(cloned_body, prefix)
    _dishrack_rewrite_asset_refs(
        cloned_body,
        mesh_map=mesh_map,
        material_map=material_map,
    )
    _mug_prepare_imported_geoms(cloned_body)

    if abs(scale_factor - 1.0) > 1e-9:
        mesh_assets = {
            mesh.get("name", ""): mesh
            for mesh in temp_asset.findall("mesh")
            if mesh.get("name")
        }
        _scale_body_subtree(
            body=cloned_body,
            factor=scale_factor,
            target_name=joint_name,
            asset_elem=temp_asset,
            mesh_assets=mesh_assets,
        )

    return list(temp_asset), cloned_body


def _mug_replace_body_contents(wrapper: _ET.Element, imported_body: _ET.Element) -> None:
    for child in list(wrapper):
        if child.tag not in {"freejoint", "joint"}:
            wrapper.remove(child)
    wrapper.append(imported_body)


def _mug_remove_inactive_bodies(root: _ET.Element, inactive_body_names: set[str]) -> None:
    if not inactive_body_names:
        return
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "body" and child.get("name") in inactive_body_names:
                parent.remove(child)


def _mug_rewrite_home_keyframe(root: _ET.Element, active_body_names: tuple[str, ...]) -> None:
    keyframe = root.find("keyframe")
    if keyframe is None:
        return
    home_key = keyframe.find("./key[@name='home']")
    if home_key is None:
        return

    raw_qpos = home_key.get("qpos", "")
    qpos_values = _parse_float_list(raw_qpos) if raw_qpos else []
    arm_qpos = qpos_values[-16:] if len(qpos_values) >= 16 else []
    object_qpos: list[float] = []
    for body_name in active_body_names:
        body = root.find(f".//body[@name='{body_name}']")
        if body is None:
            continue
        object_qpos.extend(_parse_float_list(body.get("pos", "0 0 0")))
        object_qpos.extend(_parse_float_list(body.get("quat", "1 0 0 0")))
    if object_qpos or arm_qpos:
        home_key.set("qpos", _format_float_list([*object_qpos, *arm_qpos]))


def _build_mug_scene_xml(
    *,
    task_name: str,
    mug_variant: str,
    mug_variants: list[str] | tuple[str, ...] | None = None,
    scale_states: dict[str, float],
    base_scene_xml: str | None,
    base_scene_dir: _Path | None,
) -> str:
    base_text = base_scene_xml or _MUG_BASE_SCENE_XMLS[task_name].read_text()
    base_dir = base_scene_dir or _MUG_BASE_SCENE_XMLS[task_name].parent
    if mug_variants is None:
        body_names = _MUG_TASK_BODY_NAMES[task_name]
        default_count = min(2, len(body_names))
        normalized_mug_variants = [_mug_canonical_variant_name(mug_variant)] * default_count
    else:
        normalized_mug_variants = [_mug_canonical_variant_name(str(variant)) for variant in mug_variants]
    body_names = _MUG_TASK_BODY_NAMES[task_name]
    if not 1 <= len(normalized_mug_variants) <= len(body_names):
        raise ValueError(
            f"{task_name} requires between 1 and {len(body_names)} mug variants, "
            f"got {len(normalized_mug_variants)}"
        )

    root = _ET.fromstring(base_text)
    asset_elem = root.find("asset")
    if asset_elem is None:
        raise ValueError("Mug scene XML is missing <asset>")
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Mug scene XML is missing <worldbody>")

    active_body_names = body_names[: len(normalized_mug_variants)]
    _mug_remove_inactive_bodies(root, set(body_names[len(normalized_mug_variants) :]))
    _mug_rewrite_home_keyframe(root, active_body_names)

    for instance_index, (body_name, current_mug_variant) in enumerate(
        zip(active_body_names, normalized_mug_variants)
    ):
        wrapper = worldbody.find(f".//body[@name='{body_name}']")
        if wrapper is None:
            raise ValueError(f"Mug scene XML is missing body {body_name!r}")
        joint_name = f"{body_name}_jnt"
        variant_assets, imported_body = _mug_build_object_block(
            task_name=task_name,
            variant_name=current_mug_variant,
            instance_index=instance_index,
            scale_factor=float(scale_states.get(joint_name, 1.0)),
            joint_name=joint_name,
        )
        for asset in variant_assets:
            asset_elem.append(asset)
        _mug_replace_body_contents(wrapper, imported_body)

    return _resolve_scene_xml_paths(_ET.tostring(root, encoding="unicode"), base_dir)


def _mujoco_body_in_subtree(model: mujoco.MjModel, body_id: int, root_body_id: int) -> bool:
    current = int(body_id)
    while current >= 0:
        if current == root_body_id:
            return True
        parent = int(model.body_parentid[current])
        if parent == current:
            break
        current = parent
    return False


@_lru_cache(maxsize=None)
def _mug_flip_compiled_metadata(variant_name: str) -> tuple[float, float, float, float]:
    """Return collision bounds for a mug_flip asset relative to the wrapper body."""
    variant_name = _mug_canonical_variant_name(variant_name)
    xml = _build_mug_scene_xml(
        task_name="mug_flip",
        mug_variant=variant_name,
        mug_variants=[variant_name],
        scale_states={},
        base_scene_xml=None,
        base_scene_dir=None,
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    mug_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mug_1")
    if mug_body_id < 0:
        raise ValueError(f"Compiled mug_flip scene for {variant_name} is missing mug_1")

    def collect_bounds(collision_only: bool) -> tuple[np.ndarray, np.ndarray] | None:
        mins: list[np.ndarray] = []
        maxs: list[np.ndarray] = []
        for geom_id in range(model.ngeom):
            body_id = int(model.geom_bodyid[geom_id])
            if not _mujoco_body_in_subtree(model, body_id, mug_body_id):
                continue
            if collision_only and not (
                int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id])
            ):
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if name.startswith("reg_"):
                continue
            lower, upper = _dishrack_geom_world_bounds(model, data, geom_id)
            mins.append(lower)
            maxs.append(upper)
        if not mins:
            return None
        return np.vstack(mins).min(axis=0), np.vstack(maxs).max(axis=0)

    bounds = collect_bounds(collision_only=True) or collect_bounds(collision_only=False)
    if bounds is None:
        raise ValueError(f"Compiled mug_flip scene for {variant_name} has no mug geoms")

    min_xyz, max_xyz = bounds
    mug_pos = np.asarray(data.xpos[mug_body_id], dtype=np.float64)
    rel_min = min_xyz - mug_pos
    rel_max = max_xyz - mug_pos
    half_x = max(abs(float(rel_min[0])), abs(float(rel_max[0])))
    half_y = max(abs(float(rel_min[1])), abs(float(rel_max[1])))
    return half_x, half_y, float(rel_min[2]), float(rel_max[2])


@_lru_cache(maxsize=None)
def _mug_plain_source_color_material_names(task_name: str) -> tuple[str, ...]:
    variant_dir = _mug_variant_dir(task_name, _MUG_DEFAULT_VARIANT)
    root = _ET.parse(str(variant_dir / "model.xml")).getroot()
    asset_root = root.find("asset")
    if asset_root is None:
        return ()

    names: list[str] = []
    for material in asset_root.findall("material"):
        rgba_attr = material.get("rgba")
        material_name = material.get("name")
        if not rgba_attr or not material_name:
            continue
        rgba = _parse_float_list(rgba_attr)
        if len(rgba) < 3:
            continue
        if max(rgba[:3]) - min(rgba[:3]) < 0.04 and min(rgba[:3]) > 0.75:
            continue
        names.append(material_name)
    return tuple(names)


def _mug_plain_color_material_names(task_name: str, instance_index: int) -> tuple[str, ...]:
    prefix = _mug_instance_prefix(_MUG_DEFAULT_VARIANT, instance_index)
    return tuple(
        _dishrack_prefixed_name(prefix, name)
        for name in _mug_plain_source_color_material_names(task_name)
    )


@_lru_cache(maxsize=None)
def _water_bottle_variant_names() -> list[str]:
    variants = [
        path.name
        for path in _WATER_BOTTLE_TASK_ASSET_ROOT.iterdir()
        if path.is_dir() and (path / "model.xml").exists()
    ]
    variants.sort(
        key=lambda name: (
            0,
            int(name[len("bottle_") :]),
        )
        if name.startswith("bottle_") and name[len("bottle_") :].isdigit()
        else (1, name)
    )
    if not variants:
        raise FileNotFoundError(f"No water bottle variants found under {_WATER_BOTTLE_TASK_ASSET_ROOT}")
    return variants


def _water_bottle_canonical_variant_name(variant_name: str) -> str:
    if variant_name == "current":
        return _WATER_BOTTLE_DEFAULT_VARIANT
    if variant_name.isdigit():
        return f"bottle_{variant_name}"
    if variant_name.startswith("bottle") and variant_name[6:].isdigit():
        return f"bottle_{variant_name[6:]}"
    return variant_name


def _water_bottle_variant_dir(variant_name: str) -> _Path:
    variant_name = _water_bottle_canonical_variant_name(variant_name)
    path = _WATER_BOTTLE_TASK_ASSET_ROOT / variant_name
    if not (path / "model.xml").exists():
        raise FileNotFoundError(f"Missing water bottle variant model.xml: {path}")
    return path


def _water_bottle_body_name(index: int) -> str:
    return f"bottle_{index + 1}"


def _water_bottle_joint_name(index: int) -> str:
    return f"{_water_bottle_body_name(index)}_joint"


def _water_bottle_instance_prefix(variant_name: str, instance_index: int) -> str:
    return f"bottle_{instance_index}_{variant_name}"


@_lru_cache(maxsize=None)
def _water_bottle_compiled_raw_metadata(
    variant_name: str,
) -> tuple[float, float, float, float, float, float]:
    variant_dir = _water_bottle_variant_dir(variant_name)
    model = mujoco.MjModel.from_xml_path(str(variant_dir / "model.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    min_xyz, max_xyz = _dishrack_compiled_model_bounds(model, data)
    half_xy = 0.5 * (max_xyz[:2] - min_xyz[:2])
    center_xy = 0.5 * (min_xyz[:2] + max_xyz[:2])
    return (
        float(half_xy[0]),
        float(half_xy[1]),
        -float(center_xy[0]),
        -float(center_xy[1]),
        -float(min_xyz[2]),
        float(max_xyz[2] - min_xyz[2]),
    )


@_lru_cache(maxsize=None)
def _water_bottle_compiled_metadata(variant_name: str) -> tuple[float, float, float, float]:
    half_x, half_y, _offset_x, _offset_y, _offset_z, height = _water_bottle_compiled_raw_metadata(variant_name)
    return half_x, half_y, 0.0, height


@_lru_cache(maxsize=None)
def _water_bottle_compiled_anchor_offset(variant_name: str) -> tuple[float, float, float]:
    _half_x, _half_y, offset_x, offset_y, offset_z, _height = _water_bottle_compiled_raw_metadata(variant_name)
    return offset_x, offset_y, offset_z


@_lru_cache(maxsize=None)
def _water_bottle_flat_compiled_metadata(variant_name: str) -> tuple[float, float, float]:
    half_x, half_y, _offset_x, _offset_y, _offset_z, height = _water_bottle_compiled_raw_metadata(variant_name)
    return 0.5 * height, half_y, half_x


def _quat_rotate_vector(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)
    rotated = _quat_mul(_quat_mul(q, np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)), q_conj)
    return rotated[1:]


def _water_bottle_flat_quat(yaw: float) -> np.ndarray:
    flat_quat = _quat_from_axis_angle(np.array([0.0, 1.0, 0.0], dtype=np.float64), np.pi / 2.0)
    quat = _quat_mul(_quat_from_yaw(yaw), flat_quat)
    norm = float(np.linalg.norm(quat))
    if norm > 0.0:
        quat = quat / norm
    return quat


def _water_bottle_flat_yaw_from_quat(quat: np.ndarray) -> float:
    long_axis = _quat_rotate_vector(np.asarray(quat, dtype=np.float64), np.array([0.0, 0.0, 1.0]))
    return float(np.arctan2(long_axis[1], long_axis[0]))


def _water_bottle_flat_center_from_pose(
    *,
    pos: list[float] | tuple[float, ...],
    quat: list[float] | tuple[float, ...] | np.ndarray,
    variant_name: str,
    scale_factor: float,
) -> np.ndarray:
    yaw = _water_bottle_flat_yaw_from_quat(np.asarray(quat, dtype=np.float64))
    half_length, _half_radius, _vertical_radius = _water_bottle_flat_compiled_metadata(variant_name)
    offset = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64) * half_length * float(scale_factor)
    return np.asarray(pos[:2], dtype=np.float64) + offset


def _water_bottle_prepare_imported_geoms(body: _ET.Element) -> None:
    for parent in body.iter():
        for child in list(parent):
            if child.tag != "geom":
                continue

            name = child.get("name", "")
            child_class = child.get("class", "")
            mesh_name = child.get("mesh")
            if (
                child_class == "region"
                or name.startswith("reg_")
                or mesh_name is None
            ):
                parent.remove(child)
                continue

            is_visual = child.get("contype") == "0" and child.get("conaffinity") == "0"
            child.attrib.pop("class", None)
            child.attrib.pop("mass", None)
            child.set("density", "0")
            if is_visual:
                child.set("group", "2")
                child.set("contype", "0")
                child.set("conaffinity", "0")
                continue

            child.set("group", "3")
            child.set("rgba", "0 0 0 0")
            child.set("friction", "3.0 0.03 0.003")
            child.set("condim", "6")
            child.set("solref", "0.004 1")
            child.set("solimp", "0.998 0.998 0.001")
            child.set("priority", "1")


def _water_bottle_add_inertial(
    body: _ET.Element,
    *,
    variant_name: str,
    scale_factor: float,
) -> None:
    for child in list(body):
        if child.tag == "inertial":
            body.remove(child)

    half_x, half_y, offset_x, offset_y, offset_z, height = _water_bottle_compiled_raw_metadata(variant_name)
    scale_factor = float(scale_factor)
    mass = _WATER_BOTTLE_MASS_KG
    size_x = 2.0 * half_x * scale_factor
    size_y = 2.0 * half_y * scale_factor
    size_z = height * scale_factor
    inertia = [
        mass * (size_y * size_y + size_z * size_z) / 12.0,
        mass * (size_x * size_x + size_z * size_z) / 12.0,
        mass * (size_x * size_x + size_y * size_y) / 12.0,
    ]
    com_pos = [
        -offset_x * scale_factor,
        -offset_y * scale_factor,
        -offset_z * scale_factor + 0.5 * size_z,
    ]
    inertial = _ET.Element(
        "inertial",
        pos=_format_float_list(com_pos),
        mass=_format_float_list([mass]),
        diaginertia=_format_float_list(inertia),
    )
    body.insert(0, inertial)


def _water_bottle_build_object_block(
    *,
    variant_name: str,
    instance_index: int,
    scale_factor: float,
    object_name: str,
    joint_name: str,
) -> tuple[list[_ET.Element], _ET.Element]:
    variant_name = _water_bottle_canonical_variant_name(variant_name)
    variant_dir = _water_bottle_variant_dir(variant_name)
    root = _ET.parse(str(variant_dir / "model.xml")).getroot()
    object_body = _dishrack_find_object_body(root)
    offset = np.asarray(_water_bottle_compiled_anchor_offset(variant_name), dtype=np.float64) * float(scale_factor)

    prefix = _water_bottle_instance_prefix(variant_name, instance_index)
    mesh_map: dict[str, str] = {}
    texture_map: dict[str, str] = {}
    material_map: dict[str, str] = {}
    temp_asset = _ET.Element("asset")

    asset_root = root.find("asset")
    if asset_root is not None:
        asset_children = list(asset_root)
        for asset_child in asset_children:
            local_name = _dishrack_asset_local_name(asset_child)
            if asset_child.tag == "mesh":
                mesh_map[local_name] = _dishrack_prefixed_name(prefix, local_name)
            elif asset_child.tag == "texture":
                texture_map[local_name] = _dishrack_prefixed_name(prefix, local_name)
            elif asset_child.tag == "material":
                material_map[local_name] = _dishrack_prefixed_name(prefix, local_name)

        for asset_child in asset_children:
            cloned = copy.deepcopy(asset_child)
            local_name = _dishrack_asset_local_name(asset_child)
            if asset_child.tag == "mesh":
                cloned.set("name", mesh_map[local_name])
                _dishrack_absolutize_file_attr(cloned, variant_dir)
            elif asset_child.tag == "texture":
                cloned.set("name", texture_map[local_name])
                _dishrack_absolutize_file_attr(cloned, variant_dir)
            elif asset_child.tag == "material":
                cloned.set("name", material_map[local_name])
                texture_name = asset_child.get("texture")
                if texture_name and texture_name in texture_map:
                    cloned.set("texture", texture_map[texture_name])
            temp_asset.append(cloned)

    cloned_body = copy.deepcopy(object_body)
    _dishrack_remove_dynamic_joints(cloned_body)
    _dishrack_prefix_body_names(cloned_body, prefix)
    _dishrack_rewrite_asset_refs(
        cloned_body,
        mesh_map=mesh_map,
        material_map=material_map,
    )
    _water_bottle_prepare_imported_geoms(cloned_body)
    _dishrack_shift_body(cloned_body, offset)

    if abs(scale_factor - 1.0) > 1e-9:
        mesh_assets = {
            mesh.get("name", ""): mesh
            for mesh in temp_asset.findall("mesh")
            if mesh.get("name")
        }
        _scale_body_subtree(
            body=cloned_body,
            factor=scale_factor,
            target_name=joint_name,
            asset_elem=temp_asset,
            mesh_assets=mesh_assets,
        )

    _water_bottle_add_inertial(
        cloned_body,
        variant_name=variant_name,
        scale_factor=scale_factor,
    )

    x, y = _WATER_BOTTLE_WRAPPER_XY[instance_index]
    wrapper = _ET.Element(
        "body",
        name=object_name,
        pos=_format_float_list([x, y, _WATER_BOTTLE_WRAPPER_Z]),
    )
    _ET.SubElement(wrapper, "freejoint", name=joint_name)
    wrapper.append(cloned_body)
    return list(temp_asset), wrapper


def _build_water_bottle_scene_xml(
    *,
    bottle_variants: list[str] | tuple[str, ...],
    scale_states: dict[str, float],
    base_scene_xml: str | None,
    base_scene_dir: _Path | None,
) -> str:
    if not WaterBottleRandomizer.forced_min_bottle_count <= len(bottle_variants) <= WaterBottleRandomizer.max_bottle_count:
        raise ValueError(
            "put_bottles requires between "
            f"{WaterBottleRandomizer.forced_min_bottle_count} and "
            f"{WaterBottleRandomizer.max_bottle_count} bottle variants, got {len(bottle_variants)}"
        )

    base_text = base_scene_xml or _WATER_BOTTLE_BASE_SCENE_XML.read_text()
    base_dir = base_scene_dir or _WATER_BOTTLE_BASE_SCENE_XML.parent
    task_assets: list[_ET.Element] = []
    task_bodies: list[_ET.Element] = []

    for index, variant_name in enumerate(bottle_variants):
        body_name = _water_bottle_body_name(index)
        joint_name = _water_bottle_joint_name(index)
        variant_assets, bottle_body = _water_bottle_build_object_block(
            variant_name=variant_name,
            instance_index=index,
            scale_factor=float(scale_states.get(joint_name, 1.0)),
            object_name=body_name,
            joint_name=joint_name,
        )
        task_assets.extend(variant_assets)
        task_bodies.append(bottle_body)

    xml = base_text.replace("<!-- TASK_DEFAULTS_PLACEHOLDER -->", "")
    xml = xml.replace("<!-- TASK_ASSETS_PLACEHOLDER -->", _dishrack_serialize_elements(task_assets))
    xml = xml.replace("<!-- TASK_BODY_PLACEHOLDER -->", _dishrack_serialize_elements(task_bodies))
    bin_scale = float(scale_states.get("bin_joint", 1.0))
    if abs(bin_scale - 1.0) > 1e-9:
        xml = _apply_object_scales_to_scene_xml(xml, {"bin_joint": bin_scale})
    return _resolve_scene_xml_paths(xml, base_dir)


def _inhand_asset_base(category: str) -> _Path:
    return _OBJAVERSE_BASE if category in _OBJAVERSE_CATEGORIES else _LIGHTWHEEL_BASE


def _inhand_get_variants(category: str) -> list[_Path]:
    cat_dir = _inhand_asset_base(category) / category
    variants = []
    for d in sorted(cat_dir.iterdir()):
        if not d.is_dir():
            continue
        try:
            parsed = _inhand_parse_model_xml(d)
            if parsed["col_geoms"]:
                variants.append(d)
        except Exception:
            pass
    return variants


def _inhand_parse_model_xml(variant_dir: _Path) -> dict:
    root = _ET.parse(str(variant_dir / "model.xml")).getroot()
    asset_elem = root.find("asset")
    meshes, textures, materials = [], [], []
    if asset_elem is not None:
        for mesh in asset_elem.findall("mesh"):
            extra = {k: v for k, v in mesh.attrib.items() if k not in ("name", "file")}
            meshes.append({"name": mesh.get("name", ""), "file": mesh.get("file", ""), "extra": extra})
        for tex in asset_elem.findall("texture"):
            rel = tex.get("file", "")
            if rel:
                textures.append({"name": tex.get("name", ""), "file": rel, "type": tex.get("type", "2d")})
        for mat in asset_elem.findall("material"):
            materials.append({
                "name": mat.get("name", ""), "texture": mat.get("texture", ""),
                "rgba": mat.get("rgba", ""), "shininess": mat.get("shininess", ""),
                "specular": mat.get("specular", ""),
            })
    vis_geoms, col_geoms = [], []
    worldbody = root.find("worldbody")
    if worldbody is not None:
        for geom in worldbody.iter("geom"):
            cls = geom.get("class", "")
            if geom.get("name") == "reg_bbox":
                continue
            is_visual = cls == "visual" or (geom.get("contype") == "0" and cls != "region")
            if is_visual:
                vis_geoms.append({"mesh": geom.get("mesh", ""), "material": geom.get("material", "")})
            elif cls == "collision":
                col_geoms.append({"mesh": geom.get("mesh", "")})
    return {"meshes": meshes, "textures": textures, "materials": materials,
            "vis_geoms": vis_geoms, "col_geoms": col_geoms}


def _inhand_apply_scene_transforms(xml: str, options: Any = None) -> str:
    if options is None:
        return xml
    if not (options.clean or options.mocap or options.flexible_gripper):
        return xml

    from yam_sim.scene_xml import transform_scene_xml

    transformed_xml, _ = transform_scene_xml(xml, options=options)
    return transformed_xml


def _inhand_build_xml(
    category: str,
    variant_dir: _Path,
    x: float,
    y: float,
    z: float,
    yaw: float,
    *,
    scale_factor: float = 1.0,
) -> str:
    base_text = _BASE_SCENE_XML.read_text()
    parsed = _inhand_parse_model_xml(variant_dir)
    prefix = "obj"
    cat_scale = _CATEGORY_MESH_SCALE.get(category)

    lines_asset = [f"    <!-- Object: {category}/{variant_dir.name} -->"]
    for m in parsed["meshes"]:
        extra = {**m["extra"]}
        base_scale = extra.get("scale") or cat_scale
        if base_scale or abs(scale_factor - 1.0) > 1e-9:
            extra["scale"] = _scaled_mesh_attr(base_scale, scale_factor)
        extra_str = "".join(f' {k}="{v}"' for k, v in extra.items())
        lines_asset.append(f'    <mesh file="{variant_dir / m["file"]}" name="{prefix}_{m["name"]}"{extra_str}/>')
    for t in parsed["textures"]:
        lines_asset.append(f'    <texture file="{variant_dir / t["file"]}" name="{prefix}_{t["name"]}" type="{t["type"]}"/>')
    for mat in parsed["materials"]:
        attrs = f'name="{prefix}_{mat["name"]}"'
        if mat["texture"]: attrs += f' texture="{prefix}_{mat["texture"]}"'
        if mat["rgba"]:    attrs += f' rgba="{mat["rgba"]}"'
        if mat["shininess"]: attrs += f' shininess="{mat["shininess"]}"'
        if mat["specular"]:  attrs += f' specular="{mat["specular"]}"'
        lines_asset.append(f"    <material {attrs}/>")

    w, s = np.cos(yaw / 2), np.sin(yaw / 2)
    lines_body = [
        f"    <!-- Task object: {category}/{variant_dir.name} -->",
        f'    <body name="task_object" pos="{x:.4f} {y:.4f} {z:.4f}" quat="{w:.6f} 0 0 {s:.6f}">',
        f'      <freejoint name="task_object_joint"/>',
    ]
    for vg in parsed["vis_geoms"]:
        mesh_attr = f'mesh="{prefix}_{vg["mesh"]}"' if vg["mesh"] else ""
        mat_attr = f' material="{prefix}_{vg["material"]}"' if vg["material"] else ""
        lines_body.append(
            f'      <geom type="mesh" {mesh_attr}{mat_attr} contype="0" conaffinity="0" group="2"'
            f' density="0" solimp="0.998 0.998 0.001" solref="0.001 1"/>')
    for cg in parsed["col_geoms"]:
        mesh_attr = f'mesh="{prefix}_{cg["mesh"]}"' if cg["mesh"] else ""
        lines_body.append(
            f'      <geom type="mesh" {mesh_attr} group="3" density="{_OBJ_DENSITY}"'
            f' friction="3.0 0.3 0.05" solimp="0.95 0.99 0.001" solref="0.004 1"/>')
    lines_body.append("    </body>")

    result = base_text.replace("<!-- TASK_ASSETS_PLACEHOLDER -->", "\n".join(lines_asset))
    result = result.replace("<!-- TASK_BODY_PLACEHOLDER -->", "\n".join(lines_body))
    result = result.replace('file="i2rt_yam/', f'file="{_MODELS_DIR}/assets/i2rt_yam/')
    return result


class InHandTransferRandomizer(SceneRandomizer):
    """Randomizer for inhand_transfer: swaps the entire MuJoCo model on each reset.

    Unlike other randomizers that perturb existing object positions, this one
    picks a new kitchen tool category/variant and reloads the model from XML so
    the mesh assets change.  Call ``bind_env(env)`` after construction.
    """

    perturbations: list = []  # no PerturbRange — we handle placement ourselves

    def __init__(self, *, scene_variant: str = "hybrid", scene_xml_transform_options: Any = None) -> None:
        super().__init__()
        self._env_ref = None
        self._rng = np.random.default_rng()
        # Pre-cache variant lists so we don't re-scan on every reset.
        self._variants: dict[str, list[_Path]] = {}
        self._scene_variant = scene_variant
        self._scene_xml_transform_options = scene_xml_transform_options

    def bind_env(self, env: Any) -> None:
        self._env_ref = env

    def clone(self) -> "SceneRandomizer":
        cloned = type(self)(
            scene_variant=self._scene_variant,
            scene_xml_transform_options=self._scene_xml_transform_options,
        )
        return cloned

    def _get_variants(self, category: str) -> list[_Path]:
        if category not in self._variants:
            self._variants[category] = _inhand_get_variants(category)
        return self._variants[category]

    def randomize(
        self,
        model: Any,
        data: Any,
        seed: int | None = None,
        request: Any | None = None,
    ) -> RandomizationState:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        env = self._env_ref
        categories = _INHAND_CATEGORIES
        category = categories[int(self._rng.integers(0, len(categories)))]
        variants = self._get_variants(category)
        variant_dir = variants[int(self._rng.integers(0, len(variants)))]

        side = "left" if self._rng.random() < 0.5 else "right"
        x = float(self._rng.uniform(_X_MIN, _X_MAX))
        y = float(self._rng.uniform(_Y_LEFT_MIN, _Y_LEFT_MAX) if side == "left"
                  else self._rng.uniform(_Y_RIGHT_MIN, _Y_RIGHT_MAX))
        yaw = float(self._rng.uniform(-np.pi, np.pi))
        scale_factor = float(self._rng.uniform(0.95, 1.05))

        xml = _inhand_build_xml(
            category,
            variant_dir,
            x,
            y,
            _OBJ_Z,
            yaw,
            scale_factor=scale_factor,
        )
        xml = _inhand_apply_scene_transforms(xml, self._scene_xml_transform_options)

        if env is not None:
            from yam_sim.scene_variants import apply_scene_variant

            preserved_arm_state = env._get_reset_arm_state()
            env.reload_from_xml(xml)
            apply_scene_variant(env.model, self._scene_variant)
            mujoco.mj_resetData(env.model, env.data)
            env._set_qpos_from_state(preserved_arm_state)
            jnt_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "task_object_joint")
            qadr = env.model.jnt_qposadr[jnt_id]
            w, s = np.cos(yaw / 2), np.sin(yaw / 2)
            env.data.qpos[qadr:qadr + 3] = [x, y, _OBJ_Z]
            env.data.qpos[qadr + 3:qadr + 7] = [w, 0, 0, s]
            mujoco.mj_forward(env.model, env.data)
            # Store metadata for callers
            env._inhand_category = category
            env._inhand_variant = variant_dir.name
            env._inhand_side = side

        return RandomizationState(
            seed=seed or 0,
            object_states={"task_object_joint": {"pos": [x, y, _OBJ_Z], "quat": [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]}},
            scale_states={"task_object_joint": scale_factor},
            metadata={"category": category, "variant": variant_dir.name, "side": side},
        )

    def apply(self, model: Any, data: Any, state: RandomizationState) -> None:
        env = self._env_ref
        category = str(state.metadata.get("category", ""))
        variant_name = str(state.metadata.get("variant", ""))
        if env is None or not category or not variant_name:
            super().apply(model, data, state)
            return

        variant_dir = _inhand_asset_base(category) / category / variant_name
        pose = state.object_states.get("task_object_joint", {})
        pos = pose.get("pos", [_X_MIN, _Y_LEFT_MIN, _OBJ_Z])
        quat = np.asarray(pose.get("quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)
        yaw = _yaw_from_quat(quat)
        scale_factor = float(state.scale_states.get("task_object_joint", 1.0))

        xml = _inhand_build_xml(
            category,
            variant_dir,
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
            yaw,
            scale_factor=scale_factor,
        )
        xml = _inhand_apply_scene_transforms(xml, self._scene_xml_transform_options)

        from yam_sim.scene_variants import apply_scene_variant

        preserved_arm_state = env._get_reset_arm_state()
        env.reload_from_xml(xml)
        apply_scene_variant(env.model, self._scene_variant)
        mujoco.mj_resetData(env.model, env.data)
        env._set_qpos_from_state(preserved_arm_state)
        jnt_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "task_object_joint")
        qadr = env.model.jnt_qposadr[jnt_id]
        env.data.qpos[qadr:qadr + 3] = pos
        env.data.qpos[qadr + 3:qadr + 7] = quat
        mujoco.mj_forward(env.model, env.data)
        env._inhand_category = category
        env._inhand_variant = variant_name
        env._inhand_side = state.metadata.get("side")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_WATER_BOTTLE_RANDOMIZER = WaterBottleRandomizer()

TASK_RANDOMIZERS: dict[str, SceneRandomizer] = {
    "bottles":      BottlesRandomizer(),
    "put_bottles":  _WATER_BOTTLE_RANDOMIZER,
    "water_bottles": _WATER_BOTTLE_RANDOMIZER,
    "marker":       MarkerRandomizer(),
    "pour":         PourRandomizer(),
    "drawer":       DrawerRandomizer(),
    "dishrack":     DishRackRandomizer(),
    "blocks":       BlocksRandomizer(),
    "mug_tree":     MugTreeRandomizer(),
    "mug_flip":     MugFlipRandomizer(),
    "ball_sorting": BallSortingRandomizer(),
    "chess":        ChessRandomizer(),
    "chess2":       Chess2Randomizer(),
    "sweep":        SweepRandomizer(),
    # "empty": no free-jointed objects to randomize
}
