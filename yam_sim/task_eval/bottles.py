"""Bottle-in-bin task evaluator."""

from __future__ import annotations

import re
from typing import Any

import mujoco
import numpy as np

from yam_sim.task_eval.base import TaskEvalResult
from yam_sim.task_eval.debug_spec import EvalDebugSpec, PlotSpec, ThresholdSpec
from yam_sim.task_specs import SimTaskSpec


def _quat_to_rotmat_batch(quat_batch: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternions into rotation matrices."""

    quat_batch = np.asarray(quat_batch, dtype=np.float32)
    if quat_batch.ndim != 2 or quat_batch.shape[1] != 4:
        raise ValueError(f"Expected quaternion batch shape (B, 4), got {quat_batch.shape}")

    norm = np.linalg.norm(quat_batch, axis=1, keepdims=True)
    norm = np.where(norm > 0.0, norm, 1.0)
    q = quat_batch / norm
    w, x, y, z = q.T

    return np.stack(
        [
            np.stack([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)], axis=-1),
            np.stack([2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)], axis=-1),
            np.stack([2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)], axis=-1),
        ],
        axis=1,
    ).astype(np.float32)


def _body_child_ids(model: mujoco.MjModel, parent_body_id: int) -> list[int]:
    return [
        int(body_id)
        for body_id in range(model.nbody)
        if int(model.body_parentid[body_id]) == int(parent_body_id)
    ]


def _quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    return _quat_to_rotmat_batch(np.asarray(quat, dtype=np.float32)[None, :])[0]


def _subtree_body_transforms(
    model: mujoco.MjModel,
    root_body_id: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    transforms: dict[int, tuple[np.ndarray, np.ndarray]] = {
        int(root_body_id): (np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float32))
    }
    stack = [int(root_body_id)]
    while stack:
        body_id = stack.pop()
        parent_pos, parent_rot = transforms[body_id]
        for child_id in _body_child_ids(model, body_id):
            child_rot = _quat_to_rotmat(np.asarray(model.body_quat[child_id], dtype=np.float32))
            child_pos = np.asarray(model.body_pos[child_id], dtype=np.float32)
            transforms[child_id] = (
                parent_pos + parent_rot @ child_pos,
                parent_rot @ child_rot,
            )
            stack.append(child_id)
    return transforms


def _body_subtree_ids(model: mujoco.MjModel, root_body_id: int) -> set[int]:
    body_ids = {int(root_body_id)}
    stack = [int(root_body_id)]
    while stack:
        body_id = stack.pop()
        for child_id in _body_child_ids(model, body_id):
            body_ids.add(child_id)
            stack.append(child_id)
    return body_ids


class BottlesInBinEvaluator:
    """Score the bottles task from freejoint positions in the bin's local frame."""

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
        success_count: int = 2,
    ) -> None:
        self.model = model
        self.spec = spec
        self.success_count = int(success_count)
        if self.success_count < 1:
            raise ValueError(f"success_count must be >= 1, got {self.success_count}")

        self.bottle_names, self._bottle_qpos_addrs = self._resolve_bottle_qpos_addrs(model)
        self._bin_qpos_adr = self._resolve_joint_qpos_adr(model, "bin_joint")
        self._bin_apothem = self._resolve_bin_apothem(model)
        self._bin_bottom_z = self._resolve_bin_bottom_z(model)
        self._bin_top_z = self._resolve_bin_top_z(model)
        self._nworld = 1
        self._max_bottles_in_bin = np.zeros((1,), dtype=np.int32)
        self._ever_success = np.zeros((1,), dtype=bool)

    def reset(self, *, nworld: int = 1) -> None:
        self._nworld = int(nworld)
        self._max_bottles_in_bin = np.zeros((self._nworld,), dtype=np.int32)
        self._ever_success = np.zeros((self._nworld,), dtype=bool)

    def debug_spec(self) -> dict[str, Any] | None:
        return EvalDebugSpec(
            plots=[
                PlotSpec(
                    key="num_bottles_in_bin",
                    title="Bottles In Bin",
                    color="#4fc3f7",
                    thresholds=[
                        ThresholdSpec(
                            value=float(self.success_count),
                            label=f"success >= {self.success_count}",
                            direction="gt",
                        )
                    ],
                ),
                PlotSpec(
                    key="max_bottles_in_bin_so_far",
                    title="Max Bottles So Far",
                    color="#7dd3fc",
                    thresholds=[
                        ThresholdSpec(
                            value=float(self.success_count),
                            label=f"ever hit {self.success_count}",
                            direction="gt",
                        )
                    ],
                ),
                PlotSpec(
                    key="closest_radial_margin",
                    title="Closest Radial Margin (m)",
                    color="#f59e0b",
                    thresholds=[
                        ThresholdSpec(value=0.0, label="inside footprint", direction="gt")
                    ],
                ),
                PlotSpec(
                    key="closest_height_margin",
                    title="Closest Height Margin (m)",
                    color="#a78bfa",
                    thresholds=[
                        ThresholdSpec(value=0.0, label="inside height band", direction="gt")
                    ],
                ),
                PlotSpec(
                    key="reward",
                    title="Reward",
                    color="#22c55e",
                    thresholds=[
                        ThresholdSpec(value=1.0, label="full reward", direction="gt")
                    ],
                ),
                PlotSpec(
                    key="ever_success",
                    title="Ever Success",
                    color="#34d399",
                    kind="bool",
                ),
                PlotSpec(
                    key="success",
                    title="Current Success",
                    color="#fb7185",
                    kind="bool",
                ),
            ]
        ).to_dict()

    @staticmethod
    def _resolve_joint_qpos_adr(model: mujoco.MjModel, joint_name: str) -> int:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint {joint_name!r} not found in model")
        return int(model.jnt_qposadr[joint_id])

    @staticmethod
    def _resolve_bottle_qpos_addrs(model: mujoco.MjModel) -> tuple[list[str], np.ndarray]:
        entries: list[tuple[int, str, int]] = []
        for joint_id in range(model.njnt):
            name = model.jnt(joint_id).name
            if not name:
                continue
            match = re.fullmatch(r"bottle_(\d+)_joint", name)
            if match is None:
                continue
            bottle_index = int(match.group(1))
            entries.append(
                (
                    bottle_index,
                    f"bottle_{bottle_index}",
                    int(model.jnt_qposadr[joint_id]),
                )
            )

        if not entries:
            raise ValueError("No bottle_*_joint freejoints found in model")

        entries.sort(key=lambda item: item[0])
        bottle_names = [name for _, name, _ in entries]
        addrs = np.asarray([adr for _, _, adr in entries], dtype=np.int32)
        return bottle_names, addrs

    @staticmethod
    def _resolve_bin_apothem(model: mujoco.MjModel) -> float:
        wall_distances: list[float] = []
        for geom_id in range(model.ngeom):
            name = model.geom(geom_id).name
            if name and name.startswith("bin_wall_"):
                wall_distances.append(float(np.linalg.norm(model.geom_pos[geom_id][:2])))
        if not wall_distances:
            raise ValueError("No bin wall geoms found in model")
        return float(max(wall_distances))

    @staticmethod
    def _resolve_bin_bottom_z(model: mujoco.MjModel) -> float:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bin_bottom")
        if geom_id < 0:
            return 0.0
        return float(model.geom_pos[geom_id][2])

    @staticmethod
    def _resolve_bin_top_z(model: mujoco.MjModel) -> float:
        top_values: list[float] = []
        for geom_id in range(model.ngeom):
            name = model.geom(geom_id).name
            if not name or not name.startswith("bin_wall_"):
                continue
            geom_type = int(model.geom_type[geom_id])
            if geom_type != mujoco.mjtGeom.mjGEOM_BOX:
                continue
            top_values.append(float(model.geom_pos[geom_id][2] + model.geom_size[geom_id][2]))

        if not top_values:
            raise ValueError("No bin wall box geoms found in model")
        return float(max(top_values))

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        qpos_batch = np.asarray(qpos_batch, dtype=np.float32)
        if qpos_batch.ndim != 2 or qpos_batch.shape[1] != self.model.nq:
            raise ValueError(
                f"Expected qpos batch shape (B, {self.model.nq}), got {qpos_batch.shape}"
            )
        if qpos_batch.shape[0] != self._nworld:
            self.reset(nworld=qpos_batch.shape[0])

        bin_pos = qpos_batch[:, self._bin_qpos_adr : self._bin_qpos_adr + 3]
        bin_quat = qpos_batch[:, self._bin_qpos_adr + 3 : self._bin_qpos_adr + 7]

        bottle_positions = np.stack(
            [qpos_batch[:, adr : adr + 3] for adr in self._bottle_qpos_addrs],
            axis=1,
        )
        rel_world = bottle_positions - bin_pos[:, None, :]
        rot_world_from_local = _quat_to_rotmat_batch(bin_quat)
        rel_local = np.einsum("bij,bnj->bni", np.swapaxes(rot_world_from_local, 1, 2), rel_world)

        radial_xy = np.linalg.norm(rel_local[..., :2], axis=-1)
        local_z = rel_local[..., 2]
        radial_margin = self._bin_apothem - radial_xy
        lower_height_margin = local_z - self._bin_bottom_z
        upper_height_margin = self._bin_top_z - local_z
        height_margin = np.minimum(lower_height_margin, upper_height_margin)
        in_bin_mask = (
            (radial_margin >= 0.0)
            & (lower_height_margin >= 0.0)
            & (upper_height_margin >= 0.0)
        )

        num_bottles_in_bin = in_bin_mask.sum(axis=1).astype(np.int32)
        self._max_bottles_in_bin = np.maximum(self._max_bottles_in_bin, num_bottles_in_bin)
        reward = np.clip(
            num_bottles_in_bin.astype(np.float32) / float(self.success_count),
            0.0,
            1.0,
        )
        success = num_bottles_in_bin >= self.success_count
        self._ever_success |= success
        bottles_in_bin = [
            [name for name, in_bin in zip(self.bottle_names, world_mask) if bool(in_bin)]
            for world_mask in in_bin_mask
        ]

        metrics: dict[str, Any] = {
            "num_bottles_in_bin": num_bottles_in_bin,
            "max_bottles_in_bin_so_far": self._max_bottles_in_bin.copy(),
            "ever_success": self._ever_success.copy(),
            "closest_radial_margin": radial_margin.max(axis=1).astype(np.float32),
            "closest_height_margin": height_margin.max(axis=1).astype(np.float32),
            "bottle_in_bin_mask": in_bin_mask,
            "bottles_in_bin": bottles_in_bin,
            "bottle_names": list(self.bottle_names),
            "success_count": self.success_count,
            "bin_apothem": self._bin_apothem,
            "bin_bottom_z": self._bin_bottom_z,
            "bin_top_z": self._bin_top_z,
        }
        return TaskEvalResult(reward=reward, success=success, metrics=metrics)


class PutBottlesInBinEvaluator:
    """Score put-bottles by requiring every spawned water bottle in the bin."""

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
        horizontal_margin_m: float = 0.0,
        height_margin_m: float = 0.0,
        active_z_min: float = -0.5,
    ) -> None:
        self.model = model
        self.spec = spec
        self.horizontal_margin_m = float(horizontal_margin_m)
        self.height_margin_m = float(height_margin_m)
        # Bottles parked below this z (by the batched variable-count masking) are
        # treated as absent: excluded from both the success count and the
        # denominator. With no parked bottles (single-world / faithful eval) every
        # bottle is active, so scoring is unchanged.
        self.active_z_min = float(active_z_min)
        self.bottle_names, self._bottle_qpos_addrs = self._resolve_bottle_qpos_addrs(model)
        self._bottle_center_offsets = self._resolve_bottle_center_offsets(model, self.bottle_names)
        self._bin_qpos_adr = BottlesInBinEvaluator._resolve_joint_qpos_adr(model, "bin_joint")
        (
            self._bin_radius_x,
            self._bin_radius_z,
            self._bin_bottom_y,
            self._bin_top_y,
        ) = self._resolve_bin_local_bounds(model)
        self.success_count = len(self.bottle_names)
        self._nworld = 1
        self._max_bottles_in_bin = np.zeros((1,), dtype=np.int32)
        self._ever_success = np.zeros((1,), dtype=bool)

    def reset(self, *, nworld: int = 1) -> None:
        self._nworld = int(nworld)
        self._max_bottles_in_bin = np.zeros((self._nworld,), dtype=np.int32)
        self._ever_success = np.zeros((self._nworld,), dtype=bool)

    def debug_spec(self) -> dict[str, Any] | None:
        return EvalDebugSpec(
            plots=[
                PlotSpec(
                    key="num_bottles_in_bin",
                    title="Bottles In Bin",
                    color="#4fc3f7",
                    thresholds=[
                        ThresholdSpec(
                            value=float(max(self.success_count, 1)),
                            label="all spawned bottles",
                            direction="gt",
                        )
                    ],
                ),
                PlotSpec(
                    key="max_bottles_in_bin_so_far",
                    title="Max Bottles So Far",
                    color="#7dd3fc",
                    thresholds=[
                        ThresholdSpec(
                            value=float(max(self.success_count, 1)),
                            label="ever all spawned bottles",
                            direction="gt",
                        )
                    ],
                ),
                PlotSpec(
                    key="closest_radial_margin",
                    title="Closest Radial Margin (m)",
                    color="#f59e0b",
                    thresholds=[
                        ThresholdSpec(value=0.0, label="inside footprint", direction="gt")
                    ],
                ),
                PlotSpec(
                    key="closest_height_margin",
                    title="Closest Height Margin (m)",
                    color="#a78bfa",
                    thresholds=[
                        ThresholdSpec(value=0.0, label="inside height band", direction="gt")
                    ],
                ),
                PlotSpec(
                    key="reward",
                    title="Reward",
                    color="#22c55e",
                    thresholds=[
                        ThresholdSpec(value=1.0, label="full reward", direction="gt")
                    ],
                ),
                PlotSpec(
                    key="ever_success",
                    title="Ever Success",
                    color="#34d399",
                    kind="bool",
                ),
                PlotSpec(
                    key="success",
                    title="Current Success",
                    color="#fb7185",
                    kind="bool",
                ),
            ]
        ).to_dict()

    @staticmethod
    def _resolve_bottle_qpos_addrs(model: mujoco.MjModel) -> tuple[list[str], np.ndarray]:
        entries: list[tuple[int, str, int]] = []
        for joint_id in range(model.njnt):
            name = model.jnt(joint_id).name
            if not name:
                continue
            match = re.fullmatch(r"bottle_(\d+)_joint", name)
            if match is None:
                continue
            bottle_index = int(match.group(1))
            entries.append(
                (
                    bottle_index,
                    f"bottle_{bottle_index}",
                    int(model.jnt_qposadr[joint_id]),
                )
            )
        entries.sort(key=lambda item: item[0])
        return [name for _, name, _ in entries], np.asarray(
            [adr for _, _, adr in entries],
            dtype=np.int32,
        )

    @staticmethod
    def _resolve_bottle_center_offsets(
        model: mujoco.MjModel,
        bottle_names: list[str],
    ) -> np.ndarray:
        offsets: list[np.ndarray] = []
        for bottle_name in bottle_names:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bottle_name)
            if body_id < 0:
                raise ValueError(f"Bottle body {bottle_name!r} not found in model")
            transforms = _subtree_body_transforms(model, body_id)
            weighted_sum = np.zeros(3, dtype=np.float32)
            total_mass = 0.0
            for subtree_body_id, (local_pos, local_rot) in transforms.items():
                mass = float(model.body_mass[subtree_body_id])
                if mass <= 0.0:
                    continue
                inertial_pos = np.asarray(model.body_ipos[subtree_body_id], dtype=np.float32)
                weighted_sum += mass * (local_pos + local_rot @ inertial_pos)
                total_mass += mass
            if total_mass <= 0.0:
                offsets.append(np.zeros(3, dtype=np.float32))
            else:
                offsets.append(weighted_sum / total_mass)
        if not offsets:
            return np.zeros((0, 3), dtype=np.float32)
        return np.stack(offsets, axis=0).astype(np.float32)

    @staticmethod
    def _resolve_bin_local_bounds(model: mujoco.MjModel) -> tuple[float, float, float, float]:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bin_container")
        if body_id < 0:
            raise ValueError("Body 'bin_container' not found in model")
        body_ids = _body_subtree_ids(model, body_id)
        transforms = _subtree_body_transforms(model, body_id)
        vertices: list[np.ndarray] = []
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) not in body_ids:
                continue
            if int(model.geom_group[geom_id]) != 3:
                continue
            mesh_id = int(model.geom_dataid[geom_id])
            if mesh_id < 0:
                continue
            mesh_start = int(model.mesh_vertadr[mesh_id])
            mesh_count = int(model.mesh_vertnum[mesh_id])
            mesh_vertices = np.asarray(
                model.mesh_vert[mesh_start : mesh_start + mesh_count],
                dtype=np.float32,
            )
            geom_rot = _quat_to_rotmat(np.asarray(model.geom_quat[geom_id], dtype=np.float32))
            geom_pos = np.asarray(model.geom_pos[geom_id], dtype=np.float32)
            geom_body_id = int(model.geom_bodyid[geom_id])
            body_pos, body_rot = transforms[geom_body_id]
            geom_vertices = mesh_vertices @ geom_rot.T + geom_pos
            vertices.append(geom_vertices @ body_rot.T + body_pos)
        if not vertices:
            raise ValueError("No bin collision mesh vertices found in model")
        all_vertices = np.concatenate(vertices, axis=0)
        mins = all_vertices.min(axis=0)
        maxs = all_vertices.max(axis=0)
        radius_x = max(abs(float(mins[0])), abs(float(maxs[0])))
        radius_z = max(abs(float(mins[2])), abs(float(maxs[2])))
        return radius_x, radius_z, float(mins[1]), float(maxs[1])

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        qpos_batch = np.asarray(qpos_batch, dtype=np.float32)
        if qpos_batch.ndim != 2 or qpos_batch.shape[1] != self.model.nq:
            raise ValueError(
                f"Expected qpos batch shape (B, {self.model.nq}), got {qpos_batch.shape}"
            )
        if qpos_batch.shape[0] != self._nworld:
            self.reset(nworld=qpos_batch.shape[0])

        if not self.bottle_names:
            reward = np.zeros((qpos_batch.shape[0],), dtype=np.float32)
            success = np.zeros((qpos_batch.shape[0],), dtype=bool)
            metrics = {
                "num_bottles_in_bin": np.zeros((qpos_batch.shape[0],), dtype=np.int32),
                "num_active_bottles": np.zeros((qpos_batch.shape[0],), dtype=np.int32),
                "max_bottles_in_bin_so_far": self._max_bottles_in_bin.copy(),
                "ever_success": self._ever_success.copy(),
                "bottle_in_bin_mask": np.zeros((qpos_batch.shape[0], 0), dtype=bool),
                "bottles_in_bin": [[] for _ in range(qpos_batch.shape[0])],
                "bottle_names": [],
                "success_count": 0,
            }
            return TaskEvalResult(reward=reward, success=success, metrics=metrics)

        bin_pos = qpos_batch[:, self._bin_qpos_adr : self._bin_qpos_adr + 3]
        bin_quat = qpos_batch[:, self._bin_qpos_adr + 3 : self._bin_qpos_adr + 7]
        bottle_pos = np.stack(
            [qpos_batch[:, adr : adr + 3] for adr in self._bottle_qpos_addrs],
            axis=1,
        )
        bottle_quat = np.stack(
            [qpos_batch[:, adr + 3 : adr + 7] for adr in self._bottle_qpos_addrs],
            axis=1,
        )
        bottle_rot = _quat_to_rotmat_batch(bottle_quat.reshape(-1, 4)).reshape(
            qpos_batch.shape[0],
            len(self.bottle_names),
            3,
            3,
        )
        bottle_centers = bottle_pos + np.einsum(
            "bnij,nj->bni",
            bottle_rot,
            self._bottle_center_offsets,
        )

        rot_world_from_bin = _quat_to_rotmat_batch(bin_quat)
        rel_world = bottle_centers - bin_pos[:, None, :]
        rel_bin = np.einsum("bij,bnj->bni", np.swapaxes(rot_world_from_bin, 1, 2), rel_world)

        radius_x = max(self._bin_radius_x - self.horizontal_margin_m, 1.0e-6)
        radius_z = max(self._bin_radius_z - self.horizontal_margin_m, 1.0e-6)
        normalized_radial = np.sqrt(
            (rel_bin[..., 0] / radius_x) ** 2
            + (rel_bin[..., 2] / radius_z) ** 2
        )
        radial_margin = 1.0 - normalized_radial
        lower_height_margin = rel_bin[..., 1] - (self._bin_bottom_y - self.height_margin_m)
        upper_height_margin = (self._bin_top_y + self.height_margin_m) - rel_bin[..., 1]
        height_margin = np.minimum(lower_height_margin, upper_height_margin)
        active_mask = bottle_centers[..., 2] > self.active_z_min
        in_bin_mask = (
            active_mask
            & (radial_margin >= 0.0)
            & (lower_height_margin >= 0.0)
            & (upper_height_margin >= 0.0)
        )

        num_bottles_in_bin = in_bin_mask.sum(axis=1).astype(np.int32)
        active_count = active_mask.sum(axis=1).astype(np.int32)
        self._max_bottles_in_bin = np.maximum(self._max_bottles_in_bin, num_bottles_in_bin)
        reward = num_bottles_in_bin.astype(np.float32) / np.maximum(active_count, 1).astype(np.float32)
        success = (active_count > 0) & (num_bottles_in_bin == active_count)
        self._ever_success |= success
        bottles_in_bin = [
            [name for name, in_bin in zip(self.bottle_names, world_mask) if bool(in_bin)]
            for world_mask in in_bin_mask
        ]

        metrics: dict[str, Any] = {
            "num_bottles_in_bin": num_bottles_in_bin,
            "num_active_bottles": active_count,
            "max_bottles_in_bin_so_far": self._max_bottles_in_bin.copy(),
            "ever_success": self._ever_success.copy(),
            "closest_radial_margin": radial_margin.max(axis=1).astype(np.float32),
            "closest_height_margin": height_margin.max(axis=1).astype(np.float32),
            "bottle_in_bin_mask": in_bin_mask,
            "bottles_in_bin": bottles_in_bin,
            "bottle_names": list(self.bottle_names),
            "success_count": len(self.bottle_names),
            "bin_radius_x": self._bin_radius_x,
            "bin_radius_z": self._bin_radius_z,
            "bin_bottom_y": self._bin_bottom_y,
            "bin_top_y": self._bin_top_y,
            "bottle_center_offsets": self._bottle_center_offsets.copy(),
        }
        return TaskEvalResult(reward=reward, success=success, metrics=metrics)
