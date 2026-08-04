"""Sweep-away task evaluator."""

from __future__ import annotations

import re
from typing import Any

import mujoco
import numpy as np

from yam_sim.task_eval.base import TaskEvalResult
from yam_sim.task_eval.debug_spec import EvalDebugSpec, PlotSpec, ThresholdSpec
from yam_sim.task_specs import SimTaskSpec


def _quat_to_rotmat_batch(quat_batch: np.ndarray) -> np.ndarray:
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


class SweepAwayEvaluator:
    """Score sweep success from active paper scraps ending inside the bin.

    The sweep randomizer activates the first 2-4 ``trash_*`` joints and parks the
    remaining joints below the scene.  When reset metadata is available, the
    evaluator uses the randomizer's active trash list as the episode's fixed
    active set.  Without metadata it falls back to z-height inference for
    ad-hoc qpos evaluation.
    """

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
        table_bounds: tuple[float, float, float, float] | None = None,
        active_z_min: float = -0.25,
        bin_radial_margin_m: float = 0.005,
        bin_height_margin_m: float = 0.005,
    ) -> None:
        self.model = model
        self.spec = spec
        self.table_bounds = self._resolve_table_bounds(model, table_bounds)
        self.active_z_min = float(active_z_min)
        self.bin_radial_margin_m = float(bin_radial_margin_m)
        self.bin_height_margin_m = float(bin_height_margin_m)

        self.trash_names, self._trash_qpos_addrs = self._resolve_trash_qpos_addrs(model)
        self._bin_qpos_adr = self._resolve_joint_qpos_adr(model, "bin_joint")
        self._bin_radius, self._bin_bottom_y, self._bin_top_y = self._resolve_bin_local_bounds(model)
        self._nworld = 1
        self._active_trash_mask = np.zeros((1, len(self.trash_names)), dtype=bool)
        self._fixed_active_trash_mask = False
        self._max_scraps_in_bin = np.zeros((1,), dtype=np.int32)
        self._ever_success = np.zeros((1,), dtype=bool)

    def reset(self, *, nworld: int = 1) -> None:
        self._nworld = int(nworld)
        self._active_trash_mask = np.zeros((self._nworld, len(self.trash_names)), dtype=bool)
        self._fixed_active_trash_mask = False
        self._max_scraps_in_bin = np.zeros((self._nworld,), dtype=np.int32)
        self._ever_success = np.zeros((self._nworld,), dtype=bool)

    @staticmethod
    def _normalize_trash_joint_name(joint_name: str) -> str:
        name = str(joint_name)
        return name.removesuffix("_jnt")

    def set_active_trash_joints(self, active_trash_joints_by_world: Any) -> None:
        """Fix the episode active set from sweep randomization metadata.

        ``active_trash_joints_by_world`` accepts either a single list of joint
        names for one world or a list of per-world joint-name lists.
        """

        if self._nworld == 1 and (
            not isinstance(active_trash_joints_by_world, (list, tuple))
            or (
                active_trash_joints_by_world
                and all(isinstance(item, str) for item in active_trash_joints_by_world)
            )
        ):
            per_world = [active_trash_joints_by_world]
        else:
            per_world = list(active_trash_joints_by_world)

        if len(per_world) != self._nworld:
            raise ValueError(
                f"Expected active trash metadata for {self._nworld} world(s), got {len(per_world)}"
            )

        name_to_index = {name: idx for idx, name in enumerate(self.trash_names)}
        mask = np.zeros((self._nworld, len(self.trash_names)), dtype=bool)
        for world_idx, world_joints in enumerate(per_world):
            if world_joints is None:
                continue
            for joint_name in world_joints:
                trash_name = self._normalize_trash_joint_name(joint_name)
                if trash_name not in name_to_index:
                    raise ValueError(f"Unknown sweep trash joint {joint_name!r}")
                mask[world_idx, name_to_index[trash_name]] = True

        self._active_trash_mask = mask
        self._fixed_active_trash_mask = True

    @staticmethod
    def _resolve_table_bounds(
        model: mujoco.MjModel,
        table_bounds: tuple[float, float, float, float] | None,
    ) -> tuple[float, float, float, float]:
        if table_bounds is not None:
            resolved = tuple(float(value) for value in table_bounds)
            if len(resolved) != 4:
                raise ValueError(f"table_bounds must have 4 entries, got {resolved!r}")
            return resolved

        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_plane")
        if geom_id >= 0:
            pos = np.asarray(model.geom_pos[geom_id], dtype=np.float64)
            size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
            return (
                float(pos[0] - size[0]),
                float(pos[0] + size[0]),
                float(pos[1] - size[1]),
                float(pos[1] + size[1]),
            )

        return (0.3025, 0.8975, -0.65, 0.65)

    def debug_spec(self) -> dict[str, Any] | None:
        return EvalDebugSpec(
            plots=[
                PlotSpec(
                    key="num_scraps_in_bin",
                    title="Scraps In Bin",
                    color="#4fc3f7",
                ),
                PlotSpec(
                    key="max_scraps_in_bin_so_far",
                    title="Max In Bin So Far",
                    color="#7dd3fc",
                ),
                PlotSpec(
                    key="num_active_scraps",
                    title="Active Scraps",
                    color="#f59e0b",
                ),
                PlotSpec(
                    key="reward",
                    title="Reward",
                    color="#22c55e",
                    thresholds=[
                        ThresholdSpec(value=1.0, label="all active scraps in bin", direction="gt")
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
    def _resolve_trash_qpos_addrs(model: mujoco.MjModel) -> tuple[list[str], np.ndarray]:
        entries: list[tuple[int, str, int]] = []
        for joint_id in range(model.njnt):
            joint_name = model.jnt(joint_id).name
            if not joint_name:
                continue
            match = re.fullmatch(r"trash_(\d+)_jnt", joint_name)
            if match is None:
                continue
            trash_index = int(match.group(1))
            entries.append(
                (
                    trash_index,
                    f"trash_{trash_index}",
                    int(model.jnt_qposadr[joint_id]),
                )
            )

        if not entries:
            raise ValueError("No trash_*_jnt freejoints found in model")

        entries.sort(key=lambda item: item[0])
        names = [name for _, name, _ in entries]
        addrs = np.asarray([addr for _, _, addr in entries], dtype=np.int32)
        return names, addrs

    @staticmethod
    def _resolve_bin_local_bounds(model: mujoco.MjModel) -> tuple[float, float, float]:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "gc_inner")
        if geom_id >= 0 and int(model.geom_type[geom_id]) == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = int(model.geom_dataid[geom_id])
            start = int(model.mesh_vertadr[mesh_id])
            count = int(model.mesh_vertnum[mesh_id])
            verts = np.asarray(model.mesh_vert[start : start + count], dtype=np.float64)
            geom_rot = np.empty(9, dtype=np.float64)
            mujoco.mju_quat2Mat(geom_rot, np.asarray(model.geom_quat[geom_id], dtype=np.float64))
            geom_rot = geom_rot.reshape(3, 3)
            local_points = verts @ geom_rot.T + np.asarray(model.geom_pos[geom_id], dtype=np.float64)
            radial = np.linalg.norm(local_points[:, [0, 2]], axis=1)
            return (
                float(radial.max()),
                float(local_points[:, 1].min()),
                float(local_points[:, 1].max()),
            )

        return 0.087, -0.084, 0.118

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        qpos_batch = np.asarray(qpos_batch, dtype=np.float32)
        if qpos_batch.ndim != 2 or qpos_batch.shape[1] != self.model.nq:
            raise ValueError(
                f"Expected qpos batch shape (B, {self.model.nq}), got {qpos_batch.shape}"
            )
        if qpos_batch.shape[0] != self._nworld:
            self.reset(nworld=qpos_batch.shape[0])

        trash_positions = np.stack(
            [qpos_batch[:, addr : addr + 3] for addr in self._trash_qpos_addrs],
            axis=1,
        )
        if not self._fixed_active_trash_mask:
            self._active_trash_mask |= trash_positions[..., 2] > self.active_z_min

        bin_pos = qpos_batch[:, self._bin_qpos_adr : self._bin_qpos_adr + 3]
        bin_quat = qpos_batch[:, self._bin_qpos_adr + 3 : self._bin_qpos_adr + 7]
        rel_world = trash_positions - bin_pos[:, None, :]
        rot_world_from_bin = _quat_to_rotmat_batch(bin_quat)
        rel_bin = np.einsum("bij,bnj->bni", np.swapaxes(rot_world_from_bin, 1, 2), rel_world)

        bin_radial = np.linalg.norm(rel_bin[..., [0, 2]], axis=-1)
        bin_height = rel_bin[..., 1]
        bin_radial_margin = self._bin_radius + self.bin_radial_margin_m - bin_radial
        bin_lower_margin = bin_height - (self._bin_bottom_y - self.bin_height_margin_m)
        bin_upper_margin = (self._bin_top_y + self.bin_height_margin_m) - bin_height
        bin_height_margin = np.minimum(bin_lower_margin, bin_upper_margin)
        in_bin_mask = (
            self._active_trash_mask
            & (bin_radial_margin >= 0.0)
            & (bin_lower_margin >= 0.0)
            & (bin_upper_margin >= 0.0)
        )

        x_min, x_max, y_min, y_max = self.table_bounds
        inside_table_footprint_mask = (
            (trash_positions[..., 0] >= x_min)
            & (trash_positions[..., 0] <= x_max)
            & (trash_positions[..., 1] >= y_min)
            & (trash_positions[..., 1] <= y_max)
        )
        outside_table_footprint_mask = self._active_trash_mask & ~inside_table_footprint_mask
        active_count = self._active_trash_mask.sum(axis=1).astype(np.int32)
        num_scraps_in_bin = in_bin_mask.sum(axis=1).astype(np.int32)
        num_scraps_outside_table_footprint = outside_table_footprint_mask.sum(axis=1).astype(np.int32)
        num_scraps_inside_table_footprint = (
            self._active_trash_mask & inside_table_footprint_mask
        ).sum(axis=1).astype(np.int32)

        denom = np.maximum(active_count, 1).astype(np.float32)
        reward = np.where(active_count > 0, num_scraps_in_bin.astype(np.float32) / denom, 0.0)
        success = (active_count > 0) & (num_scraps_in_bin == active_count)
        self._max_scraps_in_bin = np.maximum(self._max_scraps_in_bin, num_scraps_in_bin)
        self._ever_success |= success

        scraps_in_bin = [
            [name for name, in_bin in zip(self.trash_names, world_mask) if bool(in_bin)]
            for world_mask in in_bin_mask
        ]
        outside_table_footprint_scraps = [
            [name for name, outside_table in zip(self.trash_names, world_mask) if bool(outside_table)]
            for world_mask in outside_table_footprint_mask
        ]
        active_scraps = [
            [name for name, active in zip(self.trash_names, world_mask) if bool(active)]
            for world_mask in self._active_trash_mask
        ]

        metrics: dict[str, Any] = {
            "num_scraps_in_bin": num_scraps_in_bin,
            "num_scraps_outside_table_footprint": num_scraps_outside_table_footprint,
            "num_scraps_inside_table_footprint": num_scraps_inside_table_footprint,
            "num_active_scraps": active_count,
            "max_scraps_in_bin_so_far": self._max_scraps_in_bin.copy(),
            "ever_success": self._ever_success.copy(),
            "trash_in_bin_mask": in_bin_mask,
            "trash_outside_table_footprint_mask": outside_table_footprint_mask,
            "trash_active_mask": self._active_trash_mask.copy(),
            "fixed_active_trash_mask": self._fixed_active_trash_mask,
            "scraps_in_bin": scraps_in_bin,
            "outside_table_footprint_scraps": outside_table_footprint_scraps,
            "active_scraps": active_scraps,
            "closest_bin_radial_margin": bin_radial_margin.max(axis=1).astype(np.float32),
            "closest_bin_height_margin": bin_height_margin.max(axis=1).astype(np.float32),
            "trash_names": list(self.trash_names),
            "table_bounds": self.table_bounds,
            "active_z_min": self.active_z_min,
            "bin_radius": self._bin_radius,
            "bin_bottom_y": self._bin_bottom_y,
            "bin_top_y": self._bin_top_y,
        }
        return TaskEvalResult(reward=reward, success=success, metrics=metrics)
