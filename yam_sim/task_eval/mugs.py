"""Mug task evaluators."""

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
            np.stack(
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - z * w),
                    2.0 * (x * z + y * w),
                ],
                axis=-1,
            ),
            np.stack(
                [
                    2.0 * (x * y + z * w),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - x * w),
                ],
                axis=-1,
            ),
            np.stack(
                [
                    2.0 * (x * z - y * w),
                    2.0 * (y * z + x * w),
                    1.0 - 2.0 * (x * x + y * y),
                ],
                axis=-1,
            ),
        ],
        axis=1,
    ).astype(np.float32)


class MugFlipUprightEvaluator:
    """Score mug flip by requiring every spawned mug to be right-side-up."""

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
        upright_z_min: float = 0.75,
        active_z_min: float = -0.5,
    ) -> None:
        self.model = model
        self.spec = spec
        self.upright_z_min = float(upright_z_min)
        # Mugs parked below this z (batched variable-count masking) are absent:
        # excluded from upright count and denominator. No parked mugs => unchanged.
        self.active_z_min = float(active_z_min)
        self.mug_names, self._mug_qpos_addrs = self._resolve_mug_qpos_addrs(model)
        self.success_count = len(self.mug_names)
        self._nworld = 1
        self._max_upright_mugs = np.zeros((1,), dtype=np.int32)
        self._ever_success = np.zeros((1,), dtype=bool)

    def reset(self, *, nworld: int = 1) -> None:
        self._nworld = int(nworld)
        self._max_upright_mugs = np.zeros((self._nworld,), dtype=np.int32)
        self._ever_success = np.zeros((self._nworld,), dtype=bool)

    def debug_spec(self) -> dict[str, Any] | None:
        return EvalDebugSpec(
            plots=[
                PlotSpec(
                    key="num_upright_mugs",
                    title="Upright Mugs",
                    color="#4fc3f7",
                    thresholds=[
                        ThresholdSpec(
                            value=float(max(self.success_count, 1)),
                            label="all active mugs",
                            direction="gt",
                        )
                    ],
                ),
                PlotSpec(
                    key="max_upright_mugs_so_far",
                    title="Max Upright So Far",
                    color="#7dd3fc",
                    thresholds=[
                        ThresholdSpec(
                            value=float(max(self.success_count, 1)),
                            label="ever all active mugs",
                            direction="gt",
                        )
                    ],
                ),
                PlotSpec(
                    key="min_mug_upright_z",
                    title="Minimum Mug Upright Z",
                    color="#f59e0b",
                    thresholds=[
                        ThresholdSpec(
                            value=self.upright_z_min,
                            label="right-side-up",
                            direction="gt",
                        )
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
    def _resolve_mug_qpos_addrs(model: mujoco.MjModel) -> tuple[list[str], np.ndarray]:
        entries: list[tuple[int, str, int]] = []
        for joint_id in range(model.njnt):
            name = model.jnt(joint_id).name
            if not name:
                continue
            match = re.fullmatch(r"mug_(\d+)_jnt", name)
            if match is None:
                continue
            mug_index = int(match.group(1))
            entries.append((mug_index, f"mug_{mug_index}", int(model.jnt_qposadr[joint_id])))

        entries.sort(key=lambda item: item[0])
        return [name for _, name, _ in entries], np.asarray(
            [adr for _, _, adr in entries],
            dtype=np.int32,
        )

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        qpos_batch = np.asarray(qpos_batch, dtype=np.float32)
        if qpos_batch.ndim != 2 or qpos_batch.shape[1] != self.model.nq:
            raise ValueError(
                f"Expected qpos batch shape (B, {self.model.nq}), got {qpos_batch.shape}"
            )
        if qpos_batch.shape[0] != self._nworld:
            self.reset(nworld=qpos_batch.shape[0])

        if not self.mug_names:
            reward = np.zeros((qpos_batch.shape[0],), dtype=np.float32)
            success = np.zeros((qpos_batch.shape[0],), dtype=bool)
            metrics = {
                "num_upright_mugs": np.zeros((qpos_batch.shape[0],), dtype=np.int32),
                "num_active_mugs": np.zeros((qpos_batch.shape[0],), dtype=np.int32),
                "max_upright_mugs_so_far": self._max_upright_mugs.copy(),
                "ever_success": self._ever_success.copy(),
                "mug_upright_mask": np.zeros((qpos_batch.shape[0], 0), dtype=bool),
                "mug_upright_z": np.zeros((qpos_batch.shape[0], 0), dtype=np.float32),
                "min_mug_upright_z": np.full(
                    (qpos_batch.shape[0],),
                    np.nan,
                    dtype=np.float32,
                ),
                "upright_mugs": [[] for _ in range(qpos_batch.shape[0])],
                "mug_names": [],
                "success_count": 0,
                "upright_z_min": self.upright_z_min,
            }
            return TaskEvalResult(reward=reward, success=success, metrics=metrics)

        mug_quat = np.stack(
            [qpos_batch[:, adr + 3 : adr + 7] for adr in self._mug_qpos_addrs],
            axis=1,
        )
        mug_rot = _quat_to_rotmat_batch(mug_quat.reshape(-1, 4)).reshape(
            qpos_batch.shape[0],
            len(self.mug_names),
            3,
            3,
        )
        mug_pos_z = np.stack(
            [qpos_batch[:, adr + 2] for adr in self._mug_qpos_addrs],
            axis=1,
        )
        active_mask = mug_pos_z > self.active_z_min
        local_z_world = mug_rot[..., :, 2]
        upright_z = local_z_world[..., 2]
        upright_mask = (upright_z >= self.upright_z_min) & active_mask

        num_upright = upright_mask.sum(axis=1).astype(np.int32)
        active_count = active_mask.sum(axis=1).astype(np.int32)
        self._max_upright_mugs = np.maximum(self._max_upright_mugs, num_upright)
        reward = num_upright.astype(np.float32) / np.maximum(active_count, 1).astype(np.float32)
        success = (active_count > 0) & (num_upright == active_count)
        self._ever_success |= success
        upright_mugs = [
            [name for name, is_upright in zip(self.mug_names, world_mask) if bool(is_upright)]
            for world_mask in upright_mask
        ]

        metrics: dict[str, Any] = {
            "num_upright_mugs": num_upright,
            "num_active_mugs": active_count,
            "max_upright_mugs_so_far": self._max_upright_mugs.copy(),
            "ever_success": self._ever_success.copy(),
            "mug_upright_mask": upright_mask,
            "mug_upright_z": upright_z.astype(np.float32),
            "min_mug_upright_z": upright_z.min(axis=1).astype(np.float32),
            "upright_mugs": upright_mugs,
            "mug_names": list(self.mug_names),
            "success_count": len(self.mug_names),
            "upright_z_min": self.upright_z_min,
        }
        return TaskEvalResult(reward=reward, success=success, metrics=metrics)


class HangMugOnRackEvaluator:
    """Score the mug-tree task by counting mugs hung on the rack.

    A mug counts as *hung* when, at the evaluated state, it: (1) makes at least
    ``min_tree_contacts`` contact(s) with the static mug-tree body, (2) has its
    center at or below the top of the tree (``max_center_z``, auto-derived from
    the tree geometry), and (3) -- when ``require_off_table`` -- is not touching
    the table/floor.

    A mug that is being *held* by a gripper is never counted, even if it touches
    the tree: ``disqualify_held`` rejects any mug grasped between both fingers of
    one gripper (contact with both the ``lf`` and ``rf`` finger of the same arm).
    A single incidental finger brush does not disqualify -- only a two-finger
    grasp does -- so a genuinely-hung mug grazed by a retracting fingertip still
    counts.

    Calibrated against human gradings (11/11 exact agreement). Notes on why each
    condition matters: a single tree contact suffices (>= 2 wrongly drops mugs in
    crowded states where mug-mug contacts steal direct mug-tree contacts -- e.g.
    a genuine 3/3 where the top mug momentarily rests against a neighbour). The
    ``center <= tree-top`` bound is the load-bearing guard: it rejects a mug
    *balanced on top of* the rack (center above every peg) which touches the tree
    but is not hung -- a properly hung mug always dangles below the peg it hangs
    on, so its center sits below the topmost peg. The held-check rejects a mug the
    arm is still holding up next to a branch as the episode ends.

    Contacts are computed with a private ``MjData`` via ``mj_kinematics`` +
    ``mj_collision`` on each world's qpos, so this works identically for the
    batched live harness and for offline scoring of saved qpos trajectories.
    Per-step reward is the fraction of mugs currently hung; ``success`` requires
    all mugs hung. ``ever_success`` / ``max_hung_mugs_so_far`` are tracked across
    a rollout like the other evaluators.
    """

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
        min_tree_contacts: int = 1,
        require_off_table: bool = True,
        disqualify_held: bool = True,
        max_center_z: float | None = None,
        top_margin: float = 0.0,
        tree_body: str = "mug_tree",
        table_geoms: tuple[str, ...] = ("table_plane", "floor_collision"),
        active_z_min: float = -0.5,
    ) -> None:
        self.model = model
        self.spec = spec
        self.min_tree_contacts = int(min_tree_contacts)
        self.require_off_table = bool(require_off_table)
        self.disqualify_held = bool(disqualify_held)
        # Mugs parked below this z (batched variable-count masking) are absent:
        # excluded from hung count and denominator. No parked mugs => unchanged.
        self.active_z_min = float(active_z_min)
        self._data = mujoco.MjData(model)

        self.mug_names, self._mug_body_ids = self._resolve_mug_bodies(model)
        self.success_count = len(self.mug_names)
        self._body_to_mug = {bid: i for i, bid in enumerate(self._mug_body_ids)}

        self._tree_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tree_body)
        if self._tree_body_id < 0:
            raise ValueError(f"tree body {tree_body!r} not found in model")
        self._table_geom_ids = set()
        for name in table_geoms:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self._table_geom_ids.add(int(gid))

        # Gripper finger bodies grouped by arm side, so we can detect a two-finger
        # grasp (mug held between the lf and rf fingers of the same gripper).
        self._finger_bodies: dict[str, dict[str, set[int]]] = {
            "left": {"lf": set(), "rf": set()},
            "right": {"lf": set(), "rf": set()},
        }
        # Broader set of gripper/hand bodies (fingers + the wrist link they hang
        # off), for an "is the gripper touching this mug at all" signal used by
        # offline hang-event timing to tell a self-supported hang from one the arm
        # is still holding / pushing against the tree.
        self._gripper_bodies: set[int] = set()
        for body_id in range(model.nbody):
            fm = re.match(r"(left|right)_(lf|rf)_(down|rot)$", model.body(body_id).name or "")
            if fm is not None:
                self._finger_bodies[fm.group(1)][fm.group(2)].add(int(body_id))
            if re.search(r"(link_6|_lf_|_rf_|finger)", model.body(body_id).name or ""):
                self._gripper_bodies.add(int(body_id))

        # Top of the rack = highest tree-geom center (pegs). The tree is static,
        # so forward kinematics from the default qpos already places its geoms.
        if max_center_z is not None:
            self.max_center_z = float(max_center_z)
        else:
            mujoco.mj_kinematics(self.model, self._data)
            tree_geom_z = [
                float(self._data.geom_xpos[g][2])
                for g in range(model.ngeom)
                if int(model.geom_bodyid[g]) == self._tree_body_id
            ]
            self.max_center_z = (max(tree_geom_z) if tree_geom_z else float("inf")) + float(top_margin)

        self._nworld = 1
        self._max_hung = np.zeros((1,), dtype=np.int32)
        self._ever_success = np.zeros((1,), dtype=bool)

    def reset(self, *, nworld: int = 1) -> None:
        self._nworld = int(nworld)
        self._max_hung = np.zeros((self._nworld,), dtype=np.int32)
        self._ever_success = np.zeros((self._nworld,), dtype=bool)

    @staticmethod
    def _resolve_mug_bodies(model: mujoco.MjModel) -> tuple[list[str], np.ndarray]:
        entries: list[tuple[int, str, int]] = []
        for body_id in range(model.nbody):
            name = model.body(body_id).name
            match = re.fullmatch(r"mug_(\d+)", name or "")
            if match is None:
                continue
            entries.append((int(match.group(1)), name, body_id))
        entries.sort(key=lambda item: item[0])
        return [name for _, name, _ in entries], np.asarray(
            [bid for _, _, bid in entries], dtype=np.int32
        )

    def _hung_mask_one(
        self, qpos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (n_tree_contacts, table_contact, held, gripper_contact, center_z)."""
        data = self._data
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        # Kinematics + collision are sufficient for contacts and far cheaper than
        # a full forward dynamics solve.
        mujoco.mj_kinematics(self.model, data)
        mujoco.mj_collision(self.model, data)

        n_mugs = len(self.mug_names)
        n_tree = np.zeros((n_mugs,), dtype=np.int32)
        on_table = np.zeros((n_mugs,), dtype=bool)
        gripper_contact = np.zeros((n_mugs,), dtype=bool)
        # finger_touch[mug][side] = [lf_touched, rf_touched]
        finger_touch = [
            {"left": [False, False], "right": [False, False]} for _ in range(n_mugs)
        ]
        center_z = np.array(
            [float(data.xpos[bid][2]) for bid in self._mug_body_ids], dtype=np.float32
        )
        body_of_geom = self.model.geom_bodyid
        for c in range(data.ncon):
            con = data.contact[c]
            b1 = int(body_of_geom[con.geom1])
            b2 = int(body_of_geom[con.geom2])
            mug_idx = None
            other_body = None
            other_geom = None
            if b1 in self._body_to_mug:
                mug_idx, other_body, other_geom = self._body_to_mug[b1], b2, int(con.geom2)
            elif b2 in self._body_to_mug:
                mug_idx, other_body, other_geom = self._body_to_mug[b2], b1, int(con.geom1)
            if mug_idx is None:
                continue
            if other_body == self._tree_body_id:
                n_tree[mug_idx] += 1
            if other_geom in self._table_geom_ids:
                on_table[mug_idx] = True
            if other_body in self._gripper_bodies:
                gripper_contact[mug_idx] = True
            for side, fingers in self._finger_bodies.items():
                if other_body in fingers["lf"]:
                    finger_touch[mug_idx][side][0] = True
                if other_body in fingers["rf"]:
                    finger_touch[mug_idx][side][1] = True

        held = np.array(
            [
                any(ft[side][0] and ft[side][1] for side in ("left", "right"))
                for ft in finger_touch
            ],
            dtype=bool,
        )
        return n_tree, on_table, held, gripper_contact, center_z

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        qpos_batch = np.asarray(qpos_batch, dtype=np.float64)
        if qpos_batch.ndim != 2 or qpos_batch.shape[1] != self.model.nq:
            raise ValueError(
                f"Expected qpos batch shape (B, {self.model.nq}), got {qpos_batch.shape}"
            )
        if qpos_batch.shape[0] != self._nworld:
            self.reset(nworld=qpos_batch.shape[0])

        n_worlds = qpos_batch.shape[0]
        n_mugs = len(self.mug_names)
        active = max(n_mugs, 1)

        hung_mask = np.zeros((n_worlds, n_mugs), dtype=bool)
        active_mask = np.zeros((n_worlds, n_mugs), dtype=bool)
        n_tree_all = np.zeros((n_worlds, n_mugs), dtype=np.int32)
        center_z_all = np.zeros((n_worlds, n_mugs), dtype=np.float32)
        held_all = np.zeros((n_worlds, n_mugs), dtype=bool)
        gripper_all = np.zeros((n_worlds, n_mugs), dtype=bool)
        for w in range(n_worlds):
            n_tree, on_table, held, gripper, center_z = self._hung_mask_one(qpos_batch[w])
            n_tree_all[w] = n_tree
            center_z_all[w] = center_z
            held_all[w] = held
            gripper_all[w] = gripper
            world_active = center_z > self.active_z_min
            active_mask[w] = world_active
            mug_hung = (
                world_active
                & (n_tree >= self.min_tree_contacts)
                & (center_z <= self.max_center_z)
            )
            if self.require_off_table:
                mug_hung &= ~on_table
            if self.disqualify_held:
                mug_hung &= ~held
            hung_mask[w] = mug_hung

        active_count = active_mask.sum(axis=1).astype(np.int32)
        num_hung = hung_mask.sum(axis=1).astype(np.int32)
        self._max_hung = np.maximum(self._max_hung, num_hung)
        reward = num_hung.astype(np.float32) / np.maximum(active_count, 1).astype(np.float32)
        success = (active_count > 0) & (num_hung >= active_count)
        self._ever_success |= success
        hung_mugs = [
            [name for name, h in zip(self.mug_names, world_mask) if bool(h)]
            for world_mask in hung_mask
        ]

        metrics: dict[str, Any] = {
            "num_hung_mugs": num_hung,
            "num_active_mugs": active_count,
            "max_hung_mugs_so_far": self._max_hung.copy(),
            "ever_success": self._ever_success.copy(),
            "mug_hung_mask": hung_mask,
            "mug_tree_contacts": n_tree_all,
            "mug_center_z": center_z_all,
            "mug_held": held_all,
            "mug_gripper_contact": gripper_all,
            "hung_mugs": hung_mugs,
            "mug_names": list(self.mug_names),
            "success_count": n_mugs,
            "min_tree_contacts": self.min_tree_contacts,
            "max_center_z": self.max_center_z,
        }
        return TaskEvalResult(reward=reward, success=success, metrics=metrics)

    def debug_spec(self) -> dict[str, Any] | None:
        return EvalDebugSpec(
            plots=[
                PlotSpec(
                    key="num_hung_mugs",
                    title="Hung Mugs",
                    color="#4fc3f7",
                    thresholds=[
                        ThresholdSpec(
                            value=float(max(self.success_count, 1)),
                            label="all mugs hung",
                            direction="gt",
                        )
                    ],
                ),
                PlotSpec(
                    key="max_hung_mugs_so_far",
                    title="Max Hung So Far",
                    color="#7dd3fc",
                ),
                PlotSpec(key="reward", title="Reward", color="#22c55e"),
                PlotSpec(key="ever_success", title="Ever Success", color="#34d399", kind="bool"),
                PlotSpec(key="success", title="Current Success", color="#fb7185", kind="bool"),
            ]
        ).to_dict()
