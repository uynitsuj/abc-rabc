"""Plate task evaluators."""

from __future__ import annotations

import re
from typing import Any

import mujoco
import numpy as np

from yam_sim.task_eval.base import TaskEvalResult
from yam_sim.task_eval.debug_spec import EvalDebugSpec, PlotSpec, ThresholdSpec
from yam_sim.task_specs import SimTaskSpec


class LoadPlatesInRackEvaluator:
    """Score the dish-rack task by counting plates loaded into the rack.

    A plate counts as *loaded* when, at the evaluated state, it: (1) makes at
    least ``min_rack_contacts`` contact(s) with the dish-rack body, (2) is not
    touching the table/floor (``require_off_table``), (3) is standing roughly
    upright in a slot -- its disk normal is near-horizontal, ``|plate_z_axis_z|
    <= max_face_z`` -- and (4) has its center within the rack's horizontal
    footprint (expanded by ``footprint_margin``). A plate being *held* between
    both fingers of one gripper is never counted (``disqualify_held``).

    Why each condition matters, calibrated against the annotated ``load_plates``
    rollouts (12/12 loaded examples score 1.0; the flat-on-table start state and
    the mid-grasp lift state both score 0.0):

    * *rack contact* is the core "it's in the rack" signal -- a loaded plate
      always rests against the rack tines (loaded examples show 3-6 contacts;
      a plate sitting on the table has 0).
    * *off-table* rejects a plate still lying flat on the table (the start
      state) -- a slotted plate is held up by the rack, never the table.
    * *upright* is the load-bearing guard: a plate dumped *flat on top of* the
      rack would touch it and be off the table, but is not loaded. A slotted
      plate stands on its edge, so its face normal is near-horizontal
      (loaded ``|z_axis_z|`` ~ 0.18-0.32; a flat plate is ~1.0; a plate the arm
      is mid-lift tilting up is ~0.7).
    * *footprint* rejects a vertical plate leaning against the *outside* of the
      rack while grazing it -- a loaded plate's center sits over the rack base
      (loaded center-to-rack horizontal offset ~ 0.04-0.08; the start state is
      ~0.46 away).
    * the *held* check rejects a plate the arm is still carrying next to the
      rack as the episode ends.

    Contacts are computed with a private ``MjData`` via ``mj_kinematics`` +
    ``mj_collision`` on each world's qpos, so this works identically for the
    batched live harness and for offline scoring of saved qpos trajectories.
    Per-step reward is the fraction of plates currently loaded; ``success``
    requires all plates loaded. ``ever_success`` / ``max_loaded_so_far`` are
    tracked across a rollout like the other evaluators.
    """

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
        min_rack_contacts: int = 1,
        require_off_table: bool = True,
        disqualify_held: bool = True,
        max_face_z: float = 0.5,
        footprint_margin: float = 0.05,
        rack_body: str = "dish_rack_0_dish_rack_0_model",
        table_geoms: tuple[str, ...] = ("table_plane", "floor_collision"),
        active_z_min: float = -0.5,
    ) -> None:
        self.model = model
        self.spec = spec
        self.min_rack_contacts = int(min_rack_contacts)
        self.require_off_table = bool(require_off_table)
        self.disqualify_held = bool(disqualify_held)
        self.max_face_z = float(max_face_z)
        self.footprint_margin = float(footprint_margin)
        # Plates parked below this z (batched variable-count masking) are absent:
        # excluded from loaded count and denominator. No parked plates (single-
        # world / faithful eval) => every plate active => scoring unchanged.
        self.active_z_min = float(active_z_min)
        self._data = mujoco.MjData(model)

        self.plate_names, self._plate_body_ids = self._resolve_plate_bodies(model)
        self.success_count = len(self.plate_names)
        self._body_to_plate = {bid: i for i, bid in enumerate(self._plate_body_ids)}

        # The dish rack and plates are attached to the scene by the task
        # randomizer *after* the env is first built, so the evaluator may be
        # constructed (eagerly, in ``set_task``) against a model that has neither
        # yet. We tolerate that here -- ``reload_from_model`` rebuilds the
        # evaluator once the randomizer has attached the objects -- and simply
        # score zero whenever the rack or plates are absent.
        self._rack_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, rack_body)
        if self._rack_body_id < 0:
            # Fall back to any geom-bearing body whose name mentions a rack.
            self._rack_body_id = self._resolve_rack_body(model)
        if self._rack_body_id >= 0:
            self._rack_geom_ids = np.asarray(
                [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == self._rack_body_id],
                dtype=np.int32,
            )
        else:
            self._rack_geom_ids = np.zeros((0,), dtype=np.int32)

        self._table_geom_ids = set()
        for name in table_geoms:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self._table_geom_ids.add(int(gid))

        # Gripper finger bodies grouped by arm side, so we can detect a two-finger
        # grasp (plate held between the lf and rf fingers of the same gripper).
        self._finger_bodies: dict[str, dict[str, set[int]]] = {
            "left": {"lf": set(), "rf": set()},
            "right": {"lf": set(), "rf": set()},
        }
        # Broader set of gripper/hand bodies, for an "is the gripper touching this
        # plate at all" signal (useful for offline load-event timing).
        self._gripper_bodies: set[int] = set()
        for body_id in range(model.nbody):
            fm = re.match(r"(left|right)_(lf|rf)_(down|rot)$", model.body(body_id).name or "")
            if fm is not None:
                self._finger_bodies[fm.group(1)][fm.group(2)].add(int(body_id))
            if re.search(r"(link_6|_lf_|_rf_|finger)", model.body(body_id).name or ""):
                self._gripper_bodies.add(int(body_id))

        self._nworld = 1
        self._max_loaded = np.zeros((1,), dtype=np.int32)
        self._ever_success = np.zeros((1,), dtype=bool)

    def reset(self, *, nworld: int = 1) -> None:
        self._nworld = int(nworld)
        self._max_loaded = np.zeros((self._nworld,), dtype=np.int32)
        self._ever_success = np.zeros((self._nworld,), dtype=bool)

    @staticmethod
    def _resolve_plate_bodies(model: mujoco.MjModel) -> tuple[list[str], np.ndarray]:
        """Geom-bearing plate bodies, ordered by index, e.g. ``plate_0_..._model``."""
        entries: list[tuple[int, str, int]] = []
        for body_id in range(model.nbody):
            name = model.body(body_id).name or ""
            # Variant reload names the geom-bearing plate body ``..._object``;
            # the canonical/masked (MAX-body) path names it ``..._model``. Match both.
            match = re.fullmatch(r"plate_(\d+)_plate_\d+_(?:model|object)", name)
            if match is None:
                continue
            has_geom = any(
                int(model.geom_bodyid[g]) == body_id for g in range(model.ngeom)
            )
            if not has_geom:
                continue
            entries.append((int(match.group(1)), name, body_id))
        entries.sort(key=lambda item: item[0])
        return [name for _, name, _ in entries], np.asarray(
            [bid for _, _, bid in entries], dtype=np.int32
        )

    @staticmethod
    def _resolve_rack_body(model: mujoco.MjModel) -> int:
        for body_id in range(model.nbody):
            name = model.body(body_id).name or ""
            if "rack" not in name:
                continue
            if any(int(model.geom_bodyid[g]) == body_id for g in range(model.ngeom)):
                return body_id
        return -1

    def _loaded_mask_one(
        self, qpos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (n_rack, on_table, held, gripper_contact, face_z, in_footprint, plate_z)."""
        data = self._data
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        # Kinematics + collision are sufficient for contacts and far cheaper than
        # a full forward dynamics solve.
        mujoco.mj_kinematics(self.model, data)
        mujoco.mj_collision(self.model, data)

        n_plates = len(self.plate_names)
        n_rack = np.zeros((n_plates,), dtype=np.int32)
        on_table = np.zeros((n_plates,), dtype=bool)
        gripper_contact = np.zeros((n_plates,), dtype=bool)
        # finger_touch[plate][side] = [lf_touched, rf_touched]
        finger_touch = [
            {"left": [False, False], "right": [False, False]} for _ in range(n_plates)
        ]

        body_of_geom = self.model.geom_bodyid
        for c in range(data.ncon):
            con = data.contact[c]
            b1 = int(body_of_geom[con.geom1])
            b2 = int(body_of_geom[con.geom2])
            plate_idx = None
            other_body = None
            other_geom = None
            if b1 in self._body_to_plate:
                plate_idx, other_body, other_geom = self._body_to_plate[b1], b2, int(con.geom2)
            elif b2 in self._body_to_plate:
                plate_idx, other_body, other_geom = self._body_to_plate[b2], b1, int(con.geom1)
            if plate_idx is None:
                continue
            if other_body == self._rack_body_id:
                n_rack[plate_idx] += 1
            if other_geom in self._table_geom_ids:
                on_table[plate_idx] = True
            if other_body in self._gripper_bodies:
                gripper_contact[plate_idx] = True
            for side, fingers in self._finger_bodies.items():
                if other_body in fingers["lf"]:
                    finger_touch[plate_idx][side][0] = True
                if other_body in fingers["rf"]:
                    finger_touch[plate_idx][side][1] = True

        held = np.array(
            [
                any(ft[side][0] and ft[side][1] for side in ("left", "right"))
                for ft in finger_touch
            ],
            dtype=bool,
        )

        # Plate face orientation: the disk normal is the body's local z-axis;
        # |z-component| ~ 0 when the plate stands on edge, ~ 1 when it lies flat.
        face_z = np.array(
            [abs(float(data.xmat[bid].reshape(3, 3)[2, 2])) for bid in self._plate_body_ids],
            dtype=np.float32,
        )

        # Rack horizontal footprint from current rack-geom world positions (the
        # rack is a free body and may shift slightly), expanded by the margin.
        rack_xy = data.geom_xpos[self._rack_geom_ids][:, :2]
        lo = rack_xy.min(axis=0) - self.footprint_margin
        hi = rack_xy.max(axis=0) + self.footprint_margin
        plate_xy = np.array(
            [data.xpos[bid][:2] for bid in self._plate_body_ids], dtype=np.float32
        )
        in_footprint = np.all((plate_xy >= lo) & (plate_xy <= hi), axis=1)
        plate_z = np.array(
            [data.xpos[bid][2] for bid in self._plate_body_ids], dtype=np.float32
        )

        return n_rack, on_table, held, gripper_contact, face_z, in_footprint, plate_z

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        qpos_batch = np.asarray(qpos_batch, dtype=np.float64)
        if qpos_batch.ndim != 2 or qpos_batch.shape[1] != self.model.nq:
            raise ValueError(
                f"Expected qpos batch shape (B, {self.model.nq}), got {qpos_batch.shape}"
            )
        if qpos_batch.shape[0] != self._nworld:
            self.reset(nworld=qpos_batch.shape[0])

        n_worlds = qpos_batch.shape[0]
        n_plates = len(self.plate_names)
        active = max(n_plates, 1)

        # Rack/plates not attached yet (evaluator built before the randomizer
        # populated the scene): nothing is loadable, score zero.
        if n_plates == 0 or self._rack_body_id < 0:
            return TaskEvalResult(
                reward=np.zeros((n_worlds,), dtype=np.float32),
                success=np.zeros((n_worlds,), dtype=bool),
                metrics={
                    "num_loaded_plates": np.zeros((n_worlds,), dtype=np.int32),
                    "num_active_plates": np.full((n_worlds,), n_plates, dtype=np.int32),
                    "max_loaded_plates_so_far": self._max_loaded.copy(),
                    "ever_success": self._ever_success.copy(),
                    "plate_loaded_mask": np.zeros((n_worlds, n_plates), dtype=bool),
                    "plate_names": list(self.plate_names),
                    "success_count": n_plates,
                },
            )

        loaded_mask = np.zeros((n_worlds, n_plates), dtype=bool)
        active_mask = np.zeros((n_worlds, n_plates), dtype=bool)
        n_rack_all = np.zeros((n_worlds, n_plates), dtype=np.int32)
        face_z_all = np.zeros((n_worlds, n_plates), dtype=np.float32)
        held_all = np.zeros((n_worlds, n_plates), dtype=bool)
        gripper_all = np.zeros((n_worlds, n_plates), dtype=bool)
        footprint_all = np.zeros((n_worlds, n_plates), dtype=bool)
        for w in range(n_worlds):
            n_rack, on_table, held, gripper, face_z, in_fp, plate_z = self._loaded_mask_one(
                qpos_batch[w]
            )
            n_rack_all[w] = n_rack
            face_z_all[w] = face_z
            held_all[w] = held
            gripper_all[w] = gripper
            footprint_all[w] = in_fp
            world_active = plate_z > self.active_z_min
            active_mask[w] = world_active
            plate_loaded = (
                world_active
                & (n_rack >= self.min_rack_contacts)
                & (face_z <= self.max_face_z)
                & in_fp
            )
            if self.require_off_table:
                plate_loaded &= ~on_table
            if self.disqualify_held:
                plate_loaded &= ~held
            loaded_mask[w] = plate_loaded

        active_count = active_mask.sum(axis=1).astype(np.int32)
        num_loaded = loaded_mask.sum(axis=1).astype(np.int32)
        self._max_loaded = np.maximum(self._max_loaded, num_loaded)
        reward = num_loaded.astype(np.float32) / np.maximum(active_count, 1).astype(np.float32)
        success = (active_count > 0) & (num_loaded >= active_count)
        self._ever_success |= success
        loaded_plates = [
            [name for name, h in zip(self.plate_names, world_mask) if bool(h)]
            for world_mask in loaded_mask
        ]

        metrics: dict[str, Any] = {
            "num_loaded_plates": num_loaded,
            "num_active_plates": active_count,
            "max_loaded_plates_so_far": self._max_loaded.copy(),
            "ever_success": self._ever_success.copy(),
            "plate_loaded_mask": loaded_mask,
            "plate_rack_contacts": n_rack_all,
            "plate_face_z": face_z_all,
            "plate_in_footprint": footprint_all,
            "plate_held": held_all,
            "plate_gripper_contact": gripper_all,
            "loaded_plates": loaded_plates,
            "plate_names": list(self.plate_names),
            "success_count": n_plates,
            "min_rack_contacts": self.min_rack_contacts,
            "max_face_z": self.max_face_z,
        }
        return TaskEvalResult(reward=reward, success=success, metrics=metrics)

    def debug_spec(self) -> dict[str, Any] | None:
        return EvalDebugSpec(
            plots=[
                PlotSpec(
                    key="num_loaded_plates",
                    title="Loaded Plates",
                    color="#4fc3f7",
                    thresholds=[
                        ThresholdSpec(
                            value=float(max(self.success_count, 1)),
                            label="all plates loaded",
                            direction="gt",
                        )
                    ],
                ),
                PlotSpec(
                    key="max_loaded_plates_so_far",
                    title="Max Loaded So Far",
                    color="#7dd3fc",
                ),
                PlotSpec(key="reward", title="Reward", color="#22c55e"),
                PlotSpec(key="ever_success", title="Ever Success", color="#34d399", kind="bool"),
                PlotSpec(key="success", title="Current Success", color="#fb7185", kind="bool"),
            ]
        ).to_dict()
