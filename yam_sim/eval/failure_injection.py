"""Mid-rollout failure injection for the batched eval harness.

A failure injector is constructed per batch (after ``env.reset``) and called by
``_run_one_batch`` once per executed control step, *before* ``env.step``. It may
mutate that step's ``(num_worlds, 14)`` action slice in place (the mutated
actions are what get executed and recorded to ``seed_<n>_actions.npy``), and it
contributes per-world ``slip_*`` metrics that the harness merges into each
``EpisodeResult.metrics``.

The only injector implemented so far is :class:`GraspSlipInjector` (put_bottles
only): the first time a world has a bottle in-grasp within a configured
horizontal distance of the bin footprint, the holding gripper's action channel
is forced open for a fixed window and the episode is scored for whether/when
that bottle ends up in the bin afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import mujoco
import numpy as np

from yam_sim.task_eval.bottles import PutBottlesInBinEvaluator, _quat_to_rotmat_batch

# Policy-space gripper command that maps to the fully open finger position
# (env.py scales [0, 1] by _GRIPPER_CTRL_MAX).
_GRIPPER_OPEN_ACTION = 1.0


@dataclass(frozen=True)
class GraspSlipConfig:
    """Trigger/scoring parameters for the grasp-slip injection.

    Distances are meters, times are sim seconds. The trigger condition is:
    bottle center above ``held_z_min``, not in the bin, horizontal distance to
    the bin footprint edge in ``(0, bin_edge_distance_m]``, and bottle center
    within ``gripper_bottle_max_dist_m`` of a gripper (FK check, which also
    picks the holding arm). Excluding the region *over* the footprint
    (edge distance <= 0) keeps a forced release from trivially landing the
    bottle in the bin.
    """

    # When to force the gripper open:
    #   "bin_proximity"  -- first time a held bottle comes within
    #                       bin_edge_distance_m of the bin footprint edge.
    #   "hold_duration"  -- after the bottle has been continuously in-grasp
    #                       (aloft + near a gripper) for hold_duration_s,
    #                       regardless of where it is (FRBench-E2 style
    #                       time-based rule; drop lands early in the carry).
    trigger: str = "bin_proximity"
    hold_duration_s: float = 0.5
    # Fixed trigger distance from the bin footprint edge (bin_proximity mode).
    # Must be generous enough that a fast carry (fingers take ~0.2 s to open)
    # still releases short of the rim -- at the ~0.8 m/s carries observed in
    # eval, a 0.10 band drops bottles INTO the bin; 0.25 lands them on the table.
    bin_edge_distance_m: float = 0.25
    # Optional velocity compensation: lead the trigger by approach_speed *
    # release_lead_s (capped below) so the release point is speed-independent.
    # Disabled by default: a policy-dependent trigger rule reads as unfair in
    # cross-policy comparisons; prefer the fixed band + post-hoc landing checks.
    release_lead_s: float = 0.0
    release_lead_max_m: float = 0.25
    held_z_min: float = 0.92
    # Bottle-center-to-gripper distance that counts as "in grasp". The gripper
    # point is the midpoint of the finger body origins, and a carried bottle's
    # mass center rides ~0.08-0.11 m from it (measured from eval trajectories),
    # so this must be comfortably above that; the z/in-bin gates do the rest.
    gripper_bottle_max_dist_m: float = 0.15
    open_window_s: float = 0.5
    verify_window_s: float = 1.5
    # Bottle center below this z counts as "fell back to table level" when
    # verifying that the forced release actually produced a drop.
    drop_z_max: float = 0.88
    # Visual table extent (x_min, x_max, y_min, y_max) and top height; the
    # collision plane is infinite, so leaving this box is the practical
    # definition of "rolled off the table".
    table_bounds: tuple[float, float, float, float] = (0.3025, 0.8975, -0.65, 0.65)
    table_z: float = 0.75
    # The bin counts as knocked when it tilts or slides this far from its
    # reset pose (the arm bumping it during recovery makes success unfair to
    # score); flagged episodes should be excluded from recovery stats.
    bin_knock_tilt_deg: float = 30.0
    bin_knock_disp_m: float = 0.15


class GraspSlipInjector:
    """Force the holding gripper open once a bottle is carried near the bin.

    One injection per world per episode, at the first step the trigger
    condition holds. Post-injection the target bottle is tracked for
    verification (did it actually drop?), re-binning time, and rolling off
    the table.
    """

    def __init__(self, env: Any, config: GraspSlipConfig) -> None:
        evaluator = getattr(env, "_task_evaluator", None)
        if not isinstance(evaluator, PutBottlesInBinEvaluator):
            raise ValueError(
                "GraspSlipInjector requires the put_bottles task "
                f"(PutBottlesInBinEvaluator), got {type(evaluator).__name__}"
            )
        if config.trigger not in ("bin_proximity", "hold_duration"):
            raise ValueError(
                f"trigger must be 'bin_proximity' or 'hold_duration', got {config.trigger!r}"
            )
        self.cfg = config
        self.env = env
        self.model: mujoco.MjModel = env.model
        self._eval = evaluator
        self.num_worlds = int(env.num_worlds)

        base = getattr(env, "base_env", env)
        self._control_dt = float(getattr(base, "_physics_dt", 0.002)) * int(
            getattr(base, "_control_decimation", 17)
        )
        self._robot_names: list[str] = list(base.robot_names)
        gripper_indices = list(getattr(env, "_gripper_indices", None) or base._gripper_indices)
        if len(gripper_indices) != len(self._robot_names):
            raise ValueError(
                f"Expected one gripper action index per robot, got {gripper_indices}"
            )
        self._gripper_action_dims = gripper_indices

        self._finger_body_ids: list[tuple[int, int]] = []
        for robot in self._robot_names:
            ids = []
            for side in ("left", "right"):
                body_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, f"{robot}_link_{side}_finger"
                )
                if body_id < 0:
                    raise ValueError(f"Finger body {robot}_link_{side}_finger not found")
                ids.append(body_id)
            self._finger_body_ids.append((ids[0], ids[1]))
        self._fk_data = mujoco.MjData(self.model)

        n = self.num_worlds
        self._t = 0
        self._open_steps = max(1, round(config.open_window_s / self._control_dt))
        self._verify_steps = max(1, round(config.verify_window_s / self._control_dt))
        self._triggered = np.zeros(n, dtype=bool)
        self._bottle_idx = np.full(n, -1, dtype=np.int32)
        self._arm_idx = np.full(n, -1, dtype=np.int32)
        self._slip_step = np.full(n, -1, dtype=np.int32)
        self._slip_chunk_offset = np.full(n, -1, dtype=np.int32)
        self._force_until = np.full(n, -1, dtype=np.int32)
        self._verify_deadline = np.full(n, -1, dtype=np.int32)
        self._verified = np.zeros(n, dtype=bool)
        self._outcome: list[str | None] = [None] * n
        self._rebin_step = np.full(n, -1, dtype=np.int32)
        self._rolloff_step = np.full(n, -1, dtype=np.int32)
        self._prev_edge: np.ndarray | None = None
        self._trigger_edge = np.full(n, np.nan, dtype=np.float64)
        self._trigger_speed = np.full(n, np.nan, dtype=np.float64)
        # Where the dropped bottle comes to rest (post-hoc fairness check:
        # release energy varies with carry speed even at a fixed trigger
        # distance, so landing distributions should be compared across policies).
        self._landing_step = np.full(n, -1, dtype=np.int32)
        self._landing_edge = np.full(n, np.nan, dtype=np.float64)
        self._landing_pos = np.full((n, 3), np.nan, dtype=np.float64)
        self._prev_center = np.full((n, 3), np.nan, dtype=np.float64)
        self._slow_run = np.zeros(n, dtype=np.int32)
        self._hold_steps = max(1, round(config.hold_duration_s / self._control_dt))
        self._hold_run: np.ndarray | None = None  # (W, B), lazily sized
        self._bin_pose0: tuple[np.ndarray, np.ndarray] | None = None  # xy, up-vector
        self._bin_knock_step = np.full(n, -1, dtype=np.int32)
        self._bin_max_tilt = np.zeros(n, dtype=np.float64)
        self._bin_max_disp = np.zeros(n, dtype=np.float64)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _bottle_geometry(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bottle mass centers, in-bin mask, and horizontal distance to the bin
        footprint edge, shapes (W, B, 3), (W, B), (W, B).

        Mirrors PutBottlesInBinEvaluator.evaluate_qpos_batch's transform math
        (reusing its precomputed addresses/bounds) without touching the env
        evaluator's latched ever_success/max-so-far state.
        """
        ev = self._eval
        bin_pos = qpos[:, ev._bin_qpos_adr : ev._bin_qpos_adr + 3]
        bin_quat = qpos[:, ev._bin_qpos_adr + 3 : ev._bin_qpos_adr + 7]
        bottle_pos = np.stack(
            [qpos[:, adr : adr + 3] for adr in ev._bottle_qpos_addrs], axis=1
        )
        bottle_quat = np.stack(
            [qpos[:, adr + 3 : adr + 7] for adr in ev._bottle_qpos_addrs], axis=1
        )
        nworld, nbottle = bottle_pos.shape[:2]
        bottle_rot = _quat_to_rotmat_batch(
            bottle_quat.reshape(-1, 4).astype(np.float32)
        ).reshape(nworld, nbottle, 3, 3)
        centers = bottle_pos + np.einsum(
            "bnij,nj->bni", bottle_rot, ev._bottle_center_offsets
        )

        rot_world_from_bin = _quat_to_rotmat_batch(bin_quat.astype(np.float32))
        rel_world = centers - bin_pos[:, None, :]
        rel_bin = np.einsum(
            "bij,bnj->bni", np.swapaxes(rot_world_from_bin, 1, 2), rel_world
        )

        radius_x = max(ev._bin_radius_x - ev.horizontal_margin_m, 1.0e-6)
        radius_z = max(ev._bin_radius_z - ev.horizontal_margin_m, 1.0e-6)
        normalized_radial = np.sqrt(
            (rel_bin[..., 0] / radius_x) ** 2 + (rel_bin[..., 2] / radius_z) ** 2
        )
        in_bin = (
            (centers[..., 2] > ev.active_z_min)
            & (normalized_radial <= 1.0)
            & (rel_bin[..., 1] >= ev._bin_bottom_y - ev.height_margin_m)
            & (rel_bin[..., 1] <= ev._bin_top_y + ev.height_margin_m)
        )
        edge_dist = (normalized_radial - 1.0) * (0.5 * (radius_x + radius_z))
        return centers, in_bin, edge_dist

    def _nearest_gripper(
        self, qpos_world: np.ndarray, bottle_center: np.ndarray
    ) -> tuple[int, float]:
        """(arm index, distance) of the gripper closest to a bottle center."""
        self._fk_data.qpos[:] = qpos_world
        mujoco.mj_kinematics(self.model, self._fk_data)
        best_arm, best_dist = -1, np.inf
        for arm, (fid_a, fid_b) in enumerate(self._finger_body_ids):
            grip_center = 0.5 * (
                self._fk_data.xpos[fid_a] + self._fk_data.xpos[fid_b]
            )
            dist = float(np.linalg.norm(grip_center - bottle_center))
            if dist < best_dist:
                best_arm, best_dist = arm, dist
        return best_arm, best_dist

    def _off_table(self, center: np.ndarray) -> bool:
        x_min, x_max, y_min, y_max = self.cfg.table_bounds
        return bool(
            center[0] < x_min
            or center[0] > x_max
            or center[1] < y_min
            or center[1] > y_max
            or center[2] < self.cfg.table_z - 0.05
        )

    # ------------------------------------------------------------------
    # Harness hook
    # ------------------------------------------------------------------

    def before_step(
        self, env: Any, actions: np.ndarray, *, chunk_idx: int, step_idx: int
    ) -> None:
        """Advance the per-world state machines and apply the gripper override.

        ``actions`` is this step's (num_worlds, 14) slice and is mutated in
        place while the open window is active.
        """
        t = self._t
        qpos = env.qpos_batch()
        centers, in_bin, edge_dist = self._bottle_geometry(qpos)
        self._track_bin(qpos, t)

        for w in range(self.num_worlds):
            if not self._triggered[w]:
                continue
            b = int(self._bottle_idx[w])
            center = centers[w, b]
            if self._outcome[w] is None:
                if bool(in_bin[w, b]):
                    self._outcome[w] = "accidental_bin"
                    self._rebin_step[w] = t
                elif center[2] < self.cfg.drop_z_max:
                    self._outcome[w] = "dropped"
                    self._verified[w] = True
                elif t >= self._verify_deadline[w]:
                    self._outcome[w] = "not_released"
            if self._verified[w]:
                if self._rebin_step[w] < 0 and bool(in_bin[w, b]):
                    self._rebin_step[w] = t
                if self._rolloff_step[w] < 0 and self._off_table(center):
                    self._rolloff_step[w] = t
                if self._landing_step[w] < 0 and not bool(in_bin[w, b]):
                    prev = self._prev_center[w]
                    if not np.isnan(prev).any():
                        speed = float(np.linalg.norm(center - prev)) / self._control_dt
                        self._slow_run[w] = self._slow_run[w] + 1 if speed < 0.05 else 0
                        if self._slow_run[w] >= 8:
                            self._landing_step[w] = t
                            self._landing_edge[w] = float(edge_dist[w, b])
                            self._landing_pos[w] = center
            self._prev_center[w] = center

        if self._prev_edge is None:
            approach_speed = np.zeros_like(edge_dist)
        else:
            approach_speed = np.maximum(
                (self._prev_edge - edge_dist) / self._control_dt, 0.0
            )
        self._prev_edge = edge_dist.copy()
        lead = np.minimum(
            approach_speed * self.cfg.release_lead_s, self.cfg.release_lead_max_m
        )

        held = (centers[..., 2] > self.cfg.held_z_min) & ~in_bin
        if self.cfg.trigger == "hold_duration":
            if self._hold_run is None:
                self._hold_run = np.zeros(held.shape, dtype=np.int32)
            for w in range(self.num_worlds):
                if self._triggered[w]:
                    continue
                for b in range(held.shape[1]):
                    in_grasp = False
                    if held[w, b]:
                        arm, dist = self._nearest_gripper(qpos[w], centers[w, b])
                        in_grasp = dist <= self.cfg.gripper_bottle_max_dist_m
                    self._hold_run[w, b] = self._hold_run[w, b] + 1 if in_grasp else 0
                    if in_grasp and self._hold_run[w, b] >= self._hold_steps:
                        self._fire(w, b, arm, t, step_idx, edge_dist, approach_speed)
                        break
        else:
            candidates = (
                held
                & (edge_dist > 0.0)
                & (edge_dist <= self.cfg.bin_edge_distance_m + lead)
            )
            for w in range(self.num_worlds):
                if self._triggered[w] or not candidates[w].any():
                    continue
                for b in np.flatnonzero(candidates[w]):
                    arm, dist = self._nearest_gripper(qpos[w], centers[w, b])
                    if dist <= self.cfg.gripper_bottle_max_dist_m:
                        self._fire(w, b, arm, t, step_idx, edge_dist, approach_speed)
                        break

        forcing = self._triggered & (t < self._force_until)
        for w in np.flatnonzero(forcing):
            actions[w, self._gripper_action_dims[self._arm_idx[w]]] = _GRIPPER_OPEN_ACTION

        self._t += 1

    def _track_bin(self, qpos: np.ndarray, t: int) -> None:
        """Flag worlds whose bin has tilted or slid away from its reset pose."""
        adr = self._eval._bin_qpos_adr
        bin_xy = qpos[:, adr : adr + 2]
        bin_quat = qpos[:, adr + 3 : adr + 7].astype(np.float32)
        rot = _quat_to_rotmat_batch(bin_quat)
        up = rot[:, :, 1]  # bin local +Y is world up at the canonical pose
        if self._bin_pose0 is None:
            self._bin_pose0 = (bin_xy.copy(), up.copy())
            return
        xy0, up0 = self._bin_pose0
        disp = np.linalg.norm(bin_xy - xy0, axis=1)
        cos_tilt = np.clip(np.einsum("wi,wi->w", up, up0), -1.0, 1.0)
        tilt_deg = np.degrees(np.arccos(cos_tilt))
        self._bin_max_disp = np.maximum(self._bin_max_disp, disp)
        self._bin_max_tilt = np.maximum(self._bin_max_tilt, tilt_deg)
        knocked = (tilt_deg > self.cfg.bin_knock_tilt_deg) | (
            disp > self.cfg.bin_knock_disp_m
        )
        newly = knocked & (self._bin_knock_step < 0)
        self._bin_knock_step[newly] = t

    def _fire(
        self,
        w: int,
        b: int,
        arm: int,
        t: int,
        step_idx: int,
        edge_dist: np.ndarray,
        approach_speed: np.ndarray,
    ) -> None:
        self._triggered[w] = True
        self._bottle_idx[w] = b
        self._arm_idx[w] = arm
        self._slip_step[w] = t
        self._slip_chunk_offset[w] = step_idx
        self._force_until[w] = t + self._open_steps
        self._verify_deadline[w] = t + self._verify_steps
        self._trigger_edge[w] = float(edge_dist[w, b])
        self._trigger_speed[w] = float(approach_speed[w, b])

    def finalize(self) -> list[dict[str, Any]]:
        """Per-world slip metrics, JSON-native, ready to merge into episode metrics."""
        records: list[dict[str, Any]] = []
        dt = self._control_dt
        for w in range(self.num_worlds):
            triggered = bool(self._triggered[w])
            slip_step = int(self._slip_step[w]) if triggered else None
            rebin_step = int(self._rebin_step[w]) if self._rebin_step[w] >= 0 else None
            rolloff_step = (
                int(self._rolloff_step[w]) if self._rolloff_step[w] >= 0 else None
            )
            recovery_s = (
                (rebin_step - slip_step) * dt
                if (triggered and rebin_step is not None and self._verified[w])
                else None
            )
            records.append(
                {
                    "slip_triggered": triggered,
                    "slip_step": slip_step,
                    "slip_time_s": slip_step * dt if slip_step is not None else None,
                    "slip_chunk_offset": (
                        int(self._slip_chunk_offset[w]) if triggered else None
                    ),
                    "slip_bottle": (
                        self._eval.bottle_names[int(self._bottle_idx[w])]
                        if triggered
                        else None
                    ),
                    "slip_arm": (
                        self._robot_names[int(self._arm_idx[w])] if triggered else None
                    ),
                    "slip_outcome": self._outcome[w],
                    "slip_trigger_edge_dist_m": (
                        float(self._trigger_edge[w]) if triggered else None
                    ),
                    "slip_approach_speed_mps": (
                        float(self._trigger_speed[w]) if triggered else None
                    ),
                    "slip_landing_edge_dist_m": (
                        float(self._landing_edge[w])
                        if self._landing_step[w] >= 0
                        else None
                    ),
                    "slip_landing_pos": (
                        [round(float(v), 4) for v in self._landing_pos[w]]
                        if self._landing_step[w] >= 0
                        else None
                    ),
                    "slip_rebin_step": rebin_step,
                    "slip_recovery_time_s": recovery_s,
                    "slip_rolled_off_table": rolloff_step is not None,
                    "slip_rolloff_time_s": (
                        rolloff_step * dt if rolloff_step is not None else None
                    ),
                    "slip_bin_knocked": bool(self._bin_knock_step[w] >= 0),
                    "slip_bin_knock_time_s": (
                        float(self._bin_knock_step[w] * dt)
                        if self._bin_knock_step[w] >= 0
                        else None
                    ),
                    "slip_bin_max_tilt_deg": round(float(self._bin_max_tilt[w]), 1),
                    "slip_bin_max_disp_m": round(float(self._bin_max_disp[w]), 3),
                }
            )
        return records


def build_injector_factory(
    spec: dict[str, Any] | None,
) -> Callable[[Any], GraspSlipInjector] | None:
    """Injector factory from an EvalConfig.failure_injection mapping.

    Returns None when no injection is configured. The factory is invoked once
    per batch with the freshly reset env.
    """
    if not spec:
        return None
    params = dict(spec)
    kind = params.pop("type", "grasp_slip")
    if kind != "grasp_slip":
        raise ValueError(f"Unknown failure_injection type: {kind!r}")
    if "table_bounds" in params:
        params["table_bounds"] = tuple(float(v) for v in params["table_bounds"])
    config = GraspSlipConfig(**params)
    return lambda env: GraspSlipInjector(env, config)
