"""Shared replay state machine for viewers and headless exporters."""

from __future__ import annotations

from typing import Literal

import mujoco
import numpy as np

from yam_sim.rendering.replay.types import ReplayStateAlignment, ReplayStateKind

ReplayMode = Literal["auto", "physics", "qpos"]


class ReplaySession:
    """Owns replay stepping, reset behavior, and progress state."""

    def __init__(
        self,
        env,
        actions: np.ndarray,
        grid_ts: np.ndarray,
        *,
        replay_ctrls: np.ndarray | None = None,
        sim_states: np.ndarray | None = None,
        sim_integration_states: np.ndarray | None = None,
        sim_state_spec: int | None = None,
        sim_qvels: np.ndarray | None = None,
        sim_acts: np.ndarray | None = None,
        sim_mocap_pos: np.ndarray | None = None,
        sim_mocap_quat: np.ndarray | None = None,
        initial_scene_qpos: np.ndarray | None = None,
        initial_scene_integration_state: np.ndarray | None = None,
        initial_scene_qvel: np.ndarray | None = None,
        initial_scene_act: np.ndarray | None = None,
        initial_scene_mocap_pos: np.ndarray | None = None,
        initial_scene_mocap_quat: np.ndarray | None = None,
        sim_state_kind: ReplayStateKind = "qpos",
        sim_state_alignment: ReplayStateAlignment = "initial",
        rand_state=None,
        task: str = "",
        mode: ReplayMode = "auto",
    ):
        self.env = env
        self.model = env.model
        self.data = env.data
        self.actions = actions
        self.replay_ctrls = replay_ctrls
        self.grid_ts = grid_ts
        self.sim_states = sim_states
        self.sim_integration_states = sim_integration_states
        self.sim_state_spec = sim_state_spec
        self.sim_qvels = sim_qvels
        self.sim_acts = sim_acts
        self.sim_mocap_pos = sim_mocap_pos
        self.sim_mocap_quat = sim_mocap_quat
        self.initial_scene_qpos = initial_scene_qpos
        self.initial_scene_integration_state = initial_scene_integration_state
        self.initial_scene_qvel = initial_scene_qvel
        self.initial_scene_act = initial_scene_act
        self.initial_scene_mocap_pos = initial_scene_mocap_pos
        self.initial_scene_mocap_quat = initial_scene_mocap_quat
        self.sim_state_kind = sim_state_kind
        self.sim_state_alignment = sim_state_alignment
        self.rand_state = rand_state
        self.task = task
        self._mode: Literal["physics", "qpos"] = self._resolve_mode(mode)
        self._step_idx = 0
        self.reset()

    def _resolve_mode(self, mode: ReplayMode) -> Literal["physics", "qpos"]:
        if mode == "auto":
            return (
                "qpos"
                if self.sim_states is not None or self.sim_integration_states is not None
                else "physics"
            )
        if mode == "qpos" and self.sim_states is None and self.sim_integration_states is None:
            raise ValueError("qpos replay mode requested but no aligned sim states are available")
        return mode

    @property
    def mode(self) -> Literal["physics", "qpos"]:
        return self._mode

    @property
    def step_idx(self) -> int:
        return self._step_idx

    @property
    def has_exact_qpos(self) -> bool:
        return (
            self.sim_integration_states is not None
            or (self.sim_states is not None and self.sim_state_kind == "qpos")
        )

    @property
    def has_state_replay(self) -> bool:
        return self.sim_states is not None or self.sim_integration_states is not None

    @property
    def state_replay_label(self) -> str:
        if self.sim_integration_states is not None:
            return "integration state (exact)"
        if self.sim_state_kind == "policy_state":
            return "state (direct)"
        return "qpos (exact)"

    @property
    def current_frame_idx(self) -> int:
        if len(self.grid_ts) == 0:
            return 0
        if self.sim_state_alignment == "post_step":
            return min(max(self._step_idx - 1, 0), len(self.grid_ts) - 1)
        return min(self._step_idx, len(self.grid_ts) - 1)

    @property
    def current_timestamp(self) -> float:
        if len(self.grid_ts) == 0:
            return 0.0
        return float(self.grid_ts[self.current_frame_idx])

    @property
    def step_dt(self) -> float:
        if len(self.grid_ts) > 1:
            return float(self.grid_ts[1] - self.grid_ts[0])
        return 1.0 / 30.0

    @property
    def timeline_hz(self) -> float:
        if len(self.grid_ts) > 1 and self.duration_s > 0:
            return float((len(self.grid_ts) - 1) / self.duration_s)
        return 0.0

    @property
    def duration_s(self) -> float:
        if len(self.grid_ts) > 1:
            return float(self.grid_ts[-1] - self.grid_ts[0])
        return 0.0

    @property
    def elapsed_s(self) -> float:
        if len(self.grid_ts) == 0:
            return 0.0
        return float(self.grid_ts[self.current_frame_idx] - self.grid_ts[0])

    def _qpos_frame_count(self) -> int:
        if self.sim_integration_states is not None:
            return len(self.sim_integration_states)
        if self.sim_states is not None:
            return len(self.sim_states)
        return 0

    @property
    def total_steps(self) -> int:
        if self.mode == "qpos":
            frame_count = self._qpos_frame_count()
            if self.sim_state_alignment == "post_step":
                return frame_count
            return max(0, frame_count - 1)
        return len(self.actions)

    @property
    def is_done(self) -> bool:
        return self._step_idx >= self.total_steps

    def set_mode(self, mode: ReplayMode) -> None:
        self._mode = self._resolve_mode(mode)
        self.reset()

    def _apply_randomization(self) -> None:
        if self.rand_state is None or not self.task:
            return
        randomizer = getattr(self.env, "_task_randomizer", None)
        if randomizer is not None:
            randomizer.apply(self.model, self.data, self.rand_state)
            self.model = self.env.model
            self.data = self.env.data

    def _apply_qpos_frame(self, frame_idx: int) -> None:
        if self.sim_integration_states is not None:
            if self.sim_state_spec is None:
                raise RuntimeError("Integration-state replay requires a MuJoCo state spec")
            if len(self.sim_integration_states) <= frame_idx:
                return
            mujoco.mj_setState(
                self.model,
                self.data,
                np.asarray(self.sim_integration_states[frame_idx], dtype=np.float64),
                self.sim_state_spec,
            )
            mujoco.mj_forward(self.model, self.data)
            return
        if self.sim_states is None or len(self.sim_states) == 0:
            return
        state = self.sim_states[frame_idx]
        if self.sim_state_kind == "policy_state":
            self.env._set_qpos_from_state(state)
        else:
            self.data.qpos[: len(state)] = state
            if self.sim_qvels is not None and len(self.sim_qvels) > frame_idx:
                qvel = self.sim_qvels[frame_idx]
                self.data.qvel[: len(qvel)] = qvel
            if self.sim_acts is not None and len(self.sim_acts) > frame_idx:
                act = self.sim_acts[frame_idx]
                if len(act) > 0:
                    self.data.act[: len(act)] = act
            if self.sim_mocap_pos is not None and len(self.sim_mocap_pos) > frame_idx:
                mocap_pos = self.sim_mocap_pos[frame_idx]
                if mocap_pos.size > 0:
                    self.data.mocap_pos[: len(mocap_pos)] = mocap_pos
            if self.sim_mocap_quat is not None and len(self.sim_mocap_quat) > frame_idx:
                mocap_quat = self.sim_mocap_quat[frame_idx]
                if mocap_quat.size > 0:
                    self.data.mocap_quat[: len(mocap_quat)] = mocap_quat
        mujoco.mj_forward(self.model, self.data)

    def _has_initial_scene_state(self) -> bool:
        return any(
            value is not None
            for value in (
                self.initial_scene_integration_state,
                self.initial_scene_qpos,
                self.initial_scene_qvel,
                self.initial_scene_act,
                self.initial_scene_mocap_pos,
                self.initial_scene_mocap_quat,
            )
        )

    def _apply_initial_scene_state(self) -> None:
        if self.initial_scene_integration_state is not None:
            if self.sim_state_spec is None:
                raise RuntimeError("Initial integration-state replay requires a MuJoCo state spec")
            mujoco.mj_setState(
                self.model,
                self.data,
                np.asarray(self.initial_scene_integration_state, dtype=np.float64),
                self.sim_state_spec,
            )
            mujoco.mj_forward(self.model, self.data)
            return
        if self.initial_scene_qpos is not None and len(self.initial_scene_qpos) > 0:
            self.data.qpos[: len(self.initial_scene_qpos)] = self.initial_scene_qpos
        if self.initial_scene_qvel is not None and len(self.initial_scene_qvel) > 0:
            self.data.qvel[: len(self.initial_scene_qvel)] = self.initial_scene_qvel
        if self.initial_scene_act is not None and len(self.initial_scene_act) > 0:
            self.data.act[: len(self.initial_scene_act)] = self.initial_scene_act
        if (
            self.initial_scene_mocap_pos is not None
            and self.initial_scene_mocap_pos.size > 0
        ):
            self.data.mocap_pos[: len(self.initial_scene_mocap_pos)] = (
                self.initial_scene_mocap_pos
            )
        if (
            self.initial_scene_mocap_quat is not None
            and self.initial_scene_mocap_quat.size > 0
        ):
            self.data.mocap_quat[: len(self.initial_scene_mocap_quat)] = (
                self.initial_scene_mocap_quat
            )
        mujoco.mj_forward(self.model, self.data)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        q0 = self.env.get_init_q()
        self.data.qpos[: len(q0)] = q0
        mujoco.mj_forward(self.model, self.data)
        self._apply_randomization()
        self._step_idx = 0
        if self.mode == "physics" and self._has_initial_scene_state():
            self._apply_initial_scene_state()
        elif (
            self._qpos_frame_count() > 0
            and self.sim_state_alignment == "initial"
        ):
            self._apply_qpos_frame(0)
        else:
            mujoco.mj_forward(self.model, self.data)

    def step(self) -> bool:
        if self.mode == "qpos":
            frame_count = self._qpos_frame_count()
            if frame_count == 0:
                return False
            if self.sim_state_alignment == "post_step":
                if self._step_idx >= frame_count:
                    return False
                self._apply_qpos_frame(self._step_idx)
                self._step_idx += 1
                return True

            next_idx = self._step_idx + 1
            if next_idx >= frame_count:
                return False
            self._step_idx = next_idx
            self._apply_qpos_frame(self._step_idx)
            return True

        if self._step_idx >= len(self.actions):
            return False
        if self.replay_ctrls is not None:
            ctrl = np.asarray(self.replay_ctrls[self._step_idx], dtype=self.data.ctrl.dtype)
            self.data.ctrl[:] = ctrl
            n_substeps = int(getattr(self.env, "_control_decimation", 1))
            for _ in range(max(1, n_substeps)):
                mujoco.mj_step(self.model, self.data)
        else:
            action = self.actions[self._step_idx].astype(np.float32)
            self.env._step_single(action)
        self._step_idx += 1
        return True
