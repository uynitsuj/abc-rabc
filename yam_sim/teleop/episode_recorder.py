"""Record exact applied teleop actions plus sim state for replay debugging."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


def _copy_native_array(value) -> np.ndarray:
    """Copy an array-like value while preserving MuJoCo's native dtype."""
    return np.array(value, copy=True)


_EXACT_STATE_SPEC = int(mujoco.mjtState.mjSTATE_INTEGRATION)
_EXACT_STATE_SPEC_NAME = "mjSTATE_INTEGRATION"


def _capture_mj_state(model: mujoco.MjModel, data: mujoco.MjData, state_spec: int) -> np.ndarray:
    """Capture a MuJoCo state vector using the provided mjtState bitfield."""
    state = np.empty(mujoco.mj_stateSize(model, state_spec), dtype=np.float64)
    mujoco.mj_getState(model, data, state, state_spec)
    return state


class TeleopEpisodeRecorder:
    """Capture a self-contained episode with exact applied actions and sim state."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        task: str,
        scene: str,
        prompt: str | None,
        control_rate: float,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.task = str(task)
        self.scene = str(scene)
        self.prompt = str(prompt) if prompt is not None else None
        self.control_rate = float(control_rate)
        self.extra_metadata = dict(extra_metadata or {})

        self._started = False
        self._closed = False
        self._initial_qpos: np.ndarray | None = None
        self._initial_qvel: np.ndarray | None = None
        self._initial_act: np.ndarray | None = None
        self._initial_mocap_pos: np.ndarray | None = None
        self._initial_mocap_quat: np.ndarray | None = None
        self._initial_mj_state: np.ndarray | None = None
        self._initial_state: np.ndarray | None = None
        self._actions: list[np.ndarray] = []
        self._mj_state_frames: list[np.ndarray] = []
        self._qpos_frames: list[np.ndarray] = []
        self._ctrl_frames: list[np.ndarray] = []
        self._qvel_frames: list[np.ndarray] = []
        self._act_frames: list[np.ndarray] = []
        self._mocap_pos_frames: list[np.ndarray] = []
        self._mocap_quat_frames: list[np.ndarray] = []
        self._action_timestamps: list[float] = []
        self._qpos_timestamps: list[float] = []
        self._metadata: dict[str, Any] = {}

    def start(self, env, *, initial_state: np.ndarray | None = None) -> None:
        """Capture the initial reset state before any teleop actions are applied."""
        if self._started:
            raise RuntimeError("Recorder already started")

        self._started = True
        self._initial_qpos = _copy_native_array(env.data.qpos)
        self._initial_qvel = _copy_native_array(env.data.qvel)
        self._initial_act = _copy_native_array(env.data.act)
        self._initial_mocap_pos = _copy_native_array(env.data.mocap_pos)
        self._initial_mocap_quat = _copy_native_array(env.data.mocap_quat)
        self._initial_state = (
            _copy_native_array(initial_state)
            if initial_state is not None
            else None
        )
        self._qpos_frames = [self._initial_qpos.copy()]
        self._ctrl_frames = [_copy_native_array(env.data.ctrl)]
        self._qvel_frames = [self._initial_qvel.copy()]
        self._act_frames = [self._initial_act.copy()]
        self._mocap_pos_frames = [self._initial_mocap_pos.copy()]
        self._mocap_quat_frames = [self._initial_mocap_quat.copy()]
        self._qpos_timestamps = [float(env.data.time)]
        self._initial_mj_state = _capture_mj_state(env.model, env.data, _EXACT_STATE_SPEC)
        self._mj_state_frames = [self._initial_mj_state.copy()]

        self._metadata = {
            "format": "yam_sim_recorded_episode_v3",
            "task": self.task,
            "scene": self.scene,
            "prompt": self.prompt,
            "instruction": self.prompt,
            "control_rate": self.control_rate,
            "physics_dt": float(env.model.opt.timestep),
            "control_decimation": int(getattr(env, "_control_decimation", 1)),
            "action_dim": int(getattr(env, "single_timestep_action_dim", 14)),
            "qpos_dim": int(env.model.nq),
            "qvel_dim": int(env.model.nv),
            "ctrl_dim": int(env.model.nu),
            "act_dim": int(env.model.na),
            "nmocap": int(env.model.nmocap),
            "mj_state_spec": _EXACT_STATE_SPEC,
            "mj_state_spec_name": _EXACT_STATE_SPEC_NAME,
            "mj_state_size": int(self._initial_mj_state.size),
            "recorded_at_unix_s": float(time.time()),
        }
        if self._initial_state is not None:
            self._metadata["init_q"] = self._initial_state.tolist()
        self._metadata.update(self.extra_metadata)

    def record_step(
        self,
        action: np.ndarray,
        env,
        *,
        state: np.ndarray | None = None,
    ) -> None:
        """Append one applied control step and the resulting post-step sim state."""
        if not self._started:
            raise RuntimeError("Recorder has not been started")
        if self._closed:
            raise RuntimeError("Recorder already closed")

        action_arr = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        expected_dim = int(self._metadata.get("action_dim", action_arr.size))
        if action_arr.size != expected_dim:
            raise ValueError(
                f"Expected action dim {expected_dim}, got {action_arr.size}"
            )
        qpos_arr = _copy_native_array(env.data.qpos)
        ctrl_arr = _copy_native_array(env.data.ctrl)
        qvel_arr = _copy_native_array(env.data.qvel)
        act_arr = _copy_native_array(env.data.act)
        mocap_pos_arr = _copy_native_array(env.data.mocap_pos)
        mocap_quat_arr = _copy_native_array(env.data.mocap_quat)
        mj_state_arr = _capture_mj_state(env.model, env.data, _EXACT_STATE_SPEC)
        sim_time = float(env.data.time)

        self._actions.append(action_arr)
        self._action_timestamps.append(sim_time)
        self._mj_state_frames.append(mj_state_arr)
        self._qpos_frames.append(qpos_arr)
        self._ctrl_frames.append(ctrl_arr)
        self._qvel_frames.append(qvel_arr)
        self._act_frames.append(act_arr)
        self._mocap_pos_frames.append(mocap_pos_arr)
        self._mocap_quat_frames.append(mocap_quat_arr)
        self._qpos_timestamps.append(sim_time)
        if state is not None and "final_state" not in self._metadata:
            self._metadata["state_dim"] = int(np.asarray(state).shape[-1])

    def close(self) -> Path | None:
        """Flush the episode to disk and return the episode directory."""
        if not self._started or self._closed:
            return None
        self._closed = True

        self.output_dir.mkdir(parents=True, exist_ok=True)

        action_dim = int(self._metadata.get("action_dim", 14))
        if self._actions:
            actions = np.stack(self._actions).astype(np.float32, copy=False)
        else:
            actions = np.empty((0, action_dim), dtype=np.float32)
        mj_state = np.asarray(self._mj_state_frames)
        qpos = np.asarray(self._qpos_frames)
        ctrl = np.asarray(self._ctrl_frames)
        qvel = np.asarray(self._qvel_frames)
        act = np.asarray(self._act_frames)
        mocap_pos = np.asarray(self._mocap_pos_frames)
        mocap_quat = np.asarray(self._mocap_quat_frames)
        action_timestamps = np.asarray(self._action_timestamps, dtype=np.float64)
        qpos_timestamps = np.asarray(self._qpos_timestamps, dtype=np.float64)
        initial_qpos = (
            self._initial_qpos.copy()
            if self._initial_qpos is not None
            else qpos[0].copy()
        )
        initial_qvel = (
            self._initial_qvel.copy()
            if self._initial_qvel is not None
            else qvel[0].copy()
        )

        np.save(self.output_dir / "actions.npy", actions, allow_pickle=False)
        np.save(self.output_dir / "actions_full.npy", actions, allow_pickle=False)
        np.save(self.output_dir / "integration_state.npy", mj_state, allow_pickle=False)
        np.save(self.output_dir / "qpos.npy", qpos, allow_pickle=False)
        np.save(self.output_dir / "ctrl.npy", ctrl, allow_pickle=False)
        np.save(self.output_dir / "qvel.npy", qvel, allow_pickle=False)
        np.save(
            self.output_dir / "action_timestamps.npy",
            action_timestamps,
            allow_pickle=False,
        )
        np.save(
            self.output_dir / "qpos_timestamps.npy",
            qpos_timestamps,
            allow_pickle=False,
        )
        np.save(
            self.output_dir / "initial_qpos.npy",
            initial_qpos,
            allow_pickle=False,
        )
        np.save(
            self.output_dir / "initial_qvel.npy",
            initial_qvel,
            allow_pickle=False,
        )
        if act.ndim == 2 and act.shape[1] > 0:
            np.save(self.output_dir / "act.npy", act, allow_pickle=False)
            np.save(
                self.output_dir / "initial_act.npy",
                self._initial_act,
                allow_pickle=False,
            )
        if mocap_pos.ndim == 3 and mocap_pos.shape[1] > 0:
            np.save(self.output_dir / "mocap_pos.npy", mocap_pos, allow_pickle=False)
            np.save(
                self.output_dir / "initial_mocap_pos.npy",
                self._initial_mocap_pos,
                allow_pickle=False,
            )
        if mocap_quat.ndim == 3 and mocap_quat.shape[1] > 0:
            np.save(
                self.output_dir / "mocap_quat.npy",
                mocap_quat,
                allow_pickle=False,
            )
            np.save(
                self.output_dir / "initial_mocap_quat.npy",
                self._initial_mocap_quat,
                allow_pickle=False,
            )

        payload = dict(self._metadata)
        payload.update(
            {
                "actions_file": "actions_full.npy",
                "integration_state_file": "integration_state.npy",
                "qpos_file": "qpos.npy",
                "ctrl_file": "ctrl.npy",
                "qvel_file": "qvel.npy",
                "action_timestamps_file": "action_timestamps.npy",
                "qpos_timestamps_file": "qpos_timestamps.npy",
                "initial_qpos_file": "initial_qpos.npy",
                "initial_qvel_file": "initial_qvel.npy",
                "actions_shape": list(actions.shape),
                "integration_state_shape": list(mj_state.shape),
                "qpos_shape": list(qpos.shape),
                "ctrl_shape": list(ctrl.shape),
                "qvel_shape": list(qvel.shape),
                "actions_dtype": str(actions.dtype),
                "integration_state_dtype": str(mj_state.dtype),
                "qpos_dtype": str(qpos.dtype),
                "ctrl_dtype": str(ctrl.dtype),
                "qvel_dtype": str(qvel.dtype),
                "num_action_steps": int(len(actions)),
                "num_qpos_frames": int(len(qpos)),
            }
        )
        if act.ndim == 2 and act.shape[1] > 0:
            payload["act_file"] = "act.npy"
            payload["initial_act_file"] = "initial_act.npy"
            payload["act_shape"] = list(act.shape)
            payload["act_dtype"] = str(act.dtype)
        if mocap_pos.ndim == 3 and mocap_pos.shape[1] > 0:
            payload["mocap_pos_file"] = "mocap_pos.npy"
            payload["initial_mocap_pos_file"] = "initial_mocap_pos.npy"
            payload["mocap_pos_shape"] = list(mocap_pos.shape)
            payload["mocap_pos_dtype"] = str(mocap_pos.dtype)
        if mocap_quat.ndim == 3 and mocap_quat.shape[1] > 0:
            payload["mocap_quat_file"] = "mocap_quat.npy"
            payload["initial_mocap_quat_file"] = "initial_mocap_quat.npy"
            payload["mocap_quat_shape"] = list(mocap_quat.shape)
            payload["mocap_quat_dtype"] = str(mocap_quat.dtype)
        (self.output_dir / "config.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )
        return self.output_dir
