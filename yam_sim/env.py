"""MuJoCo simulation environment for bimanual YAM robot."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from yam_sim.config import RobotSystemConfig
from yam_sim.rendering.live import LiveRenderBackend, create_live_camera_provider
from yam_sim.task_eval import TaskEvalResult, make_task_evaluator
from yam_sim.task_registry import DEFAULT_SCENE_XML, SCENE_XMLS, get_task_randomizer, resolve_task
from yam_sim.task_specs import SimTaskSpec, maybe_get_task_spec

# Backwards-compatible aliases for older imports.
_SCENE_XMLS = dict(SCENE_XMLS)
_DEFAULT_SCENE_XML = DEFAULT_SCENE_XML

# Gripper actuator ctrl range max (from menagerie model)
_GRIPPER_CTRL_MAX = 0.0475


def project_policy_state(
    qpos: np.ndarray,
    qpos_indices: list[int],
    gripper_indices: list[int],
    *,
    dtype=np.float32,
) -> np.ndarray:
    """Project MuJoCo qpos into the 14D policy-space state vector."""
    state = np.zeros(len(qpos_indices), dtype=dtype)
    gripper_set = set(gripper_indices)
    for i, qpos_idx in enumerate(qpos_indices):
        val = qpos[qpos_idx]
        if i in gripper_set:
            val = np.clip(val / _GRIPPER_CTRL_MAX, 0.0, 1.0)
        state[i] = val
    return state


def project_policy_state_batch(
    qpos_batch: np.ndarray,
    qpos_indices: list[int],
    gripper_indices: list[int],
    *,
    dtype=np.float32,
) -> np.ndarray:
    """Project a batch of MuJoCo qpos vectors into policy-space states."""
    qpos_batch = np.asarray(qpos_batch)
    if qpos_batch.ndim != 2:
        raise ValueError(f"Expected batched qpos with shape (B, nq), got {qpos_batch.shape}")

    state = np.asarray(qpos_batch[:, qpos_indices], dtype=dtype)
    if gripper_indices:
        state[:, gripper_indices] = np.clip(
            state[:, gripper_indices] / _GRIPPER_CTRL_MAX,
            0.0,
            1.0,
        )
    return state


class MuJoCoYAMEnv(gym.Env):
    """MuJoCo-based simulation of the bimanual YAM robot.

    Actions and observations use the same 14D format as the real robot:
    [left_j1..6, left_grip, right_j1..6, right_grip].
    """

    def __init__(
        self,
        config: RobotSystemConfig,
        chunk_dim: int = 30,
        prompt: str = "fold the towel",
        render_cameras: bool = True,
        camera_backend: LiveRenderBackend = "mujoco",
        camera_gpu_id: int | None = None,
        camera_height: int = 480,
        camera_width: int = 640,
        physics_dt: float = 0.002,
        control_decimation: int = 17,  # 0.002 * 17 ≈ 0.034s ≈ 30Hz
        scene_xml: str | Path | None = None,
    ):
        super().__init__()
        self.config = config
        self.chunk_dim = chunk_dim
        self.prompt = prompt
        self._render_cameras_flag = render_cameras
        self._camera_backend: LiveRenderBackend = camera_backend
        self._camera_gpu_id = camera_gpu_id
        self._camera_height = camera_height
        self._camera_width = camera_width
        self._physics_dt = physics_dt
        self._control_decimation = control_decimation
        self._scene_xml = Path(scene_xml) if scene_xml else _DEFAULT_SCENE_XML
        self._scene_xml_string: str | None = None  # set directly for in-memory XML
        self._task_request: str = ""
        self._task: str = ""  # set by make_env after construction
        self._task_spec: SimTaskSpec | None = None
        self._task_evaluator = None
        self._arm_state_initialized = False

        # Populated by reset() when randomize=True; readable by callers.
        self._last_randomization: Any = None

        self.camera_names = list(config.cameras.keys())
        self.robot_names = list(config.robots.keys())
        self.single_timestep_action_dim = 7 * len(config.robots)  # 14
        self.action_dim = self.single_timestep_action_dim * self.chunk_dim

        # Observation space
        self.observation_space = spaces.Dict(
            {
                "images": spaces.Dict(
                    {
                        name: spaces.Box(
                            low=0,
                            high=255,
                            shape=(
                                config.cameras[name].height,
                                config.cameras[name].width,
                                3,
                            ),
                            dtype=np.uint8,
                        )
                        for name in self.camera_names
                    }
                ),
                "state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.single_timestep_action_dim,),
                    dtype=np.float32,
                ),
            }
        )

        # Action space: (chunk_dim, 14)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.chunk_dim, self.single_timestep_action_dim),
            dtype=np.float32,
        )

        self.setup_model()
        self.cur_step = 0

    # ------------------------------------------------------------------
    # MuJoCo model setup
    # ------------------------------------------------------------------

    def setup_model(self):
        if self._scene_xml_string is not None:
            # MuJoCo's from_xml_string resolves relative asset paths against
            # the process cwd, while from_xml_path resolves them against the
            # XML's directory. Randomizers that rebuild the scene on every
            # reset (scale-replay, dishrack variants, inhand_transfer) feed
            # MuJoCo an XML string; without this detour, scenes whose
            # <compiler> omits meshdir/texturedir — e.g. yam_marker_scene.xml
            # with `<compiler angle="radian"/>` — fail to find
            # assets/marker/... because cwd is wherever the caller ran from.
            # Writing to a tempfile in the original scene's directory lets
            # MuJoCo resolve assets via its usual path-load semantics.
            if self._scene_xml is not None:
                asset_root = Path(self._scene_xml).parent
            else:
                asset_root = Path(__file__).parent / "models"
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".xml", dir=str(asset_root), delete=False,
            ) as handle:
                handle.write(self._scene_xml_string)
                tmp_path = Path(handle.name)
            try:
                model = mujoco.MjModel.from_xml_path(str(tmp_path))
            finally:
                tmp_path.unlink(missing_ok=True)
        else:
            model = mujoco.MjModel.from_xml_path(str(self._scene_xml))
        self._bind_model(model)

    def _bind_model(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.model.opt.timestep = self._physics_dt
        self.data = mujoco.MjData(self.model)
        self._arm_state_initialized = False
        self._camera_provider = None

        if self._render_cameras_flag:
            if self._camera_backend == "mujoco":
                self.renderer = mujoco.Renderer(
                    self.model,
                    height=self._camera_height,
                    width=self._camera_width,
                )
            else:
                self._camera_provider = create_live_camera_provider(
                    model=self.model,
                    data=self.data,
                    backend=self._camera_backend,
                    width=self._camera_width,
                    height=self._camera_height,
                    gpu_id=self._camera_gpu_id,
                    camera_names=tuple(self.camera_names),
                )

        self._build_index_maps()

    def reload_from_xml(self, xml_string: str) -> None:
        """Swap out the MuJoCo model in-place from an XML string.

        Closes the existing renderer, loads a new model/data/renderer from
        the given XML string, and rebuilds index maps.  Does NOT call
        mj_resetData — the caller is responsible for resetting state afterward.
        """
        self._scene_xml_string = xml_string
        self._close_camera_backend()
        self.setup_model()
        self._task_evaluator = make_task_evaluator(self.model, self._task_spec)

    def reload_from_model(self, model: mujoco.MjModel) -> None:
        """Swap in a copy of an already-compiled MuJoCo model."""
        self._scene_xml_string = None
        self._close_camera_backend()
        self._bind_model(copy.deepcopy(model))
        self._task_evaluator = make_task_evaluator(self.model, self._task_spec)

    def _reset_camera_backend(self) -> None:
        if self._camera_provider is not None:
            self._camera_provider.reset()

    def _close_camera_backend(self) -> None:
        if self._camera_provider is not None:
            self._camera_provider.close()
            self._camera_provider = None
        if hasattr(self, "renderer"):
            self.renderer.close()
            del self.renderer

    def _build_index_maps(self):
        """Build mappings from 14D state/action to MuJoCo qpos/ctrl indices."""
        self._qpos_indices: list[int] = []
        self._ctrl_indices: list[int] = []
        self._gripper_indices: list[int] = []  # which of the 14 are grippers

        idx = 0
        for robot_name in self.robot_names:
            # 6 arm joints
            for j in range(1, 7):
                joint_name = f"{robot_name}_joint{j}"
                jnt_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                )
                self._qpos_indices.append(self.model.jnt_qposadr[jnt_id])

                act_name = f"{robot_name}_joint{j}"
                act_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name
                )
                self._ctrl_indices.append(act_id)
                idx += 1

            # Gripper (left_finger joint, gripper actuator)
            finger_name = f"{robot_name}_left_finger"
            finger_jnt_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, finger_name
            )
            self._qpos_indices.append(self.model.jnt_qposadr[finger_jnt_id])

            grip_act_name = f"{robot_name}_gripper"
            grip_act_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, grip_act_name
            )
            self._ctrl_indices.append(grip_act_id)
            self._gripper_indices.append(idx)
            idx += 1

        # Convert to sets for fast lookup
        self._gripper_set = set(self._gripper_indices)

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
        randomize: bool = True,
    ):
        super().reset(seed=seed)
        reset_arm_state = self._get_reset_arm_state()
        mujoco.mj_resetData(self.model, self.data)
        self._set_qpos_from_state(reset_arm_state)
        mujoco.mj_forward(self.model, self.data)

        # Object randomization: perturb free-jointed scene objects.
        # _task_randomizer (set by make_env) takes priority over TASK_RANDOMIZERS
        # so that model-swapping tasks (e.g. inhand_transfer) work correctly.
        self._last_randomization = None
        if randomize:
            randomization_request = None
            if options is not None:
                randomization_request = options.get("randomization")
            randomizer = getattr(self, "_task_randomizer", None)
            if randomizer is None and self._task:
                randomizer = get_task_randomizer(self._task)
            if randomizer is not None:
                self._last_randomization = randomizer.randomize(
                    self.model,
                    self.data,
                    seed=seed,
                    request=randomization_request,
                )

        self._reset_camera_backend()
        if self._task_evaluator is not None:
            self._task_evaluator.reset(nworld=1)
            if self._last_randomization is not None and hasattr(
                self._task_evaluator, "set_active_trash_joints"
            ):
                trash_joints = getattr(self._last_randomization, "metadata", {}).get(
                    "trash_joints"
                )
                if trash_joints is not None:
                    self._task_evaluator.set_active_trash_joints(trash_joints)

        self.cur_step = 0
        info: dict[str, Any] = {}
        if self._last_randomization is not None:
            info["randomization"] = self._last_randomization
        return self.get_obs(), info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(
            self.chunk_dim, self.single_timestep_action_dim
        )
        all_obs = []
        for i in range(self.chunk_dim):
            self._step_single(action[i])
            all_obs.append(self.get_obs())

        final_obs = all_obs[-1]
        chunk_history = self._stack_obs(all_obs)
        task_eval = self.evaluate_task()
        reward = task_eval.scalar_reward() if task_eval is not None else 0.0
        info: dict[str, Any] = {}
        if task_eval is not None:
            info["task_reward"] = reward
            info["task_success"] = task_eval.scalar_success()
            info["task_eval"] = task_eval.to_info(squeeze=True)
        return final_obs, chunk_history, reward, False, False, info

    def close(self):
        self._close_camera_backend()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_qpos_from_state(self, state: np.ndarray):
        """Write a 14D state vector into MuJoCo qpos."""
        for i, qpos_idx in enumerate(self._qpos_indices):
            val = state[i]
            if i in self._gripper_set:
                # Scale [0, 1] policy space → MuJoCo joint space
                val = val * _GRIPPER_CTRL_MAX
            self.data.qpos[qpos_idx] = val
        self._arm_state_initialized = True

    def _get_reset_arm_state(self) -> np.ndarray:
        """Return the arm state to preserve across scene resets."""
        if self._arm_state_initialized:
            return project_policy_state(
                np.asarray(self.data.qpos, dtype=np.float64),
                self._qpos_indices,
                self._gripper_indices,
            )
        return self.get_init_q().astype(np.float32, copy=True)

    def _step_single(self, action_14d: np.ndarray):
        """Apply a single 14D action and step physics."""
        ctrl = np.zeros(self.model.nu)
        for i in range(self.single_timestep_action_dim):
            val = action_14d[i]
            if i in self._gripper_set:
                # Scale [0, 1] policy space → actuator ctrl range
                val = val * _GRIPPER_CTRL_MAX
            ctrl[self._ctrl_indices[i]] = val
        self.data.ctrl[:] = ctrl

        for _ in range(self._control_decimation):
            mujoco.mj_step(self.model, self.data)

    def project_state_from_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """Project an arbitrary qpos vector into the 14D policy-space state."""
        return project_policy_state(
            qpos,
            self._qpos_indices,
            self._gripper_indices,
            dtype=np.float32,
        )

    def get_state(self) -> np.ndarray:
        """Return the current 14D policy-space state from MuJoCo qpos."""
        return self.project_state_from_qpos(self.data.qpos)

    def set_task(self, task: str) -> None:
        """Configure the task name and attach any registered evaluator."""

        resolved = resolve_task(task)
        self._task_request = task
        self._task = resolved.env_task or task
        self._task_spec = resolved.task_spec or maybe_get_task_spec(task)
        self._task_evaluator = make_task_evaluator(self.model, self._task_spec)

    def evaluate_task(self) -> TaskEvalResult | None:
        """Compute the current task reward/success from the current simulation state."""

        if self._task_evaluator is None:
            return None
        qpos_batch = np.asarray(self.data.qpos, dtype=np.float32)[None, :]
        return self._task_evaluator.evaluate_qpos_batch(qpos_batch)

    def get_obs(self) -> dict[str, Any]:
        """Return observation dict with state, images, and metadata."""
        state = self.get_state()

        # Images
        if self._render_cameras_flag:
            images = self._render_cameras()
        else:
            images = {
                name: np.zeros(
                    (3, self._camera_height, self._camera_width), dtype=np.uint8
                )
                for name in self.camera_names
            }

        masks = {name: True for name in self.camera_names}
        sim_time = self.data.time
        timestamps = {name: sim_time for name in self.camera_names}

        return {
            "images": images,
            "masks": masks,
            "state": state,
            "prompt": self.prompt,
            "camera_timestamps": timestamps,
        }

    def _render_cameras(self) -> dict[str, np.ndarray]:
        """Render all cameras and return (3, H, W) uint8 images."""
        if self._camera_provider is not None:
            frames = self._camera_provider.frames_for_step(self.cur_step, self.data.time)
            return {
                name: frame.transpose(2, 0, 1).copy()
                for name, frame in frames.items()
            }

        images = {}
        for name in self.camera_names:
            self.renderer.update_scene(self.data, camera=name)
            img = self.renderer.render()  # (H, W, 3) uint8
            images[name] = img.transpose(2, 0, 1).copy()  # → (3, H, W)
        return images

    def get_init_q(self) -> np.ndarray:
        """Return concatenated init_q across all robots as a flat 14D array."""
        return np.concatenate(
            [np.array(self.config.robots[name].init_q) for name in self.robot_names]
        )

    def _stack_obs(self, obses: list[dict]) -> dict:
        """Stack a list of observations into chunked arrays."""
        stacked: dict[str, Any] = {}
        for key in obses[0]:
            if isinstance(obses[0][key], str):
                stacked[key] = obses[0][key]
            elif isinstance(obses[0][key], dict):
                stacked[key] = {
                    k: np.stack([obs[key][k] for obs in obses])
                    for k in obses[0][key]
                }
            else:
                stacked[key] = np.stack([obs[key] for obs in obses])
        return stacked
