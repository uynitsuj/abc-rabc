"""Batched GPU-accelerated simulation environment for policy deployment."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)

from yam_sim.env import (
    MuJoCoYAMEnv,
    _GRIPPER_CTRL_MAX,
    project_policy_state_batch,
)
from yam_sim.rendering.replay.renderer import RendererWrapper, WarpReplayRuntime
from yam_sim.task_eval import TaskEvalResult, make_task_evaluator


@dataclass(frozen=True)
class WorldResetInfo:
    seed: int | None
    randomization: Any


class BatchedWarpYAMEnv:
    """Run multiple identical YAM worlds in parallel with Warp physics."""

    def __init__(
        self,
        base_env: MuJoCoYAMEnv,
        *,
        num_worlds: int,
        camera_backend: Literal["mjwarp", "madrona", "mujoco"],
        camera_gpu_id: int | None = None,
        render_cameras: bool = True,
        mask_variable_count: bool = False,
        mask_reset_options: dict | None = None,
        fixed_reset_options: dict | None = None,
        visual_seed: int | None = None,
        visual_config: dict | None = None,
    ) -> None:
        if num_worlds < 1:
            raise ValueError(f"num_worlds must be >= 1, got {num_worlds}")

        self.base_env = base_env
        self.model = base_env.model
        self.data = base_env.data
        self.config = base_env.config
        self.prompt = base_env.prompt
        self.camera_names = list(base_env.camera_names)
        self.robot_names = list(base_env.robot_names)
        self.num_worlds = num_worlds
        self.chunk_dim = base_env.chunk_dim
        self.single_timestep_action_dim = base_env.single_timestep_action_dim
        self._render_cameras = render_cameras
        self._camera_backend = camera_backend
        self._camera_gpu_id = camera_gpu_id
        self._camera_height = base_env._camera_height
        self._camera_width = base_env._camera_width
        self._control_decimation = base_env._control_decimation
        self._qpos_indices = list(base_env._qpos_indices)
        self._ctrl_indices = list(base_env._ctrl_indices)
        self._gripper_indices = list(base_env._gripper_indices)
        self._gripper_set = set(self._gripper_indices)
        self._task = base_env._task
        self._task_spec = base_env._task_spec
        self._task_evaluator = make_task_evaluator(self.model, self._task_spec)

        self._runtime = WarpReplayRuntime(
            self.model,
            self.data,
            nworld=num_worlds,
            gpu_id=camera_gpu_id,
            # MuJoCo 3.x models report nconmax/njmax as 0 (unset), so the
            # multiplier terms collapse and the FLOOR is the real allocation.
            # 512 was too small for sweep (needs ~540 with 4 scraps x 5 worlds):
            # mjwarp's narrowphase silently drops overflowing contacts, letting
            # objects interpenetrate until the solver NaNs the whole batch.
            nconmax=max(4096, int(getattr(self.model, "nconmax", 64)) * num_worlds * 16),
            njmax=max(16384, int(getattr(self.model, "njmax", 128)) * num_worlds * 64),
        )
        # Camera rendering. The mjwarp/madrona path renders all worlds on-GPU in a
        # single batched call (fast, but its images differ from the MuJoCo-GL
        # renderer used to produce the training data). The "mujoco" path renders
        # each world's observation with MuJoCo's own GL renderer from that world's
        # qpos -- physics still runs batched on mjwarp; only the observation
        # rendering is MuJoCo-GL, so policies see in-distribution images.
        self._renderer = None
        self._mjgl_renderer = None
        self._mjgl_data = None
        if render_cameras:
            if camera_backend == "mujoco":
                import mujoco

                if camera_gpu_id is not None:
                    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", str(camera_gpu_id))
                os.environ.setdefault("MUJOCO_GL", "egl")
                self._mjgl_renderer = mujoco.Renderer(
                    self.model,
                    height=self._camera_height,
                    width=self._camera_width,
                )
                self._mjgl_data = mujoco.MjData(self.model)
            else:
                self._renderer = RendererWrapper(
                    backend=camera_backend,
                    runtime=self._runtime,
                    cam_res=(self._camera_width, self._camera_height),
                    gpu_id=camera_gpu_id,
                )

        self._camera_index = {
            self.model.cam(i).name: i for i in range(self.model.ncam)
        }
        self._needs_renderer_reset = True
        self._step_count = 0
        self._episode_index = 0
        self._world_reset_info: list[WorldResetInfo] = []
        self._warned_layout_mismatch = False
        self._warned_geom_drift = False

        # Variable object-count masking (see make_batched_env). The warp model is
        # built once at MAX object count; per world we remap the natural K-object
        # state onto this MAX layout by joint name and park the unused slots.
        self._mask_variable_count = bool(mask_variable_count)
        self._mask_reset_options = dict(mask_reset_options) if mask_reset_options else {}
        # Fixed object-count eval: per-world reset options that pin the count to the
        # construction value while randomizing poses (constant nq -> standard path).
        self._fixed_reset_options = dict(fixed_reset_options) if fixed_reset_options else None
        # Grouped visual-variety eval: the deterministic visual config (mesh variant +
        # scale + color) baked into the construction model for this whole batch, plus
        # the seed that produced it. Surfaced in reset() info for per-episode recording.
        self._visual_seed = visual_seed
        self._visual_config = dict(visual_config) if visual_config else None
        self._max_joint_layout: list[dict[str, Any]] = []
        self._max_nominal_qpos: np.ndarray | None = None
        if self._mask_variable_count:
            self._max_joint_layout = self._build_joint_layout(self.model)
            self._max_nominal_qpos = np.asarray(self.data.qpos, dtype=np.float32).copy()

    def _world_seed(self, *, base_seed: int | None, world_index: int) -> int | None:
        if base_seed is None:
            return None
        return int(base_seed) + world_index

    @staticmethod
    def _build_joint_layout(model: Any) -> list[dict[str, Any]]:
        """(name, qpos addr, qpos size, is_free) for every joint, in model order."""
        import mujoco

        free = int(mujoco.mjtJoint.mjJNT_FREE)
        ball = int(mujoco.mjtJoint.mjJNT_BALL)
        layout: list[dict[str, Any]] = []
        for j in range(model.njnt):
            jtype = int(model.jnt_type[j])
            size = 7 if jtype == free else (4 if jtype == ball else 1)
            layout.append(
                {
                    "name": model.jnt(j).name,
                    "adr": int(model.jnt_qposadr[j]),
                    "size": size,
                    "free": jtype == free,
                }
            )
        return layout

    def _remap_qpos_to_max(self, live_model: Any, live_qpos: np.ndarray) -> np.ndarray:
        """Map a live (K-object) qpos onto the MAX-count warp layout by joint name.

        Joints present in the live model are copied by name; object freejoints that
        exist only in the MAX model (the unused slots for this world) are parked far
        below the scene -- staggered in x so they never collide -- mirroring
        ``SweepRandomizer._inactive_trash_states``. Evaluators treat parked
        (low-z) objects as inactive.
        """
        live_qpos = np.asarray(live_qpos, dtype=np.float32)
        live_index: dict[str, tuple[int, int]] = {}
        for jl in self._build_joint_layout(live_model):
            live_index[jl["name"]] = (jl["adr"], jl["size"])

        qpos = self._max_nominal_qpos.copy()
        parked = 0
        for jl in self._max_joint_layout:
            name, adr, size, is_free = jl["name"], jl["adr"], jl["size"], jl["free"]
            if name in live_index:
                ladr, lsize = live_index[name]
                n = min(size, lsize)
                qpos[adr : adr + n] = live_qpos[ladr : ladr + n]
            elif is_free:
                qpos[adr : adr + 3] = (-1.5 - 0.1 * parked, 0.0, -1.0)
                qpos[adr + 3 : adr + 7] = (1.0, 0.0, 0.0, 0.0)
                parked += 1
        return qpos

    def _snapshot_cpu_state(
        self, data: Any | None = None, model: Any | None = None
    ) -> dict[str, np.ndarray | None]:
        # Read from the supplied (live) model/data when given. The task
        # randomizer rebinds base_env.model/.data on every reset that reloads
        # the scene (scale/variant DR), so the cached self.data goes stale and
        # would only ever hold the un-randomized construction state.
        data = self.data if data is None else data
        model = self.model if model is None else model
        act = None
        if model.na > 0 and hasattr(data, "act"):
            act = np.asarray(data.act, dtype=np.float32).copy()
        mocap_pos = None
        mocap_quat = None
        if model.nmocap > 0:
            mocap_pos = np.asarray(data.mocap_pos, dtype=np.float32).copy()
            mocap_quat = np.asarray(data.mocap_quat, dtype=np.float32).copy()
        return {
            "qpos": np.asarray(data.qpos, dtype=np.float32).copy(),
            "qvel": np.asarray(data.qvel, dtype=np.float32).copy(),
            "ctrl": np.asarray(data.ctrl, dtype=np.float32).copy(),
            "act": act,
            "mocap_pos": mocap_pos,
            "mocap_quat": mocap_quat,
            "time": np.float32(data.time),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        randomize: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        snapshots: list[dict[str, np.ndarray | None]] = []
        reset_info: list[WorldResetInfo] = []

        warp_nq = int(self._runtime.mjm.nq)
        warp_nv = int(self._runtime.mjm.nv)
        for world_idx in range(self.num_worlds):
            world_seed = self._world_seed(base_seed=seed, world_index=world_idx)
            if self._mask_variable_count:
                # Variable-count masking: reset to the natural count with a
                # canonical (fixed) variant+scale, then remap the K-object state
                # onto the MAX warp layout and park the unused object slots.
                world_options = {"randomization": dict(self._mask_reset_options)}
                self.base_env.reset(
                    seed=world_seed, options=world_options, randomize=randomize
                )
                live_model = self.base_env.model
                live_data = self.base_env.data
                snap = self._snapshot_cpu_state(live_data, live_model)
                snap["qpos"] = self._remap_qpos_to_max(live_model, snap["qpos"])
                snap["qvel"] = np.zeros((warp_nv,), dtype=np.float32)
                snapshots.append(snap)
                reset_info.append(
                    WorldResetInfo(
                        seed=world_seed,
                        randomization=self.base_env._last_randomization,
                    )
                )
                continue

            world_options = options
            if self._fixed_reset_options is not None:
                world_options = {"randomization": dict(self._fixed_reset_options)}
            else:
                # Per-world SCALE DR cannot be represented by the shared warp/
                # render model: a scale reload keeps nq/nv (slipping the layout
                # guard below) but changes geometry, so poses sampled on the
                # rescaled scene penetrate the construction-scale geometry and
                # blow up the solver (NaN qpos -> invisible arms, skybox
                # cameras). Default it off for per-world resets unless the
                # caller explicitly opted in.
                rand = (world_options or {}).get("randomization")
                if rand is None or isinstance(rand, dict):
                    rand = dict(rand or {})
                    rand.setdefault("randomize_scales", False)
                    world_options = {**(world_options or {}), "randomization": rand}
            self.base_env.reset(seed=world_seed, options=world_options, randomize=randomize)
            # The randomizer may have rebound base_env.model/.data via a scene
            # reload. The warp runtime is built once from the construction-time
            # model, so we can only feed it a per-world state whose layout still
            # matches (pose DR keeps nq/nv fixed). If a reload changed the
            # layout (variant/count DR -- which a single shared warp model
            # cannot represent anyway) fall back to the construction buffer so
            # we never load a mismatched qpos into warp.
            live_model = self.base_env.model
            live_data = self.base_env.data
            if int(live_model.nq) == warp_nq and int(live_model.nv) == warp_nv:
                if live_model is not self.model and not self._warned_geom_drift:
                    warp_sizes = np.asarray(self._runtime.mjm.geom_size)
                    live_sizes = np.asarray(live_model.geom_size)
                    if warp_sizes.shape != live_sizes.shape or not np.allclose(
                        warp_sizes, live_sizes
                    ):
                        logger.warning(
                            "%s: per-world scene reload kept nq/nv but changed "
                            "geometry (e.g. same-layout variant/scale swap); the "
                            "shared warp model cannot represent it -- physics/"
                            "rendering may mismatch this world's sampled state",
                            type(self).__name__,
                        )
                        self._warned_geom_drift = True
                snapshots.append(self._snapshot_cpu_state(live_data, live_model))
            else:
                if not self._warned_layout_mismatch:
                    logger.warning(
                        "%s: randomizer reload changed model layout "
                        "(nq %d->%d); batched warp uses a fixed shared model, so "
                        "this world's randomization is dropped (using nominal state)",
                        type(self).__name__,
                        warp_nq,
                        int(live_model.nq),
                    )
                    self._warned_layout_mismatch = True
                snapshots.append(self._snapshot_cpu_state(self.data, self.model))
            reset_info.append(
                WorldResetInfo(
                    seed=world_seed,
                    randomization=self.base_env._last_randomization,
                )
            )

        qpos = np.stack([snap["qpos"] for snap in snapshots], axis=0)
        qvel = np.stack([snap["qvel"] for snap in snapshots], axis=0)
        ctrl = np.stack([snap["ctrl"] for snap in snapshots], axis=0)
        act = None
        if snapshots[0]["act"] is not None:
            act = np.stack([snap["act"] for snap in snapshots], axis=0)
        mocap_pos = None
        mocap_quat = None
        if snapshots[0]["mocap_pos"] is not None:
            mocap_pos = np.stack([snap["mocap_pos"] for snap in snapshots], axis=0)
            mocap_quat = np.stack([snap["mocap_quat"] for snap in snapshots], axis=0)
        time_arr = np.asarray([snap["time"] for snap in snapshots], dtype=np.float32)

        self._runtime.reset_from_mujoco()
        self._runtime.load_state_batch(
            qpos=qpos,
            qvel=qvel,
            ctrl=ctrl,
            act=act,
            mocap_pos=mocap_pos,
            mocap_quat=mocap_quat,
            time=time_arr,
        )
        self._runtime.forward()
        self._needs_renderer_reset = True
        self._step_count = 0
        self._episode_index += 1
        self._world_reset_info = reset_info
        if self._task_evaluator is not None:
            self._task_evaluator.reset(nworld=self.num_worlds)
            if hasattr(self._task_evaluator, "set_active_trash_joints"):
                active_trash_joints = [
                    getattr(item.randomization, "metadata", {}).get("trash_joints")
                    if item.randomization is not None
                    else None
                    for item in reset_info
                ]
                if any(joints is not None for joints in active_trash_joints):
                    if not all(joints is not None for joints in active_trash_joints):
                        raise ValueError("Sweep active trash metadata missing for some worlds")
                    self._task_evaluator.set_active_trash_joints(active_trash_joints)
        obs = self.get_obs()
        info = {
            "world_seeds": [item.seed for item in reset_info],
            "world_randomization": [item.randomization for item in reset_info],
            "visual_seed": self._visual_seed,
            "visual_config": self._visual_config,
        }
        return obs, info

    def _controls_from_actions(self, action_batch: np.ndarray) -> np.ndarray:
        action_batch = np.asarray(action_batch, dtype=np.float32)
        expected = (self.num_worlds, self.single_timestep_action_dim)
        if action_batch.shape != expected:
            raise ValueError(f"Expected batched actions with shape {expected}, got {action_batch.shape}")

        scaled = action_batch.copy()
        if self._gripper_indices:
            scaled[:, self._gripper_indices] *= _GRIPPER_CTRL_MAX

        ctrl = np.zeros((self.num_worlds, self.model.nu), dtype=np.float32)
        ctrl[:, self._ctrl_indices] = scaled
        return ctrl

    def _state_only_obs(self) -> dict[str, Any]:
        state = self._state_batch()
        masks = {
            name: np.zeros((self.num_worlds,), dtype=bool)
            for name in self.camera_names
        }
        if hasattr(self._runtime.d_warp, "time"):
            ts = self._runtime.d_warp.time.numpy()[: self.num_worlds].copy()
        else:
            ts = np.zeros((self.num_worlds,), dtype=np.float32)
        timestamps = {name: ts.copy() for name in self.camera_names}
        return {
            "images": {},
            "masks": masks,
            "state": state,
            "prompt": [self.prompt] * self.num_worlds,
            "camera_timestamps": timestamps,
        }

    def step(
        self,
        action_batch: np.ndarray,
        *,
        render_obs: bool = True,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        ctrl = self._controls_from_actions(action_batch)
        self._runtime.set_ctrl_batch(ctrl)
        self._runtime.step(nstep=self._control_decimation)
        self._step_count += 1

        info = {
            "sim_time": self._runtime.d_warp.time.numpy()[: self.num_worlds].copy()
            if hasattr(self._runtime.d_warp, "time")
            else None,
            "sim_step": self._step_count,
            "task_name": self._task,
        }
        task_eval = self.evaluate_task()
        reward = 0.0
        if task_eval is not None:
            reward = float(task_eval.reward.mean())
            info["task_reward"] = task_eval.reward.copy()
            info["task_success"] = task_eval.success.copy()
            info["task_eval"] = task_eval.to_info(squeeze=False)
        obs = self.get_obs() if render_obs else self._state_only_obs()
        return obs, reward, False, False, info

    def qpos_batch(self) -> np.ndarray:
        """Full MuJoCo ``qpos`` for all live worlds, shape ``(num_worlds, nq)``.

        This is the complete environment state (robot + every object freejoint),
        suitable for deterministic offline re-rendering: feed each world's qpos
        trajectory straight to the replay renderer (no physics, just visualize).
        """
        return self._runtime.d_warp.qpos.numpy()[: self.num_worlds].copy()

    def mocap_batch(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Per-world mocap pose for all live worlds, or ``None`` if the model has no
        mocap bodies. Returns ``(mocap_pos, mocap_quat)`` with shapes
        ``(num_worlds, nmocap, 3)`` / ``(num_worlds, nmocap, 4)``.

        Mocap bodies (e.g. the mug_flip tray) hold their pose in per-world ``data``
        rather than the shared model, so this is needed alongside ``qpos_batch`` to
        fully describe / re-render the scene (the tray varies per world).
        """
        if int(self.model.nmocap) <= 0:
            return None
        pos = self._runtime.d_warp.mocap_pos.numpy()[: self.num_worlds].copy()
        quat = self._runtime.d_warp.mocap_quat.numpy()[: self.num_worlds].copy()
        return pos, quat

    def _state_batch(self) -> np.ndarray:
        qpos = self.qpos_batch()
        return project_policy_state_batch(
            qpos,
            self._qpos_indices,
            self._gripper_indices,
            dtype=np.float32,
        )

    def _render_image_batch_mjgl(self) -> dict[str, np.ndarray]:
        """Render each world's cameras with MuJoCo-GL from its qpos (training renderer).

        Physics already ran batched on mjwarp; here we copy each world's qpos into a
        single MjData and render the scene cameras with ``mujoco.Renderer`` so the
        observation distribution matches the MuJoCo-GL training data.
        """
        import mujoco

        qpos = self.qpos_batch()  # (num_worlds, nq)
        # Mocap bodies (e.g. the mug_flip tray) hold their pose in per-world data, not
        # qpos -- without this the render would show every world's tray at one shared
        # default while physics has it per-world, corrupting the policy's observations.
        mocap = self.mocap_batch()  # (pos, quat) per world, or None
        out = {
            name: np.empty(
                (self.num_worlds, 3, self._camera_height, self._camera_width),
                dtype=np.uint8,
            )
            for name in self.camera_names
        }
        data = self._mjgl_data
        renderer = self._mjgl_renderer
        for w in range(self.num_worlds):
            data.qpos[:] = qpos[w]
            data.qvel[:] = 0.0
            if mocap is not None:
                data.mocap_pos[:] = mocap[0][w]
                data.mocap_quat[:] = mocap[1][w]
            # Render-only forward: the physics already ran on mjwarp; here we only
            # need world-space poses for the camera. Use the position-stage subset
            # (kinematics + com + camera/light) instead of full mj_forward so we do
            # NOT run collision detection / the constraint solver on this throwaway
            # snapshot. That keeps the image identical while avoiding the CPU
            # contact-stack overflow that mj_forward hits on contact-heavy scenes
            # (e.g. sweep_paper: thousands of scrap contacts). Collisions in the
            # actual simulation are unaffected -- they run on mjwarp during step().
            mujoco.mj_kinematics(self.model, data)
            mujoco.mj_comPos(self.model, data)
            mujoco.mj_camlight(self.model, data)
            for name in self.camera_names:
                renderer.update_scene(data, camera=name)
                out[name][w] = renderer.render().transpose(2, 0, 1)  # (H,W,3)->(3,H,W)
        return out

    def _render_image_batch(self) -> dict[str, np.ndarray]:
        if not self._render_cameras or (
            self._renderer is None and self._mjgl_renderer is None
        ):
            return {
                name: np.zeros(
                    (self.num_worlds, 3, self._camera_height, self._camera_width),
                    dtype=np.uint8,
                )
                for name in self.camera_names
            }

        if self._camera_backend == "mujoco":
            return self._render_image_batch_mjgl()

        if self._needs_renderer_reset:
            images = self._renderer.reset_numpy(actual_batch=self.num_worlds)
            self._needs_renderer_reset = False
        else:
            images = self._renderer.render_numpy(actual_batch=self.num_worlds)

        return {
            name: images[:, self._camera_index[name]].transpose(0, 3, 1, 2).copy()
            for name in self.camera_names
            if name in self._camera_index
        }

    def get_obs(self) -> dict[str, Any]:
        state = self._state_batch()
        images = self._render_image_batch()
        masks = {
            name: np.ones((self.num_worlds,), dtype=bool)
            for name in self.camera_names
        }
        if hasattr(self._runtime.d_warp, "time"):
            ts = self._runtime.d_warp.time.numpy()[: self.num_worlds].copy()
        else:
            ts = np.zeros((self.num_worlds,), dtype=np.float32)
        timestamps = {name: ts.copy() for name in self.camera_names}
        return {
            "images": images,
            "masks": masks,
            "state": state,
            "prompt": [self.prompt] * self.num_worlds,
            "camera_timestamps": timestamps,
        }

    def evaluate_task(self) -> TaskEvalResult | None:
        """Compute task reward/success for all live worlds."""

        if self._task_evaluator is None:
            return None
        qpos = self.qpos_batch()
        return self._task_evaluator.evaluate_qpos_batch(qpos)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
        if self._mjgl_renderer is not None:
            self._mjgl_renderer.close()
        self.base_env.close()

    def rollout_metadata(self) -> dict[str, Any]:
        return {
            "task": self._task,
            "prompt": self.prompt,
            "camera_names": list(self.camera_names),
            "camera_height": self._camera_height,
            "camera_width": self._camera_width,
            "num_worlds": self.num_worlds,
            "step_count": self._step_count,
            "camera_backend": self._camera_backend,
            "camera_gpu_id": self._camera_gpu_id,
            "world_seeds": [item.seed for item in self._world_reset_info],
        }
