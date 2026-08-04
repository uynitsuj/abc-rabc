"""MuJoCo-backed inverse kinematics using J-PARSE velocity IK.

Uses MuJoCo's native ``mj_jac`` for Jacobian computation and ``mj_forward``
for forward kinematics, making the MJCF model the single source of truth
for the robot's kinematic chain.  No URDF conversion required.
"""

from __future__ import annotations

from typing import Literal

import mujoco
import numpy as np

from yam_sim.ik.jparse import compute_pseudoinverse, manipulability_measure


def _rotation_matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to wxyz quaternion."""
    mat_flat = np.zeros(9, dtype=np.float64)
    mat_flat[:] = R.flatten()
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, mat_flat)
    return quat  # wxyz


def _quat_wxyz_to_rotation_matrix(wxyz: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion to 3x3 rotation matrix."""
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, np.asarray(wxyz, dtype=np.float64))
    return mat.reshape(3, 3)


def _orientation_error(target_wxyz: np.ndarray, current_wxyz: np.ndarray) -> np.ndarray:
    """Compute orientation error as a 3D rotation vector (axis * angle).

    Returns a vector in the tangent space (world frame) that rotates from
    current to target.
    """
    # Ensure unit quaternions
    tw = np.asarray(target_wxyz, dtype=np.float64)
    cw = np.asarray(current_wxyz, dtype=np.float64)
    tw = tw / (np.linalg.norm(tw) + 1e-12)
    cw = cw / (np.linalg.norm(cw) + 1e-12)

    # Shortest path
    if np.dot(tw, cw) < 0:
        tw = -tw

    # Error quaternion: q_err = q_target * q_current^{-1}
    # MuJoCo: mju_mulQuat, mju_negQuat
    cw_inv = np.array([cw[0], -cw[1], -cw[2], -cw[3]])
    q_err = np.zeros(4, dtype=np.float64)
    mujoco.mju_mulQuat(q_err, tw, cw_inv)

    # Convert to axis-angle
    angle = 2.0 * np.arccos(np.clip(q_err[0], -1.0, 1.0))
    if abs(angle) < 1e-8:
        return np.zeros(3)
    axis = q_err[1:4] / (np.sin(angle / 2.0) + 1e-12)
    return axis * angle


class MuJoCoIKSolver:
    """Velocity IK solver for a single arm using MuJoCo's Jacobian.

    Wraps the J-PARSE algorithm with MuJoCo FK and Jacobian computation.

    Args:
        model: MuJoCo model.
        data: MuJoCo data (shared with the simulation).
        arm_joint_names: List of joint names for the arm (6 revolute joints).
        tcp_body_name: Name of the end-effector body (e.g. "left_link_6").
        tcp_site_name: Optional site name to use as TCP reference instead
            of the body origin.  When set, FK and Jacobians are computed
            at the site's world position/orientation (e.g. "left_grasp_site"
            for the fingertip frame rather than the wrist).
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        arm_joint_names: list[str],
        tcp_body_name: str,
        tcp_site_name: str | None = None,
    ):
        self.model = model
        self.data = data
        self.ndof = len(arm_joint_names)

        # Resolve IDs and DOF addresses
        self.tcp_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, tcp_body_name
        )
        assert self.tcp_body_id >= 0, f"Body '{tcp_body_name}' not found"

        # Optional site for TCP reference (tooltip / grasp frame)
        self.tcp_site_id: int | None = None
        if tcp_site_name is not None:
            self.tcp_site_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, tcp_site_name
            )
            assert self.tcp_site_id >= 0, f"Site '{tcp_site_name}' not found"

        self.joint_ids: list[int] = []
        self.dof_indices: list[int] = []
        self.qpos_indices: list[int] = []
        self.joint_lower: np.ndarray = np.zeros(self.ndof)
        self.joint_upper: np.ndarray = np.zeros(self.ndof)

        for i, name in enumerate(arm_joint_names):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert jid >= 0, f"Joint '{name}' not found"
            self.joint_ids.append(jid)
            self.dof_indices.append(int(model.jnt_dofadr[jid]))
            self.qpos_indices.append(int(model.jnt_qposadr[jid]))
            self.joint_lower[i] = model.jnt_range[jid, 0]
            self.joint_upper[i] = model.jnt_range[jid, 1]

        self._dof_slice = np.array(self.dof_indices, dtype=int)

    def get_joint_config(self) -> np.ndarray:
        """Read current joint configuration from MuJoCo qpos."""
        return np.array([self.data.qpos[i] for i in self.qpos_indices])

    def get_tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Get current TCP position (3,) and orientation as wxyz quaternion (4,).

        If a tcp_site was specified, returns the site's world pose;
        otherwise returns the body origin pose.
        """
        if self.tcp_site_id is not None:
            pos = self.data.site_xpos[self.tcp_site_id].copy()
            xmat = self.data.site_xmat[self.tcp_site_id].reshape(3, 3)
        else:
            pos = self.data.xpos[self.tcp_body_id].copy()
            xmat = self.data.xmat[self.tcp_body_id].reshape(3, 3)
        wxyz = _rotation_matrix_to_quat_wxyz(xmat)
        return pos, wxyz

    def compute_jacobian(self, point: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Compute position and rotation Jacobians for the TCP.

        Uses the site world position (if configured) or body origin as the
        default reference point for Jacobian computation.

        Args:
            point: 3D point in world frame for Jacobian computation.
                Defaults to the TCP site/body position.

        Returns:
            (jacp, jacr) — each shaped (3, ndof), extracted for the arm DOFs.
        """
        if point is None:
            if self.tcp_site_id is not None:
                point = self.data.site_xpos[self.tcp_site_id]
            else:
                point = self.data.xpos[self.tcp_body_id]
        jacp_full = np.zeros((3, self.model.nv))
        jacr_full = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jacp_full, jacr_full, point, self.tcp_body_id)
        return jacp_full[:, self._dof_slice], jacr_full[:, self._dof_slice]

    def step(
        self,
        target_position: np.ndarray,
        target_wxyz: np.ndarray | None = None,
        *,
        method: Literal["jparse", "pinv", "dls"] = "jparse",
        gamma: float = 0.1,
        singular_direction_gain_position: float = 1.0,
        singular_direction_gain_angular: float = 1.0,
        position_gain: float = 5.0,
        orientation_gain: float = 1.0,
        nullspace_gain: float = 0.5,
        max_joint_velocity: float = 2.0,
        dls_damping: float = 0.05,
        dt: float = 0.02,
        home_cfg: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Run a single velocity IK step.

        Computes Jacobian, applies the chosen pseudo-inverse method, computes
        joint velocities, integrates, and clamps to joint limits.

        Args:
            target_position: Target end-effector position (3,).
            target_wxyz: Target orientation as wxyz quaternion (4,).
                If None, only position is tracked (3-DOF IK).
            method: IK method — "jparse", "pinv", or "dls".
            gamma: J-PARSE singularity threshold.
            position_gain: Proportional gain for position error.
            orientation_gain: Proportional gain for orientation error.
            nullspace_gain: Gain for nullspace motion toward home.
            max_joint_velocity: Maximum joint velocity (rad/s).
            dls_damping: Damping factor for DLS method.
            dt: Time step for integration.
            home_cfg: Home configuration for nullspace bias.
                If None, uses joint midpoints.

        Returns:
            (new_cfg, info_dict) where new_cfg is the updated joint angles.
        """
        cfg = self.get_joint_config()
        current_pos, current_wxyz = self.get_tcp_pose()

        # Position error
        target_position = np.asarray(target_position, dtype=np.float64)
        pos_error = target_position - current_pos
        pos_error_mag = float(np.linalg.norm(pos_error))

        # Compute Jacobian and build desired velocity
        jacp, jacr = self.compute_jacobian()
        position_only = target_wxyz is None

        if position_only:
            v_des = position_gain * pos_error
            jacobian = jacp
            pos_dims, ang_dims = 3, 0
        else:
            omega_error = _orientation_error(target_wxyz, current_wxyz)
            # Clamp orientation error magnitude
            omega_mag = np.linalg.norm(omega_error)
            if omega_mag > 1.0:
                omega_error = omega_error * (1.0 / omega_mag)
            v_des = np.concatenate([
                position_gain * pos_error,
                orientation_gain * omega_error,
            ])
            jacobian = np.vstack([jacp, jacr])
            pos_dims, ang_dims = 3, 3

        # Compute pseudo-inverse
        J_inv, N = compute_pseudoinverse(
            jacobian,
            method=method,
            gamma=gamma,
            dls_damping=dls_damping,
            position_dimensions=pos_dims,
            angular_dimensions=ang_dims,
            singular_direction_gain_position=singular_direction_gain_position,
            singular_direction_gain_angular=singular_direction_gain_angular,
        )

        # Primary task joint velocities
        dq = J_inv @ v_des

        # Nullspace motion toward home
        if nullspace_gain > 0:
            if home_cfg is None:
                home = (self.joint_lower + self.joint_upper) / 2.0
            else:
                home = np.asarray(home_cfg, dtype=np.float64)
            dq_null = N @ (-nullspace_gain * (cfg - home))
            dq = dq + dq_null

        # Velocity limits
        max_vel = float(np.max(np.abs(dq)))
        if max_vel > max_joint_velocity:
            dq = dq * (max_joint_velocity / max_vel)

        # Integrate
        new_cfg = cfg + dq * dt

        # Clamp to joint limits
        new_cfg = np.clip(new_cfg, self.joint_lower, self.joint_upper)

        info = {
            "position_error": pos_error_mag,
            "orientation_error": float(np.linalg.norm(omega_error)) if not position_only else 0.0,
            "max_joint_vel": max_vel,
            "manipulability": manipulability_measure(jacp),
        }

        return new_cfg, info
