"""NumPy implementation of the J-PARSE algorithm for singularity-aware velocity IK.

J-PARSE (Jacobian-based Projection Algorithm for Resolving Singularities
Effectively) computes a modified pseudo-inverse that handles singular
configurations smoothly via SVD decomposition and singular-direction
feedback.

Reference: https://github.com/chungmin99/jparse
Ported from the JAX implementation in robots_realtime (chungmin99/pyroki#85).
"""

from __future__ import annotations

from typing import Literal

import numpy as np


def jparse_pseudoinverse(
    jacobian: np.ndarray,
    gamma: float = 0.1,
    singular_direction_gain_position: float = 1.0,
    singular_direction_gain_angular: float = 1.0,
    position_dimensions: int | None = None,
    angular_dimensions: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute J-PARSE pseudo-inverse of a Jacobian matrix.

    Decomposes the Jacobian using SVD and constructs a modified pseudo-inverse
    that clamps singular values below a threshold, projects commands onto
    non-singular directions, and provides smooth feedback in singular
    directions.

    Args:
        jacobian: (m, n) Jacobian matrix.
        gamma: Singularity threshold in (0, 1). Directions with
            sigma/sigma_max < gamma are treated as singular.
        singular_direction_gain_position: Gain for position dimensions.
        singular_direction_gain_angular: Gain for angular dimensions.
        position_dimensions: Number of position rows in the Jacobian.
        angular_dimensions: Number of angular rows in the Jacobian.

    Returns:
        (J_parse, nullspace_projector) — each shaped (n, m) and (n, n).
    """
    J = np.asarray(jacobian, dtype=np.float64)
    m, n = J.shape

    if position_dimensions is None and angular_dimensions is None:
        pos_dims = m
        ang_dims = 0
    else:
        assert position_dimensions is not None and angular_dimensions is not None
        assert position_dimensions + angular_dimensions == m
        pos_dims = position_dimensions
        ang_dims = angular_dimensions

    U, S, Vt = np.linalg.svd(J, full_matrices=True)
    k = S.shape[0]  # min(m, n)

    sigma_max = np.max(S) if S.size > 0 else 1.0
    threshold = gamma * sigma_max

    non_singular = S > threshold

    # Safety singular values: clamp below threshold.
    S_safety = np.where(non_singular, S, threshold)
    # Projection singular values: keep only non-singular directions.
    S_proj = np.where(non_singular, S, 0.0)

    U_k = U[:, :k]
    Vt_k = Vt[:k, :]

    J_safety = (U_k * S_safety[None, :]) @ Vt_k
    J_proj = (U_k * S_proj[None, :]) @ Vt_k

    J_safety_pinv = np.linalg.pinv(J_safety)
    J_proj_pinv = np.linalg.pinv(J_proj)

    # Singular direction feedback gains.
    phi = np.where(non_singular, 0.0, S / (sigma_max * gamma + 1e-12))
    singular_gains = np.concatenate([
        np.full(pos_dims, singular_direction_gain_position),
        np.full(ang_dims, singular_direction_gain_angular),
    ])
    Kp = np.diag(singular_gains)
    Phi_singular = (U_k * phi[None, :]) @ U_k.T @ Kp

    J_parse = J_safety_pinv @ J_proj @ J_proj_pinv + J_safety_pinv @ Phi_singular

    # Nullspace projector: N = I - J_safety^+ @ J_safety
    nullspace = np.eye(n) - J_safety_pinv @ J_safety

    return J_parse, nullspace


def manipulability_measure(jacobian: np.ndarray) -> float:
    """Yoshikawa's manipulability measure: sqrt(det(J @ J^T))."""
    J = np.asarray(jacobian, dtype=np.float64)
    return float(np.sqrt(max(0.0, np.linalg.det(J @ J.T))))


def compute_pseudoinverse(
    jacobian: np.ndarray,
    method: Literal["jparse", "pinv", "dls"] = "jparse",
    gamma: float = 0.1,
    dls_damping: float = 0.05,
    position_dimensions: int = 3,
    angular_dimensions: int = 0,
    singular_direction_gain_position: float = 1.0,
    singular_direction_gain_angular: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pseudo-inverse and nullspace projector using the chosen method.

    Returns:
        (J_inv, nullspace_projector).
    """
    n = jacobian.shape[1]

    if method == "jparse":
        return jparse_pseudoinverse(
            jacobian,
            gamma=gamma,
            singular_direction_gain_position=singular_direction_gain_position,
            singular_direction_gain_angular=singular_direction_gain_angular,
            position_dimensions=position_dimensions,
            angular_dimensions=angular_dimensions,
        )
    elif method == "pinv":
        J_inv = np.linalg.pinv(jacobian)
        N = np.eye(n) - J_inv @ jacobian
        return J_inv, N
    elif method == "dls":
        J = jacobian
        J_inv = np.linalg.inv(J.T @ J + dls_damping**2 * np.eye(n)) @ J.T
        N = np.eye(n) - J_inv @ jacobian
        return J_inv, N
    else:
        raise ValueError(f"Unknown IK method: {method}")
