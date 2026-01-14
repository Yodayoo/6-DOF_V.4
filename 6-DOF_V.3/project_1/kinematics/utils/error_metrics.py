"""
Error metrics for pose and rotation comparisons.

Provides functions to compute position and rotation errors
between transformation matrices.
"""

import numpy as np
from typing import Tuple


def pose_error(T_current: np.ndarray, T_target: np.ndarray) -> np.ndarray:
    """
    Compute 6D pose error between current and target transformations.

    Returns a 6-element error vector [position_error; orientation_error]
    suitable for use in iterative IK solvers.

    Args:
        T_current: 4x4 current transformation matrix
        T_target: 4x4 target transformation matrix

    Returns:
        error: 6-element error vector [ep_x, ep_y, ep_z, eo_x, eo_y, eo_z]
    """
    pc, Rc = T_current[:3, 3], T_current[:3, :3]
    pt, Rt = T_target[:3, 3], T_target[:3, :3]

    # Position error
    ep = pt - pc

    # Rotation error using axis-angle representation
    R_err = Rt @ Rc.T
    cos_theta = (np.trace(R_err) - 1) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if abs(theta) < 1e-8:
        eo = np.zeros(3)
    else:
        w_hat = (R_err - R_err.T) / (2 * np.sin(theta))
        eo = np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]]) * theta

    return np.hstack([ep, eo])


def rotation_error_geodesic(R1: np.ndarray, R2: np.ndarray) -> float:
    """
    Compute geodesic distance between two rotation matrices on SO(3).

    This is the mathematically correct way to measure rotation error,
    treating rotations as elements of the Special Orthogonal group SO(3).

    Args:
        R1: First 3x3 rotation matrix
        R2: Second 3x3 rotation matrix

    Returns:
        angle: Geodesic distance in radians [0, π]
    """
    R_rel = R2.T @ R1
    cos_angle = (np.trace(R_rel) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
    angle = np.arccos(cos_angle)
    return angle


def error_check(T_solved: np.ndarray, T_true: np.ndarray) -> Tuple[float, float]:
    """
    Compute position and rotation errors between two transformation matrices.

    Uses proper SO(3) geodesic distance for rotation error instead of Frobenius norm.

    Args:
        T_solved: 4x4 transformation matrix from solver
        T_true: 4x4 ground truth transformation matrix

    Returns:
        pos_err: Position error (Euclidean distance in meters)
        rot_err: Rotation error (geodesic distance in radians)
    """
    pos_err = np.linalg.norm(T_true[:3, 3] - T_solved[:3, 3])

    # Compute rotation error using SO(3) geodesic distance
    R_true = T_true[:3, :3]
    R_solved = T_solved[:3, :3]
    rot_err = rotation_error_geodesic(R_true, R_solved)

    return pos_err, rot_err
