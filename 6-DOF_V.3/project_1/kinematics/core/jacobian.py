"""
Jacobian matrix computation for manipulators.

This module provides functions to compute the geometric Jacobian
relating joint velocities to end-effector velocities.
"""

import numpy as np
from .transforms import dh_transform


def geometric_jacobian(dh_params, joint_angles):
    """
    Compute the 6x6 geometric Jacobian for a 6-DOF manipulator.

    The Jacobian relates joint velocities to end-effector twist:
    [v; ω] = J * dq

    Args:
        dh_params: Nx4 array of DH parameters [a, alpha, d, theta_offset]
        joint_angles: N-element array of joint angles (radians)

    Returns:
        J: 6xN Jacobian matrix
        T_end: 4x4 end-effector transformation matrix
    """
    T = np.eye(4)
    origins = []
    z_axes = []

    # Compute forward kinematics and store intermediate frames
    for i in range(len(dh_params)):
        a, alpha, d, theta_offset = dh_params[i]
        theta = theta_offset + joint_angles[i]
        T = T @ dh_transform(a, alpha, d, theta)
        origins.append(T[:3, 3])
        z_axes.append(T[:3, 2])

    # End-effector position
    o_n = origins[-1]

    # Initialize Jacobian
    n_joints = len(dh_params)
    J = np.zeros((6, n_joints))

    # Base frame
    o_prev = np.array([0, 0, 0])
    z_prev = np.array([0, 0, 1])

    # First column (base joint)
    J[:3, 0] = np.cross(z_prev, o_n - o_prev)
    J[3:, 0] = z_prev

    # Remaining columns
    for i in range(1, n_joints):
        J[:3, i] = np.cross(z_axes[i - 1], o_n - origins[i - 1])
        J[3:, i] = z_axes[i - 1]

    return J, T
