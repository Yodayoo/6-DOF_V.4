"""
Forward kinematics computation for serial manipulators.

This module provides functions to compute the end-effector pose
given joint configurations.
"""

import numpy as np
from typing import Union, Sequence
from .transforms import dh_transform


def forward_kinematics(
    dh_params: np.ndarray,
    joint_angles: Union[np.ndarray, Sequence[float]]
) -> np.ndarray:
    """
    Compute forward kinematics for a serial manipulator.

    Computes the transformation from base frame to end-effector frame
    by chaining DH transformations for each joint.

    Args:
        dh_params: Nx4 array of DH parameters [a, alpha, d, theta_offset]
        joint_angles: N-element array of joint angles (radians)

    Returns:
        T: 4x4 homogeneous transformation matrix from base to end-effector
    """
    T = np.eye(4)
    n_joints = len(dh_params)

    for i in range(n_joints):
        a, alpha, d, theta_offset = dh_params[i]
        theta = theta_offset + joint_angles[i]
        T = T @ dh_transform(a, alpha, d, theta)

    return T


# Legacy alias for backward compatibility
fk_chain = forward_kinematics
