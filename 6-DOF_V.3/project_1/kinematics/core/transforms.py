"""
Transformation matrix utilities for robotic kinematics.

This module provides functions for computing transformation matrices
using Denavit-Hartenberg (DH) parameters.
"""

import numpy as np


def dh_transform(a, alpha, d, theta):
    """
    Compute a 4x4 transformation matrix using Modified DH parameters.

    The Modified DH convention defines transformations from frame i-1 to frame i.

    Args:
        a: Link length (distance along x_{i-1})
        alpha: Link twist (rotation about x_{i-1})
        d: Link offset (distance along z_i)
        theta: Joint angle (rotation about z_i)

    Returns:
        T: 4x4 homogeneous transformation matrix
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)

    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,      sa,     ca,     d],
        [0,       0,      0,     1]
    ], dtype=float)


# Legacy alias for backward compatibility
dh_T = dh_transform
