"""
Validation utilities for joint limits and reachability.

Provides functions to check joint limit violations and workspace reachability.
"""

import numpy as np


def check_joint_limits(joint_angles, q_min=None, q_max=None):
    """
    Check if joint angles are within specified limits.

    Args:
        joint_angles: N-element array of joint angles (radians)
        q_min: N-element array of minimum joint limits (None = no limit)
        q_max: N-element array of maximum joint limits (None = no limit)

    Returns:
        valid: True if all joints are within limits
        violations: List of joint indices violating limits
    """
    n = len(joint_angles)

    if q_min is None:
        q_min = -np.pi * np.ones(n)
    if q_max is None:
        q_max = np.pi * np.ones(n)

    violations = []
    for i in range(n):
        if joint_angles[i] < q_min[i] or joint_angles[i] > q_max[i]:
            violations.append(i)

    return len(violations) == 0, violations


def clamp_to_limits(joint_angles, q_min=None, q_max=None):
    """
    Clamp joint angles to specified limits.

    Args:
        joint_angles: N-element array of joint angles (radians)
        q_min: N-element array of minimum joint limits (None = -π)
        q_max: N-element array of maximum joint limits (None = π)

    Returns:
        clamped: Joint angles clamped to limits
    """
    n = len(joint_angles)

    if q_min is None:
        q_min = -np.pi * np.ones(n)
    if q_max is None:
        q_max = np.pi * np.ones(n)

    return np.minimum(np.maximum(joint_angles, q_min), q_max)


def check_reachability(target_position, dh_params, safety_margin=0.0):
    """
    Fast geometric check if a target position is within workspace.

    Computes conservative reachability bounds based on link lengths.

    Args:
        target_position: 3-element target position [x, y, z]
        dh_params: Nx4 array of DH parameters [a, alpha, d, theta_offset]
        safety_margin: Additional margin to subtract from reach (meters)

    Returns:
        reachable: True if position is likely reachable
        distance: Distance from base to target
        max_reach: Maximum reach of manipulator
    """
    # Compute link lengths
    link_lengths = [np.hypot(a, d) for (a, _, d, _) in dh_params]

    # Maximum reach (sum of all link lengths)
    max_reach = sum(link_lengths) - safety_margin

    # Minimum reach (absolute difference of longest and sum of others)
    # This is a conservative estimate
    min_reach = 0.0  # Could be improved with more detailed analysis

    # Distance to target
    distance = np.linalg.norm(target_position)

    reachable = min_reach <= distance <= max_reach

    return reachable, distance, max_reach
