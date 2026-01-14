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


def check_reachability_detailed(target_position, dh_params, z_offset=0.0):
    """
    Detailed reachability check with diagnostic information.

    Args:
        target_position: 3-element target position [x, y, z]
        dh_params: Nx4 array of DH parameters [a, alpha, d, theta_offset]
        z_offset: Base height offset (meters)

    Returns:
        result: Dictionary with keys:
            - reachable: Boolean indicating if target is reachable
            - distance: Distance from base to target
            - max_reach: Maximum reach
            - min_reach: Minimum reach
            - reason: String describing reachability status
            - warnings: List of warning messages
    """
    result = {
        'reachable': False,
        'distance': 0.0,
        'max_reach': 0.0,
        'min_reach': 0.0,
        'reason': '',
        'warnings': []
    }

    # Adjust target for base height
    adjusted_target = target_position.copy()
    adjusted_target[2] -= z_offset

    # Compute link lengths for main arm (first 3 joints)
    # For 6-DOF manipulator with spherical wrist
    d1 = dh_params[0, 2]  # Base height
    a2 = dh_params[1, 0]  # Upper arm length
    d4 = dh_params[3, 2]  # Forearm length

    # Maximum reach of main arm
    max_reach = a2 + d4
    min_reach = abs(a2 - d4)

    result['max_reach'] = max_reach
    result['min_reach'] = min_reach

    # Compute wrist center distance (2D radial + vertical)
    x, y, z = adjusted_target
    radial_dist = np.hypot(x, y)
    vertical_dist = z

    # Total distance to target
    distance = np.sqrt(radial_dist**2 + vertical_dist**2)
    result['distance'] = distance

    # Check reachability
    if distance > max_reach + 1e-6:
        result['reachable'] = False
        result['reason'] = f"Target too far: {distance:.3f}m > {max_reach:.3f}m"
        return result

    if distance < min_reach - 1e-6:
        result['reachable'] = False
        result['reason'] = f"Target too close: {distance:.3f}m < {min_reach:.3f}m"
        return result

    # Check for near-singularity conditions
    if distance > max_reach * 0.98:
        result['warnings'].append("Near full extension (singularity risk)")

    if distance < min_reach * 1.02:
        result['warnings'].append("Near minimum reach (singularity risk)")

    if radial_dist < 0.01:
        result['warnings'].append("Target near vertical axis (singularity)")

    # Target is reachable
    result['reachable'] = True
    result['reason'] = "Target within workspace"

    return result


def detect_singularity(joint_angles, dh_params, threshold=0.01):
    """
    Detect if a configuration is near a kinematic singularity.

    Singularities occur when the Jacobian loses rank, typically when:
    - Arm is fully extended or retracted
    - Wrist axes align
    - Target is on vertical axis through base

    Args:
        joint_angles: N-element array of joint angles (radians)
        dh_params: Nx4 array of DH parameters
        threshold: Determinant threshold for singularity detection

    Returns:
        is_singular: True if near singularity
        singularity_type: String describing singularity type
        det: Jacobian determinant magnitude
    """
    from ..core import geometric_jacobian

    # Compute Jacobian
    J, _ = geometric_jacobian(dh_params, joint_angles)

    # Check determinant of spatial Jacobian (6x6)
    det = abs(np.linalg.det(J))

    is_singular = det < threshold
    singularity_type = "none"

    if is_singular:
        # Analyze configuration to identify singularity type
        theta2, theta3, theta5 = joint_angles[1], joint_angles[2], joint_angles[4]

        # Elbow singularity (arm fully extended or retracted)
        if abs(theta3) < 0.1:
            singularity_type = "elbow_extended"
        elif abs(theta3 - np.pi) < 0.1 or abs(theta3 + np.pi) < 0.1:
            singularity_type = "elbow_retracted"
        # Wrist singularity (wrist axes align)
        elif abs(theta5) < 0.1 or abs(theta5 - np.pi) < 0.1:
            singularity_type = "wrist_aligned"
        # Shoulder singularity (on vertical axis)
        elif abs(theta2) < 0.1 or abs(theta2 - np.pi/2) < 0.1:
            singularity_type = "shoulder_vertical"
        else:
            singularity_type = "unknown"

    return is_singular, singularity_type, det
