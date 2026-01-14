"""
Angle manipulation utilities.

Provides functions for wrapping angles and computing angle distances.
"""

import numpy as np


def wrap_to_pi(angles):
    """
    Wrap angles to [-π, π) interval.

    Args:
        angles: Scalar or array of angles in radians

    Returns:
        Wrapped angles in [-π, π)
    """
    return (angles + np.pi) % (2 * np.pi) - np.pi


def wrap_to_2pi(angles):
    """
    Wrap angles to [0, 2π) interval.

    Args:
        angles: Scalar or array of angles in radians

    Returns:
        Wrapped angles in [0, 2π)
    """
    return angles % (2 * np.pi)


def angle_distance(angles1, angles2):
    """
    Compute the sum of absolute angle differences between two configurations.

    Uses wrapped differences to account for angle periodicity.

    Args:
        angles1: First set of angles (radians)
        angles2: Second set of angles (radians)

    Returns:
        Total angular distance (sum of absolute wrapped differences)
    """
    diff = wrap_to_pi(angles1 - angles2)
    return np.sum(np.abs(diff))
