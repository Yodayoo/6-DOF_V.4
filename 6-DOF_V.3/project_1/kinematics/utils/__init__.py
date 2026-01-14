"""
Utility functions for kinematics.

Provides angle utilities, error metrics, and validation functions.
"""

from .angle_utils import wrap_to_pi, wrap_to_2pi, angle_distance
from .error_metrics import pose_error, rotation_error_geodesic, error_check
from .validation import (
    check_joint_limits,
    clamp_to_limits,
    check_reachability,
    check_reachability_detailed,
    detect_singularity,
)

__all__ = [
    'wrap_to_pi',
    'wrap_to_2pi',
    'angle_distance',
    'pose_error',
    'rotation_error_geodesic',
    'error_check',
    'check_joint_limits',
    'clamp_to_limits',
    'check_reachability',
    'check_reachability_detailed',
    'detect_singularity',
]
