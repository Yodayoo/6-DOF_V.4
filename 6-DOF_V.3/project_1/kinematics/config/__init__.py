"""
Configuration management for kinematics.

Provides robot and solver configuration classes.
"""

from .robot_config import (
    RobotConfig,
    DEFAULT_ROBOT,
    DH,
    z_offset,
    max_r,
    q_min,
    q_max,
)
from .solver_config import (
    AnalyticalSolverConfig,
    NumericalSolverConfig,
    DEFAULT_ANALYTICAL_CONFIG,
    DEFAULT_NUMERICAL_CONFIG,
    FAST_NUMERICAL_CONFIG,
    PRECISE_NUMERICAL_CONFIG,
    ROBUST_NUMERICAL_CONFIG,
)

__all__ = [
    'RobotConfig',
    'DEFAULT_ROBOT',
    'DH',
    'z_offset',
    'max_r',
    'q_min',
    'q_max',
    'AnalyticalSolverConfig',
    'NumericalSolverConfig',
    'DEFAULT_ANALYTICAL_CONFIG',
    'DEFAULT_NUMERICAL_CONFIG',
    'FAST_NUMERICAL_CONFIG',
    'PRECISE_NUMERICAL_CONFIG',
    'ROBUST_NUMERICAL_CONFIG',
]
