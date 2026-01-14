"""
Kinematics package for 6-DOF robotic manipulator.

Provides forward and inverse kinematics solvers with both analytical
and numerical methods.

New API (recommended):
    from kinematics import Robot
    robot = Robot()
    solution = robot.inverse_kinematics(T_target, method='analytical')

Legacy API (still supported):
    from kinematics import analytical_ik_solve, numerical_ik_solve, fk_chain, DH
    solution = analytical_ik_solve(T_target, dh_params=DH)
"""

import warnings

# New modular API
from .robot import Robot
from .config import (
    RobotConfig,
    DEFAULT_ROBOT,
    DH,
    z_offset,
    max_r,
    q_min,
    q_max,
)
from .core import (
    dh_transform,
    dh_T,
    forward_kinematics,
    fk_chain,
    geometric_jacobian,
)
from .solvers import (
    IKSolution,
    AnalyticalIKSolver,
    NumericalIKSolver,
    analytical_ik_solve,
    numerical_ik_solve,
)
from .utils import (
    wrap_to_pi,
    wrap_to_2pi,
    angle_distance,
    pose_error,
    rotation_error_geodesic,
    error_check,
    check_joint_limits,
    clamp_to_limits,
    check_reachability,
    check_reachability_detailed,
    detect_singularity,
)

# Backward compatibility - import from old locations with deprecation warnings
def _deprecated_import(old_module, new_location):
    """Helper to show deprecation warning."""
    warnings.warn(
        f"Importing from 'kinematics.{old_module}' is deprecated. "
        f"Use 'from kinematics import {new_location}' instead.",
        DeprecationWarning,
        stacklevel=3
    )

# Legacy imports from settings.py
try:
    from .settings import DH as _DH_old
    from .settings import z_offset as _z_offset_old
    from .settings import max_r as _max_r_old
except ImportError:
    pass  # Old files may not exist yet

# Legacy imports from helper_func.py
try:
    from .helper_func import dh_T as _dh_T_old
    from .helper_func import wrap_to_pi as _wrap_to_pi_old
    from .helper_func import wrap_to_2pi as _wrap_to_2pi_old
    from .helper_func import error_check as _error_check_old
except ImportError:
    pass

# Legacy imports from FK_chain.py
try:
    from .FK_chain import fk_chain as _fk_chain_old
except ImportError:
    pass

# Legacy imports from solvers
try:
    from .analytical_solver import analytical_ik_solve as _analytical_old
    from .numerical_solver import numerical_ik_solve as _numerical_old
except ImportError:
    pass

__all__ = [
    # Main API
    'Robot',

    # Configuration
    'RobotConfig',
    'DEFAULT_ROBOT',
    'DH',
    'z_offset',
    'max_r',
    'q_min',
    'q_max',

    # Core functions
    'dh_transform',
    'dh_T',
    'forward_kinematics',
    'fk_chain',
    'geometric_jacobian',

    # Solvers
    'IKSolution',
    'AnalyticalIKSolver',
    'NumericalIKSolver',
    'analytical_ik_solve',
    'numerical_ik_solve',

    # Utilities
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

__version__ = '3.0.0'
