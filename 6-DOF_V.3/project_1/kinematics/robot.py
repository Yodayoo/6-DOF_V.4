"""
High-level Robot class for kinematics operations.

Provides a unified interface for forward and inverse kinematics.
"""

import numpy as np
from typing import Optional, List, Literal

from .config import RobotConfig, DEFAULT_ROBOT
from .core import forward_kinematics
from .solvers import AnalyticalIKSolver, NumericalIKSolver
from .utils import error_check


class Robot:
    """
    High-level interface for robotic manipulator kinematics.

    Provides unified access to forward and inverse kinematics solvers
    with automatic solver selection and validation.
    """

    def __init__(self, config: Optional[RobotConfig] = None):
        """
        Initialize robot with configuration.

        Args:
            config: Robot configuration (uses DEFAULT_ROBOT if None)
        """
        if config is None:
            config = DEFAULT_ROBOT

        self.config = config
        self.dh_params = config.dh_params
        self.q_min = config.q_min
        self.q_max = config.q_max

        # Initialize solvers
        self.analytical_solver = AnalyticalIKSolver(
            self.dh_params, self.q_min, self.q_max
        )
        self.numerical_solver = NumericalIKSolver(
            self.dh_params, self.q_min, self.q_max
        )

    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics.

        Args:
            joint_angles: N-element array of joint angles (radians)

        Returns:
            T: 4x4 transformation matrix from base to end-effector
        """
        return forward_kinematics(self.dh_params, joint_angles)

    def inverse_kinematics(
        self,
        T_target: np.ndarray,
        method: Literal['analytical', 'numerical', 'auto'] = 'analytical',
        q_init: Optional[np.ndarray] = None,
        validate: bool = True,
        **kwargs
    ) -> Optional[np.ndarray]:
        """
        Solve inverse kinematics for target pose.

        Args:
            T_target: 4x4 target transformation matrix
            method: Solver method ('analytical', 'numerical', or 'auto')
            q_init: Initial joint configuration guess
            validate: Verify solution with forward kinematics
            **kwargs: Additional solver-specific parameters

        Returns:
            q: Joint configuration, or None if no solution found
        """
        # Auto method: try analytical first, fall back to numerical
        if method == 'auto':
            solution = self.analytical_solver.solve(T_target, q_init=q_init, **kwargs)
            if solution is None:
                solution = self.numerical_solver.solve(T_target, q_init=q_init, **kwargs)
        elif method == 'analytical':
            solution = self.analytical_solver.solve(T_target, q_init=q_init, **kwargs)
        elif method == 'numerical':
            solution = self.numerical_solver.solve(T_target, q_init=q_init, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Validate solution if requested
        if solution is not None and validate:
            T_achieved = self.forward_kinematics(solution)
            pos_err, rot_err = error_check(T_achieved, T_target)

            # Check if solution is accurate enough
            if pos_err > 1e-3 or rot_err > 1e-2:
                # Solution quality insufficient - try numerical refinement
                if method == 'analytical':
                    solution = self.numerical_solver.solve(
                        T_target, q_init=solution, **kwargs
                    )

        return solution

    def inverse_kinematics_all(
        self,
        T_target: np.ndarray,
        **kwargs
    ) -> List[np.ndarray]:
        """
        Find all valid IK solutions.

        Args:
            T_target: 4x4 target transformation matrix
            **kwargs: Additional solver parameters

        Returns:
            solutions: List of valid joint configurations
        """
        return self.analytical_solver.solve_all(T_target, **kwargs)

    def validate_solution(
        self,
        joint_angles: np.ndarray,
        T_target: np.ndarray
    ) -> tuple[float, float]:
        """
        Validate an IK solution by computing FK and measuring error.

        Args:
            joint_angles: Joint configuration to validate
            T_target: Target transformation matrix

        Returns:
            pos_err: Position error (meters)
            rot_err: Rotation error (radians)
        """
        T_achieved = self.forward_kinematics(joint_angles)
        return error_check(T_achieved, T_target)

    @property
    def n_joints(self) -> int:
        """Number of joints."""
        return self.config.n_joints

    @property
    def max_reach(self) -> float:
        """Maximum reach of manipulator."""
        return self.config.max_reach

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Robot(name='{self.config.name}', "
            f"n_joints={self.n_joints}, "
            f"max_reach={self.max_reach:.3f}m)"
        )
