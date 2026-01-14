"""
High-level Robot class for kinematics operations.

Provides a unified interface for forward and inverse kinematics.
"""

import numpy as np
from typing import Optional, List, Literal

from .config import RobotConfig, DEFAULT_ROBOT
from .core import forward_kinematics, forward_kinematics_batch
from .solvers import AnalyticalIKSolver, NumericalIKSolver
from .solvers.ik_solution import IKSolution
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

    def forward_kinematics_batch(self, joint_angles_batch: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics for multiple configurations (batch operation).

        Efficiently computes FK for a batch of joint configurations.
        Useful for trajectory planning and workspace analysis.

        Args:
            joint_angles_batch: MxN array where M is number of configurations,
                               N is number of joints

        Returns:
            T_batch: Mx4x4 array of transformation matrices

        Example:
            >>> robot = Robot()
            >>> configs = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            ...                      [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]])
            >>> transforms = robot.forward_kinematics_batch(configs)
            >>> print(transforms.shape)  # (2, 4, 4)
        """
        return forward_kinematics_batch(self.dh_params, joint_angles_batch)

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
    ) -> List[IKSolution]:
        """
        Find all valid IK solutions with quality metrics.

        Args:
            T_target: 4x4 target transformation matrix
            **kwargs: Additional solver parameters

        Returns:
            solutions: List of IKSolution objects sorted by error
        """
        return self.analytical_solver.solve_all(T_target, **kwargs)

    def select_best_solution(
        self,
        solutions: List[IKSolution],
        q_current: Optional[np.ndarray] = None,
        prefer_config: Optional[str] = None
    ) -> Optional[IKSolution]:
        """
        Select best solution from multiple candidates.

        Args:
            solutions: List of IKSolution objects
            q_current: Current joint configuration (prefer closest)
            prefer_config: Preferred configuration name (e.g., "elbow_up")

        Returns:
            Best IKSolution, or None if no solutions
        """
        if not solutions:
            return None

        # Filter for valid solutions only
        valid_solutions = [s for s in solutions if s.is_valid]
        if not valid_solutions:
            return None

        # If configuration preference specified, filter for it
        if prefer_config:
            config_matches = [s for s in valid_solutions
                            if prefer_config in s.configuration]
            if config_matches:
                valid_solutions = config_matches

        # If current configuration provided, select closest
        if q_current is not None:
            distances = [s.distance_to(q_current) for s in valid_solutions]
            best_idx = int(np.argmin(distances))
            return valid_solutions[best_idx]

        # Otherwise, return most accurate solution
        return valid_solutions[0]  # Already sorted by error

    def validate_solution(
        self,
        joint_angles: np.ndarray,
        T_target: np.ndarray,
        pos_tol: float = 1e-4,
        rot_tol: float = 1e-3
    ) -> tuple[bool, float, float]:
        """
        Validate an IK solution by computing FK and measuring error.

        Args:
            joint_angles: Joint configuration to validate
            T_target: Target transformation matrix
            pos_tol: Position error tolerance (meters)
            rot_tol: Rotation error tolerance (radians)

        Returns:
            is_valid: Whether solution meets tolerances
            pos_err: Position error (meters)
            rot_err: Rotation error (radians)
        """
        T_achieved = self.forward_kinematics(joint_angles)
        pos_err, rot_err = error_check(T_achieved, T_target)
        is_valid = (pos_err < pos_tol) and (rot_err < rot_tol)
        return is_valid, pos_err, rot_err

    def validate_solutions(
        self,
        solutions: List[IKSolution],
        T_target: np.ndarray,
        pos_tol: float = 1e-4,
        rot_tol: float = 1e-3
    ) -> List[IKSolution]:
        """
        Filter solutions to only those meeting accuracy requirements.

        Args:
            solutions: List of IKSolution objects to validate
            T_target: Target transformation matrix
            pos_tol: Position error tolerance (meters)
            rot_tol: Rotation error tolerance (radians)

        Returns:
            Filtered list of accurate solutions
        """
        accurate_solutions = []
        for sol in solutions:
            if sol.pos_error < pos_tol and sol.rot_error < rot_tol:
                accurate_solutions.append(sol)
        return accurate_solutions

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
