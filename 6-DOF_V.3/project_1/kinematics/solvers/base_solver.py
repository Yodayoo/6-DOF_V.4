"""
Base class for inverse kinematics solvers.

Defines the interface that all IK solvers should implement.
"""

from abc import ABC, abstractmethod
import numpy as np


class IKSolver(ABC):
    """
    Abstract base class for inverse kinematics solvers.

    All IK solvers should inherit from this class and implement the solve method.
    """

    def __init__(self, dh_params, q_min=None, q_max=None):
        """
        Initialize IK solver.

        Args:
            dh_params: Nx4 array of DH parameters [a, alpha, d, theta_offset]
            q_min: N-element array of minimum joint limits (None = -π)
            q_max: N-element array of maximum joint limits (None = π)
        """
        self.dh_params = dh_params
        self.n_joints = len(dh_params)

        if q_min is None:
            q_min = -np.pi * np.ones(self.n_joints)
        if q_max is None:
            q_max = np.pi * np.ones(self.n_joints)

        self.q_min = q_min
        self.q_max = q_max

    @abstractmethod
    def solve(self, T_target, q_init=None, **kwargs):
        """
        Solve inverse kinematics for target pose.

        Args:
            T_target: 4x4 target transformation matrix
            q_init: Initial guess for joint configuration (optional)
            **kwargs: Additional solver-specific parameters

        Returns:
            q: N-element joint configuration, or None if no solution found
        """
        pass

    def solve_all(self, T_target, **kwargs):
        """
        Find all valid IK solutions for target pose.

        Args:
            T_target: 4x4 target transformation matrix
            **kwargs: Additional solver-specific parameters

        Returns:
            solutions: List of valid joint configurations
        """
        # Default implementation returns single solution
        solution = self.solve(T_target, **kwargs)
        return [solution] if solution is not None else []
