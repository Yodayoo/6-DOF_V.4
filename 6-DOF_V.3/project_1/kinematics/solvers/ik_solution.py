"""
IK solution data structure with quality metrics.

Provides a structured way to represent IK solutions with error metrics
and configuration information for solution comparison and selection.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class IKSolution:
    """
    Represents an inverse kinematics solution with quality metrics.

    Attributes:
        q: Joint configuration (radians)
        pos_error: Position error in meters
        rot_error: Rotation error in radians
        configuration: String describing the configuration (e.g., "elbow_up", "elbow_down")
        is_valid: Whether solution satisfies joint limits
        solver_method: Method used to compute solution ("analytical" or "numerical")
        iterations: Number of iterations (for numerical solver)
    """
    q: np.ndarray
    pos_error: float
    rot_error: float
    configuration: str = ""
    is_valid: bool = True
    solver_method: str = ""
    iterations: Optional[int] = None

    @property
    def total_error(self) -> float:
        """Combined error metric (position + rotation)."""
        # Weight rotation error to match position units (assuming 0.1m ~ 1rad)
        return self.pos_error + 0.1 * self.rot_error

    @property
    def is_accurate(self, pos_tol: float = 1e-4, rot_tol: float = 1e-3) -> bool:
        """Check if solution meets accuracy requirements."""
        return self.pos_error < pos_tol and self.rot_error < rot_tol

    def __lt__(self, other: 'IKSolution') -> bool:
        """Compare solutions by total error (for sorting)."""
        return self.total_error < other.total_error

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"IKSolution(config='{self.configuration}', "
            f"pos_err={self.pos_error:.2e}m, rot_err={self.rot_error:.2e}rad, "
            f"valid={self.is_valid}, method='{self.solver_method}')"
        )

    def distance_to(self, q_other: np.ndarray) -> float:
        """
        Compute configuration distance to another joint configuration.

        Args:
            q_other: Another joint configuration

        Returns:
            Sum of absolute joint angle differences
        """
        from ..utils import angle_distance
        return angle_distance(self.q, q_other)
