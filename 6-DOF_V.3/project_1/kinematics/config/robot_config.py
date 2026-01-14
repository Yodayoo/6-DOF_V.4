"""
Robot configuration using DH parameters.

Defines the kinematic structure and joint limits for the 6-DOF manipulator.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class RobotConfig:
    """
    Configuration for a robotic manipulator.

    Attributes:
        name: Robot identifier
        dh_params: Nx4 array of DH parameters [a, alpha, d, theta_offset]
        q_min: Minimum joint limits (radians)
        q_max: Maximum joint limits (radians)
        description: Human-readable description
    """
    name: str
    dh_params: np.ndarray
    q_min: Optional[np.ndarray] = None
    q_max: Optional[np.ndarray] = None
    description: str = ""

    def __post_init__(self):
        """Set default joint limits if not provided."""
        n_joints = len(self.dh_params)

        if self.q_min is None:
            self.q_min = -np.pi * np.ones(n_joints)

        if self.q_max is None:
            self.q_max = np.pi * np.ones(n_joints)

    @property
    def n_joints(self):
        """Number of joints."""
        return len(self.dh_params)

    @property
    def link_lengths(self):
        """Compute effective link lengths."""
        return np.array([np.hypot(a, d) for (a, _, d, _) in self.dh_params])

    @property
    def max_reach(self):
        """Maximum reach of the manipulator."""
        return np.sum(self.link_lengths)

    @property
    def base_height(self):
        """Height of first joint above base."""
        return self.dh_params[0, 2]  # d1


# Default 6-DOF robot configuration
DH = np.array([
    [0.00,   np.pi/2,  1.00,  0.00],     # Joint 1
    [2.50,   0.00,     0.00,  np.pi/2],  # Joint 2
    [0.00,   np.pi/2,  0.00,  np.pi/2],  # Joint 3
    [0.00,  -np.pi/2,  2.50,  0.00],     # Joint 4
    [0.00,   np.pi/2,  0.00,  0.00],     # Joint 5
    [0.00,   0.00,     0.50,  0.00],     # Joint 6
])

DEFAULT_ROBOT = RobotConfig(
    name="6DOF_Manipulator",
    dh_params=DH,
    q_min=None,  # Will default to -π
    q_max=None,  # Will default to π
    description="6-DOF serial manipulator with spherical wrist"
)

# Workspace parameters
z_offset = np.hypot(DH[0, 0], DH[0, 2])
max_r = sum(np.hypot(DH[1:4, 0], DH[1:4, 2]))

# Legacy exports for backward compatibility
q_min = None
q_max = None
