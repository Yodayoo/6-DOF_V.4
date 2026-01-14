"""
Solver configuration parameters.

Defines default parameters for IK solvers.
"""

from dataclasses import dataclass


@dataclass
class AnalyticalSolverConfig:
    """
    Configuration for analytical IK solver.

    Attributes:
        debug: Enable debug output
        return_all: Return all solutions instead of just best
    """
    debug: bool = False
    return_all: bool = False


@dataclass
class NumericalSolverConfig:
    """
    Configuration for numerical IK solver.

    Attributes:
        max_iters: Maximum number of iterations
        lam: Base damping factor for DLS
        step: Initial step size
        pos_tol: Position convergence tolerance (meters)
        rot_tol: Rotation convergence tolerance (radians)
        adaptive_step: Enable adaptive step sizing
        adaptive_damping: Enable adaptive damping
    """
    max_iters: int = 600
    lam: float = 0.01
    step: float = 0.75
    pos_tol: float = 1e-6
    rot_tol: float = 1e-4
    adaptive_step: bool = True
    adaptive_damping: bool = True


# Default configurations
DEFAULT_ANALYTICAL_CONFIG = AnalyticalSolverConfig()
DEFAULT_NUMERICAL_CONFIG = NumericalSolverConfig()


# Preset configurations for different use cases
FAST_NUMERICAL_CONFIG = NumericalSolverConfig(
    max_iters=100,
    pos_tol=1e-4,
    rot_tol=1e-3,
)

PRECISE_NUMERICAL_CONFIG = NumericalSolverConfig(
    max_iters=1000,
    pos_tol=1e-8,
    rot_tol=1e-6,
)

ROBUST_NUMERICAL_CONFIG = NumericalSolverConfig(
    max_iters=800,
    lam=0.1,
    step=0.5,
    pos_tol=1e-6,
    rot_tol=1e-4,
)
