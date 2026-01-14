"""
Numerical inverse kinematics solver using damped least squares.

Implements an iterative IK solver with adaptive damping and step sizing.
"""

import numpy as np
from .base_solver import IKSolver
from ..core import geometric_jacobian
from ..utils import pose_error, wrap_to_pi


class NumericalIKSolver(IKSolver):
    """
    Numerical IK solver using damped least squares (DLS) method.

    Uses Levenberg-Marquardt style adaptive damping and backtracking
    line search for robust convergence.
    """

    def __init__(self, dh_params, q_min=None, q_max=None,
                 max_iters=600, lam=0.01, step=0.75,
                 pos_tol=1e-6, rot_tol=1e-4, adaptive_step=True):
        """
        Initialize numerical IK solver.

        Args:
            dh_params: Nx4 array of DH parameters
            q_min: Minimum joint limits
            q_max: Maximum joint limits
            max_iters: Maximum number of iterations
            lam: Base damping factor
            step: Initial step size
            pos_tol: Position convergence tolerance (meters)
            rot_tol: Rotation convergence tolerance (radians)
            adaptive_step: Enable adaptive step sizing
        """
        super().__init__(dh_params, q_min, q_max)

        self.max_iters = max_iters
        self.lam = lam
        self.step = step
        self.pos_tol = pos_tol
        self.rot_tol = rot_tol
        self.adaptive_step = adaptive_step

    def solve(self, T_target, q_init=None, **kwargs):
        """
        Solve inverse kinematics numerically.

        Args:
            T_target: 4x4 target transformation matrix
            q_init: Initial joint configuration guess
            **kwargs: Override solver parameters

        Returns:
            q: Converged joint configuration, or None if failed
        """
        # Use provided initial guess or zeros
        if q_init is None:
            q = np.zeros(self.n_joints)
        else:
            q = q_init.copy()

        # Extract solver parameters (allow override)
        max_iters = kwargs.get('max_iters', self.max_iters)
        lam = kwargs.get('lam', self.lam)
        step = kwargs.get('step', self.step)
        pos_tol = kwargs.get('pos_tol', self.pos_tol)
        rot_tol = kwargs.get('rot_tol', self.rot_tol)
        adaptive_step = kwargs.get('adaptive_step', self.adaptive_step)

        current_step = step
        prev_error_norm = np.inf

        for k in range(max_iters):
            J, T = geometric_jacobian(self.dh_params, q)
            e = pose_error(T, T_target)

            # Separate position and rotation errors
            pos_error = np.linalg.norm(e[:3])
            rot_error = np.linalg.norm(e[3:])
            error_norm = np.linalg.norm(e)

            # Check convergence
            if pos_error < pos_tol and rot_error < rot_tol:
                return wrap_to_pi(q)

            # Compute step using damped least squares
            dq = self._dls_step(J, e, lam)

            # Adaptive step sizing with backtracking
            if adaptive_step:
                if error_norm > prev_error_norm and current_step > 0.1:
                    current_step = max(0.1, current_step * 0.7)
                elif error_norm < 0.8 * prev_error_norm and current_step < 1.0:
                    current_step = min(1.0, current_step * 1.2)

            # Update configuration
            q_new = q + current_step * dq

            # Wrap and clamp angles
            q_new = wrap_to_pi(q_new)
            q_new = np.minimum(np.maximum(q_new, self.q_min), self.q_max)

            q = q_new
            prev_error_norm = error_norm

        # Failed to converge
        return None

    def _dls_step(self, J, e, lam):
        """
        Compute damped least squares step with adaptive damping.

        Args:
            J: 6xN Jacobian matrix
            e: 6-element error vector
            lam: Base damping factor

        Returns:
            dq: Joint velocity update
        """
        JJt = J @ J.T

        # Adaptive damping based on error magnitude and conditioning
        error_norm = np.linalg.norm(e)
        try:
            condition_num = np.linalg.cond(JJt)
            condition_num = np.clip(condition_num, 1.0, 1e6)
            adaptive_lam = lam * (1.0 + error_norm) * np.sqrt(condition_num / 100.0)
            adaptive_lam = np.clip(adaptive_lam, lam * 0.1, lam * 100.0)
        except:
            adaptive_lam = lam * (1.0 + error_norm)

        # Compute pseudoinverse using damping
        damping_matrix = JJt + adaptive_lam**2 * np.eye(JJt.shape[0])
        try:
            J_dls = J.T @ np.linalg.solve(damping_matrix, np.eye(JJt.shape[0]))
        except np.linalg.LinAlgError:
            # Fallback to higher damping if singular
            damping_matrix = JJt + (adaptive_lam * 10)**2 * np.eye(JJt.shape[0])
            J_dls = J.T @ np.linalg.solve(damping_matrix, np.eye(JJt.shape[0]))

        return J_dls @ e


# Legacy function interface for backward compatibility
def numerical_ik_solve(
    T_target,
    dh_parms=None,
    q0=None,
    max_iters=600,
    tol=1e-10,
    lam=0.01,
    step=0.75,
    q_min=None,
    q_max=None,
    pos_tol=1e-6,
    rot_tol=1e-4,
    adaptive_step=True
):
    """
    Legacy interface for numerical IK solver.

    Args:
        T_target: 4x4 target transformation matrix
        dh_parms: Nx4 array of DH parameters
        q0: Initial joint configuration
        max_iters: Maximum iterations
        tol: Legacy tolerance (deprecated)
        lam: Base damping factor
        step: Initial step size
        q_min: Minimum joint limits
        q_max: Maximum joint limits
        pos_tol: Position tolerance
        rot_tol: Rotation tolerance
        adaptive_step: Enable adaptive step sizing

    Returns:
        Joint configuration
    """
    if dh_parms is None:
        from ..config import DH
        dh_parms = DH

    if q0 is None:
        q0 = np.zeros(len(dh_parms))

    solver = NumericalIKSolver(
        dh_parms, q_min, q_max,
        max_iters=max_iters,
        lam=lam,
        step=step,
        pos_tol=pos_tol,
        rot_tol=rot_tol,
        adaptive_step=adaptive_step
    )

    return solver.solve(T_target, q_init=q0)
