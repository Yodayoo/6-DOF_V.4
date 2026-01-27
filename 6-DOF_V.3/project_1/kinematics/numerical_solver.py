import numpy as np
from scipy.linalg import norm
from .helper_func import dh_T, wrap_to_pi
from .settings import DH

# -----------------------------------------------
#                Geometric Jacobian
# -----------------------------------------------
def geometric_jacobian(dh_params, q):
    """Compute 6xN Jacobian for an N-DOF arm."""
    n = len(dh_params)
    T = np.eye(4)
    origins = []
    z_axes = []

    for i in range(n):
        a, alpha, d, t0 = dh_params[i]
        T = T @ dh_T(a, alpha, d, t0 + q[i])
        origins.append(T[:3, 3].copy())
        z_axes.append(T[:3, 2].copy())

    o_n = origins[-1]
    J = np.zeros((6, n))
    o_prev = np.array([0.0, 0.0, 0.0])
    z_prev = np.array([0.0, 0.0, 1.0])

    # Base joint
    J[:3, 0] = np.cross(z_prev, o_n - o_prev)
    J[3:, 0] = z_prev

    for i in range(1, n):
        J[:3, i] = np.cross(z_axes[i - 1], o_n - origins[i - 1])
        J[3:, i] = z_axes[i - 1]

    return J, T

# -----------------------------------------------
#                    Pose error
# -----------------------------------------------
def pose_error(T_current, T_target):
    """Compute 6D pose error (position + orientation)."""
    pc, Rc = T_current[:3, 3], T_current[:3, :3]
    pt, Rt = T_target[:3, 3], T_target[:3, :3]

    ep = pt - pc

    R_err = Rt @ Rc.T
    cos_theta = (np.trace(R_err) - 1) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if abs(theta) < 1e-8:
        eo = np.zeros(3)
    else:
        w_hat = (R_err - R_err.T) / (2 * np.sin(theta))
        eo = np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]]) * theta

    return np.hstack([ep, eo])

# -----------------------------------------------
#           Damped Least Squares step
# -----------------------------------------------
def dls_step(J, e, lam=1e-3):
    """Compute DLS step using solve instead of inverse for efficiency."""
    JJt = J @ J.T
    # Use solve instead of inv for better numerical stability and performance
    # Solves (JJt + lam^2*I) * x = e, then returns J.T @ x
    regularized = JJt + lam**2 * np.eye(JJt.shape[0])
    return J.T @ np.linalg.solve(regularized, e)

# -----------------------------------------------
#            Iterative IK solver (with limits)
# -----------------------------------------------
def numerical_ik_solve(
    T_target, dh_params=DH, q0=np.zeros(6),
    max_iters=600, tol=1e-6, lam=0.01, step=0.75,
    q_min=None, q_max=None
):
    """
    Iterative IK solver using Damped Least Squares.

    Args:
        T_target: Target 4x4 transformation matrix
        dh_params: DH parameter matrix (Nx4)
        q0: Initial joint angles
        max_iters: Maximum iterations (default 600)
        tol: Convergence tolerance (default 1e-6)
        lam: Damping factor (default 0.01)
        step: Step size (default 0.75)
        q_min: Joint lower limits
        q_max: Joint upper limits

    Returns:
        Joint angles that achieve the target pose (or best effort)
    """
    q = q0.copy().astype(float)
    n = len(dh_params)

    # Default limits: [-pi, pi]
    if q_min is None:
        q_min = -np.pi * np.ones(n)
    if q_max is None:
        q_max = np.pi * np.ones(n)

    for k in range(max_iters):
        J, T = geometric_jacobian(dh_params, q)
        e = pose_error(T, T_target)

        # Check convergence
        if norm(e) < tol:
            return wrap_to_pi(q)

        dq = dls_step(J, e, lam)
        q = q + step * dq

        # Wrap and clamp to limits
        q = wrap_to_pi(q)
        q = np.clip(q, q_min, q_max)

    return wrap_to_pi(q)
