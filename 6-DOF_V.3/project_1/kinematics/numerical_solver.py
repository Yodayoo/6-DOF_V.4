import numpy as np
from scipy.linalg import norm
from .helper_func import dh_T, wrap_to_pi
from .settings import DH

# 2025-09-19

def wrap_to_2pi(q):
    """Wrap angles in radians to [0, 2*pi)."""
    return q % (2 * np.pi)

# -----------------------------------------------
#                Geometric Jacobian
# -----------------------------------------------
def geometric_jacobian(DH, q):
    """Compute 6x6 Jacobian for a 6DOF arm."""
    T = np.eye(4)
    origins = []
    z_axes = []

    for i in range(len(DH)):
        a, alpha, d, t0 = DH[i]
        T = T @ dh_T(a, alpha, d, t0 + q[i])
        origins.append(T[:3, 3])
        z_axes.append(T[:3, 2])

    o_n = origins[-1]
    J = np.zeros((6, 6))
    o_prev = np.array([0, 0, 0])
    z_prev = np.array([0, 0, 1])

    # Base joint
    J[:3, 0] = np.cross(z_prev, o_n - o_prev)
    J[3:, 0] = z_prev

    for i in range(1, 6):
        J[:3, i] = np.cross(z_axes[i - 1], o_n - origins[i - 1])
        J[3:, i] = z_axes[i - 1]

    return J, T

# -----------------------------------------------
#                    Pose error
# -----------------------------------------------
def pose_error(T_current, T_target):
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
    JJt = J @ J.T
    J_dls = J.T @ np.linalg.inv(JJt + lam**2 * np.eye(JJt.shape[0]))
    return J_dls @ e

# -----------------------------------------------
#            Iterative IK solver (with limits)
# -----------------------------------------------
def numerical_ik_solve(
    T_target, dh_parms=DH, q0 = np.zeros(6),
    max_iters=200, tol=1e-10, lam=1e-3, step=1.0,
    q_min=None, q_max=None
):
    """
    Iterative numerical IK (DLS) with joint limits.
    q_min, q_max: np.ndarray (n,) giving joint angle limits in radians.
                  If None, defaults to [-pi, pi] for all joints.
    """
    q = q0.copy()
    n = len(dh_parms)

    # Default limits: [-pi, pi]
    if q_min is None:
        q_min = -np.pi * np.ones(n)
    if q_max is None:
        q_max = np.pi * np.ones(n)

    for k in range(max_iters):
        J, T = geometric_jacobian(dh_parms, q)
        e = pose_error(T, T_target)

        # Check convergence
        if norm(e) < tol:
            return wrap_to_pi(q)

        dq = dls_step(J, e, lam)
        q = q + step * dq

        # ✅ Wrap and clamp to limits
        q = wrap_to_pi(q)
        q = np.minimum(np.maximum(q, q_min), q_max)

    return wrap_to_pi(q)