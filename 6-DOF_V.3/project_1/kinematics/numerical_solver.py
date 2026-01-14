import numpy as np
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
#           Damped Least Squares step with adaptive damping
# -----------------------------------------------
def dls_step(J, e, lam=1e-3):
    """
    Compute damped least squares step with adaptive damping.

    Args:
        J: 6xN Jacobian matrix
        e: 6x1 error vector
        lam: Base damping factor (will be adapted based on conditioning)

    Returns:
        dq: Joint velocity update
    """
    JJt = J @ J.T

    # Adaptive damping based on error magnitude and Jacobian conditioning
    error_norm = np.linalg.norm(e)
    try:
        condition_num = np.linalg.cond(JJt)
        # Increase damping if poorly conditioned or large error
        # Clamp condition number to prevent extreme values
        condition_num = np.clip(condition_num, 1.0, 1e6)
        adaptive_lam = lam * (1.0 + error_norm) * np.sqrt(condition_num / 100.0)
        # Ensure damping factor stays reasonable
        adaptive_lam = np.clip(adaptive_lam, lam * 0.1, lam * 100.0)
    except:
        adaptive_lam = lam * (1.0 + error_norm)

    # Use more numerically stable pseudoinverse
    damping_matrix = JJt + adaptive_lam**2 * np.eye(JJt.shape[0])
    try:
        J_dls = J.T @ np.linalg.solve(damping_matrix, np.eye(JJt.shape[0]))
    except np.linalg.LinAlgError:
        # Fallback to higher damping if singular
        damping_matrix = JJt + (adaptive_lam * 10)**2 * np.eye(JJt.shape[0])
        J_dls = J.T @ np.linalg.solve(damping_matrix, np.eye(JJt.shape[0]))

    return J_dls @ e

# -----------------------------------------------
#            Iterative IK solver (with limits)
# -----------------------------------------------
def numerical_ik_solve(
    T_target, dh_parms=DH, q0 = np.zeros(6),
    max_iters=600, tol=1e-10, lam=0.01, step=0.75,
    q_min=None, q_max=None,
    pos_tol=1e-6, rot_tol=1e-4,
    adaptive_step=True
):
    """
    Numerical IK solver using damped least squares with adaptive improvements.

    Args:
        T_target: 4x4 target transformation matrix
        dh_parms: DH parameters (Nx4 array)
        q0: Initial joint configuration
        max_iters: Maximum number of iterations
        tol: Legacy tolerance (used if pos_tol/rot_tol not specified)
        lam: Base damping factor
        step: Initial step size (0 < step <= 1)
        q_min: Minimum joint limits
        q_max: Maximum joint limits
        pos_tol: Position tolerance in meters (default: 1e-6)
        rot_tol: Rotation tolerance in radians (default: 1e-4)
        adaptive_step: Enable adaptive step sizing with backtracking

    Returns:
        q: Converged joint configuration (wrapped to [-pi, pi])
    """
    q = q0.copy()
    n = len(dh_parms)

    # Default limits: [-pi, pi]
    if q_min is None:
        q_min = -np.pi * np.ones(n)
    if q_max is None:
        q_max = np.pi * np.ones(n)

    current_step = step
    prev_error_norm = np.inf

    for k in range(max_iters):
        J, T = geometric_jacobian(dh_parms, q)
        e = pose_error(T, T_target)

        # Separate position and rotation errors
        pos_error = np.linalg.norm(e[:3])
        rot_error = np.linalg.norm(e[3:])
        error_norm = np.linalg.norm(e)

        # Check convergence with separate tolerances
        if pos_error < pos_tol and rot_error < rot_tol:
            return wrap_to_pi(q)

        # Compute step
        dq = dls_step(J, e, lam)

        # Adaptive step sizing with backtracking
        if adaptive_step:
            # If error increased, reduce step size
            if error_norm > prev_error_norm and current_step > 0.1:
                current_step = max(0.1, current_step * 0.7)
            # If error decreased significantly, try increasing step
            elif error_norm < 0.8 * prev_error_norm and current_step < 1.0:
                current_step = min(1.0, current_step * 1.2)

        # Update with step
        q_new = q + current_step * dq

        # Wrap to [-pi, pi] to keep angles in valid range
        q_new = wrap_to_pi(q_new)

        # Clamp to limits
        q_new = np.minimum(np.maximum(q_new, q_min), q_max)

        q = q_new
        prev_error_norm = error_norm

    # Return converged solution
    return wrap_to_pi(q)