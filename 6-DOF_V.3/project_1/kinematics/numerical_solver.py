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
    origins = [np.zeros(3)]  # Include base origin
    z_axes = [np.array([0.0, 0.0, 1.0])]  # Include base z-axis

    for i in range(n):
        a, alpha, d, t0 = dh_params[i]
        T = T @ dh_T(a, alpha, d, t0 + q[i])
        origins.append(T[:3, 3].copy())
        z_axes.append(T[:3, 2].copy())

    o_n = origins[-1]
    J = np.zeros((6, n))

    for i in range(n):
        J[:3, i] = np.cross(z_axes[i], o_n - origins[i])
        J[3:, i] = z_axes[i]

    return J, T


# -----------------------------------------------
#                    Pose error
# -----------------------------------------------
def pose_error(T_current, T_target):
    """Compute 6D pose error (position + orientation)."""
    ep = T_target[:3, 3] - T_current[:3, 3]

    # Orientation error using axis-angle
    R_err = T_target[:3, :3] @ T_current[:3, :3].T
    cos_theta = np.clip((np.trace(R_err) - 1) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-8:
        eo = np.zeros(3)
    else:
        # Extract axis from skew-symmetric part
        w_hat = (R_err - R_err.T) / (2 * np.sin(theta))
        eo = np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]]) * theta

    return np.hstack([ep, eo])


# -----------------------------------------------
#           Damped Least Squares step
# -----------------------------------------------
def dls_step(J, e, lam):
    """Compute DLS step with given damping factor."""
    JJt = J @ J.T
    regularized = JJt + lam**2 * np.eye(6)
    return J.T @ np.linalg.solve(regularized, e)


# -----------------------------------------------
#            Core iterative solver
# -----------------------------------------------
def _solve_from_initial(T_target, dh_params, q0, max_iters, tol, q_min, q_max):
    """
    Single-start iterative IK with adaptive damping.

    Returns: (solution, final_error, converged)
    """
    q = q0.copy()

    # Adaptive damping parameters
    lam = 0.01  # Initial damping
    lam_min = 1e-6
    lam_max = 10.0

    # Adaptive step size
    step = 1.0
    step_min = 0.1
    step_max = 1.5

    # Track best solution
    best_q = q.copy()
    best_err = np.inf

    # Stall detection
    stall_count = 0
    stall_threshold = 20

    for k in range(max_iters):
        J, T = geometric_jacobian(dh_params, q)
        e = pose_error(T, T_target)
        err_norm = norm(e)

        # Track best solution found
        if err_norm < best_err:
            best_err = err_norm
            best_q = q.copy()
            stall_count = 0
        else:
            stall_count += 1

        # Check convergence
        if err_norm < tol:
            return wrap_to_pi(q), err_norm, True

        # Early termination if stuck
        if stall_count > stall_threshold:
            break

        # Compute step
        dq = dls_step(J, e, lam)
        q_new = q + step * dq

        # Apply constraints
        q_new = wrap_to_pi(q_new)
        q_new = np.clip(q_new, q_min, q_max)

        # Evaluate new position
        _, T_new = geometric_jacobian(dh_params, q_new)
        e_new = pose_error(T_new, T_target)
        err_new = norm(e_new)

        # Adaptive damping (Levenberg-Marquardt style)
        if err_new < err_norm:
            # Good step - reduce damping, possibly increase step size
            q = q_new
            lam = max(lam_min, lam * 0.7)
            step = min(step_max, step * 1.1)
        else:
            # Bad step - increase damping, reduce step size
            lam = min(lam_max, lam * 2.0)
            step = max(step_min, step * 0.5)
            # Still accept the step if we're not making progress
            if stall_count > stall_threshold // 2:
                q = q_new

    return wrap_to_pi(best_q), best_err, best_err < tol


# -----------------------------------------------
#            Main IK solver with multi-restart
# -----------------------------------------------
def numerical_ik_solve(
    T_target, dh_params=DH, q0=None,
    max_iters=300, tol=1e-6,
    q_min=None, q_max=None,
    num_restarts=3
):
    """
    Iterative IK solver using Damped Least Squares with adaptive damping
    and multi-restart strategy for robustness.

    Args:
        T_target: Target 4x4 transformation matrix
        dh_params: DH parameter matrix (Nx4)
        q0: Initial joint angles (None for automatic)
        max_iters: Maximum iterations per restart (default 300)
        tol: Convergence tolerance (default 1e-6)
        q_min: Joint lower limits
        q_max: Joint upper limits
        num_restarts: Number of random restarts if initial fails (default 3)

    Returns:
        Joint angles that achieve the target pose (best found solution)
    """
    n = len(dh_params)

    # Default limits
    if q_min is None:
        q_min = -np.pi * np.ones(n)
    else:
        q_min = np.asarray(q_min)
    if q_max is None:
        q_max = np.pi * np.ones(n)
    else:
        q_max = np.asarray(q_max)

    # Default initial guess
    if q0 is None:
        q0 = np.zeros(n)
    else:
        q0 = np.asarray(q0, dtype=float)

    # Track overall best solution
    overall_best_q = None
    overall_best_err = np.inf

    # Try from provided initial guess first
    initial_guesses = [q0]

    # Add strategic restarts
    if num_restarts > 1:
        # Add zeros if different from q0
        if not np.allclose(q0, np.zeros(n)):
            initial_guesses.append(np.zeros(n))

        # Add random configurations within joint limits
        for _ in range(num_restarts - len(initial_guesses)):
            q_rand = np.random.uniform(q_min, q_max)
            initial_guesses.append(q_rand)

    # Try each initial guess
    for q_init in initial_guesses:
        sol, err, converged = _solve_from_initial(
            T_target, dh_params, q_init,
            max_iters, tol, q_min, q_max
        )

        if err < overall_best_err:
            overall_best_err = err
            overall_best_q = sol

        # Early exit if converged
        if converged:
            return sol

    return overall_best_q if overall_best_q is not None else wrap_to_pi(q0)


# -----------------------------------------------
#            Fast single-shot solver (no restarts)
# -----------------------------------------------
def numerical_ik_solve_fast(
    T_target, dh_params=DH, q0=None,
    max_iters=200, tol=1e-6,
    q_min=None, q_max=None
):
    """
    Fast single-shot IK solver without restarts.
    Use when speed is more important than guaranteed convergence.
    """
    n = len(dh_params)

    if q_min is None:
        q_min = -np.pi * np.ones(n)
    if q_max is None:
        q_max = np.pi * np.ones(n)
    if q0 is None:
        q0 = np.zeros(n)

    sol, _, _ = _solve_from_initial(
        T_target, dh_params, np.asarray(q0, dtype=float),
        max_iters, tol, np.asarray(q_min), np.asarray(q_max)
    )
    return sol
