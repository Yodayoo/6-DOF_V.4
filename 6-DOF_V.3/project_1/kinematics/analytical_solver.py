import numpy as np
from .helper_func import dh_T, wrap_to_pi
from .settings import DH

# Pre-compute DH-derived constants for default robot
_DH_CACHE = {}


def _get_dh_constants(dh_params):
    """Cache and return DH-derived constants for faster computation."""
    key = id(dh_params)
    if key not in _DH_CACHE:
        # Compute link lengths (hypot of a and d)
        d_vals = np.array([np.hypot(a, d) for (a, _, d, _) in dh_params])
        _DH_CACHE[key] = {
            'd': d_vals,
            'd1': d_vals[0],
            'd2': d_vals[1],
            'd4': d_vals[3],
            'd6': d_vals[5],
            'd2_sq': d_vals[1]**2,
            'd4_sq': d_vals[3]**2,
            'd2_d4_2': 2 * d_vals[1] * d_vals[3],
            'd2_plus_d4': d_vals[1] + d_vals[3],
            'd2_minus_d4': abs(d_vals[1] - d_vals[3]),
        }
    return _DH_CACHE[key]


def fk_dh3(MDH, sol3):
    """Forward kinematics for the first 3 joints (optimized)."""
    T = np.eye(4)
    for i in range(3):
        a, alpha, d, theta0 = MDH[i]
        T = T @ dh_T(a, alpha, d, theta0 + sol3[i])
    return T


def analytical_ik_solve(
    T_endmat,
    dh_params=DH,
    q_min=None,
    q_max=None,
    q_current=None,
    debug=False,
    return_all=False
):
    """
    Analytical (closed-form) IK solver for 6-DOF robot arm.

    Uses geometric decomposition:
    1. Solve shoulder/elbow (joints 1-3) from wrist center position
    2. Solve wrist orientation (joints 4-6) from rotation matrix

    Args:
        T_endmat: Target 4x4 transformation matrix
        dh_params: DH parameter matrix
        q_min: Joint lower limits (optional)
        q_max: Joint upper limits (optional)
        q_current: Current joint angles for nearest solution selection
        debug: Print debug information
        return_all: Return all valid solutions instead of best one

    Returns:
        Best joint angles solution, or None if unreachable
    """
    # Get cached DH constants
    c = _get_dh_constants(dh_params)

    # Extract target pose
    R_target = T_endmat[:3, :3]
    z_target = R_target[:, 2]
    p_target = T_endmat[:3, 3]

    # Wrist center position
    p_wrist = p_target - c['d6'] * z_target
    x, y, z = p_wrist

    sols3 = []
    sols = []

    # ----- Step 1: Shoulder + Elbow (theta1, theta2, theta3) -----
    theta1_base = np.arctan2(y, x)
    r_xy = np.hypot(x, y)  # Horizontal distance to wrist

    for theta1_offset in (0.0, np.pi):
        theta1 = theta1_base + theta1_offset

        # 2D problem in the arm plane
        X_2d = r_xy
        Z_2d = z - c['d1']
        dis_sq = X_2d**2 + Z_2d**2
        dis = np.sqrt(dis_sq)

        # Reachability check
        if dis > c['d2_plus_d4'] + 1e-9 or dis < c['d2_minus_d4'] - 1e-9:
            if debug:
                print(f"Out of reach for theta1={np.rad2deg(theta1):.1f} deg")
            continue

        # Elbow angle (theta3) via law of cosines
        cos_theta3 = (dis_sq - c['d2_sq'] - c['d4_sq']) / c['d2_d4_2']
        cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
        theta3_base = np.arccos(cos_theta3)

        for theta3_sign in (1.0, -1.0):
            theta3 = theta3_sign * theta3_base

            # Shoulder angle (theta2)
            sin_t3, cos_t3 = np.sin(theta3), np.cos(theta3)
            k1 = c['d2'] + c['d4'] * cos_t3
            k2 = c['d4'] * sin_t3
            theta2 = np.arctan2(Z_2d, X_2d) - np.arctan2(k2, k1) - np.pi/2

            # Adjust for back-reaching configuration
            if theta1_offset > 0.1:  # theta1 flipped by pi
                sols3.append(np.array([theta1, -theta2, -theta3]))
            else:
                sols3.append(np.array([theta1, theta2, theta3]))

    # ----- Step 2: Wrist Orientation (theta4, theta5, theta6) -----
    eps = 1e-9
    for sol3 in sols3:
        # Get rotation from base to joint 3
        T03 = fk_dh3(dh_params, sol3)
        R03 = T03[:3, :3]

        # Required wrist rotation
        R_wrist = R03.T @ R_target

        r13, r23, r33 = R_wrist[0, 2], R_wrist[1, 2], R_wrist[2, 2]
        r31, r32 = R_wrist[2, 0], R_wrist[2, 1]
        r11, r21 = R_wrist[0, 0], R_wrist[1, 0]

        th1, th2, th3 = sol3

        # theta5 (wrist bend)
        theta5 = np.arccos(np.clip(r33, -1.0, 1.0))

        if abs(np.sin(theta5)) > eps:
            # Non-singular case
            theta4 = np.arctan2(r23, r13)
            theta6 = np.arctan2(r32, -r31)
            sols.append(np.array([th1, th2, th3, theta4, theta5, theta6]))

            # Alternative solution (flip wrist)
            theta4_alt = theta4 + np.pi
            theta5_alt = -theta5
            theta6_alt = theta6 + np.pi
            sols.append(np.array([th1, th2, th3, theta4_alt, theta5_alt, theta6_alt]))
        else:
            # Singular wrist (gimbal lock) - theta4 and theta6 are coupled
            # Use the sum/difference based on sign of r33
            if r33 > 0:  # theta5 ≈ 0
                theta46_sum = np.arctan2(r21, r11)
                theta4 = theta46_sum / 2
                theta6 = theta46_sum / 2
            else:  # theta5 ≈ pi
                theta46_diff = np.arctan2(-r21, r11)
                theta4 = theta46_diff / 2
                theta6 = -theta46_diff / 2
            theta5 = 0.0 if r33 > 0 else np.pi
            sols.append(np.array([th1, th2, th3, theta4, theta5, theta6]))

    # ----- Step 3: Normalize angles to [-pi, pi] -----
    sols = [wrap_to_pi(s) for s in sols]

    # ----- Step 4: Apply joint limits -----
    if q_min is None or q_max is None:
        valid_sols = sols
    else:
        q_min = np.asarray(q_min)
        q_max = np.asarray(q_max)
        valid_sols = [s for s in sols if np.all((s >= q_min) & (s <= q_max))]

        # Fallback: find solution closest to valid range
        if len(valid_sols) == 0 and len(sols) > 0:
            violations = [np.sum(np.maximum(0, q_min - s) + np.maximum(0, s - q_max)) for s in sols]
            best_idx = int(np.argmin(violations))
            best_clipped = np.clip(sols[best_idx], q_min, q_max)
            valid_sols = [best_clipped]
            if debug:
                print("No valid solutions - using clipped fallback")

    # ----- Step 5: Select best solution -----
    if len(valid_sols) == 0:
        return None if not return_all else []

    if return_all:
        return valid_sols

    if q_current is not None:
        # Select solution closest to current configuration
        q_current = np.asarray(q_current)
        distances = [np.sum(np.abs(wrap_to_pi(s - q_current))) for s in valid_sols]
        return valid_sols[int(np.argmin(distances))]
    else:
        return valid_sols[0]
