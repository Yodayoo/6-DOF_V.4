import numpy as np
from .helper_func import dh_T, wrap_to_pi
from .settings import DH

def fk_dh3(MDH, sol3):
    """Forward kinematics for the first 3 joints."""
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

    # ---------------------------------------------
    # Extract target pose
    # ---------------------------------------------
    R_rotend = T_endmat[:3, :3]
    z_rotend = R_rotend[:, 2]
    p_posend = T_endmat[:3, 3]

    # Compact d1...d6
    d1, d2, d3, d4, d5, d6 = [np.hypot(a, d) for (a, _, d, _) in dh_params]

    # Wrist center
    p_targetpos = p_posend - d6 * z_rotend
    x, y, z = p_targetpos

    sols3, sols = [], []

    # ---------------------------------------------
    # Step 1: shoulder + elbow
    # ---------------------------------------------
    theta1_base = np.arctan2(y, x)
    theta1_candidates = [theta1_base, theta1_base + np.pi]

    for theta1 in theta1_candidates:
        X_2d = np.hypot(x, y)
        Z_2d = z - d1
        dis_target2 = X_2d**2 + Z_2d**2
        dis_target = np.sqrt(dis_target2)

        # Reachability check
        if dis_target > d2 + d4 + 1e-9 or dis_target < abs(d2 - d4) - 1e-9:
            if debug:
                print(f"⚠️ Out of reach for θ1={theta1:.3f}")
            continue

        cos_theta3 = (dis_target2 - d2**2 - d4**2) / (2 * d2 * d4)
        cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
        theta3_base = np.arccos(cos_theta3)
        theta3_candidates = [theta3_base, -theta3_base]

        for theta3 in theta3_candidates:
            theta2 = (np.arctan2(Z_2d, X_2d) - np.pi/2
                      - np.arctan2(d4*np.sin(theta3),
                                   d2 + d4*np.cos(theta3)))
            if theta1 == theta1_base + np.pi:
                sols3.append(np.array([theta1, -theta2, -theta3]))
            else:
                sols3.append(np.array([theta1, theta2, theta3, 0, 0, 0]))

    # ---------------------------------------------
    # Step 2: wrist orientation
    # ---------------------------------------------
    eps = 1e-9
    for sol3 in sols3:
        T05 = fk_dh3(dh_params, sol3)
        R05 = T05[:3, :3]
        R = R05.T @ R_rotend

        r11, r12, r13 = R[0]
        r21, r22, r23 = R[1]
        r31, r32, r33 = R[2]

        th1, th2, th3 = sol3[:3]
        q7 = np.arccos(np.clip(r33, -1.0, 1.0))

        if abs(np.sin(q7)) > eps:
            q6 = np.arctan2(r23, r13)
            q8 = np.arctan2(r32, -r31)
            sols.append(np.array([th1, th2, th3, q6, q7, q8]))
            sols.append(np.array([th1, th2, th3, q6+np.pi, -q7, q8+np.pi]))
        else:
            # singular wrist
            q6 = np.arctan2(r21, r11)
            q8 = 0.0
            q7 = 0.0
            sols.append(np.array([th1, th2, th3, q6, q7, q8]))

    # ---------------------------------------------
    # Step 3: normalize
    # ---------------------------------------------
    sols = [wrap_to_pi(s) for s in sols]

    # ---------------------------------------------
    # Step 4: apply limits (if any)
    # ---------------------------------------------
    if q_min is None or q_max is None:
        valid_sols = sols
        if debug:
            print("ℹ️ Joint limits disabled.")
    else:
        valid_sols = [s for s in sols if np.all((s >= q_min) & (s <= q_max))]
        out_of_range = [s for s in sols if not np.all((s >= q_min) & (s <= q_max))]

        # fallback if no valid
        if len(valid_sols) == 0 and len(out_of_range) > 0:
            clipped = [np.clip(s, q_min, q_max) for s in out_of_range]
            dists = [np.sum(np.abs(s - np.clip(s, q_min, q_max))) for s in out_of_range]
            best_idx = int(np.argmin(dists))
            best_clipped = np.clip(out_of_range[best_idx], q_min, q_max)
            valid_sols = [best_clipped]
            if debug:
                print("⚠️ No valid solutions — using clipped fallback.")

    # ---------------------------------------------
    # Step 5: choose closest to current q (if given)
    # ---------------------------------------------
    reachable = len(valid_sols) > 0
    if not reachable:
        return (None if not return_all else [])

    if q_current is not None:
        def angle_distance(a, b):
            """Sum of wrapped absolute differences between two joint sets."""
            diff = wrap_to_pi(a - b)
            return np.sum(np.abs(diff))
        distances = [angle_distance(s, q_current) for s in valid_sols]
        best_idx = int(np.argmin(distances))
        best_sol = valid_sols[best_idx]
    else:
        best_sol = valid_sols[0]

    if return_all:
        return valid_sols
    else:
        return best_sol