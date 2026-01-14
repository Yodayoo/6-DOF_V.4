"""
Analytical inverse kinematics solver for 6-DOF manipulator.

Uses geometric decomposition to find closed-form solutions.
"""

import numpy as np
from .base_solver import IKSolver
from ..core import dh_transform
from ..utils import wrap_to_pi, angle_distance


class AnalyticalIKSolver(IKSolver):
    """
    Analytical IK solver using geometric decomposition.

    Decomposes the 6-DOF IK problem into:
    1. Position problem (first 3 joints) - shoulder and elbow
    2. Orientation problem (last 3 joints) - spherical wrist
    """

    def solve(self, T_target, q_init=None, return_all=False, debug=False):
        """
        Solve inverse kinematics analytically.

        Args:
            T_target: 4x4 target transformation matrix
            q_init: Initial configuration for closest solution selection
            return_all: If True, return all valid solutions
            debug: If True, print diagnostic information

        Returns:
            q: Best joint configuration, or None if unreachable
            OR
            solutions: List of all valid solutions (if return_all=True)
        """
        # Extract target pose
        R_target = T_target[:3, :3]
        z_target = R_target[:, 2]
        p_target = T_target[:3, 3]

        # Compact link lengths
        d1, d2, d3, d4, d5, d6 = [
            np.hypot(a, d) for (a, _, d, _) in self.dh_params
        ]

        # Wrist center position
        p_wrist = p_target - d6 * z_target
        x, y, z = p_wrist

        sols3 = []  # Partial solutions (first 3 joints)
        sols = []   # Complete solutions (all 6 joints)

        # Step 1: Solve for shoulder and elbow (first 3 joints)
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
                    sols3.append(np.array([theta1, -theta2, -theta3, 0, 0, 0]))
                else:
                    sols3.append(np.array([theta1, theta2, theta3, 0, 0, 0]))

        # Step 2: Solve for wrist orientation (last 3 joints)
        eps = 1e-9
        for sol3 in sols3:
            T05 = self._fk_first_3_joints(sol3)
            R05 = T05[:3, :3]
            R = R05.T @ R_target

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
                # Singular wrist configuration
                q6 = np.arctan2(r21, r11)
                q8 = 0.0
                q7 = 0.0
                sols.append(np.array([th1, th2, th3, q6, q7, q8]))

        # Step 3: Normalize angles
        sols = [wrap_to_pi(s) for s in sols]

        # Step 4: Apply joint limits
        valid_sols = [s for s in sols if np.all((s >= self.q_min) & (s <= self.q_max))]

        # Fallback: clip out-of-range solutions if no valid ones found
        if len(valid_sols) == 0 and len(sols) > 0:
            out_of_range = [s for s in sols if not np.all((s >= self.q_min) & (s <= self.q_max))]
            if len(out_of_range) > 0:
                dists = [np.sum(np.abs(s - np.clip(s, self.q_min, self.q_max)))
                         for s in out_of_range]
                best_idx = int(np.argmin(dists))
                best_clipped = np.clip(out_of_range[best_idx], self.q_min, self.q_max)
                valid_sols = [best_clipped]
                if debug:
                    print("⚠️ No valid solutions — using clipped fallback.")

        # Step 5: Select best solution or return all
        if len(valid_sols) == 0:
            return [] if return_all else None

        if return_all:
            return valid_sols

        # Choose closest to current configuration if provided
        if q_init is not None:
            distances = [angle_distance(s, q_init) for s in valid_sols]
            best_idx = int(np.argmin(distances))
            return valid_sols[best_idx]
        else:
            return valid_sols[0]

    def solve_all(self, T_target, **kwargs):
        """Return all valid IK solutions."""
        return self.solve(T_target, return_all=True, **kwargs)

    def _fk_first_3_joints(self, sol3):
        """Forward kinematics for the first 3 joints."""
        T = np.eye(4)
        for i in range(3):
            a, alpha, d, theta0 = self.dh_params[i]
            T = T @ dh_transform(a, alpha, d, theta0 + sol3[i])
        return T


# Legacy function interface for backward compatibility
def analytical_ik_solve(
    T_endmat,
    dh_params=None,
    q_min=None,
    q_max=None,
    q_current=None,
    debug=False,
    return_all=False
):
    """
    Legacy interface for analytical IK solver.

    Args:
        T_endmat: 4x4 target transformation matrix
        dh_params: Nx4 array of DH parameters
        q_min: Minimum joint limits
        q_max: Maximum joint limits
        q_current: Current joint configuration
        debug: Enable debug output
        return_all: Return all solutions

    Returns:
        Joint configuration(s)
    """
    if dh_params is None:
        from ..config import DH
        dh_params = DH

    solver = AnalyticalIKSolver(dh_params, q_min, q_max)
    return solver.solve(T_endmat, q_init=q_current, debug=debug, return_all=return_all)
