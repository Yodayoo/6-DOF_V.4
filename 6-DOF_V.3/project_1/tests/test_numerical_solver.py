import numpy as np
import sys
import os

# Get the absolute path to the project root (one level above 'tests/')
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from kinematics import fk_chain, numerical_ik_solve, DH


def test_numerical_solver():
    """Test that numerical IK solver produces valid solutions."""
    test_theta = np.deg2rad([30, -45, 60, 10, 20, 15])
    T_target = fk_chain(DH, test_theta)
    theta_solved = numerical_ik_solve(T_target)

    assert theta_solved is not None, "Solver failed to find solution"
    T_check = fk_chain(DH, theta_solved)

    # Position error
    pos_err = np.linalg.norm(T_check[:3, 3] - T_target[:3, 3])

    # Rotation error - clip to avoid NaN from numerical precision issues
    cos_val = (np.trace(T_check[:3, :3].T @ T_target[:3, :3]) - 1) / 2
    cos_val = np.clip(cos_val, -1.0, 1.0)
    rot_err = np.rad2deg(np.arccos(cos_val))

    assert pos_err < 1e-5, f"High position error: {pos_err}"
    assert rot_err < 1e-2, f"High rotation error: {rot_err}"

    print(f"Position error: {pos_err:.2e} m")
    print(f"Rotation error: {rot_err:.2e} deg")


if __name__ == "__main__":
    test_numerical_solver()
    print("Numerical solver test passed!")
