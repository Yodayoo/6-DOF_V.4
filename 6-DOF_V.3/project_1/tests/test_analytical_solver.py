import numpy as np

from kinematics.FK_chain import fk_chain
from kinematics.analytical_solver import analytical_ik_solve
from kinematics.settings import DH

def test_analytical_solver():

    test_theta = np.deg2rad([30 , -45 , 60 , 10 , 20 , 15])
    T_target = fk_chain(DH , test_theta)
    theta_solved = analytical_ik_solve(T_target)

    assert theta_solved is not None ,  "Solver faied to find solution"
    T_check = fk_chain(DH , theta_solved)

    pos_err = np.linalg.norm(T_check[:3, 3] - T_target[:3, 3])
    rot_err = np.rad2deg(np.arccos((np.trace(T_check[:3,:3].T @ T_target[:3,:3]) - 1) / 2))

    assert pos_err < 1e-5, f"High position error: {pos_err}"
    assert rot_err < 1e-2, f"High rotation error: {rot_err}"

print("Analytical_solver-test passed")