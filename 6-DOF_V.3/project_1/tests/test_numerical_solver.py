import numpy as np

# At the very top of test_benchmark_pipeline.py
import sys
import os

# Get the absolute path to the project root (one level above 'tests/')
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)  # prepend to sys.path so Python finds your packages first

from kinematics.FK_chain import fk_chain
from kinematics.numerical_solver import numerical_ik_solve
from kinematics.settings import DH

def test_numerical_solver():

    test_theta = np.deg2rad([30 , -45 , 60 , 10 , 20 , 15])
    T_target = fk_chain(DH , test_theta)
    theta_solved = numerical_ik_solve(T_target)

    assert theta_solved is not None ,  "Solver failed to find solution"
    T_check = fk_chain(DH , theta_solved)

    pos_err = np.linalg.norm(T_check[:3, 3] - T_target[:3, 3])
    cos_arg = np.clip((np.trace(T_check[:3,:3].T @ T_target[:3,:3]) - 1) / 2, -1.0, 1.0)
    rot_err = np.rad2deg(np.arccos(cos_arg))

    assert pos_err < 1e-5, f"High position error: {pos_err}"
    assert rot_err < 1e-2, f"High rotation error: {rot_err}"

print("Numerical_solver-test passed")