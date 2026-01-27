import numpy as np

from .helper_func import dh_T

def fk_chain(MDH, thetas):
    """
    Compute forward kinematics for an N-DOF robot arm.

    Args:
        MDH: Nx4 array of DH parameters [a, alpha, d, theta0]
        thetas: Array of N joint angles

    Returns:
        T: 4x4 end-effector transformation matrix
    """
    T = np.eye(4)
    n = len(MDH)
    for i in range(n):
        a, alpha, d, theta0 = MDH[i]
        T = T @ dh_T(a, alpha, d, theta0 + thetas[i])
    return T
