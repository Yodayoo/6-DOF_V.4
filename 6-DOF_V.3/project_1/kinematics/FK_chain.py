import numpy as np

from .helper_func import dh_T

def fk_chain(MDH, thetas):
    T = np.eye(4)
    for i in range(6):
        a, alpha, d, theta0 = MDH[i]
        T = T @ dh_T(a, alpha, d, theta0 + thetas[i])
    return T