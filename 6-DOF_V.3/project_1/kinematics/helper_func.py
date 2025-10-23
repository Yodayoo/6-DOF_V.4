import numpy as np

def dh_T(a, alpha, d, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([[ ct, -st*ca,  st*sa, a*ct],
                     [ st,  ct*ca, -ct*sa, a*st],
                     [  0,     sa,     ca,     d],
                     [  0,      0,      0,     1]], dtype=float)

def wrap_to_pi(q):
    return (q + np.pi) % (2*np.pi) - np.pi

def wrap_to_2pi(q):
    return q % (2*np.pi)

def error_check(T_solved , T_true):
    pos_err = np.linalg.norm(T_true[:3 , 3] - T_solved[:3 , 3])
    rot_err = np.linalg.norm(T_true[:3 , :3] - T_solved[:3 , :3])

    return pos_err , rot_err