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
    """
    Compute position and rotation errors between two transformation matrices.

    Uses proper SO(3) geodesic distance for rotation error instead of Frobenius norm.

    Args:
        T_solved: 4x4 transformation matrix from solver
        T_true: 4x4 ground truth transformation matrix

    Returns:
        pos_err: Position error (Euclidean distance in meters)
        rot_err: Rotation error (geodesic distance in radians)
    """
    pos_err = np.linalg.norm(T_true[:3 , 3] - T_solved[:3 , 3])

    # Compute rotation error using SO(3) geodesic distance
    R_true = T_true[:3, :3]
    R_solved = T_solved[:3, :3]
    R_rel = R_true.T @ R_solved

    # Geodesic distance: angle = arccos((trace(R_rel) - 1) / 2)
    cos_angle = (np.trace(R_rel) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
    rot_err = np.arccos(cos_angle)

    return pos_err , rot_err