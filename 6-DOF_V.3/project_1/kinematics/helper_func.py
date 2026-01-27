import numpy as np

def dh_T(a, alpha, d, theta):
    """Build 4x4 DH transformation matrix from DH parameters."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([[ ct, -st*ca,  st*sa, a*ct],
                     [ st,  ct*ca, -ct*sa, a*st],
                     [  0,     sa,     ca,    d],
                     [  0,      0,      0,    1]], dtype=float)

def wrap_to_pi(q):
    """Wrap angles to [-pi, pi] range."""
    return (q + np.pi) % (2*np.pi) - np.pi

def wrap_to_2pi(q):
    """Wrap angles to [0, 2*pi) range."""
    return q % (2*np.pi)

def error_check(T_solved, T_true):
    """
    Calculate position and rotation error between two poses.

    Args:
        T_solved: Computed 4x4 transformation matrix
        T_true: Target 4x4 transformation matrix

    Returns:
        pos_err: Euclidean distance between positions (meters)
        rot_err: Angle-axis magnitude of rotation error (radians)
    """
    # Position error: Euclidean distance
    pos_err = np.linalg.norm(T_true[:3, 3] - T_solved[:3, 3])

    # Rotation error: angle of rotation between the two orientations
    # This is more meaningful than Frobenius norm
    R_err = T_true[:3, :3] @ T_solved[:3, :3].T
    cos_angle = (np.trace(R_err) - 1) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    rot_err = np.arccos(cos_angle)

    return pos_err, rot_err
