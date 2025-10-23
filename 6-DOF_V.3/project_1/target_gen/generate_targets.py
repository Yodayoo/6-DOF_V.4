import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kinematics import DH

    # Example: maximum reach of your 6-DOF arm
max_r = sum(np.hypot(DH[1: , 0] , DH[1: , 2]))# replace with sum of DH link lengths if needed
z_offset = np.hypot(DH[0 , 0] , DH[0 , 2])  # adjust if base is offset

# ---------------- Helper functions ---------------- #
def spherical_to_cartesian(alpha, beta, r):
    """Convert spherical coordinates to Cartesian coordinates."""
    x = r * np.cos(beta) * np.cos(alpha)
    y = r * np.cos(beta) * np.sin(alpha)
    z = r * np.sin(beta)
    return np.array([x, y, z])

def rpy_to_matrix(roll, pitch, yaw):
    """Convert roll, pitch, yaw to a rotation matrix."""
    cx, cy, cz = np.cos(roll), np.cos(pitch), np.cos(yaw)
    sx, sy, sz = np.sin(roll), np.sin(pitch), np.sin(yaw)
    R = np.array([
        [cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx],
        [sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx],
        [-sy,   cy*sx,            cy*cx]
    ])
    return R

def target_to_matrix(target, offset=np.zeros(3)):
    """
    Convert a 6D target [roll, pitch, yaw, alpha, beta, r] 
    into a 4x4 transformation matrix.
    """
    T = np.eye(4)
    roll, pitch, yaw = target[:3]
    alpha, beta, r = target[3:]

    pos = spherical_to_cartesian(alpha, beta, r) + offset
    T[:3, 3] = pos
    T[:3, :3] = rpy_to_matrix(roll, pitch, yaw)
    return T

# ---------------- Main generator ---------------- #
def generate_transformation_matrices(
        n_points=10000,
        max_radius=1.0,
        offset=np.zeros(3),
        save_path="target_matrices.npy"
    ):
    """
    Generate random 6D targets (random rotation, position within radius),
    convert to 4x4 matrices, and save as .npy file.
    """

    # Random rotations: roll, pitch, yaw in [0, 2*pi)
    rotations = np.random.uniform(0, 2*np.pi, size=(n_points, 3))

    # Random positions within a sphere of radius max_radius
    # Uniform distribution in 3D sphere
    u = np.random.uniform(0, 1, n_points)
    v = np.random.uniform(0, 1, n_points)
    w = np.random.uniform(0, 1, n_points)

    alpha = 2 * np.pi * u           # azimuth
    beta = np.arccos(2*v - 1)       # elevation angle (0..pi)
    r = max_radius * np.cbrt(w)     # cube root to ensure uniform in volume

    # Combine rotation and position
    targets = np.hstack([rotations, alpha[:, np.newaxis], beta[:, np.newaxis], r[:, np.newaxis]])

    # Convert all to 4x4 matrices
    matrices = np.array([target_to_matrix(t, offset=offset) for t in targets])

    # Save
    save_file = os.path.abspath(save_path)
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    np.save(save_file, matrices)
    print(f"\nSaved {n_points} transformation matrices at: {save_file}\n")

    return matrices

# ---------------- Run script ---------------- #
if __name__ == "__main__":


    generate_transformation_matrices(
        n_points=10_000,
        max_radius=max_r,
        offset=np.array([0.0, 0.0, z_offset]),
        save_path="data/target_archive.npy"
    )