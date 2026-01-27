import numpy as np
import sys
import os

# Use __file__ based paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from kinematics import fk_chain, DH


def generate_fk_targets(
    n_points=10000,
    joint_limits=None,
    save_path="target_matrices_fk.npy",
    show_progress=True
):
    """
    Generate random FK targets by sampling joint angles,
    computing the end-effector pose with fk_chain, and saving as .npy.

    Args:
        n_points (int): number of targets to generate
        joint_limits (list of tuples): [(min1,max1),..., (min6,max6)] in radians
        save_path (str): path to save the 4x4 matrices
        show_progress (bool): whether to show progress during generation

    Returns:
        np.ndarray: (n_points, 4, 4) array of transformation matrices
    """
    if joint_limits is None:
        # Default limits: full joint range [-pi, pi] for each joint
        joint_limits = [(-np.pi, np.pi)] * 6

    # Convert to numpy array for vectorized operations
    limits = np.array(joint_limits)
    lows = limits[:, 0]
    highs = limits[:, 1]

    # Vectorized random sampling - much faster than loop
    all_thetas = np.random.uniform(lows, highs, size=(n_points, 6))

    # Pre-allocate output array
    targets_matrices = np.zeros((n_points, 4, 4))

    # Progress reporting interval
    report_interval = max(1, n_points // 10)

    for i, thetas in enumerate(all_thetas):
        targets_matrices[i] = fk_chain(DH, thetas)

        if show_progress and (i + 1) % report_interval == 0:
            print(f"Generated {i + 1}/{n_points} targets ({100 * (i + 1) / n_points:.0f}%)")

    # Save
    if save_path:
        save_file = os.path.abspath(save_path)
        dir_path = os.path.dirname(save_file)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        np.save(save_file, targets_matrices)
        print(f"\nSaved {n_points} FK targets at: {save_file}\n")

    return targets_matrices


if __name__ == "__main__":
    # Use __file__ based path for output
    DATA_DIR = os.path.join(PROJECT_DIR, "data")
    output_path = os.path.join(DATA_DIR, "target_archive.npy")

    matrices = generate_fk_targets(
        n_points=1_000_000,
        joint_limits=[
            (-np.pi/2, np.pi/2),   # theta1
            (-np.pi/4, np.pi/4),   # theta2
            (-np.pi/2, np.pi/2),   # theta3
            (-np.pi/2, np.pi/2),   # theta4
            (-np.pi/2, np.pi/2),   # theta5
            (-np.pi/2, np.pi/2),   # theta6
        ],
        save_path=output_path
    )
