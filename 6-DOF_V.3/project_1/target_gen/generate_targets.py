import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kinematics import fk_chain, DH

# ---------------- Main generator ---------------- #
def generate_fk_targets(
        n_points=10000,
        joint_limits=None,
        save_path="target_matrices_fk.npy"
    ):
    """
    Generate random FK targets by sampling joint angles,
    computing the end-effector pose with fk_chain, and saving as .npy.

    Args:
        n_points (int): number of targets to generate
        joint_limits (list of tuples): [(min1,max1),..., (min6,max6)] in radians
        save_path (str): path to save the 4x4 matrices
    """

    if joint_limits is None:
        # Default limits: full joint range [-pi, pi] for each joint
        joint_limits = [(-np.pi, np.pi)] * 6

    targets_matrices = []

    for _ in range(n_points):
        # Sample joint angles within limits
        thetas = np.array([np.random.uniform(low, high) for (low, high) in joint_limits])

        # Compute FK
        T = fk_chain(DH ,thetas)  # assume fk_chain returns a 4x4 np.array
        targets_matrices.append(T)

    matrices = np.array(targets_matrices)

    # Save
    save_file = os.path.abspath(save_path)
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    np.save(save_file, matrices)
    print(f"\nSaved {n_points} FK targets at: {save_file}\n")

    return matrices


# ---------------- Run script ---------------- #
if __name__ == "__main__":

    # Example: default joint limits
    matrices = generate_fk_targets(
        n_points=100_000,
        joint_limits=[(-np.pi/2, np.pi/2),  # theta1
                      (-np.pi/4, np.pi/4),  # theta2
                      (-np.pi/2, np.pi/2),  # theta3
                      (-np.pi/2, np.pi/2),  # theta4
                      (-np.pi/2, np.pi/2),  # theta5
                      (-np.pi/2, np.pi/2)], # theta6
        save_path="6-DOF_V.3/project_1/data/target_archive.npy"
    )
