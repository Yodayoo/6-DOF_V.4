import numpy as np
import sys
import os

# Get the absolute path to the project root (one level above 'tests/')
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from target_gen.generate_targets import generate_fk_targets
from kinematics import max_r, z_offset, fk_chain, DH


def test_target_gen():
    """Test that generated FK targets are within robot workspace."""
    # Generate test targets with default limits
    matrices = generate_fk_targets(n_points=1000, save_path=None, show_progress=False)

    assert len(matrices) == 1000, f"Expected 1000 targets, got {len(matrices)}"
    assert matrices.shape == (1000, 4, 4), f"Unexpected shape: {matrices.shape}"

    # Verify all matrices are valid transformation matrices
    for i, T in enumerate(matrices):
        # Check it's a valid 4x4 transformation matrix
        assert T.shape == (4, 4), f"Target {i} has invalid shape: {T.shape}"

        # Check bottom row is [0, 0, 0, 1]
        assert np.allclose(T[3, :], [0, 0, 0, 1]), f"Target {i} has invalid bottom row"

        # Check rotation matrix is orthonormal (R @ R.T = I)
        R = T[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6), f"Target {i} rotation not orthonormal"

    print(f"Generated {len(matrices)} valid transformation matrices")
    print(f"Shape: {matrices.shape}")


def test_fk_consistency():
    """Test that FK produces consistent results."""
    # Test with known joint angles
    test_angles = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    T1 = fk_chain(DH, test_angles)
    T2 = fk_chain(DH, test_angles)

    assert np.allclose(T1, T2), "FK produces inconsistent results"
    print("FK consistency test passed")


if __name__ == "__main__":
    test_target_gen()
    print("Target generation test passed!")

    test_fk_consistency()
    print("FK consistency test passed!")
