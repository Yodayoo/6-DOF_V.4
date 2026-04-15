import numpy as np
import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from target_gen.generate_targets import generate_fk_targets


def test_target_gen(tmp_path):
    save_path = tmp_path / "targets.npy"
    matrices = generate_fk_targets(
        n_points=1000,
        joint_limits=[(-np.pi/2, np.pi/2)] * 6,
        save_path=str(save_path),
    )
    assert matrices.shape == (1000, 4, 4)
    # Each matrix must be a valid homogeneous transform
    for T in matrices:
        assert np.allclose(T[3], [0, 0, 0, 1])
        R = T[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-8)
