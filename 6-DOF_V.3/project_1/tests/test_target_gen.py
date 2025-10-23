import numpy as np
from target_gen.generate_targets import generate_transformation_matrices, max_r

def test_target_gen():
    
    matrices = generate_transformation_matrices(n_points=10000, offset=np.array([0.0, 0.0, 0.0]))
    assert len(matrices) == 10000

    for i, T in enumerate(matrices):
        pos = T[:3, 3]                # Extract position
        r = np.linalg.norm(pos)        # Euclidean distance from origin
        assert 0 <= r <= max_r, f"Target out of reach: r={r}, max_r={max_r}"
    
print("Target_generation-test passed")
