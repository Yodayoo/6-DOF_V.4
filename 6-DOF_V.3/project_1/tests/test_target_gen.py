import numpy as np
from target_gen.generate_targets import generate_transformation_matrices, max_r , z_offset

def test_target_gen():
    
    matrices = generate_transformation_matrices(n_points=100_000, max_radius=5 , offset=np.array([0.0, 0.0, z_offset]))
    assert len(matrices) == 100_000

    for i, T in enumerate(matrices):
        pos = T[:3, 3]         # Extract position
        r = np.linalg.norm(pos) - z_offset        # Euclidean distance from origin
        assert 0 <= r <= max_r, f"Target out of reach: r={r}, max_r={max_r} , {pos}"
    
print("Target_generation-test passed")
