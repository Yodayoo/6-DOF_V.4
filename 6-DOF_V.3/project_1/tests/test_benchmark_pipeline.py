import numpy as np
import sys
import os

# Get the absolute path to the project root (one level above 'tests/')
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from benchmark.bench_func import measure_batch_performance
from kinematics import analytical_ik_solve
from target_gen.generate_targets import generate_fk_targets


def test_benchmark_pipeline():
    """Test the complete benchmark pipeline."""
    # Generate a small set of test targets
    targets = generate_fk_targets(n_points=100, save_path=None, show_progress=False)

    # Run benchmark
    result = measure_batch_performance(
        analytical_ik_solve,
        targets,
        batch_size=100
    )

    # Verify output structure
    assert "results" in result, "Missing 'results' key in output"
    assert "thetas" in result, "Missing 'thetas' key in output"

    # Verify shapes
    assert result["results"].shape == (100, 5), f"Unexpected results shape: {result['results'].shape}"
    assert result["thetas"].shape == (100, 6), f"Unexpected thetas shape: {result['thetas'].shape}"

    print(f"Results shape: {result['results'].shape}")
    print(f"Thetas shape: {result['thetas'].shape}")


if __name__ == "__main__":
    test_benchmark_pipeline()
    print("Benchmark pipeline test passed!")
