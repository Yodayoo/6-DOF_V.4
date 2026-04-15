import numpy as np
import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from benchmark.bench_func import measure_batch_performance
from kinematics.analytical_solver import analytical_ik_solve
from target_gen.generate_targets import generate_fk_targets


def test_benchmark_pipeline(tmp_path):
    targets = generate_fk_targets(
        n_points=50,
        joint_limits=[(-np.pi/2, np.pi/2)] * 6,
        save_path=str(tmp_path / "targets.npy"),
    )
    result = measure_batch_performance(analytical_ik_solve, targets, batch_size=25)

    assert "results" in result
    assert "thetas" in result
    assert result["results"].shape == (50, 5)
    assert result["thetas"].shape == (50, 6)
