import numpy as np
from benchmark.bench_func import measure_batch_performance
from kinematics.analytical_solver import analytical_ik_solve
from target_gen.generate_targets import generate_transformation_matrices

def test_benchmark_pipline():
    targets = generate_transformation_matrices(n_points=1000)
    result = measure_batch_performance(analytical_ik_solve , targets , batch_size=1000)

    assert "results" in  result
    assert "thetas" in result