"""
Performance benchmarking script for Phase 6 optimization.

Measures baseline performance before optimization and tracks improvements.
"""

import numpy as np
import time
from kinematics import Robot, forward_kinematics, DH


def benchmark_forward_kinematics(n_iterations=1000):
    """Benchmark FK computation."""
    robot = Robot()
    q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    start = time.perf_counter()
    for _ in range(n_iterations):
        T = robot.forward_kinematics(q)
    end = time.perf_counter()

    avg_time = (end - start) / n_iterations * 1000  # ms
    return avg_time


def benchmark_batch_forward_kinematics(n_configs=100):
    """Benchmark batch FK computation."""
    robot = Robot()
    configs = np.random.uniform(-np.pi/2, np.pi/2, (n_configs, 6))

    start = time.perf_counter()
    for q in configs:
        T = robot.forward_kinematics(q)
    end = time.perf_counter()

    total_time = (end - start) * 1000  # ms
    avg_time = total_time / n_configs
    throughput = n_configs / (end - start)  # configs/sec

    return avg_time, throughput


def benchmark_analytical_ik(n_iterations=100):
    """Benchmark analytical IK solver."""
    robot = Robot()

    # Generate test targets
    targets = []
    for _ in range(n_iterations):
        q = np.random.uniform(-np.pi/2, np.pi/2, 6)
        T = robot.forward_kinematics(q)
        targets.append(T)

    start = time.perf_counter()
    for T_target in targets:
        q_solution = robot.inverse_kinematics(T_target, method='analytical')
    end = time.perf_counter()

    avg_time = (end - start) / n_iterations * 1000  # ms
    throughput = n_iterations / (end - start)  # solves/sec

    return avg_time, throughput


def benchmark_multi_solution_ik(n_iterations=50):
    """Benchmark multi-solution IK solver."""
    robot = Robot()

    # Generate test targets
    targets = []
    for _ in range(n_iterations):
        q = np.random.uniform(-np.pi/2, np.pi/2, 6)
        T = robot.forward_kinematics(q)
        targets.append(T)

    start = time.perf_counter()
    for T_target in targets:
        solutions = robot.inverse_kinematics_all(T_target)
    end = time.perf_counter()

    avg_time = (end - start) / n_iterations * 1000  # ms
    throughput = n_iterations / (end - start)  # solves/sec

    return avg_time, throughput


def benchmark_numerical_ik(n_iterations=50):
    """Benchmark numerical IK solver."""
    robot = Robot()

    # Generate test targets
    targets = []
    q_inits = []
    for _ in range(n_iterations):
        q = np.random.uniform(-np.pi/2, np.pi/2, 6)
        T = robot.forward_kinematics(q)
        targets.append(T)
        q_inits.append(q + np.random.uniform(-0.1, 0.1, 6))

    start = time.perf_counter()
    for T_target, q_init in zip(targets, q_inits):
        q_solution = robot.inverse_kinematics(
            T_target, method='numerical', q_init=q_init
        )
    end = time.perf_counter()

    avg_time = (end - start) / n_iterations * 1000  # ms
    throughput = n_iterations / (end - start)  # solves/sec

    return avg_time, throughput


def run_all_benchmarks():
    """Run all performance benchmarks."""
    print("=" * 70)
    print("Performance Benchmarks - Phase 6 Baseline")
    print("=" * 70)

    # FK benchmark
    print("\n[1] Forward Kinematics (single config)")
    fk_time = benchmark_forward_kinematics(1000)
    print(f"  Average time: {fk_time:.4f} ms")
    print(f"  Throughput: {1000/fk_time:.1f} FK/sec")

    # Batch FK benchmark
    print("\n[2] Forward Kinematics (batch)")
    batch_fk_time, batch_fk_throughput = benchmark_batch_forward_kinematics(100)
    print(f"  Average time: {batch_fk_time:.4f} ms per config")
    print(f"  Throughput: {batch_fk_throughput:.1f} FK/sec")

    # Analytical IK benchmark
    print("\n[3] Analytical IK (single solution)")
    ik_time, ik_throughput = benchmark_analytical_ik(100)
    print(f"  Average time: {ik_time:.4f} ms")
    print(f"  Throughput: {ik_throughput:.1f} solves/sec")

    # Multi-solution IK benchmark
    print("\n[4] Analytical IK (all solutions)")
    multi_ik_time, multi_ik_throughput = benchmark_multi_solution_ik(50)
    print(f"  Average time: {multi_ik_time:.4f} ms")
    print(f"  Throughput: {multi_ik_throughput:.1f} solves/sec")

    # Numerical IK benchmark
    print("\n[5] Numerical IK")
    num_ik_time, num_ik_throughput = benchmark_numerical_ik(50)
    print(f"  Average time: {num_ik_time:.4f} ms")
    print(f"  Throughput: {num_ik_throughput:.1f} solves/sec")

    print("\n" + "=" * 70)
    print("Benchmark Summary")
    print("=" * 70)
    print(f"FK (single):        {fk_time:.4f} ms  ({1000/fk_time:.1f} ops/sec)")
    print(f"FK (batch):         {batch_fk_time:.4f} ms  ({batch_fk_throughput:.1f} ops/sec)")
    print(f"IK (analytical):    {ik_time:.4f} ms  ({ik_throughput:.1f} ops/sec)")
    print(f"IK (multi-sol):     {multi_ik_time:.4f} ms  ({multi_ik_throughput:.1f} ops/sec)")
    print(f"IK (numerical):     {num_ik_time:.4f} ms  ({num_ik_throughput:.1f} ops/sec)")
    print("=" * 70)

    return {
        'fk_single': fk_time,
        'fk_batch': batch_fk_time,
        'ik_analytical': ik_time,
        'ik_multi': multi_ik_time,
        'ik_numerical': num_ik_time,
    }


if __name__ == '__main__':
    results = run_all_benchmarks()
