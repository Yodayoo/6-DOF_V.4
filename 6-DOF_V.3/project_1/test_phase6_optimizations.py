"""
Test script for Phase 6 optimizations.

Compares performance before and after optimizations.
"""

import numpy as np
import time
from kinematics import Robot


def test_batch_fk_comparison():
    """Compare loop vs batch FK performance."""
    print("=" * 70)
    print("Phase 6: Testing Batch FK Optimization")
    print("=" * 70)

    robot = Robot()
    n_configs = 100
    configs = np.random.uniform(-np.pi/2, np.pi/2, (n_configs, 6))

    # Method 1: Loop-based FK
    print("\n[Method 1] Loop-based FK:")
    start = time.perf_counter()
    results_loop = []
    for q in configs:
        T = robot.forward_kinematics(q)
        results_loop.append(T)
    end = time.perf_counter()

    time_loop = (end - start) * 1000  # ms
    throughput_loop = n_configs / (end - start)

    print(f"  Total time: {time_loop:.4f} ms")
    print(f"  Avg per config: {time_loop/n_configs:.4f} ms")
    print(f"  Throughput: {throughput_loop:.1f} FK/sec")

    # Method 2: Batch FK
    print("\n[Method 2] Batch FK:")
    start = time.perf_counter()
    results_batch = robot.forward_kinematics_batch(configs)
    end = time.perf_counter()

    time_batch = (end - start) * 1000  # ms
    throughput_batch = n_configs / (end - start)

    print(f"  Total time: {time_batch:.4f} ms")
    print(f"  Avg per config: {time_batch/n_configs:.4f} ms")
    print(f"  Throughput: {throughput_batch:.1f} FK/sec")

    # Compare results
    print("\n[Comparison]:")
    speedup = time_loop / time_batch
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Time saved: {time_loop - time_batch:.4f} ms")

    # Verify results are identical
    max_diff = 0
    for i in range(n_configs):
        diff = np.linalg.norm(results_loop[i] - results_batch[i])
        max_diff = max(max_diff, diff)

    print(f"  Max difference: {max_diff:.2e} (should be ~0)")

    if max_diff < 1e-10:
        print("  [PASS] Results are identical")
    else:
        print("  [WARN] Results differ slightly")

    print("\n" + "=" * 70)

    return {
        'time_loop': time_loop,
        'time_batch': time_batch,
        'speedup': speedup,
        'throughput_loop': throughput_loop,
        'throughput_batch': throughput_batch,
    }


def test_accuracy_preserved():
    """Verify optimization doesn't affect accuracy."""
    print("\n" + "=" * 70)
    print("Phase 6: Verifying Accuracy Preservation")
    print("=" * 70)

    robot = Robot()
    n_tests = 20

    print(f"\n  Testing {n_tests} random configurations...")

    max_pos_err = 0
    max_rot_err = 0

    for _ in range(n_tests):
        q_original = np.random.uniform(-np.pi/2, np.pi/2, 6)
        T_target = robot.forward_kinematics(q_original)

        # Solve IK
        q_solution = robot.inverse_kinematics(T_target, method='analytical')

        if q_solution is not None:
            # Verify with FK
            T_achieved = robot.forward_kinematics(q_solution)

            pos_err = np.linalg.norm(T_target[:3, 3] - T_achieved[:3, 3])
            R_err = T_target[:3, :3].T @ T_achieved[:3, :3]
            rot_err = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))

            max_pos_err = max(max_pos_err, pos_err)
            max_rot_err = max(max_rot_err, rot_err)

    print(f"  Max position error: {max_pos_err:.2e} m")
    print(f"  Max rotation error: {max_rot_err:.2e} rad")

    if max_pos_err < 1e-4 and max_rot_err < 1e-3:
        print("  [PASS] Accuracy preserved")
    else:
        print("  [FAIL] Accuracy degraded")

    print("=" * 70)

    return max_pos_err, max_rot_err


if __name__ == '__main__':
    # Test batch FK optimization
    results = test_batch_fk_comparison()

    # Verify accuracy is preserved
    test_accuracy_preserved()

    print("\n" + "=" * 70)
    print("Phase 6 Optimization Test Summary")
    print("=" * 70)
    print(f"Batch FK speedup: {results['speedup']:.2f}x")
    print(f"Throughput improvement: {results['throughput_batch']:.1f} vs {results['throughput_loop']:.1f} FK/sec")
    print("=" * 70)
