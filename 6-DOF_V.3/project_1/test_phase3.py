"""
Test script for Phase 3 refactoring.

Verifies:
1. Backward compatibility with old API
2. New Robot class API works
3. All imports are accessible
"""

import numpy as np
import sys

print("=" * 70)
print("Phase 3 Refactoring Test")
print("=" * 70)

# Test 1: Legacy imports
print("\n[Test 1] Testing legacy imports...")
try:
    from kinematics import DH, analytical_ik_solve, numerical_ik_solve, fk_chain
    from kinematics import wrap_to_pi, error_check
    print("[PASS] Legacy imports successful")
except Exception as e:
    print(f"[FAIL] Legacy imports failed: {e}")
    sys.exit(1)

# Test 2: New modular imports
print("\n[Test 2] Testing new modular imports...")
try:
    from kinematics import Robot, RobotConfig
    from kinematics import AnalyticalIKSolver, NumericalIKSolver
    from kinematics.core import forward_kinematics, geometric_jacobian
    from kinematics.utils import pose_error, check_reachability
    print("[PASS] New modular imports successful")
except Exception as e:
    print(f"[FAIL] New modular imports failed: {e}")
    sys.exit(1)

# Test 3: Robot class API
print("\n[Test 3] Testing Robot class API...")
try:
    robot = Robot()
    print(f"  Robot: {robot}")
    print(f"  Number of joints: {robot.n_joints}")
    print(f"  Max reach: {robot.max_reach:.3f}m")
    print("[PASS] Robot class works")
except Exception as e:
    print(f"[FAIL] Robot class failed: {e}")
    sys.exit(1)

# Test 4: Forward kinematics
print("\n[Test 4] Testing forward kinematics...")
try:
    q_test = np.array([0, 0, 0, 0, 0, 0])

    # Old API
    T1 = fk_chain(DH, q_test)

    # New API
    T2 = robot.forward_kinematics(q_test)

    # Compare
    diff = np.linalg.norm(T1 - T2)
    print(f"  FK difference (old vs new API): {diff:.2e}")

    if diff < 1e-10:
        print("[PASS] Forward kinematics consistent")
    else:
        print("[FAIL] Forward kinematics inconsistent!")
        sys.exit(1)
except Exception as e:
    print(f"[FAIL] Forward kinematics failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Analytical IK
print("\n[Test 5] Testing analytical IK solver...")
try:
    # Create test target
    q_target = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    T_target = robot.forward_kinematics(q_target)

    # Old API
    q_old = analytical_ik_solve(T_target, dh_params=DH, q_current=q_target)

    # New API
    q_new = robot.inverse_kinematics(T_target, method='analytical', q_init=q_target)

    if q_old is not None and q_new is not None:
        # Verify solutions
        T_old = fk_chain(DH, q_old)
        T_new = robot.forward_kinematics(q_new)

        pos_err_old, rot_err_old = error_check(T_old, T_target)
        pos_err_new, rot_err_new = error_check(T_new, T_target)

        print(f"  Old API - Pos error: {pos_err_old:.2e}m, Rot error: {rot_err_old:.2e}rad")
        print(f"  New API - Pos error: {pos_err_new:.2e}m, Rot error: {rot_err_new:.2e}rad")

        if pos_err_old < 1e-6 and pos_err_new < 1e-6:
            print("[PASS] Analytical IK solver works")
        else:
            print("[FAIL] Analytical IK solver has accuracy issues")
            sys.exit(1)
    else:
        print("[FAIL] Analytical IK solver returned None")
        sys.exit(1)
except Exception as e:
    print(f"[FAIL] Analytical IK failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Numerical IK
print("\n[Test 6] Testing numerical IK solver...")
try:
    # Use same target
    q0 = np.zeros(6)

    # Old API
    q_old = numerical_ik_solve(T_target, dh_parms=DH, q0=q0)

    # New API
    q_new = robot.inverse_kinematics(T_target, method='numerical', q_init=q0)

    if q_old is not None and q_new is not None:
        # Verify solutions
        T_old = fk_chain(DH, q_old)
        T_new = robot.forward_kinematics(q_new)

        pos_err_old, rot_err_old = error_check(T_old, T_target)
        pos_err_new, rot_err_new = error_check(T_new, T_target)

        print(f"  Old API - Pos error: {pos_err_old:.2e}m, Rot error: {rot_err_old:.2e}rad")
        print(f"  New API - Pos error: {pos_err_new:.2e}m, Rot error: {rot_err_new:.2e}rad")

        if pos_err_old < 1e-4 and pos_err_new < 1e-4:
            print("[PASS] Numerical IK solver works")
        else:
            print("[FAIL] Numerical IK solver has accuracy issues")
            sys.exit(1)
    else:
        print("[FAIL] Numerical IK solver returned None")
        sys.exit(1)
except Exception as e:
    print(f"[FAIL] Numerical IK failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Batch test
print("\n[Test 7] Running batch test on 100 random targets...")
try:
    # Load test targets
    targets = np.load("data/target_archive.npy", allow_pickle=True)[:100]

    success_count = 0
    errors_pos = []
    errors_rot = []

    for T_target in targets:
        try:
            q = robot.inverse_kinematics(T_target, method='analytical')
            if q is not None:
                T_achieved = robot.forward_kinematics(q)
                pos_err, rot_err = error_check(T_achieved, T_target)
                errors_pos.append(pos_err)
                errors_rot.append(rot_err)
                if pos_err < 1e-6 and rot_err < 1e-4:
                    success_count += 1
        except:
            pass

    success_rate = 100.0 * success_count / len(targets)
    mean_pos_err = np.mean(errors_pos) if errors_pos else float('inf')
    mean_rot_err = np.mean(errors_rot) if errors_rot else float('inf')

    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Mean position error: {mean_pos_err:.2e}m")
    print(f"  Mean rotation error: {mean_rot_err:.2e}rad")

    if success_rate > 95:
        print("[PASS] Batch test passed")
    else:
        print("[FAIL] Batch test failed - low success rate")
        sys.exit(1)
except Exception as e:
    print(f"[FAIL] Batch test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("All tests passed! Phase 3 refactoring successful.")
print("=" * 70)
