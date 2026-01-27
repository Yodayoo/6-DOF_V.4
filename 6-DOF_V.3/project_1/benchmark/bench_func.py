import numpy as np
import time
import psutil
import os

from datetime import datetime
from kinematics import fk_chain, DH
from scipy.spatial.transform import Rotation as R


def compute_error(Theta_solved, T_true, fk_func=fk_chain, dh_params=DH):
    """
    Compute position and rotation error for a solved IK solution.

    Args:
        Theta_solved: Solved joint angles
        T_true: Target transformation matrix
        fk_func: Forward kinematics function
        dh_params: DH parameters

    Returns:
        pos_error: Position error in meters
        rot_error: Rotation error in radians
    """
    if Theta_solved is None:
        return np.inf, np.inf
    T_solved = fk_func(dh_params, Theta_solved)
    pos_error = np.linalg.norm(T_true[:3, 3] - T_solved[:3, 3])
    rot_error = R.from_matrix(T_true[:3, :3].T @ T_solved[:3, :3]).magnitude()
    return pos_error, rot_error


def measure_batch_performance(solver_func, targets, batch_size=100, save_path=None):
    """
    Benchmark an IK solver on a set of target poses.

    Args:
        solver_func: IK solver function to benchmark
        targets: Array of target transformation matrices
        batch_size: Number of targets per batch for progress reporting
        save_path: Optional path to save results (timestamp will be appended)

    Returns:
        dict with:
            - "results": (n_points, 5) array [pos_err, rot_err, time, memory, cpu%]
            - "thetas": (n_points, 6) array of solved joint angles
            - "save_path": Actual path where results were saved (if save_path provided)
    """
    n_points = len(targets)
    results = np.zeros((n_points, 5))  # pos_err, rot_err, time, memory, cpu%
    thetas = np.full((n_points, 6), np.nan)
    process = psutil.Process(os.getpid())

    n_batches = int(np.ceil(n_points / batch_size))

    for i in range(0, n_points, batch_size):
        batch = targets[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"Processing batch {batch_num}/{n_batches}...")

        # CPU & memory start snapshot
        cpu_start = process.cpu_times()
        wall_start = time.time()
        mem_before = process.memory_info().rss / (1024**2)  # MB

        for j, pt in enumerate(batch):
            try:
                t0 = time.time()
                Theta_solved = solver_func(pt)
                elapsed = time.time() - t0

                if Theta_solved is None:
                    results[i + j, :] = [np.nan, np.nan, elapsed, np.nan, np.nan]
                else:
                    pos_err, rot_err = compute_error(Theta_solved, pt)
                    results[i + j, :3] = [pos_err, rot_err, elapsed]
                    thetas[i + j, :] = Theta_solved
            except Exception as e:
                print(f"Error processing target {i + j}: {e}")
                results[i + j, :] = [np.nan, np.nan, np.nan, np.nan, np.nan]

        # CPU & memory end snapshot (measured once per batch for efficiency)
        wall_end = time.time()
        cpu_end = process.cpu_times()
        mem_after = process.memory_info().rss / (1024**2)

        cpu_time_used = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
        wall_elapsed = wall_end - wall_start
        cpu_percent = 100 * cpu_time_used / wall_elapsed if wall_elapsed > 0 else 0
        mem_delta = mem_after - mem_before

        # Assign batch-level metrics to all points in the batch
        results[i:i + len(batch), 3] = mem_delta
        results[i:i + len(batch), 4] = cpu_percent

    output = {"results": results, "thetas": thetas, "save_path": None}

    if save_path is not None:
        # Add a timestamp before the file extension
        base, ext = os.path.splitext(save_path)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path_dated = f"{base}_{timestamp}{ext}"

        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(save_path_dated)), exist_ok=True)

        # Save results with timestamped filename
        np.save(save_path_dated, {"results": results, "thetas": thetas})

        print(f"\nSaved benchmark results to: {save_path_dated}")
        output["save_path"] = save_path_dated

    return output
