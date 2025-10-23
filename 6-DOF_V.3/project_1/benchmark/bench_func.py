import numpy as np
import time
import psutil
import os

from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from kinematics import fk_chain , DH


def compute_error(Theta_solved, T_true, fk_chain=fk_chain, DH=DH):
    if Theta_solved is None:
        return np.inf, np.inf
    T_solved = fk_chain(DH, Theta_solved)
    pos_error = np.linalg.norm(T_true[:3, 3] - T_solved[:3, 3])
    rot_error = R.from_matrix(T_true[:3, :3].T @ T_solved[:3, :3]).magnitude()
    return pos_error, rot_error



def measure_batch_performance(solver_func, targets, batch_size=100, save_path=None):
    n_points = len(targets)
    results = np.zeros((n_points, 5))  # pos_err, rot_err, time, memory, cpu%
    thetas = np.full((n_points, 6), np.nan)
    process = psutil.Process(os.getpid())

    for i in range(0, n_points, batch_size):
        batch = targets[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{int(np.ceil(n_points / batch_size))}...")

        # --- CPU & memory start snapshot ---
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
                    mem_now = process.memory_info().rss / (1024**2)
                    results[i + j, :4] = [pos_err, rot_err, elapsed, mem_now - mem_before]
                    thetas[i + j, :] = Theta_solved
            except Exception as e:
                print(f"Error processing target {i + j}: {e}")
                results[i + j, :] = [np.nan, np.nan, np.nan, np.nan, np.nan]

        # --- CPU & memory end snapshot ---
        wall_end = time.time()
        cpu_end = process.cpu_times()

        cpu_time_used = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
        wall_elapsed = wall_end - wall_start
        cpu_percent = 100 * cpu_time_used / wall_elapsed if wall_elapsed > 0 else 0

        # Assign this CPU% to all points in the batch
        results[i:i + len(batch), 4] = cpu_percent

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        np.save(save_path, {"results": results, "thetas": thetas})
        print(f"\n✅ Saved benchmark results to: {save_path}")

    return {"results": results, "thetas": thetas}
