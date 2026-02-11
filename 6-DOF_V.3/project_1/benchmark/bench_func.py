import numpy as np
import time
import psutil
import os
import platform

from datetime import datetime
from kinematics import (
    fk_chain, DH, q_min, q_max, N_JOINTS,
    POS_ERROR_THRESHOLD, ROT_ERROR_THRESHOLD,
    numerical_ik_solve_detailed, geometric_jacobian
)
from scipy.spatial.transform import Rotation as R


def get_cpu_info():
    """Get detailed CPU information for computing power analysis."""
    cpu_info = {
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "processor": platform.processor(),
        "architecture": platform.machine(),
    }

    # Get CPU frequency
    try:
        freq = psutil.cpu_freq()
        if freq:
            cpu_info["freq_current_mhz"] = freq.current
            cpu_info["freq_min_mhz"] = freq.min
            cpu_info["freq_max_mhz"] = freq.max
    except:
        cpu_info["freq_current_mhz"] = None
        cpu_info["freq_min_mhz"] = None
        cpu_info["freq_max_mhz"] = None

    return cpu_info


def measure_cpu_frequency():
    """Measure current CPU frequency during computation."""
    try:
        freq = psutil.cpu_freq()
        return freq.current if freq else None
    except:
        return None


def estimate_flops(n_operations, elapsed_time):
    """
    Estimate FLOPS (Floating Point Operations Per Second).

    For IK solving, we estimate operations based on:
    - Matrix multiplications (4x4 matrices): ~64 FLOPs each
    - Jacobian computation: ~200 FLOPs per joint
    - DLS solve: ~500 FLOPs
    """
    if elapsed_time <= 0:
        return 0.0
    return n_operations / elapsed_time


def get_system_power_info():
    """
    Get system power information if available.
    Returns battery/power status on laptops.
    """
    power_info = {}
    try:
        battery = psutil.sensors_battery()
        if battery:
            power_info["battery_percent"] = battery.percent
            power_info["power_plugged"] = battery.power_plugged
            power_info["battery_time_left"] = battery.secsleft if battery.secsleft != -1 else None
    except:
        pass

    return power_info


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


def compute_manipulability(dh_params, q):
    """
    Compute Yoshikawa manipulability index.

    w = sqrt(det(J * J^T))

    Higher values indicate configurations further from singularities.
    """
    try:
        J, _ = geometric_jacobian(dh_params, q)
        JJt = J @ J.T
        return np.sqrt(max(0, np.linalg.det(JJt)))
    except:
        return 0.0


def check_joint_limits(q, q_min_limits=q_min, q_max_limits=q_max):
    """
    Check if solution respects joint limits.

    Returns:
        within_limits: Boolean, True if all joints within limits
        violations: Array of violation amounts (0 if within limits)
        margin: Minimum distance to any joint limit
    """
    if q is None:
        return False, np.full(len(q_min_limits), np.inf), 0.0

    q = np.asarray(q)
    lower_violation = np.maximum(0, q_min_limits - q)
    upper_violation = np.maximum(0, q - q_max_limits)
    violations = lower_violation + upper_violation

    within_limits = np.all(violations == 0)

    # Compute margin to closest limit
    lower_margin = q - q_min_limits
    upper_margin = q_max_limits - q
    margin = np.min(np.minimum(lower_margin, upper_margin))

    return within_limits, violations, margin


def compute_workspace_region(T_target):
    """
    Categorize target position in workspace for regional analysis.

    Returns:
        region: String identifier for workspace region
        distance_from_base: Euclidean distance from robot base
        height: Z-coordinate of target
    """
    pos = T_target[:3, 3]
    x, y, z = pos

    distance_from_base = np.sqrt(x**2 + y**2)

    # Categorize by distance (inner/middle/outer)
    if distance_from_base < 1.5:
        radial = "inner"
    elif distance_from_base < 3.5:
        radial = "middle"
    else:
        radial = "outer"

    # Categorize by height (low/mid/high)
    if z < 0.5:
        vertical = "low"
    elif z < 2.0:
        vertical = "mid"
    else:
        vertical = "high"

    region = f"{radial}_{vertical}"

    return region, distance_from_base, z


def measure_batch_performance(solver_func, targets, batch_size=100, save_path=None,
                               solver_type="generic", detailed_numerical=False):
    """
    Comprehensive benchmark for IK solvers with extended metrics.

    Args:
        solver_func: IK solver function to benchmark
        targets: Array of target transformation matrices
        batch_size: Number of targets per batch for progress reporting
        save_path: Optional path to save results (timestamp will be appended)
        solver_type: "analytical" or "numerical" for type-specific metrics
        detailed_numerical: If True and solver_type="numerical", use detailed solver

    Returns:
        dict with comprehensive benchmark results
    """
    n_points = len(targets)
    process = psutil.Process(os.getpid())
    n_batches = int(np.ceil(n_points / batch_size))

    # Core metrics arrays
    results = np.zeros((n_points, 5))  # pos_err, rot_err, time, memory, cpu%
    thetas = np.full((n_points, N_JOINTS), np.nan)

    # Extended metrics
    success = np.zeros(n_points, dtype=bool)  # Met error thresholds
    joint_limit_ok = np.zeros(n_points, dtype=bool)  # Within joint limits
    joint_margin = np.zeros(n_points)  # Distance to joint limits
    manipulability = np.zeros(n_points)  # Singularity measure

    # Workspace analysis
    workspace_regions = []
    distance_from_base = np.zeros(n_points)
    target_heights = np.zeros(n_points)

    # Numerical solver specific
    iterations = np.zeros(n_points, dtype=int) if solver_type == "numerical" else None
    converged = np.zeros(n_points, dtype=bool) if solver_type == "numerical" else None
    restarts = np.zeros(n_points, dtype=int) if solver_type == "numerical" else None

    # Computing power metrics
    cpu_frequencies = []  # Track CPU frequency per batch
    batch_throughput = []  # Solves per second per batch

    # Get initial CPU info
    cpu_info = get_cpu_info()
    power_info = get_system_power_info()

    print(f"\n{'='*60}")
    print(f"Benchmarking {solver_type.upper()} solver on {n_points:,} targets")
    print(f"{'='*60}")
    print(f"\nSystem Info:")
    print(f"  CPU: {cpu_info.get('processor', 'Unknown')}")
    print(f"  Cores: {cpu_info.get('physical_cores', '?')} physical, {cpu_info.get('logical_cores', '?')} logical")
    if cpu_info.get('freq_max_mhz'):
        print(f"  Max Frequency: {cpu_info['freq_max_mhz']:.0f} MHz")
    if power_info.get('power_plugged') is not None:
        status = "Plugged in" if power_info['power_plugged'] else f"Battery ({power_info.get('battery_percent', '?')}%)"
        print(f"  Power: {status}")
    print()

    for i in range(0, n_points, batch_size):
        batch = targets[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"Processing batch {batch_num}/{n_batches}...")

        # CPU & memory start snapshot
        cpu_start = process.cpu_times()
        wall_start = time.time()
        mem_before = process.memory_info().rss / (1024**2)

        for j, pt in enumerate(batch):
            idx = i + j

            # Workspace analysis
            region, dist, height = compute_workspace_region(pt)
            workspace_regions.append(region)
            distance_from_base[idx] = dist
            target_heights[idx] = height

            try:
                t0 = time.time()
                cpu0 = time.process_time()

                # Use detailed solver for numerical if requested
                if solver_type == "numerical" and detailed_numerical:
                    result = numerical_ik_solve_detailed(pt)
                    Theta_solved = result['solution']
                    iterations[idx] = result['iterations']
                    converged[idx] = result['converged']
                    restarts[idx] = result['restarts']
                    manipulability[idx] = result['manipulability']
                else:
                    Theta_solved = solver_func(pt)
                    if Theta_solved is not None:
                        manipulability[idx] = compute_manipulability(DH, Theta_solved)

                elapsed = time.time() - t0
                cpu_elapsed = time.process_time() - cpu0
                cpu_pct = 100 * cpu_elapsed / elapsed if elapsed > 0 else 0

                if Theta_solved is None:
                    results[idx, :] = [np.nan, np.nan, elapsed, np.nan, np.nan]
                else:
                    pos_err, rot_err = compute_error(Theta_solved, pt)
                    results[idx, :3] = [pos_err, rot_err, elapsed]
                    results[idx, 4] = cpu_pct
                    thetas[idx, :] = Theta_solved

                    # Check success criteria
                    success[idx] = (pos_err < POS_ERROR_THRESHOLD and
                                   rot_err < ROT_ERROR_THRESHOLD)

                    # Check joint limits
                    within_limits, _, margin = check_joint_limits(Theta_solved)
                    joint_limit_ok[idx] = within_limits
                    joint_margin[idx] = margin

            except Exception as e:
                print(f"Error processing target {idx}: {e}")
                results[idx, :] = [np.nan, np.nan, np.nan, np.nan, np.nan]

        # CPU & memory end snapshot
        wall_end = time.time()
        cpu_end = process.cpu_times()
        mem_after = process.memory_info().rss / (1024**2)

        cpu_time_used = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
        wall_elapsed = wall_end - wall_start
        cpu_percent = 100 * cpu_time_used / wall_elapsed if wall_elapsed > 0 else 0
        mem_delta = mem_after - mem_before

        results[i:i + len(batch), 3] = mem_delta

        # Track computing power metrics
        current_freq = measure_cpu_frequency()
        if current_freq:
            cpu_frequencies.append(current_freq)

        # Calculate throughput (solves per second)
        if wall_elapsed > 0:
            throughput = len(batch) / wall_elapsed
            batch_throughput.append(throughput)

    # Compute summary statistics
    valid_mask = ~np.isnan(results[:, 0])

    # Computing power summary
    computing_power = {
        "cpu_info": cpu_info,
        "power_info": power_info,
        "cpu_frequencies": np.array(cpu_frequencies) if cpu_frequencies else None,
        "batch_throughput": np.array(batch_throughput) if batch_throughput else None,
    }

    # Calculate computing power statistics
    if cpu_frequencies:
        computing_power["freq_mean_mhz"] = float(np.mean(cpu_frequencies))
        computing_power["freq_min_mhz"] = float(np.min(cpu_frequencies))
        computing_power["freq_max_mhz"] = float(np.max(cpu_frequencies))

    if batch_throughput:
        computing_power["throughput_mean"] = float(np.mean(batch_throughput))
        computing_power["throughput_max"] = float(np.max(batch_throughput))
        computing_power["throughput_min"] = float(np.min(batch_throughput))

    # Estimate total FLOPS (rough estimate based on solver type)
    valid_times = results[valid_mask, 2]
    total_time = np.sum(valid_times) if len(valid_times) > 0 else 0

    # Rough FLOP estimates per IK solve:
    # - Analytical: ~500 FLOPs (direct computation)
    # - Numerical: ~5000 FLOPs per iteration (Jacobian + DLS)
    if solver_type == "analytical":
        flops_per_solve = 500
    else:
        avg_iters = np.mean(iterations[valid_mask]) if iterations is not None and valid_mask.any() else 50
        flops_per_solve = int(5000 * avg_iters)

    total_flops = int(np.sum(valid_mask)) * flops_per_solve
    if total_time > 0:
        computing_power["estimated_mflops"] = (total_flops / total_time) / 1e6
        computing_power["total_flops"] = total_flops
    else:
        computing_power["estimated_mflops"] = 0
        computing_power["total_flops"] = 0

    summary = compute_summary_statistics(
        results, success, joint_limit_ok, manipulability,
        iterations, converged, restarts, workspace_regions, distance_from_base,
        solver_type, valid_mask, computing_power
    )

    # Build output dictionary
    output = {
        "results": results,
        "thetas": thetas,
        "success": success,
        "joint_limit_ok": joint_limit_ok,
        "joint_margin": joint_margin,
        "manipulability": manipulability,
        "workspace_regions": np.array(workspace_regions),
        "distance_from_base": distance_from_base,
        "target_heights": target_heights,
        "computing_power": computing_power,
        "summary": summary,
        "save_path": None
    }

    # Add numerical-specific metrics
    if solver_type == "numerical":
        output["iterations"] = iterations
        output["converged"] = converged
        output["restarts"] = restarts

    # Print summary
    print_summary(summary, solver_type)

    if save_path is not None:
        base, ext = os.path.splitext(save_path)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path_dated = f"{base}_{timestamp}{ext}"
        os.makedirs(os.path.dirname(os.path.abspath(save_path_dated)), exist_ok=True)
        np.save(save_path_dated, output)
        print(f"\nSaved benchmark results to: {save_path_dated}")
        output["save_path"] = save_path_dated

    return output


def compute_summary_statistics(results, success, joint_limit_ok, manipulability,
                                iterations, converged, restarts, workspace_regions, distance_from_base,
                                solver_type, valid_mask, computing_power=None):
    """Compute comprehensive summary statistics."""

    pos_errors = results[valid_mask, 0]
    rot_errors = results[valid_mask, 1]
    times = results[valid_mask, 2]

    summary = {
        # Sample info
        "n_total": len(results),
        "n_valid": int(np.sum(valid_mask)),
        "n_failed": int(np.sum(~valid_mask)),

        # Success rate
        "success_rate": float(np.mean(success[valid_mask]) * 100) if valid_mask.any() else 0,
        "joint_limit_compliance": float(np.mean(joint_limit_ok[valid_mask]) * 100) if valid_mask.any() else 0,

        # Position error statistics
        "pos_error_mean": float(np.mean(pos_errors)) if len(pos_errors) > 0 else np.nan,
        "pos_error_std": float(np.std(pos_errors)) if len(pos_errors) > 0 else np.nan,
        "pos_error_median": float(np.median(pos_errors)) if len(pos_errors) > 0 else np.nan,
        "pos_error_p95": float(np.percentile(pos_errors, 95)) if len(pos_errors) > 0 else np.nan,
        "pos_error_p99": float(np.percentile(pos_errors, 99)) if len(pos_errors) > 0 else np.nan,
        "pos_error_max": float(np.max(pos_errors)) if len(pos_errors) > 0 else np.nan,

        # Rotation error statistics
        "rot_error_mean": float(np.mean(rot_errors)) if len(rot_errors) > 0 else np.nan,
        "rot_error_std": float(np.std(rot_errors)) if len(rot_errors) > 0 else np.nan,
        "rot_error_median": float(np.median(rot_errors)) if len(rot_errors) > 0 else np.nan,
        "rot_error_p95": float(np.percentile(rot_errors, 95)) if len(rot_errors) > 0 else np.nan,
        "rot_error_p99": float(np.percentile(rot_errors, 99)) if len(rot_errors) > 0 else np.nan,
        "rot_error_max": float(np.max(rot_errors)) if len(rot_errors) > 0 else np.nan,

        # Time statistics
        "time_mean": float(np.mean(times)) if len(times) > 0 else np.nan,
        "time_std": float(np.std(times)) if len(times) > 0 else np.nan,
        "time_median": float(np.median(times)) if len(times) > 0 else np.nan,
        "time_p95": float(np.percentile(times, 95)) if len(times) > 0 else np.nan,
        "time_total": float(np.sum(times)) if len(times) > 0 else np.nan,

        # Manipulability statistics
        "manipulability_mean": float(np.mean(manipulability[valid_mask])) if valid_mask.any() else np.nan,
        "manipulability_min": float(np.min(manipulability[valid_mask])) if valid_mask.any() else np.nan,
    }

    # Workspace region analysis
    unique_regions = np.unique(workspace_regions)
    region_stats = {}
    for region in unique_regions:
        region_mask = np.array(workspace_regions) == region
        combined_mask = valid_mask & region_mask
        if combined_mask.any():
            region_stats[region] = {
                "count": int(np.sum(combined_mask)),
                "success_rate": float(np.mean(success[combined_mask]) * 100),
                "pos_error_mean": float(np.mean(results[combined_mask, 0])),
                "time_mean": float(np.mean(results[combined_mask, 2])),
            }
    summary["workspace_analysis"] = region_stats

    # Distance correlation
    if valid_mask.any() and len(pos_errors) > 1:
        valid_dist = distance_from_base[valid_mask]
        corr = np.corrcoef(valid_dist, pos_errors)[0, 1]
        summary["error_distance_correlation"] = float(corr) if not np.isnan(corr) else 0.0

    # Numerical solver specific
    if solver_type == "numerical" and iterations is not None:
        valid_iters = iterations[valid_mask]
        summary["iterations_mean"] = float(np.mean(valid_iters)) if len(valid_iters) > 0 else np.nan
        summary["iterations_median"] = float(np.median(valid_iters)) if len(valid_iters) > 0 else np.nan
        summary["iterations_max"] = int(np.max(valid_iters)) if len(valid_iters) > 0 else 0
        summary["convergence_rate"] = float(np.mean(converged[valid_mask]) * 100) if valid_mask.any() else 0
        summary["avg_restarts"] = float(np.mean(restarts[valid_mask])) if valid_mask.any() else np.nan

    # Computing power metrics
    if computing_power:
        summary["computing_power"] = {
            "cpu_physical_cores": computing_power.get("cpu_info", {}).get("physical_cores"),
            "cpu_logical_cores": computing_power.get("cpu_info", {}).get("logical_cores"),
            "cpu_freq_max_mhz": computing_power.get("cpu_info", {}).get("freq_max_mhz"),
            "freq_mean_mhz": computing_power.get("freq_mean_mhz"),
            "freq_during_min_mhz": computing_power.get("freq_min_mhz"),
            "freq_during_max_mhz": computing_power.get("freq_max_mhz"),
            "throughput_mean_solves_per_sec": computing_power.get("throughput_mean"),
            "throughput_max_solves_per_sec": computing_power.get("throughput_max"),
            "estimated_mflops": computing_power.get("estimated_mflops"),
            "total_flops": computing_power.get("total_flops"),
        }

    return summary


def print_summary(summary, solver_type):
    """Print formatted summary statistics."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY - {solver_type.upper()} SOLVER")
    print(f"{'='*60}")

    print(f"\n--- Sample Info ---")
    print(f"Total targets:     {summary['n_total']:,}")
    print(f"Valid solutions:   {summary['n_valid']:,}")
    print(f"Failed:            {summary['n_failed']:,}")

    print(f"\n--- Success Metrics ---")
    print(f"Success rate:           {summary['success_rate']:.2f}%")
    print(f"Joint limit compliance: {summary['joint_limit_compliance']:.2f}%")

    print(f"\n--- Position Error (meters) ---")
    print(f"Mean:     {summary['pos_error_mean']:.2e}")
    print(f"Std:      {summary['pos_error_std']:.2e}")
    print(f"Median:   {summary['pos_error_median']:.2e}")
    print(f"95th %:   {summary['pos_error_p95']:.2e}")
    print(f"99th %:   {summary['pos_error_p99']:.2e}")
    print(f"Max:      {summary['pos_error_max']:.2e}")

    print(f"\n--- Rotation Error (radians) ---")
    print(f"Mean:     {summary['rot_error_mean']:.2e}")
    print(f"Std:      {summary['rot_error_std']:.2e}")
    print(f"Median:   {summary['rot_error_median']:.2e}")
    print(f"95th %:   {summary['rot_error_p95']:.2e}")
    print(f"99th %:   {summary['rot_error_p99']:.2e}")
    print(f"Max:      {summary['rot_error_max']:.2e}")

    print(f"\n--- Computation Time (seconds) ---")
    print(f"Mean:     {summary['time_mean']:.6f}")
    print(f"Std:      {summary['time_std']:.6f}")
    print(f"Median:   {summary['time_median']:.6f}")
    print(f"95th %:   {summary['time_p95']:.6f}")
    print(f"Total:    {summary['time_total']:.2f}")

    print(f"\n--- Manipulability ---")
    print(f"Mean:     {summary['manipulability_mean']:.4f}")
    print(f"Min:      {summary['manipulability_min']:.4f}")

    if solver_type == "numerical":
        print(f"\n--- Numerical Solver Metrics ---")
        print(f"Convergence rate:   {summary.get('convergence_rate', 0):.2f}%")
        print(f"Mean iterations:    {summary.get('iterations_mean', 0):.1f}")
        print(f"Median iterations:  {summary.get('iterations_median', 0):.1f}")
        print(f"Max iterations:     {summary.get('iterations_max', 0)}")
        print(f"Avg restarts:       {summary.get('avg_restarts', 0):.2f}")

    print(f"\n--- Error vs Distance Correlation ---")
    print(f"Correlation:  {summary.get('error_distance_correlation', 0):.4f}")

    print(f"\n--- Workspace Region Analysis ---")
    for region, stats in summary.get('workspace_analysis', {}).items():
        print(f"  {region}:")
        print(f"    Count: {stats['count']}, Success: {stats['success_rate']:.1f}%, "
              f"Pos Error: {stats['pos_error_mean']:.2e}")

    # Computing power metrics
    cp = summary.get('computing_power', {})
    if cp:
        print(f"\n--- Computing Power ---")
        if cp.get('cpu_physical_cores'):
            print(f"CPU Cores:        {cp['cpu_physical_cores']} physical, {cp.get('cpu_logical_cores', '?')} logical")
        if cp.get('cpu_freq_max_mhz'):
            print(f"CPU Max Freq:     {cp['cpu_freq_max_mhz']:.0f} MHz")
        if cp.get('freq_mean_mhz'):
            print(f"Avg Freq During:  {cp['freq_mean_mhz']:.0f} MHz")
        if cp.get('throughput_mean_solves_per_sec'):
            print(f"Throughput (mean): {cp['throughput_mean_solves_per_sec']:.1f} solves/sec")
            print(f"Throughput (max):  {cp['throughput_max_solves_per_sec']:.1f} solves/sec")
        if cp.get('estimated_mflops'):
            print(f"Est. Performance: {cp['estimated_mflops']:.2f} MFLOPS")
        if cp.get('total_flops'):
            total_gflops = cp['total_flops'] / 1e9
            print(f"Total FLOPs:      {total_gflops:.2f} GFLOPS")

    print(f"\n{'='*60}\n")


# Legacy function for backward compatibility
def measure_batch_performance_simple(solver_func, targets, batch_size=100, save_path=None):
    """
    Simple benchmark (original interface) for backward compatibility.
    """
    return measure_batch_performance(
        solver_func, targets, batch_size, save_path,
        solver_type="generic", detailed_numerical=False
    )
