"""
Comprehensive Solver Comparison Analysis

Compares analytical and numerical IK solvers with:
- Descriptive statistics for all metrics
- Statistical significance tests (t-test, Mann-Whitney U)
- Effect size calculations (Cohen's d)
- Workspace region analysis
- Formatted output tables
"""

import numpy as np
import os
import sys
from scipy import stats
from tabulate import tabulate
from datetime import datetime

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")

sys.path.insert(0, PROJECT_DIR)


def load_latest_results(solver_type):
    """Load the most recent benchmark results for a solver."""
    folder = os.path.join(DATA_DIR, f"{solver_type}_solutions")

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Results folder not found: {folder}")

    # Find most recent results file
    files = [f for f in os.listdir(folder) if f.endswith('.npy')]
    if not files:
        raise FileNotFoundError(f"No .npy files found in {folder}")

    # Sort by modification time, get most recent
    files_with_time = [(f, os.path.getmtime(os.path.join(folder, f))) for f in files]
    files_with_time.sort(key=lambda x: x[1], reverse=True)
    latest_file = files_with_time[0][0]

    filepath = os.path.join(folder, latest_file)
    print(f"Loading {solver_type} results from: {latest_file}")

    data = np.load(filepath, allow_pickle=True).item()
    return data


def compute_statistics(arr, name=""):
    """Compute comprehensive statistics for an array."""
    arr = np.asarray(arr)
    valid = arr[~np.isnan(arr)]

    if len(valid) == 0:
        return {
            "n": 0, "mean": np.nan, "std": np.nan, "median": np.nan,
            "min": np.nan, "max": np.nan, "p25": np.nan, "p75": np.nan,
            "p95": np.nan, "p99": np.nan, "iqr": np.nan
        }

    return {
        "n": len(valid),
        "mean": np.mean(valid),
        "std": np.std(valid),
        "median": np.median(valid),
        "min": np.min(valid),
        "max": np.max(valid),
        "p25": np.percentile(valid, 25),
        "p75": np.percentile(valid, 75),
        "p95": np.percentile(valid, 95),
        "p99": np.percentile(valid, 99),
        "iqr": np.percentile(valid, 75) - np.percentile(valid, 25)
    }


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    g1 = np.asarray(group1)[~np.isnan(group1)]
    g2 = np.asarray(group2)[~np.isnan(group2)]

    if len(g1) < 2 or len(g2) < 2:
        return np.nan

    n1, n2 = len(g1), len(g2)
    var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(g1) - np.mean(g2)) / pooled_std


def interpret_cohens_d(d):
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def interpret_p_value(p):
    """Interpret p-value significance."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def format_pvalue(p):
    """
    Format p-value for display, handling extremely small values.

    Standard .2e format shows 0.00e+00 for values < ~1e-308.
    This function provides better representation for tiny p-values.
    """
    if p is None or np.isnan(p):
        return "-"

    if p == 0:
        return "< 1e-308"

    if p < 1e-300:
        # For extremely small values, compute log10 manually
        log_p = np.log10(p) if p > 0 else -np.inf
        if np.isinf(log_p):
            return "< 1e-308"
        return f"~1e{int(log_p)}"

    if p < 1e-100:
        # Show more precision for very small values
        log_p = np.log10(p)
        mantissa = p / (10 ** int(np.floor(log_p)))
        return f"{mantissa:.1f}e{int(np.floor(log_p))}"

    # Standard scientific notation for normal small values
    return f"{p:.2e}"


def statistical_comparison(analytical_arr, numerical_arr, metric_name):
    """
    Perform comprehensive statistical comparison between two groups.

    Returns dict with test results and interpretations.
    """
    a = np.asarray(analytical_arr)[~np.isnan(analytical_arr)]
    n = np.asarray(numerical_arr)[~np.isnan(numerical_arr)]

    result = {
        "metric": metric_name,
        "analytical_n": len(a),
        "numerical_n": len(n),
    }

    if len(a) < 2 or len(n) < 2:
        result.update({
            "t_statistic": np.nan, "t_pvalue": np.nan,
            "u_statistic": np.nan, "u_pvalue": np.nan,
            "cohens_d": np.nan, "effect_interpretation": "insufficient data"
        })
        return result

    # Welch's t-test (doesn't assume equal variance)
    t_stat, t_pvalue = stats.ttest_ind(a, n, equal_var=False)

    # Mann-Whitney U test (non-parametric alternative)
    try:
        u_stat, u_pvalue = stats.mannwhitneyu(a, n, alternative='two-sided')
    except ValueError:
        u_stat, u_pvalue = np.nan, np.nan

    # Effect size
    d = cohens_d(a, n)

    result.update({
        "t_statistic": t_stat,
        "t_pvalue": t_pvalue,
        "t_significance": interpret_p_value(t_pvalue),
        "u_statistic": u_stat,
        "u_pvalue": u_pvalue,
        "u_significance": interpret_p_value(u_pvalue),
        "cohens_d": d,
        "effect_interpretation": interpret_cohens_d(d),
        "better_solver": "Analytical" if np.mean(a) < np.mean(n) else "Numerical"
    })

    return result


def format_value(val, fmt="e", precision=2):
    """Format a value for display."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "-"
    if fmt == "e":
        return f"{val:.{precision}e}"
    elif fmt == "f":
        return f"{val:.{precision}f}"
    elif fmt == "pct":
        return f"{val:.{precision}f}%"
    elif fmt == "int":
        return f"{int(val):,}"
    return str(val)


def generate_comparison_report(analytical_data, numerical_data):
    """Generate comprehensive comparison report."""

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("IK SOLVER COMPARISON REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)

    # Extract core arrays
    a_results = analytical_data["results"]
    n_results = numerical_data["results"]

    a_success = analytical_data.get("success", np.zeros(len(a_results), dtype=bool))
    n_success = numerical_data.get("success", np.zeros(len(n_results), dtype=bool))

    a_manip = analytical_data.get("manipulability", np.zeros(len(a_results)))
    n_manip = numerical_data.get("manipulability", np.zeros(len(n_results)))

    # ============================================================
    # SECTION 1: SAMPLE OVERVIEW
    # ============================================================
    report_lines.append("\n" + "=" * 80)
    report_lines.append("1. SAMPLE OVERVIEW")
    report_lines.append("=" * 80)

    a_valid = ~np.isnan(a_results[:, 0])
    n_valid = ~np.isnan(n_results[:, 0])

    overview_table = [
        ["Total Targets", format_value(len(a_results), "int"), format_value(len(n_results), "int")],
        ["Valid Solutions", format_value(np.sum(a_valid), "int"), format_value(np.sum(n_valid), "int")],
        ["Failed Solutions", format_value(np.sum(~a_valid), "int"), format_value(np.sum(~n_valid), "int")],
        ["Success Rate", format_value(np.mean(a_success) * 100, "pct"), format_value(np.mean(n_success) * 100, "pct")],
    ]

    # Add joint limit compliance if available
    if "joint_limit_ok" in analytical_data:
        a_jl = analytical_data["joint_limit_ok"]
        n_jl = numerical_data.get("joint_limit_ok", np.zeros_like(a_jl))
        overview_table.append([
            "Joint Limit Compliance",
            format_value(np.mean(a_jl[a_valid]) * 100, "pct"),
            format_value(np.mean(n_jl[n_valid]) * 100, "pct")
        ])

    report_lines.append("\n" + tabulate(overview_table,
                                         headers=["Metric", "Analytical", "Numerical"],
                                         tablefmt="grid"))

    # ============================================================
    # SECTION 2: POSITION ERROR ANALYSIS
    # ============================================================
    report_lines.append("\n" + "=" * 80)
    report_lines.append("2. POSITION ERROR ANALYSIS (meters)")
    report_lines.append("=" * 80)

    a_pos = a_results[a_valid, 0]
    n_pos = n_results[n_valid, 0]

    a_stats = compute_statistics(a_pos)
    n_stats = compute_statistics(n_pos)

    pos_table = [
        ["N (valid)", format_value(a_stats["n"], "int"), format_value(n_stats["n"], "int")],
        ["Mean", format_value(a_stats["mean"], "e"), format_value(n_stats["mean"], "e")],
        ["Std Dev", format_value(a_stats["std"], "e"), format_value(n_stats["std"], "e")],
        ["Median", format_value(a_stats["median"], "e"), format_value(n_stats["median"], "e")],
        ["Min", format_value(a_stats["min"], "e"), format_value(n_stats["min"], "e")],
        ["Max", format_value(a_stats["max"], "e"), format_value(n_stats["max"], "e")],
        ["25th Percentile", format_value(a_stats["p25"], "e"), format_value(n_stats["p25"], "e")],
        ["75th Percentile", format_value(a_stats["p75"], "e"), format_value(n_stats["p75"], "e")],
        ["95th Percentile", format_value(a_stats["p95"], "e"), format_value(n_stats["p95"], "e")],
        ["99th Percentile", format_value(a_stats["p99"], "e"), format_value(n_stats["p99"], "e")],
        ["IQR", format_value(a_stats["iqr"], "e"), format_value(n_stats["iqr"], "e")],
    ]

    report_lines.append("\n" + tabulate(pos_table,
                                         headers=["Statistic", "Analytical", "Numerical"],
                                         tablefmt="grid"))

    # Statistical test
    pos_comparison = statistical_comparison(a_pos, n_pos, "Position Error")
    report_lines.append(f"\nStatistical Comparison:")
    report_lines.append(f"  Welch's t-test: t = {pos_comparison['t_statistic']:.4f}, p = {format_pvalue(pos_comparison['t_pvalue'])} {pos_comparison['t_significance']}")
    report_lines.append(f"  Mann-Whitney U: U = {pos_comparison['u_statistic']:.0f}, p = {format_pvalue(pos_comparison['u_pvalue'])} {pos_comparison['u_significance']}")
    report_lines.append(f"  Cohen's d: {pos_comparison['cohens_d']:.4f} ({pos_comparison['effect_interpretation']} effect)")
    report_lines.append(f"  Better Solver: {pos_comparison['better_solver']} (lower error)")

    # ============================================================
    # SECTION 3: ROTATION ERROR ANALYSIS
    # ============================================================
    report_lines.append("\n" + "=" * 80)
    report_lines.append("3. ROTATION ERROR ANALYSIS (radians)")
    report_lines.append("=" * 80)

    a_rot = a_results[a_valid, 1]
    n_rot = n_results[n_valid, 1]

    a_stats = compute_statistics(a_rot)
    n_stats = compute_statistics(n_rot)

    rot_table = [
        ["N (valid)", format_value(a_stats["n"], "int"), format_value(n_stats["n"], "int")],
        ["Mean", format_value(a_stats["mean"], "e"), format_value(n_stats["mean"], "e")],
        ["Std Dev", format_value(a_stats["std"], "e"), format_value(n_stats["std"], "e")],
        ["Median", format_value(a_stats["median"], "e"), format_value(n_stats["median"], "e")],
        ["Min", format_value(a_stats["min"], "e"), format_value(n_stats["min"], "e")],
        ["Max", format_value(a_stats["max"], "e"), format_value(n_stats["max"], "e")],
        ["95th Percentile", format_value(a_stats["p95"], "e"), format_value(n_stats["p95"], "e")],
        ["99th Percentile", format_value(a_stats["p99"], "e"), format_value(n_stats["p99"], "e")],
    ]

    report_lines.append("\n" + tabulate(rot_table,
                                         headers=["Statistic", "Analytical", "Numerical"],
                                         tablefmt="grid"))

    rot_comparison = statistical_comparison(a_rot, n_rot, "Rotation Error")
    report_lines.append(f"\nStatistical Comparison:")
    report_lines.append(f"  Welch's t-test: t = {rot_comparison['t_statistic']:.4f}, p = {format_pvalue(rot_comparison['t_pvalue'])} {rot_comparison['t_significance']}")
    report_lines.append(f"  Mann-Whitney U: U = {rot_comparison['u_statistic']:.0f}, p = {format_pvalue(rot_comparison['u_pvalue'])} {rot_comparison['u_significance']}")
    report_lines.append(f"  Cohen's d: {rot_comparison['cohens_d']:.4f} ({rot_comparison['effect_interpretation']} effect)")
    report_lines.append(f"  Better Solver: {rot_comparison['better_solver']} (lower error)")

    # ============================================================
    # SECTION 4: COMPUTATION TIME ANALYSIS
    # ============================================================
    report_lines.append("\n" + "=" * 80)
    report_lines.append("4. COMPUTATION TIME ANALYSIS (seconds)")
    report_lines.append("=" * 80)

    a_time = a_results[a_valid, 2]
    n_time = n_results[n_valid, 2]

    a_stats = compute_statistics(a_time)
    n_stats = compute_statistics(n_time)

    time_table = [
        ["Mean", format_value(a_stats["mean"], "e"), format_value(n_stats["mean"], "e")],
        ["Std Dev", format_value(a_stats["std"], "e"), format_value(n_stats["std"], "e")],
        ["Median", format_value(a_stats["median"], "e"), format_value(n_stats["median"], "e")],
        ["Min", format_value(a_stats["min"], "e"), format_value(n_stats["min"], "e")],
        ["Max", format_value(a_stats["max"], "e"), format_value(n_stats["max"], "e")],
        ["95th Percentile", format_value(a_stats["p95"], "e"), format_value(n_stats["p95"], "e")],
        ["Total Time", format_value(np.sum(a_time), "f", 2) + "s", format_value(np.sum(n_time), "f", 2) + "s"],
        ["Speedup Factor", "-", format_value(np.mean(n_time) / np.mean(a_time) if np.mean(a_time) > 0 else np.nan, "f", 1) + "x slower"],
    ]

    report_lines.append("\n" + tabulate(time_table,
                                         headers=["Statistic", "Analytical", "Numerical"],
                                         tablefmt="grid"))

    time_comparison = statistical_comparison(a_time, n_time, "Computation Time")
    report_lines.append(f"\nStatistical Comparison:")
    report_lines.append(f"  Welch's t-test: t = {time_comparison['t_statistic']:.4f}, p = {format_pvalue(time_comparison['t_pvalue'])} {time_comparison['t_significance']}")
    report_lines.append(f"  Cohen's d: {time_comparison['cohens_d']:.4f} ({time_comparison['effect_interpretation']} effect)")
    report_lines.append(f"  Faster Solver: {time_comparison['better_solver']}")

    # ============================================================
    # SECTION 5: MANIPULABILITY ANALYSIS
    # ============================================================
    report_lines.append("\n" + "=" * 80)
    report_lines.append("5. MANIPULABILITY INDEX ANALYSIS")
    report_lines.append("=" * 80)
    report_lines.append("(Higher = further from singularities)")

    a_stats = compute_statistics(a_manip[a_valid])
    n_stats = compute_statistics(n_manip[n_valid])

    manip_table = [
        ["Mean", format_value(a_stats["mean"], "f", 4), format_value(n_stats["mean"], "f", 4)],
        ["Std Dev", format_value(a_stats["std"], "f", 4), format_value(n_stats["std"], "f", 4)],
        ["Median", format_value(a_stats["median"], "f", 4), format_value(n_stats["median"], "f", 4)],
        ["Min", format_value(a_stats["min"], "f", 4), format_value(n_stats["min"], "f", 4)],
        ["Max", format_value(a_stats["max"], "f", 4), format_value(n_stats["max"], "f", 4)],
    ]

    report_lines.append("\n" + tabulate(manip_table,
                                         headers=["Statistic", "Analytical", "Numerical"],
                                         tablefmt="grid"))

    manip_comparison = statistical_comparison(a_manip[a_valid], n_manip[n_valid], "Manipulability")
    report_lines.append(f"\nStatistical Comparison:")
    report_lines.append(f"  Welch's t-test: t = {manip_comparison['t_statistic']:.4f}, p = {format_pvalue(manip_comparison['t_pvalue'])} {manip_comparison['t_significance']}")
    report_lines.append(f"  Cohen's d: {manip_comparison['cohens_d']:.4f} ({manip_comparison['effect_interpretation']} effect)")

    # ============================================================
    # SECTION 6: JOINT LIMITS ANALYSIS
    # ============================================================
    if "joint_limit_ok" in analytical_data and "joint_margin" in analytical_data:
        report_lines.append("\n" + "=" * 80)
        report_lines.append("6. JOINT LIMITS ANALYSIS")
        report_lines.append("=" * 80)
        report_lines.append("(Joint margin = minimum distance to any joint limit)")

        a_jl_ok = analytical_data["joint_limit_ok"]
        n_jl_ok = numerical_data.get("joint_limit_ok", np.zeros_like(a_jl_ok))
        a_margin = analytical_data["joint_margin"]
        n_margin = numerical_data.get("joint_margin", np.zeros_like(a_margin))

        a_margin_stats = compute_statistics(a_margin[a_valid])
        n_margin_stats = compute_statistics(n_margin[n_valid])

        jl_table = [
            ["Within Limits (%)", format_value(np.mean(a_jl_ok[a_valid]) * 100, "pct"),
             format_value(np.mean(n_jl_ok[n_valid]) * 100, "pct")],
            ["Mean Margin (rad)", format_value(a_margin_stats["mean"], "f", 4),
             format_value(n_margin_stats["mean"], "f", 4)],
            ["Min Margin (rad)", format_value(a_margin_stats["min"], "f", 4),
             format_value(n_margin_stats["min"], "f", 4)],
            ["Max Margin (rad)", format_value(a_margin_stats["max"], "f", 4),
             format_value(n_margin_stats["max"], "f", 4)],
            ["Std Dev (rad)", format_value(a_margin_stats["std"], "f", 4),
             format_value(n_margin_stats["std"], "f", 4)],
        ]

        report_lines.append("\n" + tabulate(jl_table,
                                             headers=["Statistic", "Analytical", "Numerical"],
                                             tablefmt="grid"))

        # Statistical comparison for joint margin
        jl_comparison = statistical_comparison(a_margin[a_valid], n_margin[n_valid], "Joint Margin")
        report_lines.append(f"\nStatistical Comparison (Joint Margin):")
        report_lines.append(f"  Welch's t-test: t = {jl_comparison['t_statistic']:.4f}, p = {format_pvalue(jl_comparison['t_pvalue'])} {jl_comparison['t_significance']}")
        report_lines.append(f"  Cohen's d: {jl_comparison['cohens_d']:.4f} ({jl_comparison['effect_interpretation']} effect)")
        report_lines.append(f"  Better Joint Margin: {'Analytical' if np.mean(a_margin[a_valid]) > np.mean(n_margin[n_valid]) else 'Numerical'} (higher is better)")

    # ============================================================
    # SECTION 7: MEMORY & CPU RESOURCE USAGE
    # ============================================================
    # Memory is in column 3, CPU% is in column 4 of results
    a_mem = a_results[a_valid, 3]
    n_mem = n_results[n_valid, 3]
    a_cpu = a_results[a_valid, 4]
    n_cpu = n_results[n_valid, 4]

    # Only show if we have valid data
    if not np.all(np.isnan(a_mem)) or not np.all(np.isnan(n_mem)):
        report_lines.append("\n" + "=" * 80)
        report_lines.append("7. MEMORY & CPU RESOURCE USAGE")
        report_lines.append("=" * 80)

        a_mem_stats = compute_statistics(a_mem)
        n_mem_stats = compute_statistics(n_mem)
        a_cpu_stats = compute_statistics(a_cpu)
        n_cpu_stats = compute_statistics(n_cpu)

        resource_table = [
            ["Memory Delta Mean (MB)", format_value(a_mem_stats["mean"], "f", 2),
             format_value(n_mem_stats["mean"], "f", 2)],
            ["Memory Delta Max (MB)", format_value(a_mem_stats["max"], "f", 2),
             format_value(n_mem_stats["max"], "f", 2)],
            ["CPU Utilization Mean (%)", format_value(a_cpu_stats["mean"], "f", 1),
             format_value(n_cpu_stats["mean"], "f", 1)],
            ["CPU Utilization Max (%)", format_value(a_cpu_stats["max"], "f", 1),
             format_value(n_cpu_stats["max"], "f", 1)],
        ]

        report_lines.append("\n" + tabulate(resource_table,
                                             headers=["Metric", "Analytical", "Numerical"],
                                             tablefmt="grid"))

    # ============================================================
    # SECTION 8: NUMERICAL SOLVER SPECIFIC METRICS
    # ============================================================
    if "iterations" in numerical_data and numerical_data["iterations"] is not None:
        report_lines.append("\n" + "=" * 80)
        report_lines.append("8. NUMERICAL SOLVER SPECIFIC METRICS")
        report_lines.append("=" * 80)

        n_iters = numerical_data["iterations"][n_valid]
        n_conv = numerical_data.get("converged", np.zeros_like(n_iters, dtype=bool))[n_valid]
        n_restarts = numerical_data.get("restarts", np.zeros_like(n_iters))[n_valid]

        iter_stats = compute_statistics(n_iters)

        num_table = [
            ["Convergence Rate", "-", format_value(np.mean(n_conv) * 100, "pct")],
            ["Mean Iterations", "-", format_value(iter_stats["mean"], "f", 1)],
            ["Median Iterations", "-", format_value(iter_stats["median"], "f", 1)],
            ["Max Iterations", "-", format_value(iter_stats["max"], "int")],
            ["Mean Restarts", "-", format_value(np.mean(n_restarts), "f", 2)],
        ]

        report_lines.append("\n" + tabulate(num_table,
                                             headers=["Metric", "Analytical", "Numerical"],
                                             tablefmt="grid"))

    # ============================================================
    # SECTION 9: WORKSPACE REGION ANALYSIS
    # ============================================================
    if "workspace_regions" in analytical_data:
        report_lines.append("\n" + "=" * 80)
        report_lines.append("9. WORKSPACE REGION ANALYSIS")
        report_lines.append("=" * 80)

        a_regions = analytical_data["workspace_regions"]
        n_regions = numerical_data.get("workspace_regions", a_regions)

        unique_regions = np.unique(a_regions)

        region_table = []
        for region in sorted(unique_regions):
            a_mask = (a_regions == region) & a_valid
            n_mask = (n_regions == region) & n_valid

            if np.sum(a_mask) > 0 and np.sum(n_mask) > 0:
                a_region_success = np.mean(a_success[a_mask]) * 100
                n_region_success = np.mean(n_success[n_mask]) * 100
                a_region_err = np.mean(a_results[a_mask, 0])
                n_region_err = np.mean(n_results[n_mask, 0])

                region_table.append([
                    region,
                    format_value(np.sum(a_mask), "int"),
                    format_value(a_region_success, "pct"),
                    format_value(a_region_err, "e"),
                    format_value(n_region_success, "pct"),
                    format_value(n_region_err, "e"),
                ])

        report_lines.append("\n" + tabulate(region_table,
                                             headers=["Region", "Count", "A.Success", "A.PosErr", "N.Success", "N.PosErr"],
                                             tablefmt="grid"))

    # ============================================================
    # SECTION 10: DISTANCE & HEIGHT CORRELATION ANALYSIS
    # ============================================================
    if "distance_from_base" in analytical_data and "target_heights" in analytical_data:
        report_lines.append("\n" + "=" * 80)
        report_lines.append("10. DISTANCE & HEIGHT CORRELATION ANALYSIS")
        report_lines.append("=" * 80)
        report_lines.append("(Correlation between target position and solver performance)")

        a_dist = analytical_data["distance_from_base"][a_valid]
        n_dist = numerical_data.get("distance_from_base", analytical_data["distance_from_base"])[n_valid]
        a_height = analytical_data["target_heights"][a_valid]
        n_height = numerical_data.get("target_heights", analytical_data["target_heights"])[n_valid]

        # Distance statistics
        a_dist_stats = compute_statistics(a_dist)
        n_dist_stats = compute_statistics(n_dist)
        a_height_stats = compute_statistics(a_height)
        n_height_stats = compute_statistics(n_height)

        dist_table = [
            ["Mean Distance from Base (m)", format_value(a_dist_stats["mean"], "f", 2),
             format_value(n_dist_stats["mean"], "f", 2)],
            ["Min Distance (m)", format_value(a_dist_stats["min"], "f", 2),
             format_value(n_dist_stats["min"], "f", 2)],
            ["Max Distance (m)", format_value(a_dist_stats["max"], "f", 2),
             format_value(n_dist_stats["max"], "f", 2)],
            ["Mean Target Height (m)", format_value(a_height_stats["mean"], "f", 2),
             format_value(n_height_stats["mean"], "f", 2)],
            ["Min Height (m)", format_value(a_height_stats["min"], "f", 2),
             format_value(n_height_stats["min"], "f", 2)],
            ["Max Height (m)", format_value(a_height_stats["max"], "f", 2),
             format_value(n_height_stats["max"], "f", 2)],
        ]

        report_lines.append("\n" + tabulate(dist_table,
                                             headers=["Metric", "Analytical", "Numerical"],
                                             tablefmt="grid"))

        # Correlation analysis
        report_lines.append("\nCorrelation with Position Error:")

        # Distance vs error correlation
        a_pos_err = a_results[a_valid, 0]
        n_pos_err = n_results[n_valid, 0]

        if len(a_dist) > 1 and len(a_pos_err) > 1:
            a_dist_corr = np.corrcoef(a_dist, a_pos_err)[0, 1]
            report_lines.append(f"  Analytical: Distance vs Error correlation = {a_dist_corr:.4f}")
        if len(n_dist) > 1 and len(n_pos_err) > 1:
            n_dist_corr = np.corrcoef(n_dist, n_pos_err)[0, 1]
            report_lines.append(f"  Numerical:  Distance vs Error correlation = {n_dist_corr:.4f}")

        # Height vs error correlation
        if len(a_height) > 1 and len(a_pos_err) > 1:
            a_height_corr = np.corrcoef(a_height, a_pos_err)[0, 1]
            report_lines.append(f"  Analytical: Height vs Error correlation   = {a_height_corr:.4f}")
        if len(n_height) > 1 and len(n_pos_err) > 1:
            n_height_corr = np.corrcoef(n_height, n_pos_err)[0, 1]
            report_lines.append(f"  Numerical:  Height vs Error correlation   = {n_height_corr:.4f}")

        # Interpretation
        report_lines.append("\nInterpretation:")
        report_lines.append("  |r| < 0.1: negligible, 0.1-0.3: weak, 0.3-0.5: moderate, > 0.5: strong")

    # ============================================================
    # SECTION 11: STATISTICAL TESTS SUMMARY
    # ============================================================
    report_lines.append("\n" + "=" * 80)
    report_lines.append("11. STATISTICAL TESTS SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    report_lines.append("Effect size (Cohen's d): |d|<0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large")

    all_comparisons = [
        statistical_comparison(a_results[a_valid, 0], n_results[n_valid, 0], "Position Error"),
        statistical_comparison(a_results[a_valid, 1], n_results[n_valid, 1], "Rotation Error"),
        statistical_comparison(a_results[a_valid, 2], n_results[n_valid, 2], "Computation Time"),
        statistical_comparison(a_manip[a_valid], n_manip[n_valid], "Manipulability"),
    ]

    # Add joint margin comparison if available
    if "joint_margin" in analytical_data:
        a_margin = analytical_data["joint_margin"][a_valid]
        n_margin = numerical_data.get("joint_margin", np.zeros_like(a_margin))[n_valid]
        jm_comp = statistical_comparison(a_margin, n_margin, "Joint Margin")
        # For joint margin, higher is better (more distance from limits)
        jm_comp["better_solver"] = "Analytical" if np.mean(a_margin) > np.mean(n_margin) else "Numerical"
        all_comparisons.append(jm_comp)

    summary_table = []
    for comp in all_comparisons:
        summary_table.append([
            comp["metric"],
            format_pvalue(comp['t_pvalue']),
            comp.get("t_significance", "-"),
            format_pvalue(comp['u_pvalue']),
            comp.get("u_significance", "-"),
            f"{comp['cohens_d']:.3f}" if not np.isnan(comp['cohens_d']) else "-",
            comp.get("effect_interpretation", "-"),
            comp.get("better_solver", "-"),
        ])

    report_lines.append("\n" + tabulate(summary_table,
                                         headers=["Metric", "t-test p", "Sig", "U-test p", "Sig", "Cohen's d", "Effect", "Better"],
                                         tablefmt="grid"))

    # ============================================================
    # SECTION 12: COMPUTING POWER ANALYSIS
    # ============================================================
    a_cp = analytical_data.get("computing_power", {})
    n_cp = numerical_data.get("computing_power", {})

    if a_cp or n_cp:
        report_lines.append("\n" + "=" * 80)
        report_lines.append("12. COMPUTING POWER ANALYSIS")
        report_lines.append("=" * 80)

        # System info
        a_cpu_info = a_cp.get("cpu_info", {})
        n_cpu_info = n_cp.get("cpu_info", {})

        if a_cpu_info or n_cpu_info:
            cpu_info = a_cpu_info if a_cpu_info else n_cpu_info
            report_lines.append(f"\nSystem Information:")
            report_lines.append(f"  CPU: {cpu_info.get('processor', 'Unknown')}")
            report_lines.append(f"  Physical Cores: {cpu_info.get('physical_cores', '?')}")
            report_lines.append(f"  Logical Cores: {cpu_info.get('logical_cores', '?')}")
            if cpu_info.get('freq_max_mhz'):
                report_lines.append(f"  Max Frequency: {cpu_info['freq_max_mhz']:.0f} MHz")

        # Throughput comparison
        power_table = []

        # CPU frequency during benchmark
        a_freq = a_cp.get("freq_mean_mhz")
        n_freq = n_cp.get("freq_mean_mhz")
        if a_freq or n_freq:
            power_table.append([
                "Avg CPU Freq (MHz)",
                format_value(a_freq, "f", 0) if a_freq else "-",
                format_value(n_freq, "f", 0) if n_freq else "-"
            ])

        # Throughput
        a_throughput = a_cp.get("throughput_mean")
        n_throughput = n_cp.get("throughput_mean")
        if a_throughput or n_throughput:
            power_table.append([
                "Mean Throughput (solves/sec)",
                format_value(a_throughput, "f", 1) if a_throughput else "-",
                format_value(n_throughput, "f", 1) if n_throughput else "-"
            ])

        a_throughput_max = a_cp.get("throughput_max")
        n_throughput_max = n_cp.get("throughput_max")
        if a_throughput_max or n_throughput_max:
            power_table.append([
                "Max Throughput (solves/sec)",
                format_value(a_throughput_max, "f", 1) if a_throughput_max else "-",
                format_value(n_throughput_max, "f", 1) if n_throughput_max else "-"
            ])

        # Estimated MFLOPS
        a_mflops = a_cp.get("estimated_mflops")
        n_mflops = n_cp.get("estimated_mflops")
        if a_mflops or n_mflops:
            power_table.append([
                "Est. Performance (MFLOPS)",
                format_value(a_mflops, "f", 2) if a_mflops else "-",
                format_value(n_mflops, "f", 2) if n_mflops else "-"
            ])

        # Total FLOPs
        a_flops = a_cp.get("total_flops")
        n_flops = n_cp.get("total_flops")
        if a_flops or n_flops:
            a_gflops = a_flops / 1e9 if a_flops else None
            n_gflops = n_flops / 1e9 if n_flops else None
            power_table.append([
                "Total Computation (GFLOPS)",
                format_value(a_gflops, "f", 2) if a_gflops else "-",
                format_value(n_gflops, "f", 2) if n_gflops else "-"
            ])

        if power_table:
            report_lines.append("\n" + tabulate(power_table,
                                                 headers=["Metric", "Analytical", "Numerical"],
                                                 tablefmt="grid"))

        # Throughput comparison
        if a_throughput and n_throughput:
            ratio = a_throughput / n_throughput if n_throughput > 0 else float('inf')
            report_lines.append(f"\nThroughput Comparison:")
            report_lines.append(f"  Analytical throughput: {a_throughput:.1f} solves/sec")
            report_lines.append(f"  Numerical throughput:  {n_throughput:.1f} solves/sec")
            report_lines.append(f"  Analytical is {ratio:.1f}x faster in throughput")

    # ============================================================
    # SECTION 13: CONCLUSIONS
    # ============================================================
    report_lines.append("\n" + "=" * 80)
    report_lines.append("13. CONCLUSIONS")
    report_lines.append("=" * 80)

    # Calculate all comparison metrics
    a_mean_pos = np.mean(a_results[a_valid, 0])
    n_mean_pos = np.mean(n_results[n_valid, 0])
    a_mean_rot = np.mean(a_results[a_valid, 1])
    n_mean_rot = np.mean(n_results[n_valid, 1])
    a_mean_time = np.mean(a_results[a_valid, 2])
    n_mean_time = np.mean(n_results[n_valid, 2])
    a_success_rate = np.mean(a_success) * 100
    n_success_rate = np.mean(n_success) * 100
    a_mean_manip = np.mean(a_manip[a_valid])
    n_mean_manip = np.mean(n_manip[n_valid])

    # Track winners for overall summary
    winners = {"Analytical": 0, "Numerical": 0, "Tie": 0}

    # 1. Position Error (lower is better)
    pos_winner = "Analytical" if a_mean_pos < n_mean_pos else "Numerical"
    if a_mean_pos == n_mean_pos:
        pos_winner = "Tie"
    winners[pos_winner] += 1
    report_lines.append(f"\n1. POSITION ERROR (lower is better): {pos_winner}")
    report_lines.append(f"   - Analytical: {a_mean_pos:.2e} m")
    report_lines.append(f"   - Numerical:  {n_mean_pos:.2e} m")
    if a_mean_pos > 0 and n_mean_pos > 0:
        ratio = n_mean_pos / a_mean_pos
        report_lines.append(f"   - Ratio: Numerical is {ratio:.1f}x higher error")

    # 2. Rotation Error (lower is better)
    rot_winner = "Analytical" if a_mean_rot < n_mean_rot else "Numerical"
    if a_mean_rot == n_mean_rot:
        rot_winner = "Tie"
    winners[rot_winner] += 1
    report_lines.append(f"\n2. ROTATION ERROR (lower is better): {rot_winner}")
    report_lines.append(f"   - Analytical: {a_mean_rot:.2e} rad")
    report_lines.append(f"   - Numerical:  {n_mean_rot:.2e} rad")
    if a_mean_rot > 0 and n_mean_rot > 0:
        ratio = n_mean_rot / a_mean_rot
        report_lines.append(f"   - Ratio: Numerical is {ratio:.1f}x higher error")

    # 3. Computation Time (lower is better)
    time_winner = "Analytical" if a_mean_time < n_mean_time else "Numerical"
    if a_mean_time == n_mean_time:
        time_winner = "Tie"
    winners[time_winner] += 1
    report_lines.append(f"\n3. COMPUTATION TIME (lower is better): {time_winner}")
    report_lines.append(f"   - Analytical: {a_mean_time*1000:.3f} ms")
    report_lines.append(f"   - Numerical:  {n_mean_time*1000:.3f} ms")
    if a_mean_time > 0:
        ratio = n_mean_time / a_mean_time
        report_lines.append(f"   - Ratio: Numerical is {ratio:.1f}x slower")

    # 4. Success Rate (higher is better)
    success_winner = "Analytical" if a_success_rate > n_success_rate else "Numerical"
    if a_success_rate == n_success_rate:
        success_winner = "Tie"
    winners[success_winner] += 1
    report_lines.append(f"\n4. SUCCESS RATE (higher is better): {success_winner}")
    report_lines.append(f"   - Analytical: {a_success_rate:.2f}%")
    report_lines.append(f"   - Numerical:  {n_success_rate:.2f}%")

    # 5. Joint Limit Compliance (higher is better)
    if "joint_limit_ok" in analytical_data:
        a_jl_rate = np.mean(analytical_data["joint_limit_ok"][a_valid]) * 100
        n_jl_rate = np.mean(numerical_data.get("joint_limit_ok", np.zeros_like(a_valid))[n_valid]) * 100
        jl_winner = "Analytical" if a_jl_rate > n_jl_rate else "Numerical"
        if a_jl_rate == n_jl_rate:
            jl_winner = "Tie"
        winners[jl_winner] += 1
        report_lines.append(f"\n5. JOINT LIMIT COMPLIANCE (higher is better): {jl_winner}")
        report_lines.append(f"   - Analytical: {a_jl_rate:.2f}%")
        report_lines.append(f"   - Numerical:  {n_jl_rate:.2f}%")
        if a_jl_rate > 0:
            ratio = n_jl_rate / a_jl_rate
            report_lines.append(f"   - Ratio: Numerical is {ratio:.1f}x better compliance")

    # 6. Manipulability (higher is better - further from singularities)
    manip_winner = "Analytical" if a_mean_manip > n_mean_manip else "Numerical"
    if abs(a_mean_manip - n_mean_manip) < 0.01:
        manip_winner = "Tie"
    winners[manip_winner] += 1
    report_lines.append(f"\n6. MANIPULABILITY (higher is better): {manip_winner}")
    report_lines.append(f"   - Analytical: {a_mean_manip:.4f}")
    report_lines.append(f"   - Numerical:  {n_mean_manip:.4f}")

    # 7. Memory Usage (lower is better)
    a_mem = np.mean(a_results[a_valid, 3])
    n_mem = np.mean(n_results[n_valid, 3])
    if not np.isnan(a_mem) and not np.isnan(n_mem):
        mem_winner = "Analytical" if a_mem < n_mem else "Numerical"
        if abs(a_mem - n_mem) < 0.01:
            mem_winner = "Tie"
        winners[mem_winner] += 1
        report_lines.append(f"\n7. MEMORY USAGE (lower is better): {mem_winner}")
        report_lines.append(f"   - Analytical: {a_mem:.2f} MB")
        report_lines.append(f"   - Numerical:  {n_mem:.2f} MB")

    # 8. Throughput (higher is better)
    a_throughput = a_cp.get("throughput_mean") if a_cp else None
    n_throughput = n_cp.get("throughput_mean") if n_cp else None
    if a_throughput and n_throughput:
        throughput_winner = "Analytical" if a_throughput > n_throughput else "Numerical"
        winners[throughput_winner] += 1
        report_lines.append(f"\n8. THROUGHPUT (higher is better): {throughput_winner}")
        report_lines.append(f"   - Analytical: {a_throughput:.1f} solves/sec")
        report_lines.append(f"   - Numerical:  {n_throughput:.1f} solves/sec")
        if n_throughput > 0:
            ratio = a_throughput / n_throughput
            report_lines.append(f"   - Ratio: Analytical is {ratio:.1f}x faster")

    # 9. Convergence (numerical only)
    if "converged" in numerical_data and numerical_data["converged"] is not None:
        n_conv_rate = np.mean(numerical_data["converged"][n_valid]) * 100
        report_lines.append(f"\n9. NUMERICAL CONVERGENCE:")
        report_lines.append(f"   - Convergence rate: {n_conv_rate:.2f}%")
        if "iterations" in numerical_data:
            n_mean_iters = np.mean(numerical_data["iterations"][n_valid])
            report_lines.append(f"   - Mean iterations: {n_mean_iters:.1f}")

    # Overall Summary
    report_lines.append("\n" + "-" * 40)
    report_lines.append("OVERALL SUMMARY")
    report_lines.append("-" * 40)
    report_lines.append(f"\nCategories won:")
    report_lines.append(f"  - Analytical: {winners['Analytical']}")
    report_lines.append(f"  - Numerical:  {winners['Numerical']}")
    report_lines.append(f"  - Tie:        {winners['Tie']}")

    overall_winner = "Analytical" if winners["Analytical"] > winners["Numerical"] else "Numerical"
    if winners["Analytical"] == winners["Numerical"]:
        overall_winner = "Tie"

    report_lines.append(f"\nOverall Winner: {overall_winner}")

    # Summary recommendation
    report_lines.append("\nRECOMMENDATION:")
    report_lines.append("  - Use ANALYTICAL solver for: High precision, real-time applications,")
    report_lines.append("    speed-critical tasks, and when machine-precision accuracy is needed.")
    report_lines.append("  - Use NUMERICAL solver for: Better joint limit compliance,")
    report_lines.append("    cases where analytical solution doesn't exist, and when")
    report_lines.append("    exploring alternative configurations is beneficial.")

    report_lines.append("\n" + "=" * 80)

    return "\n".join(report_lines)


def save_comparison_data(analytical_data, numerical_data, output_path):
    """Save comparison data for further analysis or plotting."""

    a_valid = ~np.isnan(analytical_data["results"][:, 0])
    n_valid = ~np.isnan(numerical_data["results"][:, 0])

    comparison_data = {
        "analytical": {
            "pos_error": analytical_data["results"][a_valid, 0],
            "rot_error": analytical_data["results"][a_valid, 1],
            "time": analytical_data["results"][a_valid, 2],
            "success": analytical_data.get("success", np.zeros(len(a_valid)))[a_valid],
            "manipulability": analytical_data.get("manipulability", np.zeros(len(a_valid)))[a_valid],
        },
        "numerical": {
            "pos_error": numerical_data["results"][n_valid, 0],
            "rot_error": numerical_data["results"][n_valid, 1],
            "time": numerical_data["results"][n_valid, 2],
            "success": numerical_data.get("success", np.zeros(len(n_valid)))[n_valid],
            "manipulability": numerical_data.get("manipulability", np.zeros(len(n_valid)))[n_valid],
            "iterations": numerical_data.get("iterations", np.zeros(len(n_valid)))[n_valid] if "iterations" in numerical_data else None,
            "converged": numerical_data.get("converged", np.zeros(len(n_valid)))[n_valid] if "converged" in numerical_data else None,
        },
        "timestamp": datetime.now().isoformat(),
    }

    np.save(output_path, comparison_data)
    print(f"Saved comparison data to: {output_path}")


def main():
    """Main entry point."""
    print("\n" + "=" * 80)
    print("LOADING BENCHMARK RESULTS")
    print("=" * 80 + "\n")

    try:
        analytical_data = load_latest_results("analytical")
        numerical_data = load_latest_results("numerical")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run the benchmark scripts first:")
        print("  python benchmark/benchmark_analytical.py")
        print("  python benchmark/benchmark_numerical.py")
        return

    # Generate report
    report = generate_comparison_report(analytical_data, numerical_data)
    print(report)

    # Save report to file
    report_path = os.path.join(DATA_DIR, "solver_comparison_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    # Save comparison data for plotting
    data_path = os.path.join(DATA_DIR, "solver_comparison_data.npy")
    save_comparison_data(analytical_data, numerical_data, data_path)


if __name__ == "__main__":
    main()
