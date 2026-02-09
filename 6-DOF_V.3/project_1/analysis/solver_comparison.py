"""
Solver Comparison Analysis

Compares analytical and numerical IK solvers on 5 core metrics:
- Position error
- Rotation error
- Computation time
- Memory usage
- CPU usage
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

    files = [f for f in os.listdir(folder) if f.endswith('.npy')]
    if not files:
        raise FileNotFoundError(f"No .npy files found in {folder}")

    files_with_time = [(f, os.path.getmtime(os.path.join(folder, f))) for f in files]
    files_with_time.sort(key=lambda x: x[1], reverse=True)
    latest_file = files_with_time[0][0]

    filepath = os.path.join(folder, latest_file)
    print(f"Loading {solver_type} results from: {latest_file}")

    data = np.load(filepath, allow_pickle=True).item()
    return data


def compute_statistics(arr):
    """Compute descriptive statistics for an array."""
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
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(g1) - np.mean(g2)) / pooled_std


def interpret_cohens_d(d):
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
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def format_pvalue(p):
    if p is None or np.isnan(p):
        return "-"
    if p == 0:
        return "< 1e-308"
    if p < 1e-300:
        log_p = np.log10(p) if p > 0 else -np.inf
        if np.isinf(log_p):
            return "< 1e-308"
        return f"~1e{int(log_p)}"
    if p < 1e-100:
        log_p = np.log10(p)
        mantissa = p / (10 ** int(np.floor(log_p)))
        return f"{mantissa:.1f}e{int(np.floor(log_p))}"
    return f"{p:.2e}"


def statistical_comparison(analytical_arr, numerical_arr, metric_name):
    """Perform statistical comparison between two groups."""
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

    t_stat, t_pvalue = stats.ttest_ind(a, n, equal_var=False)

    try:
        u_stat, u_pvalue = stats.mannwhitneyu(a, n, alternative='two-sided')
    except ValueError:
        u_stat, u_pvalue = np.nan, np.nan

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
    """Generate comparison report for 5 core metrics."""

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("IK SOLVER COMPARISON REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)

    a_results = analytical_data["results"]
    n_results = numerical_data["results"]

    a_valid = ~np.isnan(a_results[:, 0])
    n_valid = ~np.isnan(n_results[:, 0])

    # ============================================================
    # SECTION 1: SAMPLE OVERVIEW
    # ============================================================
    report_lines.append("\n" + "=" * 80)
    report_lines.append("1. SAMPLE OVERVIEW")
    report_lines.append("=" * 80)

    overview_table = [
        ["Total Targets", format_value(len(a_results), "int"), format_value(len(n_results), "int")],
        ["Valid Solutions", format_value(np.sum(a_valid), "int"), format_value(np.sum(n_valid), "int")],
        ["Failed Solutions", format_value(np.sum(~a_valid), "int"), format_value(np.sum(~n_valid), "int")],
    ]

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
    # SECTION 5: MEMORY & CPU RESOURCE USAGE
    # ============================================================
    a_mem = a_results[a_valid, 3]
    n_mem = n_results[n_valid, 3]
    a_cpu = a_results[a_valid, 4]
    n_cpu = n_results[n_valid, 4]

    report_lines.append("\n" + "=" * 80)
    report_lines.append("5. MEMORY & CPU RESOURCE USAGE")
    report_lines.append("=" * 80)

    a_mem_stats = compute_statistics(a_mem)
    n_mem_stats = compute_statistics(n_mem)
    a_cpu_stats = compute_statistics(a_cpu)
    n_cpu_stats = compute_statistics(n_cpu)

    resource_table = [
        ["Memory Delta Mean (MB)", format_value(a_mem_stats["mean"], "f", 2),
         format_value(n_mem_stats["mean"], "f", 2)],
        ["Memory Delta Std (MB)", format_value(a_mem_stats["std"], "f", 2),
         format_value(n_mem_stats["std"], "f", 2)],
        ["Memory Delta Max (MB)", format_value(a_mem_stats["max"], "f", 2),
         format_value(n_mem_stats["max"], "f", 2)],
        ["CPU Utilization Mean (%)", format_value(a_cpu_stats["mean"], "f", 1),
         format_value(n_cpu_stats["mean"], "f", 1)],
        ["CPU Utilization Std (%)", format_value(a_cpu_stats["std"], "f", 1),
         format_value(n_cpu_stats["std"], "f", 1)],
        ["CPU Utilization Max (%)", format_value(a_cpu_stats["max"], "f", 1),
         format_value(n_cpu_stats["max"], "f", 1)],
    ]

    report_lines.append("\n" + tabulate(resource_table,
                                         headers=["Metric", "Analytical", "Numerical"],
                                         tablefmt="grid"))

    mem_comparison = statistical_comparison(a_mem, n_mem, "Memory Usage")
    cpu_comparison = statistical_comparison(a_cpu, n_cpu, "CPU Usage")
    report_lines.append(f"\nMemory Statistical Comparison:")
    report_lines.append(f"  Welch's t-test: t = {mem_comparison['t_statistic']:.4f}, p = {format_pvalue(mem_comparison['t_pvalue'])} {mem_comparison['t_significance']}")
    report_lines.append(f"  Cohen's d: {mem_comparison['cohens_d']:.4f} ({mem_comparison['effect_interpretation']} effect)")
    report_lines.append(f"\nCPU Statistical Comparison:")
    report_lines.append(f"  Welch's t-test: t = {cpu_comparison['t_statistic']:.4f}, p = {format_pvalue(cpu_comparison['t_pvalue'])} {cpu_comparison['t_significance']}")
    report_lines.append(f"  Cohen's d: {cpu_comparison['cohens_d']:.4f} ({cpu_comparison['effect_interpretation']} effect)")

    # ============================================================
    # SECTION 6: STATISTICAL TESTS SUMMARY
    # ============================================================
    report_lines.append("\n" + "=" * 80)
    report_lines.append("6. STATISTICAL TESTS SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    report_lines.append("Effect size (Cohen's d): |d|<0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large")

    all_comparisons = [
        pos_comparison,
        rot_comparison,
        time_comparison,
        mem_comparison,
        cpu_comparison,
    ]

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
    # SECTION 7: CONCLUSIONS
    # ============================================================
    report_lines.append("\n" + "=" * 80)
    report_lines.append("7. CONCLUSIONS")
    report_lines.append("=" * 80)

    a_mean_pos = np.mean(a_results[a_valid, 0])
    n_mean_pos = np.mean(n_results[n_valid, 0])
    a_mean_rot = np.mean(a_results[a_valid, 1])
    n_mean_rot = np.mean(n_results[n_valid, 1])
    a_mean_time = np.mean(a_results[a_valid, 2])
    n_mean_time = np.mean(n_results[n_valid, 2])
    a_mean_mem = np.mean(a_mem)
    n_mean_mem = np.mean(n_mem)
    a_mean_cpu = np.mean(a_cpu)
    n_mean_cpu = np.mean(n_cpu)

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

    # 4. Memory Usage (lower is better)
    mem_winner = "Analytical" if a_mean_mem < n_mean_mem else "Numerical"
    if abs(a_mean_mem - n_mean_mem) < 0.01:
        mem_winner = "Tie"
    winners[mem_winner] += 1
    report_lines.append(f"\n4. MEMORY USAGE (lower is better): {mem_winner}")
    report_lines.append(f"   - Analytical: {a_mean_mem:.2f} MB")
    report_lines.append(f"   - Numerical:  {n_mean_mem:.2f} MB")

    # 5. CPU Usage (lower is better)
    cpu_winner = "Analytical" if a_mean_cpu < n_mean_cpu else "Numerical"
    if abs(a_mean_cpu - n_mean_cpu) < 0.5:
        cpu_winner = "Tie"
    winners[cpu_winner] += 1
    report_lines.append(f"\n5. CPU USAGE (lower is better): {cpu_winner}")
    report_lines.append(f"   - Analytical: {a_mean_cpu:.1f}%")
    report_lines.append(f"   - Numerical:  {n_mean_cpu:.1f}%")

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

    report_lines.append("\n" + "=" * 80)

    return "\n".join(report_lines)


def save_comparison_data(analytical_data, numerical_data, output_path):
    """Save comparison data for plotting."""

    a_valid = ~np.isnan(analytical_data["results"][:, 0])
    n_valid = ~np.isnan(numerical_data["results"][:, 0])

    comparison_data = {
        "analytical": {
            "pos_error": analytical_data["results"][a_valid, 0],
            "rot_error": analytical_data["results"][a_valid, 1],
            "time": analytical_data["results"][a_valid, 2],
            "memory": analytical_data["results"][a_valid, 3],
            "cpu": analytical_data["results"][a_valid, 4],
        },
        "numerical": {
            "pos_error": numerical_data["results"][n_valid, 0],
            "rot_error": numerical_data["results"][n_valid, 1],
            "time": numerical_data["results"][n_valid, 2],
            "memory": numerical_data["results"][n_valid, 3],
            "cpu": numerical_data["results"][n_valid, 4],
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

    report = generate_comparison_report(analytical_data, numerical_data)
    print(report)

    report_path = os.path.join(DATA_DIR, "solver_comparison_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    data_path = os.path.join(DATA_DIR, "solver_comparison_data.npy")
    save_comparison_data(analytical_data, numerical_data, data_path)


if __name__ == "__main__":
    main()
