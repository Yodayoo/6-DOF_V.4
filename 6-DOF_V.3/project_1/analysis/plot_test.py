import numpy as np
import matplotlib.pyplot as plt
import os , sys

from tabulate import tabulate
from scipy.stats import ttest_ind
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

analytical = np.load("6-DOF_V.3/project_1/data/compiled_solutions/analytical_results_combined.npy" , allow_pickle=True).item()
numerical = np.load("6-DOF_V.3/project_1/data/compiled_solutions/numerical_results_combined.npy" , allow_pickle=True).item()

analytical_results = np.array(analytical["results"])
numerical_results = np.array(numerical["results"])



n_min = min(len(analytical_results), len(numerical_results))
err_analytical = np.vstack(analytical_results[:n_min]).astype(float)
err_numerical  = np.vstack(numerical_results[:n_min]).astype(float)

metrics = ['Position Error [m]', 'Rotation Error [rad]', 'Time [s]', 'Memory [MB]', 'CPU [%]']
colors = ['tab:blue', 'tab:orange']



def clean(arr):
    arr = np.ravel(arr.astype(float))
    return arr[np.isfinite(arr) & (arr > 0)]

def valid_ratio(arr):
    total = arr.size
    valid = np.count_nonzero(np.isfinite(arr))
    return valid / total if total > 0 else 0

def stats(arr):
    if arr.size == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    q1, q3 = np.percentile(arr, [25, 75])
    geo_mean = np.exp(np.mean(np.log(arr))) if np.all(arr > 0) else np.nan
    median = np.median(arr)
    return np.mean(arr), geo_mean, median, np.std(arr), np.max(arr) - np.min(arr), q3 - q1

def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return np.nan
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else np.nan


table_a, table_n, table_p = [], [], []

for i, name in enumerate(metrics):
    data_a = clean(err_analytical[:, i])
    data_n = clean(err_numerical[:, i])
    ratio_a = valid_ratio(err_analytical[:, i])
    ratio_n = valid_ratio(err_numerical[:, i])

    mean_a, gmean_a, median_a, std_a, spread_a, iqr_a = stats(data_a)
    mean_n, gmean_n, median_n, std_n, spread_n, iqr_n = stats(data_n)

    if len(data_a) > 1 and len(data_n) > 1:
        t_stat, p_val = ttest_ind(data_a, data_n, equal_var=False, nan_policy='omit')
        d_val = cohens_d(data_a, data_n)
    else:
        t_stat, p_val, d_val = np.nan, np.nan, np.nan

    table_a.append([name, f"{mean_a:.3e}", f"{gmean_a:.3e}", f"{median_a:.3e}", f"{std_a:.3e}", f"{iqr_a:.3e}", f"{spread_a:.3e}", f"{ratio_a*100:.1f}%"])
    table_n.append([name, f"{mean_n:.3e}", f"{gmean_n:.3e}", f"{median_n:.3e}", f"{std_n:.3e}", f"{iqr_n:.3e}", f"{spread_n:.3e}", f"{ratio_n*100:.1f}%"])

    significance = "Significant" if (not np.isnan(p_val) and p_val < 0.05) else "Not sig."
    if np.isnan(d_val):
        effect = "n/a"
    else:
        ad = abs(d_val)
        effect = "negligible" if ad < 0.2 else "small" if ad < 0.5 else "medium" if ad < 0.8 else "large"
    table_p.append([name, f"{t_stat:.3f}", f"{p_val:.3e}", f"{d_val:.3f}", effect, significance])


summary_lines = [
    "="*70,
    " Analytical Solver Performance Summary",
    "="*70,
    tabulate(table_a, headers=["Metric", "Mean", "Geo Mean", "Median", "Std Dev", "IQR", "Spread", "Valid %"], tablefmt="fancy_grid"),
    "",
    "="*70,
    " Numerical Solver Performance Summary",
    "="*70,
    tabulate(table_n, headers=["Metric", "Mean", "Geo Mean", "Median", "Std Dev", "IQR", "Spread", "Valid %"], tablefmt="fancy_grid"),
    "",
    "="*70,
    " Hypothesis Test (Two-sample t-test , analytical - numerical)",
    "="*70,
    tabulate(table_p, headers=["Metric", "t-statistic", "p-value", "Cohen's d", "Effect size", "Result"], tablefmt="fancy_grid"),
]
summary_text = "\n".join(summary_lines)
print("\n" + summary_text)

summary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(f"Total points measured: {n_min}\n\n")
    f.write(summary_text + "\n")
print(f"\n📝 Saved metrics summary to: {summary_path}")



from matplotlib.lines import Line2D

total_points = len(err_analytical)
fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "histograms")
os.makedirs(fig_dir, exist_ok=True)

use_log = {'Position Error [m]', 'Rotation Error [rad]', 'Time [s]', 'Memory [MB]'}

for i, name in enumerate(metrics):
    data_a = clean(err_analytical[:, i])
    data_n = clean(err_numerical[:, i])

    if len(data_a) == 0 and len(data_n) == 0:
        print(f"Skipping {name}: no valid data")
        continue

    fig, ax = plt.subplots(figsize=(10, 5))

    lo = min([d.min() for d in (data_a, data_n) if len(d) > 0])
    hi = max([d.max() for d in (data_a, data_n) if len(d) > 0])
    bins = np.logspace(np.log10(lo), np.log10(hi), 100) if name in use_log else np.linspace(lo, hi, 100)

    ax.hist(data_a, bins=bins, alpha=0.5, color=colors[0], label='Analytical', edgecolor='black')
    ax.hist(data_n, bins=bins, alpha=0.5, color=colors[1], label='Numerical', edgecolor='black')

    if name in use_log:
        ax.set_xscale('log')

    # Central-tendency lines per solver
    for d, color in [(data_a, 'blue'), (data_n, 'darkorange')]:
        if len(d) == 0:
            continue
        ax.axvline(np.mean(d), color=color, linestyle='dashed', linewidth=1.2)
        if np.all(d > 0):
            ax.axvline(np.exp(np.mean(np.log(d))), color=color, linestyle='dotted', linewidth=1.2)
        ax.axvline(np.median(d), color=color, linestyle='solid', linewidth=1.0, alpha=0.7)

    # Combined legend outside plot area: solver colors + line-style meanings
    legend_entries = [
        Line2D([0], [0], color=colors[0], lw=6, alpha=0.5, label='Analytical'),
        Line2D([0], [0], color=colors[1], lw=6, alpha=0.5, label='Numerical'),
        Line2D([0], [0], color='gray', linestyle='dashed', label='Arithmetic mean'),
        Line2D([0], [0], color='gray', linestyle='dotted', label='Geometric mean'),
        Line2D([0], [0], color='gray', linestyle='solid',  label='Median'),
    ]
    ax.legend(handles=legend_entries, loc='center left', bbox_to_anchor=(1.02, 0.5),
              borderaxespad=0., frameon=True)

    ax.set_title(f"{name}\nTotal points: {total_points}")
    ax.set_xlabel(name)
    ax.set_ylabel("Count")
    ax.grid(True, linestyle='--', alpha=0.5, which='both')

    safe_name = name.split(" [")[0].lower().replace(" ", "_")
    out_path = os.path.join(fig_dir, f"{safe_name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"🖼️  Saved: {out_path}")
