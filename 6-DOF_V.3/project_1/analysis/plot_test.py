import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from tabulate import tabulate
from scipy.stats import ttest_ind

# Use __file__ based paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")

sys.path.insert(0, PROJECT_DIR)

# Load from compiled_solutions directory
analytical_path = os.path.join(DATA_DIR, "compiled_solutions", "analytical_results_combined.npy")
numerical_path = os.path.join(DATA_DIR, "compiled_solutions", "numerical_results_combined.npy")

analytical = np.load(analytical_path, allow_pickle=True).item()
numerical = np.load(numerical_path, allow_pickle=True).item()

analytical_results = np.array(analytical["results"])
numerical_results = np.array(numerical["results"])

n_min = min(len(analytical_results), len(numerical_results))
err_analytical = analytical_results[:n_min]
err_numerical = numerical_results[:n_min]

metrics = ['Position Error [m]', 'Rotation Error [rad]', 'Time [s]', 'Memory [MB]', 'CPU [%]']
colors = ['tab:blue', 'tab:orange']


def clean(arr, require_positive=True):
    """
    Clean array by removing non-finite values.

    Args:
        arr: Input array
        require_positive: If True, also filter out non-positive values

    Returns:
        Cleaned 1D array
    """
    arr = np.ravel(arr.astype(float))
    mask = np.isfinite(arr)
    if require_positive:
        mask = mask & (arr > 0)
    return arr[mask]


def valid_ratio(arr):
    """Calculate ratio of finite values in array."""
    total = arr.size
    valid = np.count_nonzero(np.isfinite(arr))
    return valid / total if total > 0 else 0


def stats(arr):
    """Calculate mean, std, and range of array."""
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    return np.mean(arr), np.std(arr), np.max(arr) - np.min(arr)


table_a, table_n, table_p = [], [], []

for i, name in enumerate(metrics):
    # Memory can be zero or negative (GC), so don't require positive
    require_pos = i < 3  # Only require positive for error and time metrics

    data_a = clean(err_analytical[:, i], require_positive=require_pos)
    data_n = clean(err_numerical[:, i], require_positive=require_pos)
    ratio_a = valid_ratio(err_analytical[:, i])
    ratio_n = valid_ratio(err_numerical[:, i])

    mean_a, std_a, spread_a = stats(data_a)
    mean_n, std_n, spread_n = stats(data_n)

    if len(data_a) > 1 and len(data_n) > 1:
        t_stat, p_val = ttest_ind(data_a, data_n, equal_var=False, nan_policy='omit')
    else:
        t_stat, p_val = np.nan, np.nan

    table_a.append([name, f"{mean_a:.3e}", f"{std_a:.3e}", f"{spread_a:.3e}", f"{ratio_a*100:.1f}%"])
    table_n.append([name, f"{mean_n:.3e}", f"{std_n:.3e}", f"{spread_n:.3e}", f"{ratio_n*100:.1f}%"])

    significance = "Significant" if (not np.isnan(p_val) and p_val < 0.05) else "Not sig."
    table_p.append([name, f"{t_stat:.3f}", f"{p_val:.3e}", significance])


print("\n" + "=" * 70)
print(" Analytical Solver Performance Summary")
print("=" * 70)
print(tabulate(table_a, headers=["Metric", "Mean", "Std Dev", "Spread", "Valid %"], tablefmt="fancy_grid"))

print("\n" + "=" * 70)
print(" Numerical Solver Performance Summary")
print("=" * 70)
print(tabulate(table_n, headers=["Metric", "Mean", "Std Dev", "Spread", "Valid %"], tablefmt="fancy_grid"))

print("\n" + "=" * 70)
print(" Hypothesis Test (Two-sample t-test, analytical vs numerical)")
print("=" * 70)
print(tabulate(table_p, headers=["Metric", "t-statistic", "p-value", "Result"], tablefmt="fancy_grid"))


# Visualization
fig, axs = plt.subplots(3, 2, figsize=(14, 10))
axs = axs.flatten()

total_points = len(err_analytical)
plt.suptitle(f'Analytical vs Numerical Solver Performance\nTotal points measured: {total_points}', fontsize=14)

for i, name in enumerate(metrics):
    ax = axs[i]
    raw_a = err_analytical[:, i]
    raw_n = err_numerical[:, i]

    # Memory can be zero or negative, so don't require positive
    require_pos = i < 3
    data_a = clean(raw_a, require_positive=require_pos)
    data_n = clean(raw_n, require_positive=require_pos)

    if len(data_a) == 0 and len(data_n) == 0:
        ax.set_title(f"{name}\n(no valid data)")
        ax.axis('off')
        continue

    # Determine bins - use log scale for error/time metrics, linear for CPU/memory
    use_log = i < 3 and data_a.min() > 0 and data_n.min() > 0

    if use_log:
        if len(data_a) > 0 and len(data_n) > 0:
            bins = np.logspace(np.log10(min(data_a.min(), data_n.min())),
                               np.log10(max(data_a.max(), data_n.max())), 100)
        elif len(data_a) > 0:
            bins = np.logspace(np.log10(data_a.min()), np.log10(data_a.max()), 100)
        else:
            bins = np.logspace(np.log10(data_n.min()), np.log10(data_n.max()), 100)
        ax.set_xscale('log')
    else:
        # Linear bins
        all_data = np.concatenate([data_a, data_n]) if len(data_a) > 0 and len(data_n) > 0 else (data_a if len(data_a) > 0 else data_n)
        bins = np.linspace(all_data.min(), all_data.max(), 100)

    # Plot histograms
    if len(data_a) > 0:
        ax.hist(data_a, bins=bins, alpha=0.5, color=colors[0], label='Analytical', edgecolor='black')
    if len(data_n) > 0:
        ax.hist(data_n, bins=bins, alpha=0.5, color=colors[1], label='Numerical', edgecolor='black')

    # Mean lines
    if len(data_a) > 0:
        ax.axvline(np.mean(data_a), color='blue', linestyle='dashed', linewidth=1)
    if len(data_n) > 0:
        ax.axvline(np.mean(data_n), color='orange', linestyle='dashed', linewidth=1)

    ax.set_title(name)
    ax.grid(True, linestyle='--', alpha=0.5, which='both')
    ax.legend(fontsize=8, loc='upper right')

# Hide unused subplot (6th spot)
for j in range(len(metrics), len(axs)):
    axs[j].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()
