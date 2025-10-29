import numpy as np
import matplotlib.pyplot as plt
import os , sys

from tabulate import tabulate
from scipy.stats import ttest_ind
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

path = "6-DOF_V.3/project_1/data/compiled_solutions/"

analytical = np.load(path+"analytical_results_combined.npy" , allow_pickle=True).item()
numerical = np.load(path+"numerical_results_combined.npy" , allow_pickle=True).item()

analytical_results = np.array(analytical["results"])
numerical_results = np.array(numerical["results"])



n_min = min(len(analytical_results), len(numerical_results))
err_analytical = np.vstack(analytical_results[:n_min])
err_numerical  = np.vstack(numerical_results[:n_min])

metrics = ['Position Error [m]', 'Rotation Error [rad]', 'Time [s]', 'CPU [%]', 'Memory [MB]']
colors = ['tab:blue', 'tab:orange']



def clean(arr):
    arr = np.ravel(arr)
    arr = np.array(arr, dtype=float)  # force numeric conversion
    return arr[np.isfinite(arr) & (arr > 0)]

def valid_ratio(arr):
    arr = np.array(arr, dtype=float)  # force numeric conversion
    total = arr.size
    valid = np.count_nonzero(np.isfinite(arr))
    return valid / total if total > 0 else 0


def stats(arr):
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    return np.mean(arr), np.std(arr), np.max(arr) - np.min(arr)


table_a, table_n, table_p = [], [], []

for i, name in enumerate(metrics):
    data_a = clean(err_analytical[:, i])
    data_n = clean(err_numerical[:, i])
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


print("\n" + "="*70)
print(" Analytical Solver Performance Summary")
print("="*70)
print(tabulate(table_a, headers=["Metric", "Mean", "Std Dev", "Spread", "Valid %"], tablefmt="fancy_grid"))

print("\n" + "="*70)
print(" Numerical Solver Performance Summary")
print("="*70)
print(tabulate(table_n, headers=["Metric", "Mean", "Std Dev", "Spread", "Valid %"], tablefmt="fancy_grid"))

print("\n" + "="*70)
print(" Hypothesis Test (Two-sample t-test , analytical - numerical)")
print("="*70)
print(tabulate(table_p, headers=["Metric", "t-statistic", "p-value", "Result"], tablefmt="fancy_grid"))



total_points = len(err_analytical)

for i, name in enumerate(metrics):
    plt.figure(figsize=(8, 5))
    raw_a = err_analytical[:, i]
    raw_n = err_numerical[:, i]

    data_a = clean(raw_a)
    data_n = clean(raw_n)

    if len(data_a) == 0 and len(data_n) == 0:
        plt.title(f"{name}\n(no valid data)")
        plt.axis('off')
        continue

    # Determine bins
    if len(data_a) > 0 and len(data_n) > 0:
        bins = np.logspace(np.log10(min(data_a.min(), data_n.min())),
                           np.log10(max(data_a.max(), data_n.max())), 100)
    elif len(data_a) > 0:
        bins = np.logspace(np.log10(data_a.min()), np.log10(data_a.max()), 100)
    else:
        bins = np.logspace(np.log10(data_n.min()), np.log10(data_n.max()), 100)

    # Plot histograms
    plt.hist(data_a, bins=bins, alpha=0.5, color=colors[0], label='Analytical', edgecolor='black')
    plt.hist(data_n, bins=bins, alpha=0.5, color=colors[1], label='Numerical', edgecolor='black')

    plt.title(f"{name}\nAnalytical vs Numerical Solver Performance (Linear Scale)\nTotal points measured: {total_points}")

    # Log scale for first 4 metrics
    if i < 4:
        plt.xscale('log')
        plt.title(f"{name}\nAnalytical vs Numerical Solver Performance (Log Scale)\nTotal points measured: {total_points}")

    # Mean lines
    if len(data_a) > 0:
        plt.axvline(np.mean(data_a), color='blue', linestyle='dashed', linewidth=1)
    if len(data_n) > 0:
        plt.axvline(np.mean(data_n), color='orange', linestyle='dashed', linewidth=1)

    #plt.title(f"{name}\nAnalytical vs Numerical Solver Performance (Log Scale)\nTotal points measured: {total_points}")
    plt.xlabel(name)
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.5, which='both')
    plt.legend(fontsize=7, loc='upper left')
    plt.tight_layout()
    plt.show()
