"""
Visualization Script for Solver Comparison

Generates plots comparing analytical and numerical IK solvers on 5 metrics:
- Position error
- Rotation error
- Computation time
- Memory usage
- CPU usage
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUTPUT_DIR = os.path.join(DATA_DIR, "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color scheme
ANALYTICAL_COLOR = "#58a6ff"
NUMERICAL_COLOR = "#f0883e"


def load_comparison_data():
    """Load comparison data from file."""
    data_path = os.path.join(DATA_DIR, "solver_comparison_data.npy")

    if not os.path.exists(data_path):
        print(f"Comparison data not found at: {data_path}")
        print("Please run solver_comparison.py first.")
        return None

    return np.load(data_path, allow_pickle=True).item()


def plot_error_histograms(data, save=True):
    """Plot position and rotation error histograms."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Position Error
    ax1 = axes[0]
    a_pos = data["analytical"]["pos_error"]
    n_pos = data["numerical"]["pos_error"]

    bins = np.logspace(np.log10(max(1e-16, min(a_pos.min(), n_pos.min()))),
                       np.log10(max(a_pos.max(), n_pos.max())), 50)

    ax1.hist(a_pos, bins=bins, alpha=0.7, label=f'Analytical (n={len(a_pos):,})',
             color=ANALYTICAL_COLOR, edgecolor='white', linewidth=0.5)
    ax1.hist(n_pos, bins=bins, alpha=0.7, label=f'Numerical (n={len(n_pos):,})',
             color=NUMERICAL_COLOR, edgecolor='white', linewidth=0.5)

    ax1.set_xscale('log')
    ax1.set_xlabel('Position Error (m)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Position Error Distribution', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax1.axvline(np.mean(a_pos), color=ANALYTICAL_COLOR, linestyle='--', linewidth=2)
    ax1.axvline(np.mean(n_pos), color=NUMERICAL_COLOR, linestyle='--', linewidth=2)

    # Rotation Error
    ax2 = axes[1]
    a_rot = data["analytical"]["rot_error"]
    n_rot = data["numerical"]["rot_error"]

    bins = np.logspace(np.log10(max(1e-16, min(a_rot.min(), n_rot.min()))),
                       np.log10(max(a_rot.max(), n_rot.max())), 50)

    ax2.hist(a_rot, bins=bins, alpha=0.7, label=f'Analytical (n={len(a_rot):,})',
             color=ANALYTICAL_COLOR, edgecolor='white', linewidth=0.5)
    ax2.hist(n_rot, bins=bins, alpha=0.7, label=f'Numerical (n={len(n_rot):,})',
             color=NUMERICAL_COLOR, edgecolor='white', linewidth=0.5)

    ax2.set_xscale('log')
    ax2.set_xlabel('Rotation Error (rad)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Rotation Error Distribution', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    ax2.axvline(np.mean(a_rot), color=ANALYTICAL_COLOR, linestyle='--', linewidth=2)
    ax2.axvline(np.mean(n_rot), color=NUMERICAL_COLOR, linestyle='--', linewidth=2)

    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "error_histograms.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")

    return fig


def plot_time_comparison(data, save=True):
    """Plot computation time comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    a_time = data["analytical"]["time"]
    n_time = data["numerical"]["time"]

    # Histogram
    ax1 = axes[0]
    max_time = max(np.percentile(a_time, 99), np.percentile(n_time, 99))
    bins = np.linspace(0, max_time, 50)

    ax1.hist(a_time, bins=bins, alpha=0.7, label='Analytical',
             color=ANALYTICAL_COLOR, edgecolor='white', linewidth=0.5)
    ax1.hist(n_time, bins=bins, alpha=0.7, label='Numerical',
             color=NUMERICAL_COLOR, edgecolor='white', linewidth=0.5)

    ax1.set_xlabel('Computation Time (s)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Computation Time Distribution', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Box plot
    ax2 = axes[1]
    bp = ax2.boxplot([a_time, n_time], tick_labels=['Analytical', 'Numerical'], patch_artist=True)

    bp['boxes'][0].set_facecolor(ANALYTICAL_COLOR)
    bp['boxes'][1].set_facecolor(NUMERICAL_COLOR)
    for box in bp['boxes']:
        box.set_alpha(0.7)

    ax2.set_ylabel('Computation Time (s)', fontsize=12)
    ax2.set_title('Computation Time Box Plot', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    ax2.annotate(f'Mean: {np.mean(a_time)*1000:.2f}ms', xy=(1, np.mean(a_time)),
                 xytext=(1.3, np.mean(a_time)), fontsize=10, color=ANALYTICAL_COLOR)
    ax2.annotate(f'Mean: {np.mean(n_time)*1000:.2f}ms', xy=(2, np.mean(n_time)),
                 xytext=(2.1, np.mean(n_time)), fontsize=10, color=NUMERICAL_COLOR)

    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "time_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")

    return fig


def plot_resource_usage(data, save=True):
    """Plot memory and CPU usage comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Memory Usage
    ax1 = axes[0]
    a_mem = data["analytical"]["memory"]
    n_mem = data["numerical"]["memory"]

    # Filter out NaN
    a_mem = a_mem[~np.isnan(a_mem)]
    n_mem = n_mem[~np.isnan(n_mem)]

    bp1 = ax1.boxplot([a_mem, n_mem], tick_labels=['Analytical', 'Numerical'], patch_artist=True)
    bp1['boxes'][0].set_facecolor(ANALYTICAL_COLOR)
    bp1['boxes'][1].set_facecolor(NUMERICAL_COLOR)
    for box in bp1['boxes']:
        box.set_alpha(0.7)

    ax1.set_ylabel('Memory Delta (MB)', fontsize=12)
    ax1.set_title('Memory Usage per Batch', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    ax1.annotate(f'Mean: {np.mean(a_mem):.2f} MB', xy=(1, np.mean(a_mem)),
                 xytext=(1.3, np.mean(a_mem)), fontsize=10, color=ANALYTICAL_COLOR)
    ax1.annotate(f'Mean: {np.mean(n_mem):.2f} MB', xy=(2, np.mean(n_mem)),
                 xytext=(2.1, np.mean(n_mem)), fontsize=10, color=NUMERICAL_COLOR)

    # CPU Usage
    ax2 = axes[1]
    a_cpu = data["analytical"]["cpu"]
    n_cpu = data["numerical"]["cpu"]

    a_cpu = a_cpu[~np.isnan(a_cpu)]
    n_cpu = n_cpu[~np.isnan(n_cpu)]

    bp2 = ax2.boxplot([a_cpu, n_cpu], tick_labels=['Analytical', 'Numerical'], patch_artist=True)
    bp2['boxes'][0].set_facecolor(ANALYTICAL_COLOR)
    bp2['boxes'][1].set_facecolor(NUMERICAL_COLOR)
    for box in bp2['boxes']:
        box.set_alpha(0.7)

    ax2.set_ylabel('CPU Utilization (%)', fontsize=12)
    ax2.set_title('CPU Usage per Batch', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    ax2.annotate(f'Mean: {np.mean(a_cpu):.1f}%', xy=(1, np.mean(a_cpu)),
                 xytext=(1.3, np.mean(a_cpu)), fontsize=10, color=ANALYTICAL_COLOR)
    ax2.annotate(f'Mean: {np.mean(n_cpu):.1f}%', xy=(2, np.mean(n_cpu)),
                 xytext=(2.1, np.mean(n_cpu)), fontsize=10, color=NUMERICAL_COLOR)

    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "resource_usage.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")

    return fig


def plot_summary_dashboard(data, save=True):
    """Create a summary dashboard with all 5 metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. Mean Position Error
    ax1 = axes[0, 0]
    a_pos = np.mean(data["analytical"]["pos_error"])
    n_pos = np.mean(data["numerical"]["pos_error"])
    bars = ax1.bar(['Analytical', 'Numerical'], [a_pos, n_pos],
                   color=[ANALYTICAL_COLOR, NUMERICAL_COLOR], alpha=0.8)
    ax1.set_ylabel('Position Error (m)')
    ax1.set_title('Mean Position Error', fontweight='bold')
    ax1.set_yscale('log')
    for bar, val in zip(bars, [a_pos, n_pos]):
        ax1.annotate(f'{val:.2e}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 5), textcoords='offset points', ha='center', fontweight='bold', fontsize=9)

    # 2. Mean Rotation Error
    ax2 = axes[0, 1]
    a_rot = np.mean(data["analytical"]["rot_error"])
    n_rot = np.mean(data["numerical"]["rot_error"])
    bars = ax2.bar(['Analytical', 'Numerical'], [a_rot, n_rot],
                   color=[ANALYTICAL_COLOR, NUMERICAL_COLOR], alpha=0.8)
    ax2.set_ylabel('Rotation Error (rad)')
    ax2.set_title('Mean Rotation Error', fontweight='bold')
    ax2.set_yscale('log')
    for bar, val in zip(bars, [a_rot, n_rot]):
        ax2.annotate(f'{val:.2e}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 5), textcoords='offset points', ha='center', fontweight='bold', fontsize=9)

    # 3. Mean Computation Time
    ax3 = axes[0, 2]
    a_time = np.mean(data["analytical"]["time"]) * 1000
    n_time = np.mean(data["numerical"]["time"]) * 1000
    bars = ax3.bar(['Analytical', 'Numerical'], [a_time, n_time],
                   color=[ANALYTICAL_COLOR, NUMERICAL_COLOR], alpha=0.8)
    ax3.set_ylabel('Mean Time (ms)')
    ax3.set_title('Computation Time', fontweight='bold')
    for bar, val in zip(bars, [a_time, n_time]):
        ax3.annotate(f'{val:.2f}ms', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 5), textcoords='offset points', ha='center', fontweight='bold', fontsize=9)

    # 4. Position Error Distribution
    ax4 = axes[1, 0]
    a_pos_data = data["analytical"]["pos_error"]
    n_pos_data = data["numerical"]["pos_error"]
    bins = np.logspace(np.log10(max(1e-16, min(a_pos_data.min(), n_pos_data.min()))),
                       np.log10(max(a_pos_data.max(), n_pos_data.max())), 40)
    ax4.hist(a_pos_data, bins=bins, alpha=0.7, label='Analytical', color=ANALYTICAL_COLOR)
    ax4.hist(n_pos_data, bins=bins, alpha=0.7, label='Numerical', color=NUMERICAL_COLOR)
    ax4.set_xscale('log')
    ax4.set_xlabel('Position Error (m)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Position Error Distribution', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Memory Usage
    ax5 = axes[1, 1]
    a_mem = data["analytical"]["memory"]
    n_mem = data["numerical"]["memory"]
    a_mem = a_mem[~np.isnan(a_mem)]
    n_mem = n_mem[~np.isnan(n_mem)]
    a_mean_mem = np.mean(a_mem)
    n_mean_mem = np.mean(n_mem)
    bars = ax5.bar(['Analytical', 'Numerical'], [a_mean_mem, n_mean_mem],
                   color=[ANALYTICAL_COLOR, NUMERICAL_COLOR], alpha=0.8)
    ax5.set_ylabel('Memory Delta (MB)')
    ax5.set_title('Mean Memory Usage', fontweight='bold')
    for bar, val in zip(bars, [a_mean_mem, n_mean_mem]):
        ax5.annotate(f'{val:.2f} MB', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 5), textcoords='offset points', ha='center', fontweight='bold', fontsize=9)

    # 6. CPU Usage
    ax6 = axes[1, 2]
    a_cpu = data["analytical"]["cpu"]
    n_cpu = data["numerical"]["cpu"]
    a_cpu = a_cpu[~np.isnan(a_cpu)]
    n_cpu = n_cpu[~np.isnan(n_cpu)]
    a_mean_cpu = np.mean(a_cpu)
    n_mean_cpu = np.mean(n_cpu)
    bars = ax6.bar(['Analytical', 'Numerical'], [a_mean_cpu, n_mean_cpu],
                   color=[ANALYTICAL_COLOR, NUMERICAL_COLOR], alpha=0.8)
    ax6.set_ylabel('CPU Utilization (%)')
    ax6.set_title('Mean CPU Usage', fontweight='bold')
    for bar, val in zip(bars, [a_mean_cpu, n_mean_cpu]):
        ax6.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 5), textcoords='offset points', ha='center', fontweight='bold', fontsize=9)

    fig.suptitle('IK Solver Comparison Dashboard', fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "summary_dashboard.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")

    return fig


def main():
    """Generate all comparison plots."""
    print("\n" + "=" * 60)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 60 + "\n")

    data = load_comparison_data()
    if data is None:
        return

    print(f"Loaded data with {len(data['analytical']['pos_error']):,} analytical samples")
    print(f"and {len(data['numerical']['pos_error']):,} numerical samples\n")

    print("Generating plots...")
    plot_error_histograms(data)
    plot_time_comparison(data)
    plot_resource_usage(data)
    plot_summary_dashboard(data)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
