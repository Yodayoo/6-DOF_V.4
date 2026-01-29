"""
Visualization Script for Solver Comparison

Generates publication-quality plots comparing analytical and numerical IK solvers.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUTPUT_DIR = os.path.join(DATA_DIR, "plots")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color scheme
ANALYTICAL_COLOR = "#58a6ff"  # Blue
NUMERICAL_COLOR = "#f0883e"   # Orange


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

    # Position Error Histogram
    ax1 = axes[0]
    a_pos = data["analytical"]["pos_error"]
    n_pos = data["numerical"]["pos_error"]

    # Use log scale bins for better visualization
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

    # Add vertical lines for means
    ax1.axvline(np.mean(a_pos), color=ANALYTICAL_COLOR, linestyle='--', linewidth=2,
                label=f'Analytical mean: {np.mean(a_pos):.2e}')
    ax1.axvline(np.mean(n_pos), color=NUMERICAL_COLOR, linestyle='--', linewidth=2,
                label=f'Numerical mean: {np.mean(n_pos):.2e}')

    # Rotation Error Histogram
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

    ax1.hist(a_time, bins=bins, alpha=0.7, label=f'Analytical',
             color=ANALYTICAL_COLOR, edgecolor='white', linewidth=0.5)
    ax1.hist(n_time, bins=bins, alpha=0.7, label=f'Numerical',
             color=NUMERICAL_COLOR, edgecolor='white', linewidth=0.5)

    ax1.set_xlabel('Computation Time (s)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Computation Time Distribution', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Box plot
    ax2 = axes[1]
    box_data = [a_time, n_time]
    bp = ax2.boxplot(box_data, labels=['Analytical', 'Numerical'], patch_artist=True)

    bp['boxes'][0].set_facecolor(ANALYTICAL_COLOR)
    bp['boxes'][1].set_facecolor(NUMERICAL_COLOR)

    for box in bp['boxes']:
        box.set_alpha(0.7)

    ax2.set_ylabel('Computation Time (s)', fontsize=12)
    ax2.set_title('Computation Time Box Plot', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add mean annotations
    ax2.annotate(f'Mean: {np.mean(a_time):.4f}s', xy=(1, np.mean(a_time)),
                 xytext=(1.3, np.mean(a_time)), fontsize=10, color=ANALYTICAL_COLOR)
    ax2.annotate(f'Mean: {np.mean(n_time):.4f}s', xy=(2, np.mean(n_time)),
                 xytext=(2.1, np.mean(n_time)), fontsize=10, color=NUMERICAL_COLOR)

    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "time_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")

    return fig


def plot_success_comparison(data, save=True):
    """Plot success rate comparison."""
    fig, ax = plt.subplots(figsize=(8, 6))

    a_success = np.mean(data["analytical"]["success"]) * 100
    n_success = np.mean(data["numerical"]["success"]) * 100

    x = np.arange(2)
    bars = ax.bar(x, [a_success, n_success], color=[ANALYTICAL_COLOR, NUMERICAL_COLOR],
                  alpha=0.8, edgecolor='white', linewidth=2)

    ax.set_xticks(x)
    ax.set_xticklabels(['Analytical', 'Numerical'], fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Solver Success Rate Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, [a_success, n_success]):
        ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points', ha='center', fontsize=14,
                    fontweight='bold')

    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "success_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")

    return fig


def plot_percentile_comparison(data, save=True):
    """Plot percentile comparison for errors."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    percentiles = [50, 75, 90, 95, 99]

    # Position Error Percentiles
    ax1 = axes[0]
    a_pos = data["analytical"]["pos_error"]
    n_pos = data["numerical"]["pos_error"]

    a_percentiles = [np.percentile(a_pos, p) for p in percentiles]
    n_percentiles = [np.percentile(n_pos, p) for p in percentiles]

    x = np.arange(len(percentiles))
    width = 0.35

    bars1 = ax1.bar(x - width/2, a_percentiles, width, label='Analytical',
                    color=ANALYTICAL_COLOR, alpha=0.8)
    bars2 = ax1.bar(x + width/2, n_percentiles, width, label='Numerical',
                    color=NUMERICAL_COLOR, alpha=0.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{p}th' for p in percentiles])
    ax1.set_xlabel('Percentile', fontsize=12)
    ax1.set_ylabel('Position Error (m)', fontsize=12)
    ax1.set_title('Position Error Percentiles', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, axis='y')

    # Rotation Error Percentiles
    ax2 = axes[1]
    a_rot = data["analytical"]["rot_error"]
    n_rot = data["numerical"]["rot_error"]

    a_percentiles = [np.percentile(a_rot, p) for p in percentiles]
    n_percentiles = [np.percentile(n_rot, p) for p in percentiles]

    bars1 = ax2.bar(x - width/2, a_percentiles, width, label='Analytical',
                    color=ANALYTICAL_COLOR, alpha=0.8)
    bars2 = ax2.bar(x + width/2, n_percentiles, width, label='Numerical',
                    color=NUMERICAL_COLOR, alpha=0.8)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{p}th' for p in percentiles])
    ax2.set_xlabel('Percentile', fontsize=12)
    ax2.set_ylabel('Rotation Error (rad)', fontsize=12)
    ax2.set_title('Rotation Error Percentiles', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "percentile_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")

    return fig


def plot_iterations_distribution(data, save=True):
    """Plot iteration count distribution for numerical solver."""
    if data["numerical"]["iterations"] is None:
        print("No iteration data available")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    iterations = data["numerical"]["iterations"]
    converged = data["numerical"]["converged"] if data["numerical"]["converged"] is not None else np.ones_like(iterations, dtype=bool)

    # Histogram
    ax1 = axes[0]
    ax1.hist(iterations, bins=50, alpha=0.8, color=NUMERICAL_COLOR,
             edgecolor='white', linewidth=0.5)
    ax1.axvline(np.mean(iterations), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(iterations):.1f}')
    ax1.axvline(np.median(iterations), color='green', linestyle='--', linewidth=2,
                label=f'Median: {np.median(iterations):.1f}')

    ax1.set_xlabel('Iterations', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Iteration Count Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Convergence pie chart
    ax2 = axes[1]
    conv_count = np.sum(converged)
    non_conv_count = len(converged) - conv_count

    sizes = [conv_count, non_conv_count]
    labels = [f'Converged\n({conv_count:,})', f'Not Converged\n({non_conv_count:,})']
    colors = ['#2ea043', '#da3633']
    explode = (0.02, 0.02)

    ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=False, startangle=90)
    ax2.set_title('Convergence Rate', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "iterations_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")

    return fig


def plot_summary_dashboard(data, save=True):
    """Create a summary dashboard with key metrics."""
    fig = plt.figure(figsize=(16, 12))

    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Success Rate Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    a_success = np.mean(data["analytical"]["success"]) * 100
    n_success = np.mean(data["numerical"]["success"]) * 100
    bars = ax1.bar(['Analytical', 'Numerical'], [a_success, n_success],
                   color=[ANALYTICAL_COLOR, NUMERICAL_COLOR], alpha=0.8)
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Success Rate', fontweight='bold')
    ax1.set_ylim(0, 105)
    for bar, val in zip(bars, [a_success, n_success]):
        ax1.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords='offset points', ha='center', fontweight='bold')

    # 2. Mean Computation Time
    ax2 = fig.add_subplot(gs[0, 1])
    a_time = np.mean(data["analytical"]["time"]) * 1000  # Convert to ms
    n_time = np.mean(data["numerical"]["time"]) * 1000
    bars = ax2.bar(['Analytical', 'Numerical'], [a_time, n_time],
                   color=[ANALYTICAL_COLOR, NUMERICAL_COLOR], alpha=0.8)
    ax2.set_ylabel('Mean Time (ms)')
    ax2.set_title('Computation Time', fontweight='bold')
    for bar, val in zip(bars, [a_time, n_time]):
        ax2.annotate(f'{val:.2f}ms', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords='offset points', ha='center', fontweight='bold')

    # 3. Mean Position Error
    ax3 = fig.add_subplot(gs[0, 2])
    a_pos = np.mean(data["analytical"]["pos_error"]) * 1000  # Convert to mm
    n_pos = np.mean(data["numerical"]["pos_error"]) * 1000
    bars = ax3.bar(['Analytical', 'Numerical'], [a_pos, n_pos],
                   color=[ANALYTICAL_COLOR, NUMERICAL_COLOR], alpha=0.8)
    ax3.set_ylabel('Mean Position Error (mm)')
    ax3.set_title('Position Accuracy', fontweight='bold')
    for bar, val in zip(bars, [a_pos, n_pos]):
        ax3.annotate(f'{val:.4f}mm', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords='offset points', ha='center', fontweight='bold', fontsize=9)

    # 4. Position Error Distribution
    ax4 = fig.add_subplot(gs[1, :2])
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

    # 5. Time Distribution (box plot)
    ax5 = fig.add_subplot(gs[1, 2])
    bp = ax5.boxplot([data["analytical"]["time"], data["numerical"]["time"]],
                     labels=['Analytical', 'Numerical'], patch_artist=True)
    bp['boxes'][0].set_facecolor(ANALYTICAL_COLOR)
    bp['boxes'][1].set_facecolor(NUMERICAL_COLOR)
    for box in bp['boxes']:
        box.set_alpha(0.7)
    ax5.set_ylabel('Computation Time (s)')
    ax5.set_title('Time Distribution', fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Percentile Comparison
    ax6 = fig.add_subplot(gs[2, :])
    percentiles = [50, 75, 90, 95, 99]
    x = np.arange(len(percentiles))
    width = 0.35

    a_pct = [np.percentile(data["analytical"]["pos_error"], p) for p in percentiles]
    n_pct = [np.percentile(data["numerical"]["pos_error"], p) for p in percentiles]

    ax6.bar(x - width/2, a_pct, width, label='Analytical', color=ANALYTICAL_COLOR, alpha=0.8)
    ax6.bar(x + width/2, n_pct, width, label='Numerical', color=NUMERICAL_COLOR, alpha=0.8)
    ax6.set_xticks(x)
    ax6.set_xticklabels([f'{p}th' for p in percentiles])
    ax6.set_xlabel('Percentile')
    ax6.set_ylabel('Position Error (m)')
    ax6.set_title('Position Error Percentiles', fontweight='bold')
    ax6.legend()
    ax6.set_yscale('log')
    ax6.grid(True, alpha=0.3, axis='y')

    # Main title
    fig.suptitle('IK Solver Comparison Dashboard', fontsize=16, fontweight='bold', y=0.98)

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

    # Generate all plots
    print("Generating plots...")
    plot_error_histograms(data)
    plot_time_comparison(data)
    plot_success_comparison(data)
    plot_percentile_comparison(data)
    plot_iterations_distribution(data)
    plot_summary_dashboard(data)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("\nTo view plots interactively, run with plt.show():")
    print("  python -c \"from analysis.plot_comparison import *; main(); import matplotlib.pyplot as plt; plt.show()\"")


if __name__ == "__main__":
    main()
