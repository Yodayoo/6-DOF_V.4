import numpy as np
import os
import sys

# Use __file__ based paths for reliability
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# Add project root to path
sys.path.insert(0, PROJECT_DIR)

from kinematics import numerical_ik_solve
from benchmark.bench_func import measure_batch_performance

# Load targets
targets_path = os.path.join(DATA_DIR, "target_archive.npy")
targets = np.load(targets_path, allow_pickle=True)

# Output path (timestamp will be appended by measure_batch_performance)
output_path = os.path.join(DATA_DIR, "numerical_solutions", "numerical_results.npy")

# Run benchmark
data = measure_batch_performance(
    numerical_ik_solve,
    targets,
    batch_size=1_000,
    save_path=output_path
)

# Use returned data directly (no need to reload from file)
print("Results shape:", data["results"].shape)  # (n_points, 5)
print("Thetas shape:", data["thetas"].shape)    # (n_points, 6)
if data["save_path"]:
    print(f"Saved to: {data['save_path']}")
