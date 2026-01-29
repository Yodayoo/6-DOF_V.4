import numpy as np
import os
import sys

# Use __file__ based paths for reliability
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# Add project root to path
sys.path.insert(0, PROJECT_DIR)

from kinematics import analytical_ik_solve
from benchmark.bench_func import measure_batch_performance

# Load targets
targets_path = os.path.join(DATA_DIR, "target_archive.npy")
targets = np.load(targets_path, allow_pickle=True)

# Output path (timestamp will be appended by measure_batch_performance)
output_path = os.path.join(DATA_DIR, "analytical_solutions", "analytical_results.npy")

# Run comprehensive benchmark
data = measure_batch_performance(
    analytical_ik_solve,
    targets,
    batch_size=1_000,
    save_path=output_path,
    solver_type="analytical"
)

# Print extended results info
print("\n--- Data Arrays ---")
print(f"Results shape:      {data['results'].shape}")
print(f"Thetas shape:       {data['thetas'].shape}")
print(f"Success array:      {data['success'].shape}")
print(f"Manipulability:     {data['manipulability'].shape}")
print(f"Workspace regions:  {len(data['workspace_regions'])} entries")

if data["save_path"]:
    print(f"\nSaved to: {data['save_path']}")
