import numpy as np
import os , sys

path = "data/analytical_results.npy"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

targets = np.load("data/target_archive.npy" , allow_pickle=True)

from kinematics import analytical_ik_solve
from bench_func import measure_batch_performance

data = measure_batch_performance(analytical_ik_solve , targets , batch_size=1_000 , save_path=path)

data = np.load(path, allow_pickle=True).item()
print("Results shape:", data["results"].shape)  # (n_points, 4)
print("Thetas shape:", data["thetas"].shape)    # (n_points, 6)