import numpy as np
import os , sys

path = "6-DOF_V.3/project_1/data/numerical_solutions/numerical_results.npy"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

targets = np.load("6-DOF_V.3/project_1/data/target_archive.npy" , allow_pickle=True)

from kinematics import numerical_ik_solve
from bench_func import measure_batch_performance

data = measure_batch_performance(numerical_ik_solve , targets , batch_size=100 , save_path=path)

print("Results shape:", data["results"].shape)  # (n_points, 5)
print("Thetas shape:", data["thetas"].shape)    # (n_points, 6)