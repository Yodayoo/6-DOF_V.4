import os
import glob
import numpy as np

def combine_benchmarks(base_pattern, output_path):
    """
    Combines multiple benchmark .npy files into one.
    
    Args:
        base_pattern (str): e.g. "data/analytical_results_*.npy"
        output_path (str): e.g. "data/analytical_results_combined.npy"
    """
    files = sorted(glob.glob(base_pattern), key=os.path.getmtime)
    if not files:
        print(f"⚠️ No files found matching pattern: {base_pattern}")
        return

    print("\n","="*70 +"\n" , f"📂 Found {len(files)} files to combine:")
    for f in files:
        print("   └─", os.path.basename(f))

    all_results, all_thetas = [], []

    for f in files:
        data = np.load(f, allow_pickle=True).item()
        results = data.get("results", [])
        thetas  = data.get("thetas", [])
        all_results.extend(results)
        all_thetas.extend(thetas)

    combined = {"results": np.array(all_results, dtype=object),
                "thetas": np.array(all_thetas, dtype=object)}

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.save(output_path, combined)

    print(f"\n✅ Combined {len(files)} files into: {output_path}")
    print(f"   Total points: {len(all_results)}")


if __name__ == "__main__":
    # Adjust to your file structure
    data_dir = "6-DOF_V.3/project_1/data"

    analytical_pattern = os.path.join(data_dir, "analytical_solutions/analytical_results_*.npy")
    numerical_pattern  = os.path.join(data_dir, "numerical_solutions/numerical_results_*.npy")

    analytical_out = os.path.join(data_dir, "compiled_solutions/analytical_results_combined.npy")
    numerical_out  = os.path.join(data_dir, "compiled_solutions/numerical_results_combined.npy")

    combine_benchmarks(analytical_pattern, analytical_out)
    combine_benchmarks(numerical_pattern, numerical_out)
