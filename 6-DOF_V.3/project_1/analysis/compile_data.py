import os
import glob
import numpy as np

# Use __file__ based paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")


def combine_benchmarks(base_pattern, output_path):
    """
    Combines multiple benchmark .npy files into one.

    Args:
        base_pattern (str): e.g. "data/analytical_results_*.npy"
        output_path (str): e.g. "data/analytical_results_combined.npy"

    Returns:
        dict: Combined results or None if no files found
    """
    files = sorted(glob.glob(base_pattern), key=os.path.getmtime)
    if not files:
        print(f"No files found matching pattern: {base_pattern}")
        return None

    print("\n" + "=" * 70)
    print(f"Found {len(files)} files to combine:")
    for f in files:
        print("   -", os.path.basename(f))

    all_results = []
    all_thetas = []

    for f in files:
        data = np.load(f, allow_pickle=True).item()
        results = data.get("results", np.array([]))
        thetas = data.get("thetas", np.array([]))

        # Handle both array and list formats
        if isinstance(results, np.ndarray) and results.ndim == 2:
            all_results.append(results)
        elif len(results) > 0:
            all_results.append(np.array(results))

        if isinstance(thetas, np.ndarray) and thetas.ndim == 2:
            all_thetas.append(thetas)
        elif len(thetas) > 0:
            all_thetas.append(np.array(thetas))

    # Use vstack to maintain numerical dtype (not dtype=object)
    if all_results:
        combined_results = np.vstack(all_results)
    else:
        combined_results = np.array([])

    if all_thetas:
        combined_thetas = np.vstack(all_thetas)
    else:
        combined_thetas = np.array([])

    combined = {
        "results": combined_results,
        "thetas": combined_thetas
    }

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.save(output_path, combined)

    print(f"\nCombined {len(files)} files into: {output_path}")
    print(f"   Total points: {len(combined_results)}")

    return combined


if __name__ == "__main__":
    analytical_pattern = os.path.join(DATA_DIR, "analytical_solutions", "analytical_results_*.npy")
    numerical_pattern = os.path.join(DATA_DIR, "numerical_solutions", "numerical_results_*.npy")

    analytical_out = os.path.join(DATA_DIR, "compiled_solutions", "analytical_results_combined.npy")
    numerical_out = os.path.join(DATA_DIR, "compiled_solutions", "numerical_results_combined.npy")

    combine_benchmarks(analytical_pattern, analytical_out)
    combine_benchmarks(numerical_pattern, numerical_out)
