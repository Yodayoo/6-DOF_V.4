"""
Web server for visualizing 6-DOF robot target poses and solver results.
Run with: python app.py
Then open http://localhost:5000 in your browser.
"""

import numpy as np
import os
import sys
from flask import Flask, render_template, jsonify, request

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from kinematics import fk_chain, DH, analytical_ik_solve, numerical_ik_solve
from simulation import TrajectorySimulator

app = Flask(__name__)

# Global storage
_data = None
_simulator = TrajectorySimulator(interpolation_steps=15)
_results_path = os.path.join(PROJECT_DIR, "data", "solver_results_100k.npy")
_target_path = os.path.join(PROJECT_DIR, "data", "target_archive.npy")


def load_data(max_display=5000):
    """Load solver results and targets."""
    global _data

    # Try to load full results first
    if os.path.exists(_results_path):
        print(f"Loading solver results from {_results_path}...")
        raw = np.load(_results_path, allow_pickle=True).item()

        targets = raw['targets']
        total = len(targets)

        # Subsample for display
        if total > max_display:
            indices = np.linspace(0, total-1, max_display, dtype=int)
        else:
            indices = np.arange(total)

        _data = {
            'targets': targets[indices],
            'analytical_solutions': [raw['analytical_solutions'][i] for i in indices],
            'analytical_errors': [raw['analytical_errors'][i] for i in indices],
            'numerical_solutions': [raw['numerical_solutions'][i] for i in indices],
            'numerical_errors': [raw['numerical_errors'][i] for i in indices],
            'total_count': total,
            'display_count': len(indices),
            'analytical_time': raw.get('analytical_time', 0),
            'numerical_time': raw.get('numerical_time', 0),
            'all_analytical_errors': raw['analytical_errors'],
            'all_numerical_errors': raw['numerical_errors']
        }
        return len(indices)

    # Fallback to just targets
    elif os.path.exists(_target_path):
        print(f"Loading targets from {_target_path}...")
        targets = np.load(_target_path, allow_pickle=True)
        total = len(targets)

        if total > max_display:
            indices = np.linspace(0, total-1, max_display, dtype=int)
            targets = targets[indices]

        _data = {
            'targets': targets,
            'analytical_solutions': [None] * len(targets),
            'analytical_errors': [None] * len(targets),
            'numerical_solutions': [None] * len(targets),
            'numerical_errors': [None] * len(targets),
            'total_count': total,
            'display_count': len(targets)
        }
        return len(targets)

    return 0


def get_robot_frames(thetas):
    """Get all joint frames for visualization."""
    frames = [np.eye(4)]
    T = np.eye(4)

    from kinematics.helper_func import dh_T

    for i in range(6):
        a, alpha, d, theta0 = DH[i]
        T = T @ dh_T(a, alpha, d, theta0 + thetas[i])
        frames.append(T.copy())

    return frames


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/targets')
def get_targets():
    """Get all target positions with error data for visualization."""
    if _data is None:
        load_data()

    if _data is None or len(_data['targets']) == 0:
        return jsonify({'error': 'No data loaded'})

    positions = []
    analytical_errors = []
    numerical_errors = []

    for i, T in enumerate(_data['targets']):
        pos = T[:3, 3].tolist()
        positions.append(pos)

        a_err = _data['analytical_errors'][i]
        n_err = _data['numerical_errors'][i]

        analytical_errors.append(float(a_err) if a_err is not None else None)
        numerical_errors.append(float(n_err) if n_err is not None else None)

    return jsonify({
        'total_count': _data.get('total_count', len(positions)),
        'display_count': len(positions),
        'positions': positions,
        'analytical_errors': analytical_errors,
        'numerical_errors': numerical_errors
    })


@app.route('/api/stats')
def get_stats():
    """Get detailed statistics about solver performance."""
    if _data is None:
        load_data()

    if _data is None:
        return jsonify({'error': 'No data loaded'})

    # Use full error arrays for stats
    all_a_err = _data.get('all_analytical_errors', _data['analytical_errors'])
    all_n_err = _data.get('all_numerical_errors', _data['numerical_errors'])

    a_errors = np.array([e for e in all_a_err if e is not None and np.isfinite(e)])
    n_errors = np.array([e for e in all_n_err if e is not None and np.isfinite(e)])

    positions = np.array([T[:3, 3] for T in _data['targets']])

    stats = {
        'total_count': _data.get('total_count', len(_data['targets'])),
        'display_count': _data.get('display_count', len(_data['targets'])),
        'bounds': {
            'x': [float(positions[:, 0].min()), float(positions[:, 0].max())],
            'y': [float(positions[:, 1].min()), float(positions[:, 1].max())],
            'z': [float(positions[:, 2].min()), float(positions[:, 2].max())]
        }
    }

    if len(a_errors) > 0:
        stats['analytical'] = {
            'valid': int(len(a_errors)),
            'mean_error': float(np.mean(a_errors)),
            'max_error': float(np.max(a_errors)),
            'under_1um': int(np.sum(a_errors < 1e-6)),
            'under_1mm': int(np.sum(a_errors < 1e-3)),
            'time': _data.get('analytical_time', 0)
        }

    if len(n_errors) > 0:
        stats['numerical'] = {
            'valid': int(len(n_errors)),
            'mean_error': float(np.mean(n_errors)),
            'max_error': float(np.max(n_errors)),
            'under_1um': int(np.sum(n_errors < 1e-6)),
            'under_1mm': int(np.sum(n_errors < 1e-3)),
            'time': _data.get('numerical_time', 0)
        }

    return jsonify(stats)


@app.route('/api/robot/<int:target_idx>')
def get_robot_pose(target_idx):
    """Get robot arm configuration for a specific target."""
    if _data is None:
        load_data()

    if _data is None or target_idx >= len(_data['targets']):
        return jsonify({'error': 'Invalid target index'})

    T_target = _data['targets'][target_idx]

    result = {
        'target_idx': target_idx,
        'target_position': T_target[:3, 3].tolist()
    }

    # Get pre-computed analytical solution or solve
    a_sol = _data['analytical_solutions'][target_idx]
    if a_sol is None:
        a_sol = analytical_ik_solve(T_target)

    if a_sol is not None:
        frames = get_robot_frames(a_sol)
        result['analytical'] = {
            'angles': a_sol.tolist(),
            'angles_deg': (np.rad2deg(a_sol)).tolist(),
            'joint_positions': [f[:3, 3].tolist() for f in frames],
            'error': float(_data['analytical_errors'][target_idx]) if _data['analytical_errors'][target_idx] else None
        }

    # Get pre-computed numerical solution or solve
    n_sol = _data['numerical_solutions'][target_idx]
    if n_sol is None:
        n_sol = numerical_ik_solve(T_target)

    if n_sol is not None:
        frames = get_robot_frames(n_sol)
        result['numerical'] = {
            'angles': n_sol.tolist(),
            'angles_deg': (np.rad2deg(n_sol)).tolist(),
            'joint_positions': [f[:3, 3].tolist() for f in frames],
            'error': float(_data['numerical_errors'][target_idx]) if _data['numerical_errors'][target_idx] else None
        }

    return jsonify(result)


@app.route('/api/error_histogram')
def get_error_histogram():
    """Get histogram data for error distribution."""
    if _data is None:
        load_data()

    all_a_err = _data.get('all_analytical_errors', _data['analytical_errors'])
    all_n_err = _data.get('all_numerical_errors', _data['numerical_errors'])

    a_errors = np.array([e for e in all_a_err if e is not None and np.isfinite(e) and e > 0])
    n_errors = np.array([e for e in all_n_err if e is not None and np.isfinite(e) and e > 0])

    # Log-scale bins
    bins = np.logspace(-16, -1, 50)

    a_hist, _ = np.histogram(a_errors, bins=bins)
    n_hist, _ = np.histogram(n_errors, bins=bins)

    return jsonify({
        'bins': bins.tolist(),
        'analytical': a_hist.tolist(),
        'numerical': n_hist.tolist()
    })


# ============================================
# Trajectory Simulation API
# ============================================

@app.route('/simulation')
def simulation_page():
    """Serve the simulation page."""
    return render_template('simulation.html')


@app.route('/api/sim/reset', methods=['POST'])
def sim_reset():
    """Reset the simulation to initial state."""
    _simulator.reset()
    return jsonify({'status': 'ok', 'message': 'Simulation reset'})


@app.route('/api/sim/state')
def sim_state():
    """Get current simulation state."""
    return jsonify(_simulator.get_current_state())


@app.route('/api/sim/add_target', methods=['POST'])
def sim_add_target():
    """Generate and add a new target point."""
    result = _simulator.add_new_target()
    return jsonify(result)


@app.route('/api/sim/trajectory/<int:target_idx>')
def sim_trajectory(target_idx):
    """Get interpolated trajectory for moving to a target."""
    result = _simulator.get_movement_trajectory(target_idx)
    return jsonify(result)


@app.route('/api/sim/clear_old', methods=['POST'])
def sim_clear_old():
    """Clear old targets, keeping only recent ones."""
    keep = request.json.get('keep', 3) if request.json else 3
    _simulator.clear_old_targets(keep_last=keep)
    return jsonify({'status': 'ok', 'remaining': len(_simulator.targets)})


if __name__ == '__main__':
    print("Loading data...")
    count = load_data(max_display=5000)
    print(f"Loaded {count} poses for display")

    if _data:
        print(f"Total targets: {_data.get('total_count', count)}")

    print("\nStarting web server...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)
