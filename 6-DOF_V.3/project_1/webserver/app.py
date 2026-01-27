"""
Web server for visualizing 6-DOF robot target poses.
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

app = Flask(__name__)

# Global storage for loaded targets
_targets = None
_target_path = os.path.join(PROJECT_DIR, "data", "target_archive.npy")


def load_targets(max_points=1000):
    """Load target poses from file, limiting to max_points for performance."""
    global _targets
    if os.path.exists(_target_path):
        all_targets = np.load(_target_path, allow_pickle=True)
        # Subsample if too many
        if len(all_targets) > max_points:
            indices = np.linspace(0, len(all_targets)-1, max_points, dtype=int)
            _targets = all_targets[indices]
        else:
            _targets = all_targets
        return len(_targets)
    return 0


def get_robot_frames(thetas):
    """Get all joint frames for visualization."""
    frames = [np.eye(4)]  # Base frame
    T = np.eye(4)

    from kinematics.helper_func import dh_T

    for i in range(6):
        a, alpha, d, theta0 = DH[i]
        T = T @ dh_T(a, alpha, d, theta0 + thetas[i])
        frames.append(T.copy())

    return frames


@app.route('/')
def index():
    """Serve the main visualization page."""
    return render_template('index.html')


@app.route('/api/targets')
def get_targets():
    """Get all target positions for visualization."""
    if _targets is None:
        load_targets()

    if _targets is None or len(_targets) == 0:
        return jsonify({'error': 'No targets loaded', 'positions': []})

    # Extract positions and orientations
    positions = []
    orientations = []

    for T in _targets:
        pos = T[:3, 3].tolist()
        # Extract z-axis of end-effector for orientation arrow
        z_axis = T[:3, 2].tolist()
        positions.append(pos)
        orientations.append(z_axis)

    return jsonify({
        'count': len(positions),
        'positions': positions,
        'orientations': orientations
    })


@app.route('/api/robot/<int:target_idx>')
def get_robot_pose(target_idx):
    """Get robot arm configuration for a specific target."""
    if _targets is None:
        load_targets()

    if _targets is None or target_idx >= len(_targets):
        return jsonify({'error': 'Invalid target index'})

    T_target = _targets[target_idx]

    # Solve IK
    thetas = analytical_ik_solve(T_target)
    if thetas is None:
        return jsonify({'error': 'IK failed for this target'})

    # Get all joint frames
    frames = get_robot_frames(thetas)

    # Extract joint positions
    joint_positions = [f[:3, 3].tolist() for f in frames]

    # Target position
    target_pos = T_target[:3, 3].tolist()
    target_z = T_target[:3, 2].tolist()

    return jsonify({
        'target_idx': target_idx,
        'joint_angles': thetas.tolist(),
        'joint_positions': joint_positions,
        'target_position': target_pos,
        'target_orientation': target_z
    })


@app.route('/api/solve', methods=['POST'])
def solve_ik():
    """Solve IK for a custom target pose."""
    data = request.json

    # Build transformation matrix from position and orientation
    pos = np.array(data.get('position', [2, 0, 2]))

    # Simple target matrix (pointing down)
    T_target = np.eye(4)
    T_target[:3, 3] = pos

    # Solve with both methods
    analytical_sol = analytical_ik_solve(T_target)
    numerical_sol = numerical_ik_solve(T_target)

    result = {
        'target_position': pos.tolist()
    }

    if analytical_sol is not None:
        frames_a = get_robot_frames(analytical_sol)
        result['analytical'] = {
            'angles': analytical_sol.tolist(),
            'joint_positions': [f[:3, 3].tolist() for f in frames_a]
        }

    if numerical_sol is not None:
        frames_n = get_robot_frames(numerical_sol)
        result['numerical'] = {
            'angles': numerical_sol.tolist(),
            'joint_positions': [f[:3, 3].tolist() for f in frames_n]
        }

    return jsonify(result)


@app.route('/api/stats')
def get_stats():
    """Get statistics about loaded targets."""
    if _targets is None:
        load_targets()

    if _targets is None:
        return jsonify({'error': 'No targets loaded'})

    positions = np.array([T[:3, 3] for T in _targets])

    return jsonify({
        'count': len(_targets),
        'bounds': {
            'x': [float(positions[:, 0].min()), float(positions[:, 0].max())],
            'y': [float(positions[:, 1].min()), float(positions[:, 1].max())],
            'z': [float(positions[:, 2].min()), float(positions[:, 2].max())]
        },
        'centroid': positions.mean(axis=0).tolist()
    })


if __name__ == '__main__':
    print("Loading target poses...")
    count = load_targets(max_points=2000)
    print(f"Loaded {count} target poses")
    print("\nStarting web server...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)
