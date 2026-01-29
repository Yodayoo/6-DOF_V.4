"""
Trajectory simulation for comparing analytical and numerical IK solvers.

Simulates robot movement between randomly generated target points,
showing how each solver tracks the trajectory.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
import sys
import os

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from kinematics import fk_chain, DH, analytical_ik_solve, numerical_ik_solve
from kinematics.helper_func import dh_T


class TrajectorySimulator:
    """
    Simulates robot arm movement between target points using both solvers.
    """

    def __init__(self, interpolation_steps: int = 20):
        """
        Initialize the trajectory simulator.

        Args:
            interpolation_steps: Number of intermediate poses between targets
        """
        self.interpolation_steps = interpolation_steps
        self.dh_params = DH

        # Current state
        self.current_angles = np.zeros(6)  # Start at home position
        self.current_target_idx = 0
        self.targets: List[np.ndarray] = []
        self.history: List[Dict] = []

        # Solver states (track each solver's current angles independently)
        self.analytical_angles = np.zeros(6)
        self.numerical_angles = np.zeros(6)

        # Animation state
        self.current_step = 0
        self.is_moving = False

        # Workspace limits (based on robot reach)
        self.workspace_min = np.array([-0.8, -0.8, 0.1])
        self.workspace_max = np.array([0.8, 0.8, 1.2])

    def reset(self):
        """Reset simulator to initial state."""
        self.current_angles = np.zeros(6)
        self.analytical_angles = np.zeros(6)
        self.numerical_angles = np.zeros(6)
        self.current_target_idx = 0
        self.targets = []
        self.history = []
        self.current_step = 0
        self.is_moving = False

    def get_robot_frames(self, angles: np.ndarray) -> List[np.ndarray]:
        """Get all joint frames for given angles."""
        frames = [np.eye(4)]
        T = np.eye(4)

        for i in range(6):
            a, alpha, d, theta0 = self.dh_params[i]
            T = T @ dh_T(a, alpha, d, theta0 + angles[i])
            frames.append(T.copy())

        return frames

    def get_end_effector_pos(self, angles: np.ndarray) -> np.ndarray:
        """Get end-effector position for given angles."""
        T = fk_chain(self.dh_params, angles)
        return T[:3, 3]

    def generate_random_target(self) -> np.ndarray:
        """Generate a random reachable target pose."""
        # Generate random joint angles within limits
        q_random = np.random.uniform(-np.pi * 0.8, np.pi * 0.8, 6)

        # Get the resulting pose
        T_target = fk_chain(self.dh_params, q_random)

        return T_target

    def generate_target_near_current(self, max_distance: float = 0.3) -> np.ndarray:
        """Generate a target within max_distance of current end-effector."""
        # Get current end-effector position
        if len(self.targets) > 0:
            current_pos = self.targets[-1][:3, 3]
        else:
            T_current = fk_chain(self.dh_params, self.current_angles)
            current_pos = T_current[:3, 3]

        # Try to generate a reachable target nearby
        for _ in range(50):  # Max attempts
            # Random direction
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)

            # Random distance
            distance = np.random.uniform(0.1, max_distance)

            # New position
            new_pos = current_pos + direction * distance

            # Clamp to workspace
            new_pos = np.clip(new_pos, self.workspace_min, self.workspace_max)

            # Create target pose (keep orientation simple for now)
            T_target = np.eye(4)
            T_target[:3, 3] = new_pos

            # Random rotation around Z
            angle = np.random.uniform(-np.pi, np.pi)
            c, s = np.cos(angle), np.sin(angle)
            T_target[:3, :3] = np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, -1]  # Tool pointing down
            ])

            # Check if reachable by trying analytical solver
            sol = analytical_ik_solve(T_target)
            if sol is not None:
                return T_target

        # Fallback: use FK from random angles
        return self.generate_random_target()

    def add_new_target(self) -> Dict:
        """Generate and add a new target point."""
        if len(self.targets) == 0:
            # First target: generate from home position
            T_target = self.generate_random_target()
        else:
            # Subsequent targets: generate near current
            T_target = self.generate_target_near_current()

        self.targets.append(T_target)
        self.current_target_idx = len(self.targets) - 1

        # Solve with both solvers
        analytical_sol = analytical_ik_solve(T_target)
        numerical_sol = numerical_ik_solve(T_target, q0=self.numerical_angles)

        # Compute errors
        analytical_error = None
        numerical_error = None

        if analytical_sol is not None:
            T_achieved = fk_chain(self.dh_params, analytical_sol)
            analytical_error = float(np.linalg.norm(T_achieved[:3, 3] - T_target[:3, 3]))

        if numerical_sol is not None:
            T_achieved = fk_chain(self.dh_params, numerical_sol)
            numerical_error = float(np.linalg.norm(T_achieved[:3, 3] - T_target[:3, 3]))

        return {
            'target_idx': self.current_target_idx,
            'target_position': T_target[:3, 3].tolist(),
            'target_matrix': T_target.tolist(),
            'analytical_solution': analytical_sol.tolist() if analytical_sol is not None else None,
            'numerical_solution': numerical_sol.tolist() if numerical_sol is not None else None,
            'analytical_error': analytical_error,
            'numerical_error': numerical_error
        }

    def interpolate_angles(self, q_start: np.ndarray, q_end: np.ndarray,
                          num_steps: int) -> List[np.ndarray]:
        """Linear interpolation between two joint configurations."""
        trajectory = []
        for i in range(num_steps + 1):
            t = i / num_steps
            q = q_start + t * (q_end - q_start)
            trajectory.append(q)
        return trajectory

    def get_movement_trajectory(self, target_idx: int) -> Dict:
        """
        Get interpolated trajectory for both solvers moving to a target.

        Returns trajectory data for animation.
        """
        if target_idx >= len(self.targets):
            return {'error': 'Invalid target index'}

        T_target = self.targets[target_idx]

        # Solve with both solvers
        analytical_sol = analytical_ik_solve(T_target)
        numerical_sol = numerical_ik_solve(T_target, q0=self.numerical_angles)

        # Generate trajectories
        analytical_traj = []
        numerical_traj = []

        if analytical_sol is not None:
            traj = self.interpolate_angles(
                self.analytical_angles, analytical_sol, self.interpolation_steps
            )
            for q in traj:
                frames = self.get_robot_frames(q)
                positions = [f[:3, 3].tolist() for f in frames]
                analytical_traj.append({
                    'angles': q.tolist(),
                    'joint_positions': positions,
                    'end_effector': positions[-1]
                })
            # Update current angles
            self.analytical_angles = analytical_sol.copy()

        if numerical_sol is not None:
            traj = self.interpolate_angles(
                self.numerical_angles, numerical_sol, self.interpolation_steps
            )
            for q in traj:
                frames = self.get_robot_frames(q)
                positions = [f[:3, 3].tolist() for f in frames]
                numerical_traj.append({
                    'angles': q.tolist(),
                    'joint_positions': positions,
                    'end_effector': positions[-1]
                })
            # Update current angles
            self.numerical_angles = numerical_sol.copy()

        # Compute final errors
        analytical_error = None
        numerical_error = None

        if analytical_sol is not None:
            T_achieved = fk_chain(self.dh_params, analytical_sol)
            analytical_error = float(np.linalg.norm(T_achieved[:3, 3] - T_target[:3, 3]))

        if numerical_sol is not None:
            T_achieved = fk_chain(self.dh_params, numerical_sol)
            numerical_error = float(np.linalg.norm(T_achieved[:3, 3] - T_target[:3, 3]))

        return {
            'target_idx': target_idx,
            'target_position': T_target[:3, 3].tolist(),
            'analytical_trajectory': analytical_traj,
            'numerical_trajectory': numerical_traj,
            'analytical_error': analytical_error,
            'numerical_error': numerical_error,
            'num_steps': self.interpolation_steps + 1
        }

    def get_current_state(self) -> Dict:
        """Get current simulator state."""
        # Get current robot frames
        analytical_frames = self.get_robot_frames(self.analytical_angles)
        numerical_frames = self.get_robot_frames(self.numerical_angles)

        return {
            'current_target_idx': self.current_target_idx,
            'num_targets': len(self.targets),
            'targets': [t[:3, 3].tolist() for t in self.targets],
            'analytical_angles': self.analytical_angles.tolist(),
            'numerical_angles': self.numerical_angles.tolist(),
            'analytical_positions': [f[:3, 3].tolist() for f in analytical_frames],
            'numerical_positions': [f[:3, 3].tolist() for f in numerical_frames]
        }

    def clear_old_targets(self, keep_last: int = 3):
        """Remove old targets, keeping only the last N."""
        if len(self.targets) > keep_last:
            remove_count = len(self.targets) - keep_last
            self.targets = self.targets[remove_count:]
            self.current_target_idx = max(0, self.current_target_idx - remove_count)

            # Update history
            if len(self.history) > keep_last:
                self.history = self.history[-keep_last:]
