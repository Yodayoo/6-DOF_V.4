"""
Integration tests for Robot class.

Tests end-to-end functionality of the Robot class API.
"""

import numpy as np
import pytest
from kinematics import Robot


class TestRobotClassIntegration:
    """Integration tests for Robot class."""

    @pytest.fixture
    def robot(self):
        """Create robot fixture."""
        return Robot()

    def test_full_ik_fk_cycle(self, robot):
        """Test complete IK->FK cycle."""
        # Generate random reachable target
        q_original = np.random.uniform(-np.pi/2, np.pi/2, 6)
        T_target = robot.forward_kinematics(q_original)

        # Solve IK
        q_solution = robot.inverse_kinematics(T_target, method='analytical')
        assert q_solution is not None

        # Verify with FK
        is_valid, pos_err, rot_err = robot.validate_solution(q_solution, T_target)

        assert is_valid
        assert pos_err < 1e-4
        assert rot_err < 1e-3

    def test_batch_solving(self, robot):
        """Test solving multiple targets."""
        n_targets = 10
        success_count = 0

        for _ in range(n_targets):
            q = np.random.uniform(-np.pi/2, np.pi/2, 6)
            T_target = robot.forward_kinematics(q)

            q_solution = robot.inverse_kinematics(T_target, method='analytical')

            if q_solution is not None:
                is_valid, pos_err, rot_err = robot.validate_solution(
                    q_solution, T_target
                )
                if is_valid:
                    success_count += 1

        # Should solve most targets successfully
        success_rate = success_count / n_targets
        assert success_rate > 0.95

    def test_all_solutions_workflow(self, robot):
        """Test workflow with multiple solutions."""
        q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        T_target = robot.forward_kinematics(q)

        # Get all solutions
        solutions = robot.inverse_kinematics_all(T_target)

        assert len(solutions) > 1

        # Validate all solutions
        accurate_solutions = robot.validate_solutions(solutions, T_target)

        # Most should be accurate
        accuracy_rate = len(accurate_solutions) / len(solutions)
        assert accuracy_rate > 0.9

        # Select best solution
        best = robot.select_best_solution(solutions, q_current=q)

        assert best is not None
        assert best.is_valid

    def test_configuration_preference(self, robot):
        """Test configuration preference in solution selection."""
        q = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        T_target = robot.forward_kinematics(q)

        solutions = robot.inverse_kinematics_all(T_target)

        # Try selecting with elbow_up preference
        elbow_up_sol = robot.select_best_solution(
            solutions, prefer_config="elbow_up"
        )

        if elbow_up_sol:
            assert "elbow_up" in elbow_up_sol.configuration

    def test_unreachable_handling(self, robot):
        """Test handling of unreachable targets."""
        # Target far outside workspace
        T_unreachable = np.eye(4)
        T_unreachable[:3, 3] = [10.0, 10.0, 10.0]

        solutions = robot.inverse_kinematics_all(T_unreachable)

        # Should return empty list
        assert len(solutions) == 0

        # Single solution should return None
        q_solution = robot.inverse_kinematics(T_unreachable, method='analytical')
        assert q_solution is None

    def test_solver_switching(self, robot):
        """Test switching between analytical and numerical solvers."""
        q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        T_target = robot.forward_kinematics(q)

        # Analytical solver
        q_analytical = robot.inverse_kinematics(T_target, method='analytical')

        # Numerical solver
        q_numerical = robot.inverse_kinematics(
            T_target, method='numerical', q_init=q
        )

        # Both should produce valid solutions
        assert q_analytical is not None
        assert q_numerical is not None

        # Both should be accurate
        _, pos_err_a, rot_err_a = robot.validate_solution(q_analytical, T_target)
        _, pos_err_n, rot_err_n = robot.validate_solution(q_numerical, T_target)

        assert pos_err_a < 1e-4
        assert pos_err_n < 1e-4

    def test_robot_properties(self, robot):
        """Test robot property accessors."""
        assert robot.n_joints == 6
        assert robot.max_reach > 0
        assert robot.config is not None


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""

    def test_trajectory_following(self):
        """Test following a trajectory with IK."""
        robot = Robot()

        # Define waypoints
        q_start = np.array([0, 0, 0, 0, 0, 0])
        q_end = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        # Interpolate trajectory in joint space
        n_points = 10
        trajectory = []

        for alpha in np.linspace(0, 1, n_points):
            q_interp = (1 - alpha) * q_start + alpha * q_end
            T = robot.forward_kinematics(q_interp)
            trajectory.append(T)

        # Solve IK for each waypoint
        q_current = q_start
        solutions = []

        for T_waypoint in trajectory:
            # Solve IK with preference for current config
            all_sols = robot.inverse_kinematics_all(T_waypoint)
            best_sol = robot.select_best_solution(all_sols, q_current=q_current)

            assert best_sol is not None
            solutions.append(best_sol.q)
            q_current = best_sol.q

        # All solutions should be valid
        assert len(solutions) == n_points

        # Solutions should form smooth trajectory (no large jumps)
        for i in range(1, len(solutions)):
            jump = np.linalg.norm(solutions[i] - solutions[i-1])
            assert jump < 1.0  # Reasonable threshold

    def test_workspace_exploration(self):
        """Test exploring workspace systematically."""
        robot = Robot()

        # Sample workspace grid
        x_range = np.linspace(0.3, 0.7, 3)
        y_range = np.linspace(-0.2, 0.2, 3)
        z_range = np.linspace(0.2, 0.6, 3)

        reachable_count = 0
        total_count = 0

        for x in x_range:
            for y in y_range:
                for z in z_range:
                    total_count += 1

                    T_target = np.eye(4)
                    T_target[:3, 3] = [x, y, z]

                    solutions = robot.inverse_kinematics_all(T_target)

                    if len(solutions) > 0:
                        reachable_count += 1

        # Should reach reasonable portion of tested points
        reachability_rate = reachable_count / total_count
        assert reachability_rate > 0.5

    def test_accuracy_benchmark(self):
        """Test accuracy across many random configurations."""
        robot = Robot()

        n_tests = 50
        pos_errors = []
        rot_errors = []

        for _ in range(n_tests):
            q_original = np.random.uniform(-np.pi/2, np.pi/2, 6)
            T_target = robot.forward_kinematics(q_original)

            q_solution = robot.inverse_kinematics(T_target, method='analytical')

            if q_solution is not None:
                _, pos_err, rot_err = robot.validate_solution(q_solution, T_target)
                pos_errors.append(pos_err)
                rot_errors.append(rot_err)

        # Check statistics
        mean_pos_err = np.mean(pos_errors)
        mean_rot_err = np.mean(rot_errors)
        max_pos_err = np.max(pos_errors)
        max_rot_err = np.max(rot_errors)

        # All errors should be small
        assert mean_pos_err < 1e-5
        assert mean_rot_err < 1e-4
        assert max_pos_err < 1e-4
        assert max_rot_err < 1e-3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
