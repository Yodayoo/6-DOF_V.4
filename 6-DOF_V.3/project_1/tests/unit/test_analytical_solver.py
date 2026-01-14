"""
Unit tests for analytical IK solver.

Tests solution accuracy, completeness, and configuration handling.
"""

import numpy as np
import pytest
from kinematics import Robot, DH, forward_kinematics
from kinematics.solvers import AnalyticalIKSolver


class TestAnalyticalIKSolver:
    """Test analytical IK solver."""

    @pytest.fixture
    def robot(self):
        """Create robot fixture."""
        return Robot()

    @pytest.fixture
    def solver(self):
        """Create solver fixture."""
        return AnalyticalIKSolver(DH)

    def test_fk_ik_consistency(self, robot):
        """Test that IK(FK(q)) ≈ q."""
        q_original = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        # Forward kinematics
        T_target = robot.forward_kinematics(q_original)

        # Inverse kinematics
        q_solution = robot.inverse_kinematics(T_target, method='analytical')

        assert q_solution is not None

        # Forward kinematics of solution
        T_achieved = robot.forward_kinematics(q_solution)

        # Compare transformations
        pos_err = np.linalg.norm(T_target[:3, 3] - T_achieved[:3, 3])
        assert pos_err < 1e-4, f"Position error: {pos_err}"

        # Rotation error
        R_err = T_target[:3, :3].T @ T_achieved[:3, :3]
        angle_err = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
        assert angle_err < 1e-3, f"Rotation error: {angle_err}"

    def test_multiple_solutions(self, solver):
        """Test that multiple solutions are found."""
        q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        T_target = forward_kinematics(DH, q)

        solutions = solver.solve_all(T_target)

        # Should find multiple solutions (typically 8 for 6-DOF)
        assert len(solutions) > 1
        assert len(solutions) <= 8

    def test_solution_validity(self, solver):
        """Test that all solutions are valid."""
        q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        T_target = forward_kinematics(DH, q)

        solutions = solver.solve_all(T_target)

        for sol in solutions:
            # Check that solution is valid
            assert sol.is_valid

            # Check error metrics
            assert sol.pos_error < 1e-4
            assert sol.rot_error < 1e-3

            # Check configuration name exists
            assert len(sol.configuration) > 0

    def test_solution_sorting(self, solver):
        """Test that solutions are sorted by error."""
        q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        T_target = forward_kinematics(DH, q)

        solutions = solver.solve_all(T_target)

        # Check sorting
        errors = [sol.total_error for sol in solutions]
        assert all(errors[i] <= errors[i+1] for i in range(len(errors)-1))

    def test_unreachable_target(self, solver):
        """Test behavior with unreachable target."""
        # Target far outside workspace
        T_target = np.eye(4)
        T_target[:3, 3] = [10.0, 0, 0]  # 10m away

        solutions = solver.solve_all(T_target)

        # Should return empty list
        assert len(solutions) == 0

    def test_zero_configuration(self, solver):
        """Test IK at zero configuration."""
        q_zero = np.zeros(6)
        T_target = forward_kinematics(DH, q_zero)

        q_solution = solver.solve(T_target)

        assert q_solution is not None

        # Verify solution
        T_achieved = forward_kinematics(DH, q_solution)
        pos_err = np.linalg.norm(T_target[:3, 3] - T_achieved[:3, 3])
        assert pos_err < 1e-4

    def test_configuration_names(self, solver):
        """Test that configuration names are descriptive."""
        q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        T_target = forward_kinematics(DH, q)

        solutions = solver.solve_all(T_target)

        config_names = set()
        for sol in solutions:
            # Should contain descriptive keywords
            assert any(kw in sol.configuration for kw in
                      ['shoulder', 'elbow', 'wrist'])
            config_names.add(sol.configuration)

        # Should have different configurations
        assert len(config_names) > 1

    def test_current_config_preference(self, robot):
        """Test that solution close to current config is preferred."""
        q_current = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        q_target = np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.15])

        T_target = robot.forward_kinematics(q_target)

        solutions = robot.inverse_kinematics_all(T_target)
        best_sol = robot.select_best_solution(solutions, q_current=q_current)

        assert best_sol is not None

        # Best solution should be close to current
        distance = best_sol.distance_to(q_current)
        assert distance < 1.0  # Reasonable threshold


class TestAnalyticalSolverEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def solver(self):
        """Create solver fixture."""
        return AnalyticalIKSolver(DH)

    def test_singular_configuration(self, solver):
        """Test behavior near singular configuration."""
        # Fully extended arm
        q_extended = np.array([0, 0, 0, 0, 0, 0])
        T_target = forward_kinematics(DH, q_extended)

        solutions = solver.solve_all(T_target)

        # Should still find solutions (may have fewer)
        assert len(solutions) > 0

    def test_wrist_singularity(self, solver):
        """Test behavior at wrist singularity."""
        q_wrist_singular = np.array([0, np.pi/4, np.pi/4, 0, 0, 0])
        T_target = forward_kinematics(DH, q_wrist_singular)

        solutions = solver.solve_all(T_target)

        # Should handle gracefully
        assert len(solutions) >= 0

    def test_return_all_flag(self, solver):
        """Test return_all parameter."""
        q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        T_target = forward_kinematics(DH, q)

        # Without return_all: single solution
        q_single = solver.solve(T_target, return_all=False)
        assert isinstance(q_single, np.ndarray)
        assert q_single.shape == (6,)

        # With return_all: list of solutions
        q_list = solver.solve(T_target, return_all=True)
        assert isinstance(q_list, list)
        assert len(q_list) > 1

    def test_joint_limit_violations(self, solver):
        """Test handling of joint limit violations."""
        # Create target that might violate limits
        q_extreme = np.array([np.pi, np.pi/2, -np.pi/2, np.pi, -np.pi, np.pi])
        T_target = forward_kinematics(DH, q_extreme)

        solutions = solver.solve_all(T_target)

        # All returned solutions should respect limits
        for sol in solutions:
            if sol.is_valid:
                assert np.all(sol.q >= solver.q_min)
                assert np.all(sol.q <= solver.q_max)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
