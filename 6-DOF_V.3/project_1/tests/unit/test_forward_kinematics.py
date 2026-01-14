"""
Unit tests for forward kinematics.

Tests FK computation and accuracy.
"""

import numpy as np
import pytest
from kinematics import DH, forward_kinematics, Robot


class TestForwardKinematics:
    """Test forward kinematics computation."""

    def test_zero_configuration(self):
        """Test FK at zero configuration."""
        q = np.zeros(6)
        T = forward_kinematics(DH, q)

        # Should produce valid transformation
        assert T.shape == (4, 4)
        assert abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-10

        # Position should be deterministic for zero config
        # (depends on DH parameters)
        assert T[3, 3] == 1.0  # Homogeneous coordinate

    def test_robot_class_consistency(self):
        """Test that Robot class FK matches direct FK."""
        robot = Robot()
        q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        T1 = forward_kinematics(DH, q)
        T2 = robot.forward_kinematics(q)

        np.testing.assert_allclose(T1, T2, atol=1e-10)

    def test_deterministic_output(self):
        """Test that same input produces same output."""
        q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        T1 = forward_kinematics(DH, q)
        T2 = forward_kinematics(DH, q)

        np.testing.assert_allclose(T1, T2, atol=1e-15)

    def test_rotation_matrix_properties(self):
        """Test that FK produces valid rotation matrices."""
        q = np.random.uniform(-np.pi, np.pi, 6)
        T = forward_kinematics(DH, q)

        R = T[:3, :3]

        # Orthogonality: R^T @ R = I
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10)

        # Determinant should be 1
        det = np.linalg.det(R)
        assert abs(det - 1.0) < 1e-10

    def test_different_configurations(self):
        """Test FK for various configurations."""
        test_configs = [
            np.zeros(6),
            np.ones(6) * 0.1,
            np.array([np.pi/4, 0, 0, 0, 0, 0]),
            np.array([0, np.pi/2, 0, 0, 0, 0]),
            np.random.uniform(-np.pi, np.pi, 6),
        ]

        for q in test_configs:
            T = forward_kinematics(DH, q)

            # All should produce valid transforms
            assert T.shape == (4, 4)
            assert abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-10
            assert T[3, 3] == 1.0

    def test_joint_angle_wrapping(self):
        """Test that angles differing by 2π produce same result."""
        q1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        q2 = q1 + 2*np.pi

        T1 = forward_kinematics(DH, q1)
        T2 = forward_kinematics(DH, q2)

        np.testing.assert_allclose(T1, T2, atol=1e-10)

    def test_single_joint_motion(self):
        """Test moving single joints."""
        q_base = np.zeros(6)

        for i in range(6):
            q = q_base.copy()
            q[i] = np.pi / 4

            T = forward_kinematics(DH, q)

            # Should produce different result than zero config
            T_zero = forward_kinematics(DH, q_base)
            assert np.linalg.norm(T - T_zero) > 1e-6

    def test_output_dtype(self):
        """Test that output has correct data type."""
        q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        T = forward_kinematics(DH, q)

        assert T.dtype == np.float64

    def test_position_reachability(self):
        """Test that FK positions are within manipulator reach."""
        robot = Robot()
        max_reach = robot.max_reach

        # Test random configurations
        for _ in range(10):
            q = np.random.uniform(-np.pi, np.pi, 6)
            T = forward_kinematics(DH, q)
            position = T[:3, 3]
            distance = np.linalg.norm(position)

            # Distance should be within max reach
            assert distance <= max_reach + 0.1  # Small tolerance


class TestForwardKinematicsEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_extreme_joint_angles(self):
        """Test with joint angles at limits."""
        q_max = np.array([np.pi] * 6)
        q_min = np.array([-np.pi] * 6)

        T_max = forward_kinematics(DH, q_max)
        T_min = forward_kinematics(DH, q_min)

        # Should produce valid transforms
        assert abs(np.linalg.det(T_max[:3, :3]) - 1.0) < 1e-10
        assert abs(np.linalg.det(T_min[:3, :3]) - 1.0) < 1e-10

    def test_list_input(self):
        """Test that list input works."""
        q_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        T = forward_kinematics(DH, q_list)

        assert T.shape == (4, 4)

    def test_numerical_stability(self):
        """Test numerical stability with small perturbations."""
        q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        T1 = forward_kinematics(DH, q)

        # Small perturbation
        q_perturbed = q + 1e-10
        T2 = forward_kinematics(DH, q_perturbed)

        # Results should be very close
        np.testing.assert_allclose(T1, T2, atol=1e-8)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
