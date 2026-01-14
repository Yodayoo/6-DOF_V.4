"""
Unit tests for transformation matrix utilities.

Tests DH transformation computation and matrix properties.
"""

import numpy as np
import pytest
from kinematics.core import dh_transform


class TestDHTransform:
    """Test DH transformation matrix computation."""

    def test_identity_transform(self):
        """Test that zero parameters produce identity matrix."""
        T = dh_transform(0, 0, 0, 0)
        np.testing.assert_allclose(T, np.eye(4), atol=1e-10)

    def test_pure_translation_x(self):
        """Test translation along x-axis."""
        a = 0.5
        T = dh_transform(a, 0, 0, 0)
        expected_position = np.array([a, 0, 0])
        np.testing.assert_allclose(T[:3, 3], expected_position, atol=1e-10)
        np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-10)

    def test_pure_translation_z(self):
        """Test translation along z-axis."""
        d = 0.3
        T = dh_transform(0, 0, d, 0)
        expected_position = np.array([0, 0, d])
        np.testing.assert_allclose(T[:3, 3], expected_position, atol=1e-10)
        np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-10)

    def test_pure_rotation_z(self):
        """Test rotation about z-axis."""
        theta = np.pi / 4
        T = dh_transform(0, 0, 0, theta)

        # Rotation matrix around z-axis
        c, s = np.cos(theta), np.sin(theta)
        expected_R = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])

        np.testing.assert_allclose(T[:3, :3], expected_R, atol=1e-10)
        np.testing.assert_allclose(T[:3, 3], np.zeros(3), atol=1e-10)

    def test_pure_rotation_x(self):
        """Test rotation about x-axis (alpha parameter)."""
        alpha = np.pi / 3
        T = dh_transform(0, alpha, 0, 0)

        # Rotation matrix around x-axis
        c, s = np.cos(alpha), np.sin(alpha)
        expected_R = np.array([
            [1,  0, 0],
            [0,  c, -s],
            [0,  s, c]
        ])

        np.testing.assert_allclose(T[:3, :3], expected_R, atol=1e-10)

    def test_homogeneous_property(self):
        """Test that bottom row is always [0, 0, 0, 1]."""
        T = dh_transform(0.5, np.pi/6, 0.3, np.pi/4)
        expected_bottom = np.array([0, 0, 0, 1])
        np.testing.assert_allclose(T[3, :], expected_bottom, atol=1e-10)

    def test_rotation_matrix_orthogonality(self):
        """Test that rotation part is orthogonal (R^T @ R = I)."""
        T = dh_transform(0.5, np.pi/6, 0.3, np.pi/4)
        R = T[:3, :3]
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10)

    def test_rotation_matrix_determinant(self):
        """Test that rotation part has determinant = 1."""
        T = dh_transform(0.5, np.pi/6, 0.3, np.pi/4)
        R = T[:3, :3]
        det = np.linalg.det(R)
        assert abs(det - 1.0) < 1e-10

    def test_combined_transform(self):
        """Test combination of all parameters."""
        a, alpha, d, theta = 0.5, np.pi/6, 0.3, np.pi/4
        T = dh_transform(a, alpha, d, theta)

        # Result should be 4x4
        assert T.shape == (4, 4)

        # Should be valid transformation matrix
        assert abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-10

    def test_chaining_property(self):
        """Test that chaining transforms works correctly."""
        T1 = dh_transform(0.5, 0, 0, 0)
        T2 = dh_transform(0, 0, 0.3, 0)
        T_combined = T1 @ T2

        # Should produce translation [0.5, 0, 0.3]
        expected_position = np.array([0.5, 0, 0.3])
        np.testing.assert_allclose(T_combined[:3, 3], expected_position, atol=1e-10)


class TestDHParameterEdgeCases:
    """Test edge cases and special values."""

    def test_large_angles(self):
        """Test with angles > 2π."""
        theta = 3 * np.pi
        T1 = dh_transform(0, 0, 0, theta)
        T2 = dh_transform(0, 0, 0, theta - 2*np.pi)
        np.testing.assert_allclose(T1, T2, atol=1e-10)

    def test_negative_lengths(self):
        """Test with negative link lengths."""
        T = dh_transform(-0.5, 0, -0.3, 0)
        # Should work without errors
        assert T.shape == (4, 4)

    def test_very_small_angles(self):
        """Test numerical stability with very small angles."""
        theta = 1e-12
        T = dh_transform(0, 0, 0, theta)
        # Should be very close to identity
        np.testing.assert_allclose(T, np.eye(4), atol=1e-10)

    def test_pi_angles(self):
        """Test with π angles (common edge case)."""
        T = dh_transform(0, np.pi, 0, np.pi)
        # Should produce valid transformation
        assert abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
