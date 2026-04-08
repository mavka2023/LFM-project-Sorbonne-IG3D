import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lfm.interpolation import n2, dn2, interp_mac_2d, interp_mac_2d_grad
from lfm.grid import MACGrid2D


class TestN2Basis:
    """Tests for the N2 basis function."""

    def test_symmetry(self):
        x = np.linspace(-2, 2, 1000)
        assert np.allclose(n2(x), n2(-x))

    def test_support(self):
        x = np.array([-2.0, -1.5, 1.5, 2.0])
        assert np.allclose(n2(x), 0.0)

    def test_peak(self):
        assert np.isclose(n2(np.array([0.0])), 0.75)

    def test_partition_of_unity(self):
        for x in np.linspace(0, 1, 50):
            shifts = np.array([x - i for i in range(-2, 4)])
            total = np.sum(n2(shifts))
            assert np.isclose(total, 1.0, atol=1e-10), \
                f"Partition of unity violated at x={x}: sum={total}"


class TestDN2:
    """Tests for the N2 derivative."""

    def test_antisymmetry(self):
        x = np.linspace(-2, 2, 1000)
        assert np.allclose(dn2(x), -dn2(-x), atol=1e-10)

    def test_finite_difference(self):
        eps = 1e-6
        x = np.linspace(-1.4, 1.4, 100)
        fd = (n2(x + eps) - n2(x - eps)) / (2 * eps)
        analytical = dn2(x)
        assert np.allclose(fd, analytical, atol=1e-4)


class TestInterp2D:
    """Tests for 2D MAC interpolation."""

    def test_constant_field(self):
        grid = MACGrid2D(16, 16, 1.0 / 16)
        u_x = np.ones(grid.face_shape(0)) * 3.0
        u_y = np.ones(grid.face_shape(1)) * 7.0

        np.random.seed(0)
        px = np.random.uniform(2 * grid.dx, 14 * grid.dx, 50)
        py = np.random.uniform(2 * grid.dx, 14 * grid.dx, 50)

        vx, vy = interp_mac_2d(grid, u_x, u_y, px, py)
        assert np.allclose(vx, 3.0, atol=1e-6), f"Max error: {np.max(np.abs(vx - 3.0))}"
        assert np.allclose(vy, 7.0, atol=1e-6), f"Max error: {np.max(np.abs(vy - 7.0))}"

    def test_linear_field(self):
        grid = MACGrid2D(16, 16, 1.0 / 16)

        px_f, py_f = grid.face_positions(0)
        u_x = 2.0 * px_f + 3.0 * py_f

        px_f, py_f = grid.face_positions(1)
        u_y = -1.0 * px_f + 5.0 * py_f

        np.random.seed(1)
        qx = np.random.uniform(3 * grid.dx, 13 * grid.dx, 30)
        qy = np.random.uniform(3 * grid.dx, 13 * grid.dx, 30)

        vx, vy = interp_mac_2d(grid, u_x, u_y, qx, qy)
        expected_vx = 2.0 * qx + 3.0 * qy
        expected_vy = -1.0 * qx + 5.0 * qy

        assert np.allclose(vx, expected_vx, atol=1e-4), \
            f"ux error: max={np.max(np.abs(vx - expected_vx))}"
        assert np.allclose(vy, expected_vy, atol=1e-4), \
            f"uy error: max={np.max(np.abs(vy - expected_vy))}"

    def test_gradient_constant(self):
        """Gradient of a constant field should be zero."""
        grid = MACGrid2D(16, 16, 1.0 / 16)
        u_x = np.ones(grid.face_shape(0)) * 5.0
        u_y = np.ones(grid.face_shape(1)) * 3.0

        px = np.array([0.5 * grid.dx * 8])
        py = np.array([0.5 * grid.dx * 8])

        (vx, vy), grad = interp_mac_2d_grad(grid, u_x, u_y, px, py)
        assert np.allclose(grad, 0.0, atol=1e-4), f"Gradient not zero: {grad}"

    def test_gradient_linear(self):
        """Gradient of a linear field should be the known constant slope."""
        grid = MACGrid2D(16, 16, 1.0 / 16)
        dx = grid.dx

        px_f, py_f = grid.face_positions(0)
        u_x = 2.0 * px_f

        px_f, py_f = grid.face_positions(1)
        u_y = 3.0 * py_f

        qx = np.array([8 * dx])
        qy = np.array([8 * dx])

        (vx, vy), grad = interp_mac_2d_grad(grid, u_x, u_y, qx, qy)

        assert np.isclose(grad[0, 0, 0], 2.0, atol=0.1), f"du_x/dx = {grad[0,0,0]}"
        assert np.isclose(grad[1, 1, 0], 3.0, atol=0.1), f"du_y/dy = {grad[1,1,0]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
