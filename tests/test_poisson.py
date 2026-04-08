import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lfm.grid import MACGrid2D
from lfm.boundary import BoundaryCondition2D
from lfm.poisson import build_laplacian_2d, solve_pressure_2d


class TestPoisson2D:
    """Tests for the 2D Poisson solver."""

    def test_zero_rhs(self):
        grid = MACGrid2D(16, 16, 1.0 / 16)
        bc = BoundaryCondition2D(grid)
        bc.set_wall_bc()

        A, is_dof = build_laplacian_2d(grid, bc.is_bc_x, bc.is_bc_y)
        rhs = np.zeros((16, 16))
        p = solve_pressure_2d(grid, rhs, A, is_dof, pure_neumann=True)

        assert np.max(np.abs(p)) < 1e-10, f"Non-zero solution for zero RHS: max={np.max(np.abs(p))}"

    def test_laplacian_symmetry(self):
        """The Laplacian matrix should be symmetric."""
        grid = MACGrid2D(8, 8, 1.0 / 8)
        bc = BoundaryCondition2D(grid)
        bc.set_wall_bc()

        A, _ = build_laplacian_2d(grid, bc.is_bc_x, bc.is_bc_y)
        diff = A - A.T
        assert diff.nnz == 0 or np.max(np.abs(diff.toarray())) < 1e-14, \
            "Laplacian is not symmetric"

    def test_convergence(self):
        grid = MACGrid2D(16, 16, 1.0 / 16)
        bc = BoundaryCondition2D(grid)
        bc.set_wall_bc()

        A, is_dof = build_laplacian_2d(grid, bc.is_bc_x, bc.is_bc_y)

        px, py = grid.cell_center_positions()
        rhs = np.sin(2 * np.pi * px) * np.sin(2 * np.pi * py)

        p = solve_pressure_2d(grid, rhs, A, is_dof, tol=1e-10, pure_neumann=True)

        residual = A @ p.ravel() - rhs.ravel()
        residual[~is_dof] = 0.0
        max_residual = np.max(np.abs(residual))
        assert max_residual < 1e-6, f"Solver did not converge: max residual = {max_residual}"


class TestDivergenceFree:
    """Tests that projection makes velocity divergence-free."""

    def test_projection_removes_divergence(self):
        from lfm.projection import project_2d, calc_divergence_2d

        grid = MACGrid2D(16, 16, 1.0 / 16)
        bc = BoundaryCondition2D(grid)
        bc.set_wall_bc()

        u_x = np.random.randn(*grid.face_shape(0)) * 0.1
        u_y = np.random.randn(*grid.face_shape(1)) * 0.1

        div_before = calc_divergence_2d(grid, u_x, u_y)
        assert np.max(np.abs(div_before)) > 1e-4, "Test field should have non-zero divergence"

        project_2d(grid, u_x, u_y, bc, tol=1e-10)

        div_after = calc_divergence_2d(grid, u_x, u_y)
        max_div = np.max(np.abs(div_after))
        assert max_div < 1e-6, f"Projection failed: max|div| = {max_div}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
