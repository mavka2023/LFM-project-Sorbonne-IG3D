import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lfm.grid import MACGrid2D
from lfm.boundary import BoundaryCondition2D
from lfm.lfm_solver import LFMSolver


class TestDivergenceFreeInSimulation:
    """Test that divergence stays near zero during simulation."""

    def test_divergence_after_steps(self):
        grid = MACGrid2D(32, 32, 1.0 / 32)
        bc = BoundaryCondition2D(grid)
        bc.set_wall_bc()

        solver = LFMSolver(grid, reinit_every=5, viscosity=0.0)

        px, py = grid.face_positions(0)
        cx, cy = 0.5, 0.5
        rx, ry = px - cx, py - cy
        r2 = rx**2 + ry**2 + 1e-10
        u_x = -ry / r2 * (1 - np.exp(-r2 / 0.05**2))

        px, py = grid.face_positions(1)
        rx, ry = px - cx, py - cy
        r2 = rx**2 + ry**2 + 1e-10
        u_y = rx / r2 * (1 - np.exp(-r2 / 0.05**2))

        solver.set_initial_velocity(u_x, u_y)
        solver.set_boundary_conditions(bc)

        dt = 0.001
        for step in range(20):
            solver.step(dt)
            max_div = solver.compute_max_divergence()
            assert max_div < 1e-4, \
                f"Divergence too large at step {step}: max|div| = {max_div}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
