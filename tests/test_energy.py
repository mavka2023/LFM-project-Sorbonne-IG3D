import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lfm.grid import MACGrid2D
from lfm.boundary import BoundaryCondition2D
from lfm.lfm_solver import LFMSolver


class TestEnergyConservation:
    """Test approximate kinetic energy conservation for inviscid flows."""

    def test_energy_stable(self):
        grid = MACGrid2D(32, 32, 1.0 / 32)
        bc = BoundaryCondition2D(grid)
        bc.set_wall_bc()

        solver = LFMSolver(grid, reinit_every=5, viscosity=0.0, use_bfecc_clamp=True)

        px, py = grid.face_positions(0)
        u_x = -np.cos(2 * np.pi * px) * np.sin(2 * np.pi * py) * 0.1

        px, py = grid.face_positions(1)
        u_y = np.sin(2 * np.pi * px) * np.cos(2 * np.pi * py) * 0.1

        solver.set_initial_velocity(u_x, u_y)
        solver.set_boundary_conditions(bc)

        initial_energy = solver.compute_kinetic_energy()
        assert initial_energy > 0, "Initial energy should be positive"

        dt = 0.005
        for step in range(50):
            solver.step(dt)

        final_energy = solver.compute_kinetic_energy()
        ratio = final_energy / initial_energy

        print(f"  Energy ratio: {ratio:.4f} (initial={initial_energy:.6f}, final={final_energy:.6f})")
        assert 0.2 < ratio < 2.0, \
            f"Energy not stable: ratio={ratio:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
