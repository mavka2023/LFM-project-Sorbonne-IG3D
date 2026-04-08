"""
3D Rising Smoke Plume
"""

import numpy as np
import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lfm.grid import MACGrid3D
from lfm.boundary import BoundaryCondition3D
from lfm.lfm_solver import LFMSolver
from lfm.advection import advect_center_rk2_3d

def run_simulation(
    resolution: int = 32,
    num_steps: int = 128,
    reinit_every: int = 2,
    save_every: int = 2,
    output_dir: str = "output_3d_smoke_plume",
    buoyancy_alpha: float = 0.5,
    buoyancy_beta: float = 2.0,
    inlet_radius: float = 0.08,
    inlet_velocity: float = 1.5,
):
    # Domain setup
    nx = resolution
    ny = resolution * 2
    nz = resolution
    domain_x = 1.0
    dx = domain_x / resolution

    grid = MACGrid3D(nx, ny, nz, dx)

    # Boundary conditions
    bc = BoundaryCondition3D(grid)
    bc.set_wall_bc(
        x_neg=0.0, x_pos=0.0,
        y_neg=0.0, y_pos=0.0,
        z_neg=0.0, z_pos=0.0,
    )

    # Velocity and scalar fields
    u_x = grid.zeros_faces(0)
    u_y = grid.zeros_faces(1)
    u_z = grid.zeros_faces(2)
    smoke = grid.zeros_centers()
    temperature = np.zeros(grid.shape, dtype=np.float64)

    # Inlet mask
    px, py, pz = grid.cell_center_positions()
    cx = domain_x / 2
    cz = domain_x / 2
    inlet_mask = ((px - cx)**2 + (pz - cz)**2 < inlet_radius**2) & (py < 3 * dx)

    # Solver init
    solver = LFMSolver(
        grid,
        reinit_every=reinit_every,
        viscosity=0.0,
        density=1.0,
        use_bfecc_clamp=True,
    )
    solver.set_initial_velocity(u_x, u_y, u_z)
    solver.set_boundary_conditions(bc)

    dt = 0.3 * dx / max(inlet_velocity, 1.0)

    os.makedirs(output_dir, exist_ok=True)
    smoke_frames = []
    vort_frames = []

    for step in range(num_steps):
        # Source injection
        smoke[inlet_mask] = 1.0
        temperature[inlet_mask] = 1.0

        py_face_y = grid.face_positions(1)[1]
        px_face_y = grid.face_positions(1)[0]
        pz_face_y = grid.face_positions(1)[2]
        inlet_face_mask = (
            ((px_face_y - cx)**2 + (pz_face_y - cz)**2 < inlet_radius**2) &
            (py_face_y < 2 * dx)
        )
        solver.u_y[inlet_face_mask] = inlet_velocity

        # Buoyancy force
        buoyancy_y = (-buoyancy_alpha * smoke + buoyancy_beta * temperature)
        buoyancy_y_face = (buoyancy_y[:, :-1, :] + buoyancy_y[:, 1:, :]) / 2.0
        solver.u_y[:, 1:ny, :] += dt * buoyancy_y_face

        # Physics step
        solver.step(dt)

        # Advect scalars
        smoke = advect_center_rk2_3d(grid, smoke, solver.u_x, solver.u_y, solver.u_z, dt)
        temperature = advect_center_rk2_3d(grid, temperature, solver.u_x, solver.u_y, solver.u_z, dt)

        temperature *= 0.995
        smoke = np.clip(smoke, 0.0, 1.0)
        temperature = np.clip(temperature, 0.0, 1.0)

        # Storage
        if step % save_every == 0 or step == num_steps - 1:
            smoke_frames.append(smoke.copy())
            from visualization.viz_3d import compute_vorticity_magnitude_3d
            vort_mag = compute_vorticity_magnitude_3d(solver.u_x, solver.u_y, solver.u_z, dx)
            vort_frames.append(vort_mag.copy())

    # Visualization and VTK Export
    from visualization.viz_3d import plot_slices, save_vtk, export_vtk_sequence

    plot_slices(
        smoke_frames[-1], dx=dx,
        title=f"Smoke Plume — Step {num_steps}",
        cmap="hot",
        save_path=os.path.join(output_dir, "final_smoke_slices.png"),
        show=False,
    )

    plot_slices(
        vort_frames[-1], dx=dx,
        title=f"Smoke Plume Vorticity — Step {num_steps}",
        cmap="inferno",
        save_path=os.path.join(output_dir, "final_vort_slices.png"),
        show=False,
    )

    save_vtk(
        smoke_frames[-1], dx,
        os.path.join(output_dir, "final_smoke.vtk"),
        field_name="smoke_density",
    )

    export_vtk_sequence(
        smoke_frames, dx, output_dir,
        prefix="smoke", field_name="smoke_density",
    )

    export_vtk_sequence(
        vort_frames, dx, output_dir,
        prefix="vorticity", field_name="vorticity_magnitude",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Rising Smoke Plume")
    parser.add_argument("--resolution", type=int, default=32, help="Grid resolution N")
    parser.add_argument("--steps", type=int, default=80, help="Number of steps")
    args = parser.parse_args()

    run_simulation(resolution=args.resolution, num_steps=args.steps)