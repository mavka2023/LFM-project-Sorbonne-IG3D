"""
3D Jet in Crossflow (JICF)
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
    num_steps: int = 250,
    reinit_every: int = 2,
    save_every: int = 2,
    output_dir: str = "output_3d_jet_crossflow",
    crossflow_vel: float = 1.0,
    jet_vel: float = 3.0,
    jet_radius_frac: float = 0.08,
    viscosity: float = 0.002,
):
    # Domain setup
    nx = resolution * 2
    ny = resolution
    nz = resolution
    domain_y = 1.0
    dx = domain_y / resolution

    jet_radius = jet_radius_frac * domain_y
    grid = MACGrid3D(nx, ny, nz, dx)

    # Boundary conditions
    bc = BoundaryCondition3D(grid)
    bc.set_wall_bc(
        x_neg=crossflow_vel, x_pos=crossflow_vel,
        y_neg=0.0, y_pos=0.0,
        z_neg=0.0, z_pos=0.0,
    )

    # Initial velocity: uniform crossflow
    u_x = np.ones(grid.face_shape(0), dtype=np.float64) * crossflow_vel
    u_y = grid.zeros_faces(1)
    u_z = grid.zeros_faces(2)

    # Jet and smoke source masks
    domain_x = nx * dx
    domain_z = nz * dx
    jet_cx = domain_x * 0.25
    jet_cz = domain_z / 2

    py_face_y = grid.face_positions(1)[1]
    px_face_y = grid.face_positions(1)[0]
    pz_face_y = grid.face_positions(1)[2]
    jet_face_mask = (
        ((px_face_y - jet_cx)**2 + (pz_face_y - jet_cz)**2 < jet_radius**2) &
        (py_face_y < 2 * dx)
    )

    smoke = grid.zeros_centers()
    px_c, py_c, pz_c = grid.cell_center_positions()
    smoke_source_mask = (
        ((px_c - jet_cx)**2 + (pz_c - jet_cz)**2 < jet_radius**2) &
        (py_c < 3 * dx)
    )

    # Solver initialization
    solver = LFMSolver(
        grid,
        reinit_every=reinit_every,
        viscosity=viscosity,
        density=1.0,
        use_bfecc_clamp=True,
    )
    solver.set_initial_velocity(u_x, u_y, u_z)
    solver.set_boundary_conditions(bc)

    dt = 0.3 * dx / max(crossflow_vel, jet_vel)

    os.makedirs(output_dir, exist_ok=True)
    smoke_frames = []
    vort_frames = []

    for step in range(num_steps):
        # Continuous jet injection
        solver.u_y[jet_face_mask] = jet_vel
        solver.u_x[0, :, :] = crossflow_vel
        smoke[smoke_source_mask] = 1.0

        # Simulation step
        solver.step(dt)

        # Advect tracer
        smoke = advect_center_rk2_3d(grid, smoke, solver.u_x, solver.u_y, solver.u_z, dt)
        smoke = np.clip(smoke, 0.0, 1.0)

        # Diagnostics storage
        if step % save_every == 0 or step == num_steps - 1:
            smoke_frames.append(smoke.copy())
            from visualization.viz_3d import compute_vorticity_magnitude_3d
            vort_mag = compute_vorticity_magnitude_3d(solver.u_x, solver.u_y, solver.u_z, dx)
            vort_frames.append(vort_mag.copy())

    # Visualization and VTK export
    from visualization.viz_3d import plot_slices, save_vtk, export_vtk_sequence

    plot_slices(
        smoke_frames[-1], dx=dx,
        title=f"Jet Crossflow — Smoke, Step {num_steps}",
        cmap="hot",
        save_path=os.path.join(output_dir, "final_smoke_slices.png"),
        show=False,
    )

    plot_slices(
        vort_frames[-1], dx=dx,
        title=f"Jet Crossflow — |ω|, Step {num_steps}",
        cmap="inferno",
        save_path=os.path.join(output_dir, "final_vort_slices.png"),
        show=False,
    )

    save_vtk(
        smoke_frames[-1], dx,
        os.path.join(output_dir, "final_smoke.vtk"),
        field_name="smoke_density",
    )
    save_vtk(
        vort_frames[-1], dx,
        os.path.join(output_dir, "final_vorticity.vtk"),
        field_name="vorticity_magnitude",
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
    parser = argparse.ArgumentParser(description="3D Jet in Crossflow")
    parser.add_argument("--resolution", type=int, default=32, help="Grid resolution N")
    parser.add_argument("--steps", type=int, default=250, help="Number of steps")
    parser.add_argument("--jet-vel", type=float, default=3.0, help="Jet velocity")
    args = parser.parse_args()

    run_simulation(
        resolution=args.resolution,
        num_steps=args.steps,
        jet_vel=args.jet_vel,
    )