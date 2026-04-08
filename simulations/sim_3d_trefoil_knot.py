"""
3D Trefoil Knot Vortex
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


def trefoil_curve(t):
    """
    Evaluates the parametric curve of a trefoil knot.
    """
    x = np.sin(t) + 2.0 * np.sin(2.0 * t)
    y = np.cos(t) - 2.0 * np.cos(2.0 * t)
    z = -np.sin(3.0 * t)
    return x, y, z


def trefoil_tangent(t):
    """
    Computes the analytical tangent vector of the trefoil knot.
    """
    tx = np.cos(t) + 4.0 * np.cos(2.0 * t)
    ty = -np.sin(t) + 4.0 * np.sin(2.0 * t)
    tz = -3.0 * np.cos(3.0 * t)
    return tx, ty, tz


def create_trefoil_velocity(
    grid: MACGrid3D,
    center: np.ndarray,
    scale: float,
    core_radius: float,
    strength: float,
    n_segments: int = 300,
):
    """
    Constructs the initial velocity field for the trefoil knot vortex tube.

    """
    center = np.array(center, dtype=np.float64)

    u_x = grid.zeros_faces(0)
    u_y = grid.zeros_faces(1)
    u_z = grid.zeros_faces(2)

    # Parametric discretization
    t_params = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    dt_param = 2 * np.pi / n_segments

    knot_x, knot_y, knot_z = trefoil_curve(t_params)
    tang_x, tang_y, tang_z = trefoil_tangent(t_params)

    # Apply scaling and translation
    knot_x = knot_x * scale + center[0]
    knot_y = knot_y * scale + center[1]
    knot_z = knot_z * scale + center[2]

    # Compute normalized tangents and segment arc lengths
    tang_mag = np.sqrt(tang_x**2 + tang_y**2 + tang_z**2)
    tang_x_n = tang_x / tang_mag
    tang_y_n = tang_y / tang_mag
    tang_z_n = tang_z / tang_mag

    ds = tang_mag * dt_param * scale

    # Accumulate Biot-Savart velocity contributions for each staggered face
    for face_axis in range(3):
        pos = grid.face_positions(face_axis)
        px, py, pz = pos[0], pos[1], pos[2]

        vx_total = np.zeros_like(px)
        vy_total = np.zeros_like(px)
        vz_total = np.zeros_like(px)

        for seg in range(n_segments):
            rx = px - knot_x[seg]
            ry = py - knot_y[seg]
            rz = pz - knot_z[seg]
            r2 = rx**2 + ry**2 + rz**2

            denom = (r2 + core_radius**2) ** 1.5
            coeff = strength * ds[seg] / (4.0 * np.pi * denom)

            cx = tang_y_n[seg] * rz - tang_z_n[seg] * ry
            cy = tang_z_n[seg] * rx - tang_x_n[seg] * rz
            cz = tang_x_n[seg] * ry - tang_y_n[seg] * rx

            vx_total += coeff * cx
            vy_total += coeff * cy
            vz_total += coeff * cz

        if face_axis == 0:
            u_x += vx_total
        elif face_axis == 1:
            u_y += vy_total
        elif face_axis == 2:
            u_z += vz_total

    return u_x, u_y, u_z


def run_simulation(
    resolution: int = 32,
    num_steps: int = 250,
    reinit_every: int = 2,
    save_every: int = 2,
    output_dir: str = "output_3d_trefoil_knot",
):
    """
    Executes the 3D Trefoil Knot topological reconnection simulation.
    """
    nx = resolution
    ny = resolution
    nz = resolution
    domain_size = 1.0
    dx = domain_size / resolution

    grid = MACGrid3D(nx, ny, nz, dx)

    knot_scale = domain_size * 0.12  
    core_radius = domain_size * 0.04 
    strength = 4.0                  
    knot_center = [domain_size / 2, domain_size / 2, domain_size / 2]

    t0 = time.time()

    u_x, u_y, u_z = create_trefoil_velocity(
        grid, knot_center, knot_scale, core_radius, strength,
        n_segments=200,
    )

    bc = BoundaryCondition3D(grid)
    bc.set_wall_bc()

    solver = LFMSolver(
        grid,
        reinit_every=reinit_every,
        viscosity=0.0,
        density=1.0,
        use_bfecc_clamp=True,
    )
    solver.set_initial_velocity(u_x, u_y, u_z)
    solver.set_boundary_conditions(bc)

    max_vel = max(
        np.max(np.abs(u_x)), np.max(np.abs(u_y)),
        np.max(np.abs(u_z)), 1e-6
    )
    dt = 0.3 * dx / max_vel

    os.makedirs(output_dir, exist_ok=True)
    vorticity_frames = []

    start_time = time.time()

    for step in range(num_steps):
        solver.step(dt)

        if step % save_every == 0 or step == num_steps - 1:
            from visualization.viz_3d import compute_vorticity_magnitude_3d
            vort_mag = compute_vorticity_magnitude_3d(
                solver.u_x, solver.u_y, solver.u_z, dx)
            vorticity_frames.append(vort_mag.copy())
            
            max_div = solver.compute_max_divergence()
            energy = solver.compute_kinetic_energy()

    total_time = time.time() - start_time

    from visualization.viz_3d import plot_slices, save_vtk, export_vtk_sequence

    plot_slices(
        vorticity_frames[-1], dx=dx,
        title=f"Trefoil Knot — Step {num_steps}, |ω|",
        save_path=os.path.join(output_dir, "final_slices.png"),
        show=False,
    )

    save_vtk(
        vorticity_frames[-1], dx,
        os.path.join(output_dir, "final_vorticity.vtk"),
        field_name="vorticity_magnitude",
    )

    export_vtk_sequence(
        vorticity_frames, dx, output_dir,
        prefix="vorticity", field_name="vorticity_magnitude",
    )
    print(f"\n  Output saved to {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Trefoil Knot Vortex")
    parser.add_argument("--resolution", type=int, default=32, help="Grid resolution N")
    parser.add_argument("--steps", type=int, default=250, help="Number of steps")
    args = parser.parse_args()

    run_simulation(resolution=args.resolution, num_steps=args.steps)