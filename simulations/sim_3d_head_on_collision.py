"""
3D Head-On Vortex Ring Collision
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


def create_vortex_ring_velocity(
    grid: MACGrid3D,
    center: np.ndarray,
    axis: np.ndarray,
    ring_radius: float,
    core_radius: float,
    strength: float,
):
    axis = np.array(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    center = np.array(center, dtype=np.float64)

    u_x = grid.zeros_faces(0)
    u_y = grid.zeros_faces(1)
    u_z = grid.zeros_faces(2)

    for face_axis in range(3):
        pos = grid.face_positions(face_axis)
        px, py, pz = pos[0], pos[1], pos[2]

        rx = px - center[0]
        ry = py - center[1]
        rz = pz - center[2]

        r_along = rx * axis[0] + ry * axis[1] + rz * axis[2]

        perp_x = rx - r_along * axis[0]
        perp_y = ry - r_along * axis[1]
        perp_z = rz - r_along * axis[2]
        rho = np.sqrt(perp_x**2 + perp_y**2 + perp_z**2 + 1e-12)

        d_ring = np.sqrt((rho - ring_radius)**2 + r_along**2)

        omega_mag = (strength / (np.pi * core_radius**2)) * \
                    np.exp(-d_ring**2 / core_radius**2)

        rho_hat_x = perp_x / (rho + 1e-12)
        rho_hat_y = perp_y / (rho + 1e-12)
        rho_hat_z = perp_z / (rho + 1e-12)

        tang_x = axis[1] * rho_hat_z - axis[2] * rho_hat_y
        tang_y = axis[2] * rho_hat_x - axis[0] * rho_hat_z
        tang_z = axis[0] * rho_hat_y - axis[1] * rho_hat_x

        vel_factor = omega_mag * core_radius**2 / (d_ring**2 + core_radius**2 + 1e-12)

        core_r_x = (rho - ring_radius) * rho_hat_x + r_along * axis[0]
        core_r_y = (rho - ring_radius) * rho_hat_y + r_along * axis[1]
        core_r_z = (rho - ring_radius) * rho_hat_z + r_along * axis[2]

        vx = tang_y * core_r_z - tang_z * core_r_y
        vy = tang_z * core_r_x - tang_x * core_r_z
        vz = tang_x * core_r_y - tang_y * core_r_x
        v_mag = np.sqrt(vx**2 + vy**2 + vz**2 + 1e-12)

        scale = vel_factor / (v_mag + 1e-12)
        vx *= scale * strength
        vy *= scale * strength
        vz *= scale * strength

        if face_axis == 0:
            u_x += vx
        elif face_axis == 1:
            u_y += vy
        elif face_axis == 2:
            u_z += vz

    return u_x, u_y, u_z


def run_simulation(
    resolution: int = 32,
    num_steps: int = 200,
    reinit_every: int = 2,
    save_every: int = 2,
    output_dir: str = "output_3d_head_on_collision",
):
    nx = resolution
    ny = resolution
    nz = resolution * 2
    domain_size = 1.0
    dx = domain_size / resolution

    grid = MACGrid3D(nx, ny, nz, dx)

    ring_radius = domain_size * 0.2
    core_radius = domain_size * 0.04
    strength = 5.0
    ring_axis = [0.0, 0.0, 1.0]

    center_xy = [domain_size / 2, domain_size / 2]
    collision_gap = domain_size * 0.4

    z1 = nz * dx * 0.5 - collision_gap / 2
    center1 = [center_xy[0], center_xy[1], z1]

    z2 = nz * dx * 0.5 + collision_gap / 2
    center2 = [center_xy[0], center_xy[1], z2]

    t0 = time.time()

    u_x1, u_y1, u_z1 = create_vortex_ring_velocity(
        grid, center1, ring_axis, ring_radius, core_radius, +strength)
    u_x2, u_y2, u_z2 = create_vortex_ring_velocity(
        grid, center2, ring_axis, ring_radius, core_radius, -strength)

    u_x = u_x1 + u_x2
    u_y = u_y1 + u_y2
    u_z = u_z1 + u_z2

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
        title=f"Head-On Collision — Step {num_steps}, |ω|",
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Head-On Vortex Ring Collision")
    parser.add_argument("--resolution", type=int, default=32, help="Grid resolution N")
    parser.add_argument("--steps", type=int, default=200, help="Number of steps")
    args = parser.parse_args()

    run_simulation(resolution=args.resolution, num_steps=args.steps)