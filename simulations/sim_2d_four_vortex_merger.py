"""
2D Four-Vortex Merger
"""
import numpy as np
import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lfm.grid import MACGrid2D
from lfm.boundary import BoundaryCondition2D
from lfm.lfm_solver import LFMSolver

def _add_vortex(grid, u_x, u_y, cx, cy, gamma, sigma):
    px, py = grid.face_positions(0)
    rx = px - cx
    ry = py - cy
    r2 = rx**2 + ry**2 + 1e-10
    factor = gamma / (2 * np.pi * r2) * (1 - np.exp(-r2 / sigma**2))
    u_x += -ry * factor

    px, py = grid.face_positions(1)
    rx = px - cx
    ry = py - cy
    r2 = rx**2 + ry**2 + 1e-10
    factor = gamma / (2 * np.pi * r2) * (1 - np.exp(-r2 / sigma**2))
    u_y += rx * factor

def run_simulation(
    resolution: int = 128,
    num_steps: int = 500,
    reinit_every: int = 2,
    save_every: int = 2,
    output_dir: str = "output_2d_four_vortex_merger",
):
    nx = resolution
    ny = resolution
    domain_size = 1.0
    dx = domain_size / resolution

    grid = MACGrid2D(nx, ny, dx)

    strength = 1.0
    core_radius = 0.04
    sep = 0.15
    cx, cy = domain_size / 2, domain_size / 2
    vortex_positions = [
        (cx - sep, cy - sep),
        (cx + sep, cy - sep),
        (cx - sep, cy + sep),
        (cx + sep, cy + sep),
    ]

    u_x = grid.zeros_faces(0)
    u_y = grid.zeros_faces(1)

    for vx, vy in vortex_positions:
        _add_vortex(grid, u_x, u_y, vx, vy, strength, core_radius)

    bc = BoundaryCondition2D(grid)
    bc.set_wall_bc()

    solver = LFMSolver(
        grid,
        reinit_every=reinit_every,
        viscosity=0.0,
        density=1.0,
        use_bfecc_clamp=True,
    )
    solver.set_initial_velocity(u_x, u_y)
    solver.set_boundary_conditions(bc)

    max_vel = max(np.max(np.abs(u_x)), np.max(np.abs(u_y)), 1e-6)
    dt = 0.4 * dx / max_vel

    os.makedirs(output_dir, exist_ok=True)
    vorticity_frames = []

    start_time = time.time()

    for step in range(num_steps):
        solver.step(dt)

        if step % save_every == 0 or step == num_steps - 1:
            vort = solver.compute_vorticity_2d()
            vorticity_frames.append(vort.copy())
            solver.compute_max_divergence()
            solver.compute_kinetic_energy()

    from visualization.viz_2d import plot_vorticity, create_animation

    plot_vorticity(
        vorticity_frames[-1], dx=dx,
        title=f"Four-Vortex Merger — Step {num_steps}",
        save_path=os.path.join(output_dir, "final_vorticity.png"),
        show=False,
    )

    import matplotlib.pyplot as plt
    n_snapshots = min(6, len(vorticity_frames))
    indices = np.linspace(0, len(vorticity_frames) - 1, n_snapshots, dtype=int)
    fig, axes = plt.subplots(1, n_snapshots, figsize=(4 * n_snapshots, 4))
    if n_snapshots == 1:
        axes = [axes]

    abs_max = max(np.max(np.abs(f)) for f in vorticity_frames)
    for idx, ax in zip(indices, axes):
        ax.imshow(
            vorticity_frames[idx].T, origin='lower', cmap='RdBu_r',
            vmin=-abs_max, vmax=abs_max,
            extent=[0, nx * dx, 0, ny * dx],
            aspect='equal', interpolation='bilinear',
        )
        ax.set_title(f"Step {idx * save_every}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "snapshots.png"), dpi=150, bbox_inches='tight')
    plt.close()

    try:
        create_animation(
            vorticity_frames, dx=dx, fps=30,
            title="Four-Vortex Merger",
            save_path=os.path.join(output_dir, "vorticity.gif"),
        )
    except Exception as e:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D Four-Vortex Merger")
    parser.add_argument("--resolution", type=int, default=128, help="Grid resolution N")
    parser.add_argument("--steps", type=int, default=500, help="Number of steps")
    args = parser.parse_args()

    run_simulation(resolution=args.resolution, num_steps=args.steps)