"""
3D Visualization Utilities for LFM Fluid Simulations.

Provides PyVista/VTK-based 3D rendering functions.
"""

import numpy as np
import os
from typing import Optional, Tuple, List


def compute_vorticity_magnitude_3d(u_x, u_y, u_z, dx):
    """
    Compute vorticity magnitude |ω| = |∇ × u| on a 3D MAC grid.

    First averages face velocities to cell centers, then computes the
    curl using np.gradient (central differences). This guarantees all
    arrays have shape (nx, ny, nz) without staggered-grid shape conflicts.

    """
    ux_c = (u_x[:-1, :, :] + u_x[1:, :, :]) / 2.0
    uy_c = (u_y[:, :-1, :] + u_y[:, 1:, :]) / 2.0
    uz_c = (u_z[:, :, :-1] + u_z[:, :, 1:]) / 2.0

    duz_dy = np.gradient(uz_c, dx, axis=1)
    duy_dz = np.gradient(uy_c, dx, axis=2)

    dux_dz = np.gradient(ux_c, dx, axis=2)
    duz_dx = np.gradient(uz_c, dx, axis=0)

    duy_dx = np.gradient(uy_c, dx, axis=0)
    dux_dy = np.gradient(ux_c, dx, axis=1)

    omega_x = duz_dy - duy_dz
    omega_y = dux_dz - duz_dx
    omega_z = duy_dx - dux_dy

    return np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)


def save_vtk(
    field: np.ndarray,
    dx: float,
    filename: str,
    field_name: str = "vorticity",
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    """
    Save a 3D scalar field as a VTK structured grid file (.vtk).
    This can be loaded in ParaView
    """
    nx, ny, nz = field.shape

    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"LFM {field_name}\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write(f"ORIGIN {origin[0]} {origin[1]} {origin[2]}\n")
        f.write(f"SPACING {dx} {dx} {dx}\n")
        f.write(f"POINT_DATA {nx * ny * nz}\n")
        f.write(f"SCALARS {field_name} float 1\n")
        f.write("LOOKUP_TABLE default\n")

        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    f.write(f"{field[i, j, k]:.6e}\n")

    print(f"  VTK saved: {filename} ({nx}×{ny}×{nz})")


def plot_slices(
    field: np.ndarray,
    dx: float = 1.0,
    title: str = "3D Field Slices",
    cmap: str = "inferno",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot orthogonal slices through a 3D field using matplotlib.
    """
    import matplotlib.pyplot as plt

    nx, ny, nz = field.shape
    cx, cy, cz = nx // 2, ny // 2, nz // 2

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    vmax = np.max(np.abs(field))
    if vmax == 0:
        vmax = 1.0

    # XY slice (z = center)
    im0 = axes[0].imshow(
        field[:, :, cz].T, origin='lower', cmap=cmap,
        extent=[0, nx * dx, 0, ny * dx],
        vmin=0, vmax=vmax, aspect='equal', interpolation='bilinear',
    )
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    axes[0].set_title(f"XY slice (z={cz*dx:.2f})")
    plt.colorbar(im0, ax=axes[0])

    # XZ slice (y = center)
    im1 = axes[1].imshow(
        field[:, cy, :].T, origin='lower', cmap=cmap,
        extent=[0, nx * dx, 0, nz * dx],
        vmin=0, vmax=vmax, aspect='equal', interpolation='bilinear',
    )
    axes[1].set_xlabel("x"); axes[1].set_ylabel("z")
    axes[1].set_title(f"XZ slice (y={cy*dx:.2f})")
    plt.colorbar(im1, ax=axes[1])

    # YZ slice (x = center)
    im2 = axes[2].imshow(
        field[cx, :, :].T, origin='lower', cmap=cmap,
        extent=[0, ny * dx, 0, nz * dx],
        vmin=0, vmax=vmax, aspect='equal', interpolation='bilinear',
    )
    axes[2].set_xlabel("y"); axes[2].set_ylabel("z")
    axes[2].set_title(f"YZ slice (x={cx*dx:.2f})")
    plt.colorbar(im2, ax=axes[2])

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def export_vtk_sequence(
    frames: List[np.ndarray],
    dx: float,
    output_dir: str,
    prefix: str = "vorticity",
    field_name: str = "vorticity",
):
    """
    Export a sequence of 3D fields as numbered VTK files.

    """
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        path = os.path.join(output_dir, f"{prefix}_{i:04d}.vtk")
        save_vtk(frame, dx, path, field_name)
    print(f"  Exported {len(frames)} VTK frames to {output_dir}/")
