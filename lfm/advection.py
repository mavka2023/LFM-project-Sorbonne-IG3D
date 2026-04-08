"""
RK2 Semi-Lagrangian Advection on MAC Staggered Grids.
Implements the semi-Lagrangian advection used for computing midpoint
velocities in the leapfrog scheme. The velocity at a grid point is
updated by backtracing a virtual particle to its departure point and
interpolating the source field there.

"""

import numpy as np
from typing import Tuple, List
from .grid import MACGrid2D, MACGrid3D
from .interpolation import (
    interp_mac_2d, interp_face_2d, _interp_n2_scalar_2d,
    interp_mac_3d, _interp_n2_scalar_3d,
)


def advect_rk2_2d(
    grid: MACGrid2D,
    src: np.ndarray,
    u_x: np.ndarray,
    u_y: np.ndarray,
    dt: float,
    axis: int,
) -> np.ndarray:
    """
    For each face center, we backtrace through the velocity field to find
    where the fluid came from, then interpolate the source field there.

    """
    pos_x, pos_y = grid.face_positions(axis)

    vx1, vy1 = interp_mac_2d(grid, u_x, u_y, pos_x, pos_y)

    mid_x = pos_x - 0.5 * dt * vx1
    mid_y = pos_y - 0.5 * dt * vy1
    vx2, vy2 = interp_mac_2d(grid, u_x, u_y, mid_x, mid_y)

    dep_x = pos_x - dt * vx2
    dep_y = pos_y - dt * vy2

    offsets = [(0.0, 0.5), (0.5, 0.0)]
    return _interp_n2_scalar_2d(src, dep_x, dep_y, offsets[axis], grid.dx, grid.shape)


def advect_rk2_3d(
    grid: MACGrid3D,
    src: np.ndarray,
    u_x: np.ndarray,
    u_y: np.ndarray,
    u_z: np.ndarray,
    dt: float,
    axis: int,
) -> np.ndarray:
    """
    Advect a 3D face-stored field using RK2 semi-Lagrangian backtracing.

    """
    pos_x, pos_y, pos_z = grid.face_positions(axis)

    vx1, vy1, vz1 = interp_mac_3d(grid, u_x, u_y, u_z, pos_x, pos_y, pos_z)
    mid_x = pos_x - 0.5 * dt * vx1
    mid_y = pos_y - 0.5 * dt * vy1
    mid_z = pos_z - 0.5 * dt * vz1
    vx2, vy2, vz2 = interp_mac_3d(grid, u_x, u_y, u_z, mid_x, mid_y, mid_z)

    dep_x = pos_x - dt * vx2
    dep_y = pos_y - dt * vy2
    dep_z = pos_z - dt * vz2

    offsets_3d = [(0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)]
    return _interp_n2_scalar_3d(src, dep_x, dep_y, dep_z, offsets_3d[axis], grid.dx)


def advect_center_rk2_2d(
    grid: MACGrid2D,
    src: np.ndarray,
    u_x: np.ndarray,
    u_y: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Advect a 2D cell-centered scalar field (e.g., smoke density, temperature).

    """
    pos_x, pos_y = grid.cell_center_positions()

    vx1, vy1 = interp_mac_2d(grid, u_x, u_y, pos_x, pos_y)
    mid_x = pos_x - 0.5 * dt * vx1
    mid_y = pos_y - 0.5 * dt * vy1
    vx2, vy2 = interp_mac_2d(grid, u_x, u_y, mid_x, mid_y)
    dep_x = pos_x - dt * vx2
    dep_y = pos_y - dt * vy2

    return _interp_n2_scalar_2d(src, dep_x, dep_y, (0.5, 0.5), grid.dx, grid.shape)


def advect_center_rk2_3d(
    grid: MACGrid3D,
    src: np.ndarray,
    u_x: np.ndarray,
    u_y: np.ndarray,
    u_z: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Advect a 3D cell-centered scalar field (e.g., smoke density, temperature).

    """
    pos_x, pos_y, pos_z = grid.cell_center_positions()

    vx1, vy1, vz1 = interp_mac_3d(grid, u_x, u_y, u_z, pos_x, pos_y, pos_z)
    mid_x = pos_x - 0.5 * dt * vx1
    mid_y = pos_y - 0.5 * dt * vy1
    mid_z = pos_z - 0.5 * dt * vz1
    vx2, vy2, vz2 = interp_mac_3d(grid, u_x, u_y, u_z, mid_x, mid_y, mid_z)
    dep_x = pos_x - dt * vx2
    dep_y = pos_y - dt * vy2
    dep_z = pos_z - dt * vz2

    return _interp_n2_scalar_3d(src, dep_x, dep_y, dep_z, (0.5, 0.5, 0.5), grid.dx)
