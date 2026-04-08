"""
Pressure Projection — Making the velocity field divergence-free.

The projection step is the core mechanism for enforcing the incompressibility
constraint ∇·u = 0. Given a velocity field that may have non-zero divergence
(e.g., after advection):

  1. Compute the divergence:  div(u) = ∂u_x/∂x + ∂u_y/∂y [+ ∂u_z/∂z]
  2. Solve the Poisson equation:  Δp = div(u)
  3. Subtract the pressure gradient:  u ← u - ∇p

On a MAC staggered grid, the divergence is naturally computed as the sum of
face velocity differences across each cell, and the pressure gradient is
simply the difference of pressures between adjacent cells.
"""

import numpy as np
from typing import Tuple, Optional
from .grid import MACGrid2D, MACGrid3D
from .boundary import BoundaryCondition2D, BoundaryCondition3D
from .poisson import (
    build_laplacian_2d, build_laplacian_3d,
    solve_pressure_2d, solve_pressure_3d,
)


def calc_divergence_2d(
    grid: MACGrid2D,
    u_x: np.ndarray,
    u_y: np.ndarray,
) -> np.ndarray:
    """
    Compute the divergence of a 2D MAC velocity field.

    div(u) at cell (i,j) = [u_x(i+1,j) - u_x(i,j) + u_y(i,j+1) - u_y(i,j)] / dx

    Note: The original code computes div without dividing by dx (it's absorbed
    into the Poisson equation scaling).  Following the same convention — the
    Laplacian matrix is also scale-free.

    """
    # Following the sign convention from CalcDivKernel:
    # b = u_x[left] - u_x[right] + u_y[bottom] - u_y[top]
    div = (u_x[:-1, :] - u_x[1:, :] +
           u_y[:, :-1] - u_y[:, 1:])
    return div


def apply_pressure_2d(
    grid: MACGrid2D,
    u_x: np.ndarray,
    u_y: np.ndarray,
    pressure: np.ndarray,
    bc: BoundaryCondition2D,
):
    """
    Apply pressure correction to make velocity divergence-free.
    """
    nx, ny = grid.nx, grid.ny

    # X-faces: u_x[i,j] += p[i-1,j] - p[i,j]
    for i in range(nx + 1):
        if i == 0 or i == nx:
            pass
        for j in range(ny):
            if not bc.is_bc_x[i, j]:
                left_p = pressure[i - 1, j] if i > 0 else 0.0
                right_p = pressure[i, j] if i < nx else 0.0
                u_x[i, j] += left_p - right_p

    # Y-faces: u_y[i,j] += p[i,j-1] - p[i,j]
    for i in range(nx):
        for j in range(ny + 1):
            if not bc.is_bc_y[i, j]:
                bottom_p = pressure[i, j - 1] if j > 0 else 0.0
                top_p = pressure[i, j] if j < ny else 0.0
                u_y[i, j] += bottom_p - top_p


def apply_pressure_2d_vectorized(
    grid: MACGrid2D,
    u_x: np.ndarray,
    u_y: np.ndarray,
    pressure: np.ndarray,
    bc: BoundaryCondition2D,
):
    """
    Vectorized version of apply_pressure_2d for better performance.
    """
    nx, ny = grid.nx, grid.ny

    p_left = np.zeros((nx + 1, ny), dtype=np.float64)
    p_right = np.zeros((nx + 1, ny), dtype=np.float64)
    p_left[1:, :] = pressure  # p[i-1, j] for face i
    p_right[:nx, :] = pressure  # p[i, j] for face i

    dp_x = p_left - p_right
    mask_x = ~bc.is_bc_x
    u_x[mask_x] += dp_x[mask_x]

    # Y-gradient
    p_bottom = np.zeros((nx, ny + 1), dtype=np.float64)
    p_top = np.zeros((nx, ny + 1), dtype=np.float64)
    p_bottom[:, 1:] = pressure
    p_top[:, :ny] = pressure

    dp_y = p_bottom - p_top
    mask_y = ~bc.is_bc_y
    u_y[mask_y] += dp_y[mask_y]


def project_2d(
    grid: MACGrid2D,
    u_x: np.ndarray,
    u_y: np.ndarray,
    bc: BoundaryCondition2D,
    laplacian_cache: dict = None,
    tol: float = 1e-6,
    max_iter: int = 500,
) -> np.ndarray:
    """
    Full 2D pressure projection pipeline.

    """
    # Step 1: Apply BCs
    bc.apply(u_x, u_y)

    # Step 2: Compute divergence
    div = calc_divergence_2d(grid, u_x, u_y)

    # Step 3: Build or retrieve Laplacian
    if laplacian_cache is not None and 'A' in laplacian_cache:
        A = laplacian_cache['A']
        is_dof = laplacian_cache['is_dof']
    else:
        A, is_dof = build_laplacian_2d(grid, bc.is_bc_x, bc.is_bc_y)
        if laplacian_cache is not None:
            laplacian_cache['A'] = A
            laplacian_cache['is_dof'] = is_dof

    div_flat = div.ravel()
    div_flat[~is_dof] = 0.0

    # Step 4: Solve Poisson
    pressure = solve_pressure_2d(grid, div, A, is_dof, tol=tol, max_iter=max_iter)

    # Step 5: Apply pressure correction
    apply_pressure_2d_vectorized(grid, u_x, u_y, pressure, bc)

    return pressure


def calc_divergence_3d(
    grid: MACGrid3D,
    u_x: np.ndarray,
    u_y: np.ndarray,
    u_z: np.ndarray,
) -> np.ndarray:
    """
    Compute the divergence of a 3D MAC velocity field.

    """
    div = (u_x[:-1, :, :] - u_x[1:, :, :] +
           u_y[:, :-1, :] - u_y[:, 1:, :] +
           u_z[:, :, :-1] - u_z[:, :, 1:])
    return div


def apply_pressure_3d_vectorized(
    grid: MACGrid3D,
    u_x: np.ndarray,
    u_y: np.ndarray,
    u_z: np.ndarray,
    pressure: np.ndarray,
    bc: BoundaryCondition3D,
):
    """
    Vectorized pressure correction for 3D.

    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    # X-gradient
    p_left = np.zeros((nx+1, ny, nz), dtype=np.float64)
    p_right = np.zeros((nx+1, ny, nz), dtype=np.float64)
    p_left[1:, :, :] = pressure
    p_right[:nx, :, :] = pressure
    dp_x = p_left - p_right
    mask_x = ~bc.is_bc_x
    u_x[mask_x] += dp_x[mask_x]

    # Y-gradient
    p_bottom = np.zeros((nx, ny+1, nz), dtype=np.float64)
    p_top = np.zeros((nx, ny+1, nz), dtype=np.float64)
    p_bottom[:, 1:, :] = pressure
    p_top[:, :ny, :] = pressure
    dp_y = p_bottom - p_top
    mask_y = ~bc.is_bc_y
    u_y[mask_y] += dp_y[mask_y]

    # Z-gradient
    p_back = np.zeros((nx, ny, nz+1), dtype=np.float64)
    p_front = np.zeros((nx, ny, nz+1), dtype=np.float64)
    p_back[:, :, 1:] = pressure
    p_front[:, :, :nz] = pressure
    dp_z = p_back - p_front
    mask_z = ~bc.is_bc_z
    u_z[mask_z] += dp_z[mask_z]


def project_3d(
    grid: MACGrid3D,
    u_x: np.ndarray,
    u_y: np.ndarray,
    u_z: np.ndarray,
    bc: BoundaryCondition3D,
    laplacian_cache: dict = None,
    tol: float = 1e-6,
    max_iter: int = 500,
) -> np.ndarray:
    """
    Full 3D pressure projection pipeline.
    """
    bc.apply(u_x, u_y, u_z)

    div = calc_divergence_3d(grid, u_x, u_y, u_z)

    if laplacian_cache is not None and 'A' in laplacian_cache:
        A = laplacian_cache['A']
        is_dof = laplacian_cache['is_dof']
    else:
        A, is_dof = build_laplacian_3d(grid, bc.is_bc_x, bc.is_bc_y, bc.is_bc_z)
        if laplacian_cache is not None:
            laplacian_cache['A'] = A
            laplacian_cache['is_dof'] = is_dof

    div_flat = div.ravel()
    div_flat[~is_dof] = 0.0

    pressure = solve_pressure_3d(grid, div, A, is_dof, tol=tol, max_iter=max_iter)

    apply_pressure_3d_vectorized(grid, u_x, u_y, u_z, pressure, bc)

    return pressure
