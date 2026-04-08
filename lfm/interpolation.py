"""
Quadratic B-Spline (N2) Interpolation for MAC 
Provides C1 continuity, which is critical for accurate velocity gradient computation
needed by the flow map Jacobian evolution.

"""

import numpy as np
from typing import Tuple, Union, Optional


def n2(x: np.ndarray) -> np.ndarray:
    """
    Evaluates the quadratic B-spline basis function.
    Compact support on [-1.5, 1.5].
    """
    ax = np.abs(x)
    result = np.zeros_like(x)
    # Inner region: |x| < 0.5
    inner = ax < 0.5
    result[inner] = 0.75 - ax[inner] ** 2
    # Outer region: 0.5 <= |x| < 1.5
    outer = (ax >= 0.5) & (ax < 1.5)
    result[outer] = 0.5 * (1.5 - ax[outer]) ** 2
    return result


def dn2(x: np.ndarray) -> np.ndarray:
    """
    Evaluates the derivative of the quadratic B-spline basis function.
    """
    ax = np.abs(x)
    result = np.zeros_like(x)
    inner = ax < 0.5
    result[inner] = -2.0 * x[inner]
    outer = (ax >= 0.5) & (ax < 1.5)
    result[outer] = x[outer] - 1.5 * np.sign(x[outer])
    return result


# 2D Interpolation Routines
def _interp_n2_scalar_2d(
    field: np.ndarray,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    field_offset: Tuple[float, float],
    dx: float,
    grid_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Interpolate a 2D scalar field using N2 B-splines.
    """
    inv_dx = 1.0 / dx
    eps = 1e-4

    # Convert world positions to index space relative to the field's staggering
    ix = pos_x * inv_dx - field_offset[0]
    iy = pos_y * inv_dx - field_offset[1]

    max_ix = field.shape[0] - 1.5 - eps
    max_iy = field.shape[1] - 1.5 - eps
    ix = np.clip(ix, 0.5 + eps, max_ix)
    iy = np.clip(iy, 0.5 + eps, max_iy)

    # Base index for the 3x3 stencil
    base_i = np.floor(ix - 0.5).astype(int)
    base_j = np.floor(iy - 0.5).astype(int)

    result = np.zeros_like(pos_x)
    for di in range(3):
        for dj in range(3):
            wi = n2(ix - (base_i + di))
            wj = n2(iy - (base_j + dj))
            si = np.clip(base_i + di, 0, field.shape[0] - 1)
            sj = np.clip(base_j + dj, 0, field.shape[1] - 1)
            result += field[si, sj] * wi * wj

    return result


def _interp_n2_scalar_2d_grad(
    field: np.ndarray,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    field_offset: Tuple[float, float],
    dx: float,
    grid_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate a 2D scalar field and its gradient using N2 B-splines.
    """
    inv_dx = 1.0 / dx
    eps = 1e-4

    ix = pos_x * inv_dx - field_offset[0]
    iy = pos_y * inv_dx - field_offset[1]

    max_ix = field.shape[0] - 1.5 - eps
    max_iy = field.shape[1] - 1.5 - eps
    ix = np.clip(ix, 0.5 + eps, max_ix)
    iy = np.clip(iy, 0.5 + eps, max_iy)

    base_i = np.floor(ix - 0.5).astype(int)
    base_j = np.floor(iy - 0.5).astype(int)

    value = np.zeros_like(pos_x)
    grad_x = np.zeros_like(pos_x)
    grad_y = np.zeros_like(pos_x)

    for di in range(3):
        for dj in range(3):
            wi = n2(ix - (base_i + di))
            wj = n2(iy - (base_j + dj))
            dwi = dn2(ix - (base_i + di))
            dwj = dn2(iy - (base_j + dj))
            si = np.clip(base_i + di, 0, field.shape[0] - 1)
            sj = np.clip(base_j + dj, 0, field.shape[1] - 1)
            val = field[si, sj]
            value += val * wi * wj
            grad_x += val * dwi * wj
            grad_y += val * wi * dwj

    # Convert gradient from index-space to world-space
    grad_x *= inv_dx
    grad_y *= inv_dx
    return value, grad_x, grad_y


def interp_mac_2d(
    grid,
    u_x: np.ndarray,
    u_y: np.ndarray,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate a 2D MAC velocity field at arbitrary positions.
    """
    vel_x = _interp_n2_scalar_2d(u_x, pos_x, pos_y, (0.0, 0.5), grid.dx, grid.shape)
    vel_y = _interp_n2_scalar_2d(u_y, pos_x, pos_y, (0.5, 0.0), grid.dx, grid.shape)
    return vel_x, vel_y


def interp_mac_2d_grad(
    grid,
    u_x: np.ndarray,
    u_y: np.ndarray,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Interpolate a 2D MAC velocity field AND its gradient at arbitrary positions.

    """
    vx, dvx_dx, dvx_dy = _interp_n2_scalar_2d_grad(
        u_x, pos_x, pos_y, (0.0, 0.5), grid.dx, grid.shape
    )
    vy, dvy_dx, dvy_dy = _interp_n2_scalar_2d_grad(
        u_y, pos_x, pos_y, (0.5, 0.0), grid.dx, grid.shape
    )

    # Build the Jacobian tensor: grad[i, j] = du_i / dx_j
    grad = np.array([[dvx_dx, dvx_dy],
                     [dvy_dx, dvy_dy]])

    return (vx, vy), grad


def interp_face_2d(
    grid,
    field: np.ndarray,
    axis: int,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
) -> np.ndarray:
    """
    Interpolate a single face-stored scalar field at arbitrary positions.

    """
    offsets = [(0.0, 0.5), (0.5, 0.0)]
    return _interp_n2_scalar_2d(field, pos_x, pos_y, offsets[axis], grid.dx, grid.shape)


# 3D Interpolation Routines
def _interp_n2_scalar_3d(
    field: np.ndarray,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    pos_z: np.ndarray,
    field_offset: Tuple[float, float, float],
    dx: float,
) -> np.ndarray:
    """
    Interpolate a 3D scalar field using N2 B-splines over a 3x3x3 stencil.

    """
    inv_dx = 1.0 / dx
    eps = 1e-4

    ix = pos_x * inv_dx - field_offset[0]
    iy = pos_y * inv_dx - field_offset[1]
    iz = pos_z * inv_dx - field_offset[2]

    ix = np.clip(ix, 0.5 + eps, field.shape[0] - 1.5 - eps)
    iy = np.clip(iy, 0.5 + eps, field.shape[1] - 1.5 - eps)
    iz = np.clip(iz, 0.5 + eps, field.shape[2] - 1.5 - eps)

    base_i = np.floor(ix - 0.5).astype(int)
    base_j = np.floor(iy - 0.5).astype(int)
    base_k = np.floor(iz - 0.5).astype(int)

    result = np.zeros_like(pos_x)
    for di in range(3):
        wi = n2(ix - (base_i + di))
        si = np.clip(base_i + di, 0, field.shape[0] - 1)
        for dj in range(3):
            wj = n2(iy - (base_j + dj))
            sj = np.clip(base_j + dj, 0, field.shape[1] - 1)
            for dk in range(3):
                wk = n2(iz - (base_k + dk))
                sk = np.clip(base_k + dk, 0, field.shape[2] - 1)
                result += field[si, sj, sk] * wi * wj * wk

    return result


def _interp_n2_scalar_3d_grad(
    field: np.ndarray,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    pos_z: np.ndarray,
    field_offset: Tuple[float, float, float],
    dx: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate a 3D scalar field and its gradient using N2 B-splines.

    """
    inv_dx = 1.0 / dx
    eps = 1e-4

    ix = pos_x * inv_dx - field_offset[0]
    iy = pos_y * inv_dx - field_offset[1]
    iz = pos_z * inv_dx - field_offset[2]

    ix = np.clip(ix, 0.5 + eps, field.shape[0] - 1.5 - eps)
    iy = np.clip(iy, 0.5 + eps, field.shape[1] - 1.5 - eps)
    iz = np.clip(iz, 0.5 + eps, field.shape[2] - 1.5 - eps)

    base_i = np.floor(ix - 0.5).astype(int)
    base_j = np.floor(iy - 0.5).astype(int)
    base_k = np.floor(iz - 0.5).astype(int)

    value = np.zeros_like(pos_x)
    gx = np.zeros_like(pos_x)
    gy = np.zeros_like(pos_x)
    gz = np.zeros_like(pos_x)

    for di in range(3):
        wi = n2(ix - (base_i + di))
        dwi = dn2(ix - (base_i + di))
        si = np.clip(base_i + di, 0, field.shape[0] - 1)
        for dj in range(3):
            wj = n2(iy - (base_j + dj))
            dwj = dn2(iy - (base_j + dj))
            sj = np.clip(base_j + dj, 0, field.shape[1] - 1)
            for dk in range(3):
                wk = n2(iz - (base_k + dk))
                dwk = dn2(iz - (base_k + dk))
                sk = np.clip(base_k + dk, 0, field.shape[2] - 1)
                val = field[si, sj, sk]
                value += val * wi * wj * wk
                gx += val * dwi * wj * wk
                gy += val * wi * dwj * wk
                gz += val * wi * wj * dwk

    gx *= inv_dx
    gy *= inv_dx
    gz *= inv_dx
    return value, gx, gy, gz


def interp_mac_3d(
    grid,
    u_x: np.ndarray,
    u_y: np.ndarray,
    u_z: np.ndarray,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    pos_z: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate a 3D MAC velocity field at arbitrary positions.

    """
    vel_x = _interp_n2_scalar_3d(u_x, pos_x, pos_y, pos_z, (0.0, 0.5, 0.5), grid.dx)
    vel_y = _interp_n2_scalar_3d(u_y, pos_x, pos_y, pos_z, (0.5, 0.0, 0.5), grid.dx)
    vel_z = _interp_n2_scalar_3d(u_z, pos_x, pos_y, pos_z, (0.5, 0.5, 0.0), grid.dx)
    return vel_x, vel_y, vel_z


def interp_mac_3d_grad(
    grid,
    u_x: np.ndarray,
    u_y: np.ndarray,
    u_z: np.ndarray,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    pos_z: np.ndarray,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """
    Interpolate a 3D MAC velocity field AND its 3x3 Jacobian at arbitrary positions.

    """
    vx, dvx_dx, dvx_dy, dvx_dz = _interp_n2_scalar_3d_grad(
        u_x, pos_x, pos_y, pos_z, (0.0, 0.5, 0.5), grid.dx
    )
    vy, dvy_dx, dvy_dy, dvy_dz = _interp_n2_scalar_3d_grad(
        u_y, pos_x, pos_y, pos_z, (0.5, 0.0, 0.5), grid.dx
    )
    vz, dvz_dx, dvz_dy, dvz_dz = _interp_n2_scalar_3d_grad(
        u_z, pos_x, pos_y, pos_z, (0.5, 0.5, 0.0), grid.dx
    )

    grad = np.array([
        [dvx_dx, dvx_dy, dvx_dz],
        [dvy_dx, dvy_dy, dvy_dz],
        [dvz_dx, dvz_dy, dvz_dz],
    ])

    return (vx, vy, vz), grad


def interp_center_2d(
    grid,
    field: np.ndarray,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
) -> np.ndarray:
    """
    Interpolate a cell-centered 2D scalar field at arbitrary positions.

    """
    return _interp_n2_scalar_2d(field, pos_x, pos_y, (0.5, 0.5), grid.dx, grid.shape)


def interp_center_3d(
    grid,
    field: np.ndarray,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    pos_z: np.ndarray,
) -> np.ndarray:
    """
    Interpolate a cell-centered 3D scalar field at arbitrary positions.

    """
    return _interp_n2_scalar_3d(
        field, pos_x, pos_y, pos_z, (0.5, 0.5, 0.5), grid.dx
    )
