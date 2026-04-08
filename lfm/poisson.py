"""
Pressure Poisson Solver 
Solves the Poisson equation  Δp = div(u)  for pressure, which is then
used to subtract ∇p from the velocity to make it divergence-free.

The original implementation uses a custom matrix-free AMGPCG solver on GPU.
It was replaced with SciPy's sparse CG solver with
Jacobi preconditioning. 
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, LinearOperator
from typing import Tuple, Optional
from .grid import MACGrid2D, MACGrid3D


def build_laplacian_2d(
    grid: MACGrid2D,
    is_bc_x: np.ndarray,
    is_bc_y: np.ndarray,
) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """
    Constructs the 2D sparse Laplacian matrix, incorporating Neumann boundary conditions.
    
    Faces marked as boundaries (is_bc=True) remove the connection between adjacent
    cells, effectively enforcing zero flux through solid walls.
    """
    nx, ny = grid.nx, grid.ny
    N = nx * ny

    def idx(i, j):
        """Convert (i,j) cell indices to linear index."""
        return i * ny + j

    # Build coefficient data for sparse matrix using COO format
    rows = []
    cols = []
    vals = []
    is_dof = np.zeros(N, dtype=bool)

    for i in range(nx):
        for j in range(ny):
            k = idx(i, j)
            diag = 0.0

            # x- neighbor: face at (i, j)
            if not is_bc_x[i, j] and i > 0:
                rows.append(k)
                cols.append(idx(i - 1, j))
                vals.append(-1.0)
                diag += 1.0

            # x+ neighbor: face at (i+1, j)
            if not is_bc_x[i + 1, j] and i < nx - 1:
                rows.append(k)
                cols.append(idx(i + 1, j))
                vals.append(-1.0)
                diag += 1.0

            # y- neighbor: face at (i, j)
            if not is_bc_y[i, j] and j > 0:
                rows.append(k)
                cols.append(idx(i, j - 1))
                vals.append(-1.0)
                diag += 1.0

            # y+ neighbor: face at (i, j+1)
            if not is_bc_y[i, j + 1] and j < ny - 1:
                rows.append(k)
                cols.append(idx(i, j + 1))
                vals.append(-1.0)
                diag += 1.0

            if is_bc_x[i, j]:
                pass 
            elif i == 0:
                diag += 1.0  # Neumann: gradient = 0 at boundary
            if is_bc_x[i + 1, j]:
                pass
            elif i == nx - 1:
                diag += 1.0

            if is_bc_y[i, j]:
                pass
            elif j == 0:
                diag += 1.0
            if is_bc_y[i, j + 1]:
                pass
            elif j == ny - 1:
                diag += 1.0

            rows.append(k)
            cols.append(k)
            vals.append(diag)

            is_dof[k] = diag > 0

    A = sparse.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    return A, is_dof


def build_laplacian_3d(
    grid: MACGrid3D,
    is_bc_x: np.ndarray,
    is_bc_y: np.ndarray,
    is_bc_z: np.ndarray,
) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """
    Build the 3D Laplacian matrix for the Poisson equation.

    Same logic as 2D but extended to 6-connected stencil.
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    N = nx * ny * nz

    def idx(i, j, k):
        return i * ny * nz + j * nz + k

    rows = []
    cols = []
    vals = []
    is_dof = np.zeros(N, dtype=bool)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                linear = idx(i, j, k)
                diag = 0.0

                # x-
                if not is_bc_x[i, j, k]:
                    if i > 0:
                        rows.append(linear)
                        cols.append(idx(i-1, j, k))
                        vals.append(-1.0)
                    diag += 1.0

                # x+
                if not is_bc_x[i+1, j, k]:
                    if i < nx-1:
                        rows.append(linear)
                        cols.append(idx(i+1, j, k))
                        vals.append(-1.0)
                    diag += 1.0

                # y-
                if not is_bc_y[i, j, k]:
                    if j > 0:
                        rows.append(linear)
                        cols.append(idx(i, j-1, k))
                        vals.append(-1.0)
                    diag += 1.0

                # y+
                if not is_bc_y[i, j+1, k]:
                    if j < ny-1:
                        rows.append(linear)
                        cols.append(idx(i, j+1, k))
                        vals.append(-1.0)
                    diag += 1.0

                # z-
                if not is_bc_z[i, j, k]:
                    if k > 0:
                        rows.append(linear)
                        cols.append(idx(i, j, k-1))
                        vals.append(-1.0)
                    diag += 1.0

                # z+
                if not is_bc_z[i, j, k+1]:
                    if k < nz-1:
                        rows.append(linear)
                        cols.append(idx(i, j, k+1))
                        vals.append(-1.0)
                    diag += 1.0

                rows.append(linear)
                cols.append(linear)
                vals.append(diag)
                is_dof[linear] = diag > 0

    A = sparse.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    return A, is_dof


def solve_pressure_2d(
    grid: MACGrid2D,
    rhs: np.ndarray,
    A: sparse.csr_matrix,
    is_dof: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 500,
    pure_neumann: bool = True,
) -> np.ndarray:
    """
    Solve the pressure Poisson equation Ap = rhs using CG with Jacobi preconditioning.
    For pure Neumann problems, the solution is recentered to zero mean.
    """
    b = rhs.ravel()

    b[~is_dof] = 0.0

    diag = A.diagonal()
    diag_inv = np.where(np.abs(diag) > 1e-14, 1.0 / diag, 0.0)
    M = sparse.diags(diag_inv)

    x, info = cg(A, b, rtol=tol, maxiter=max_iter, M=M)

    if info != 0:
        print(f"[Poisson solver] CG did not converge: info={info}")

    # Recenter for pure Neumann
    if pure_neumann:
        n_dof = np.sum(is_dof)
        if n_dof > 0:
            x -= np.sum(x[is_dof]) / n_dof

    return x.reshape(grid.nx, grid.ny)


def solve_pressure_3d(
    grid: MACGrid3D,
    rhs: np.ndarray,
    A: sparse.csr_matrix,
    is_dof: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 500,
    pure_neumann: bool = True,
) -> np.ndarray:
    """
    Solve the 3D pressure Poisson equation.
    """
    b = rhs.ravel()
    b[~is_dof] = 0.0

    diag = A.diagonal()
    diag_inv = np.where(np.abs(diag) > 1e-14, 1.0 / diag, 0.0)
    M = sparse.diags(diag_inv)

    x, info = cg(A, b, rtol=tol, maxiter=max_iter, M=M)

    if info != 0:
        print(f"[Poisson solver] CG did not converge: info={info}")

    if pure_neumann:
        n_dof = np.sum(is_dof)
        if n_dof > 0:
            x -= np.sum(x[is_dof]) / n_dof

    return x.reshape(grid.nx, grid.ny, grid.nz)
