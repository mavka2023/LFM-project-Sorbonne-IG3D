"""
Boundary Condition Management for MAC Grids.

Handles boundary conditions for the LFM solver:
  - Wall BCs.
  - Inlet/outlet BCs.
  - Solid obstacle BCs.
"""

import numpy as np
from typing import Tuple, Optional
from .grid import MACGrid2D, MACGrid3D


class BoundaryCondition2D:
    """
    Stores per-face boolean masks (is_bc) and prescribed values (bc_val)
    for each face family (x-faces and y-faces).

    """

    def __init__(self, grid: MACGrid2D):
        self.grid = grid
        self.is_bc_x = np.zeros(grid.face_shape(0), dtype=bool)
        self.is_bc_y = np.zeros(grid.face_shape(1), dtype=bool)
        self.bc_val_x = np.zeros(grid.face_shape(0), dtype=np.float64)
        self.bc_val_y = np.zeros(grid.face_shape(1), dtype=np.float64)

    def set_wall_bc(
        self,
        x_neg: float = 0.0,
        x_pos: float = 0.0,
        y_neg: float = 0.0,
        y_pos: float = 0.0,
    ):
        """
        Set no-slip wall boundary conditions on domain edges.

        """
        nx, ny = self.grid.nx, self.grid.ny

        # Left and right x-face walls
        self.is_bc_x[0, :] = True
        self.bc_val_x[0, :] = x_neg
        self.is_bc_x[nx, :] = True
        self.bc_val_x[nx, :] = x_pos

        # Bottom and top y-face walls
        self.is_bc_y[:, 0] = True
        self.bc_val_y[:, 0] = y_neg
        self.is_bc_y[:, ny] = True
        self.bc_val_y[:, ny] = y_pos

    def set_obstacle_bc(self, sdf: np.ndarray):
        """
        Set no-slip boundary conditions from a signed distance field.
        Cells where sdf < 0 are considered solid. All faces touching
        a solid cell are marked as boundary with zero velocity.

        """
        nx, ny = self.grid.nx, self.grid.ny
        solid = sdf < 0.0

        for i in range(nx):
            for j in range(ny):
                if solid[i, j]:
                    # Mark all faces touching this solid cell
                    self.is_bc_x[i, j] = True
                    self.is_bc_x[i + 1, j] = True
                    self.is_bc_y[i, j] = True
                    self.is_bc_y[i, j + 1] = True
                    self.bc_val_x[i, j] = 0.0
                    self.bc_val_x[i + 1, j] = 0.0
                    self.bc_val_y[i, j] = 0.0
                    self.bc_val_y[i, j + 1] = 0.0

    def set_inlet(self, velocity: float, angle_deg: float = 0.0):
        """
        Set inlet velocity on domain walls with specified speed and angle.
        """
        angle_rad = np.radians(angle_deg)
        vx = velocity * np.cos(angle_rad)
        vy = velocity * np.sin(angle_rad)

        # Apply to x-walls
        self.bc_val_x[0, :] = vx
        self.bc_val_x[self.grid.nx, :] = vx

        # Apply to y-walls
        self.bc_val_y[:, 0] = vy
        self.bc_val_y[:, self.grid.ny] = vy

    def apply(self, u_x: np.ndarray, u_y: np.ndarray):
        """
        Enforce boundary conditions on the velocity field.

        """
        u_x[self.is_bc_x] = self.bc_val_x[self.is_bc_x]
        u_y[self.is_bc_y] = self.bc_val_y[self.is_bc_y]


class BoundaryCondition3D:
    """
    Boundary conditions for a 3D MAC grid.
    """

    def __init__(self, grid: MACGrid3D):
        self.grid = grid
        self.is_bc_x = np.zeros(grid.face_shape(0), dtype=bool)
        self.is_bc_y = np.zeros(grid.face_shape(1), dtype=bool)
        self.is_bc_z = np.zeros(grid.face_shape(2), dtype=bool)
        self.bc_val_x = np.zeros(grid.face_shape(0), dtype=np.float64)
        self.bc_val_y = np.zeros(grid.face_shape(1), dtype=np.float64)
        self.bc_val_z = np.zeros(grid.face_shape(2), dtype=np.float64)

    def set_wall_bc(
        self,
        x_neg: float = 0.0, x_pos: float = 0.0,
        y_neg: float = 0.0, y_pos: float = 0.0,
        z_neg: float = 0.0, z_pos: float = 0.0,
    ):
        """Set no-slip wall BCs on 6 domain faces."""
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz

        self.is_bc_x[0, :, :] = True
        self.bc_val_x[0, :, :] = x_neg
        self.is_bc_x[nx, :, :] = True
        self.bc_val_x[nx, :, :] = x_pos

        self.is_bc_y[:, 0, :] = True
        self.bc_val_y[:, 0, :] = y_neg
        self.is_bc_y[:, ny, :] = True
        self.bc_val_y[:, ny, :] = y_pos

        self.is_bc_z[:, :, 0] = True
        self.bc_val_z[:, :, 0] = z_neg
        self.is_bc_z[:, :, nz] = True
        self.bc_val_z[:, :, nz] = z_pos

    def set_obstacle_bc(self, sdf: np.ndarray):
        """
        Set no-slip BCs from a 3D signed distance field.

        """
        solid = sdf < 0.0
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if solid[i, j, k]:
                        self.is_bc_x[i, j, k] = True
                        self.is_bc_x[i+1, j, k] = True
                        self.is_bc_y[i, j, k] = True
                        self.is_bc_y[i, j+1, k] = True
                        self.is_bc_z[i, j, k] = True
                        self.is_bc_z[i, j, k+1] = True
                        self.bc_val_x[i, j, k] = 0.0
                        self.bc_val_x[i+1, j, k] = 0.0
                        self.bc_val_y[i, j, k] = 0.0
                        self.bc_val_y[i, j+1, k] = 0.0
                        self.bc_val_z[i, j, k] = 0.0
                        self.bc_val_z[i, j, k+1] = 0.0

    def set_inlet(self, velocity: float, angle_deg: float = 0.0):
        """Set inlet velocity on domain walls."""
        angle_rad = np.radians(angle_deg)
        vx = velocity * np.cos(angle_rad)
        vy = velocity * np.sin(angle_rad)

        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz

        self.bc_val_x[0, :, :] = vx
        self.bc_val_x[nx, :, :] = vx
        self.bc_val_y[:, 0, :] = vy
        self.bc_val_y[:, ny, :] = vy

    def apply(self, u_x: np.ndarray, u_y: np.ndarray, u_z: np.ndarray):
        """Enforce boundary conditions on velocity field in-place."""
        u_x[self.is_bc_x] = self.bc_val_x[self.is_bc_x]
        u_y[self.is_bc_y] = self.bc_val_y[self.is_bc_y]
        u_z[self.is_bc_z] = self.bc_val_z[self.is_bc_z]
