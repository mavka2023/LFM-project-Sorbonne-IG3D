"""
MAC Staggered Grid.

Provides the MAC grid data structures for 2D and 3D incompressible fluid simulations. 
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class MACGrid2D:
    """
    2D MAC staggered grid representation.
    
    Attributes:
        nx (int): Cell count in x-direction.
        ny (int): Cell count in y-direction.
        dx (float): Uniform cell spacing.
        origin (Tuple[float, float]): World-space origin (bottom-left).
    """
    nx: int
    ny: int
    dx: float = 1.0
    origin: Tuple[float, float] = (0.0, 0.0)

    @property
    def ndim(self) -> int:
        return 2

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.nx, self.ny)

    @property
    def domain_size(self) -> Tuple[float, float]:
        return (self.nx * self.dx, self.ny * self.dx)

    def face_shape(self, axis: int) -> Tuple[int, int]:
        """Returns the shape of the face-centered grid for a given axis (0=x, 1=y)."""
        if axis == 0:
            return (self.nx + 1, self.ny)
        elif axis == 1:
            return (self.nx, self.ny + 1)
        else:
            raise ValueError(f"Invalid axis {axis} for 2D grid")

    def zeros_faces(self, axis: int) -> np.ndarray:
        """Allocates a zero-initialized array for face-centered data."""
        return np.zeros(self.face_shape(axis), dtype=np.float64)

    def zeros_centers(self) -> np.ndarray:
        """Allocates a zero-initialized array for cell-centered data."""
        return np.zeros(self.shape, dtype=np.float64)

    def face_positions(self, axis: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes world-space coordinates of face centers for the specified axis.
        """
        shape = self.face_shape(axis)
        if axis == 0:
            ix = np.arange(shape[0])
            iy = np.arange(shape[1])
            px, py = np.meshgrid(ix * self.dx, iy * self.dx + 0.5 * self.dx, indexing='ij')
        else:
            ix = np.arange(shape[0])
            iy = np.arange(shape[1])
            px, py = np.meshgrid(ix * self.dx + 0.5 * self.dx, iy * self.dx, indexing='ij')
        return px + self.origin[0], py + self.origin[1]

    def cell_center_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes world-space coordinates of cell centers.
        """
        ix = np.arange(self.nx)
        iy = np.arange(self.ny)
        px, py = np.meshgrid(
            ix * self.dx + 0.5 * self.dx,
            iy * self.dx + 0.5 * self.dx,
            indexing='ij'
        )
        return px + self.origin[0], py + self.origin[1]


@dataclass
class MACGrid3D:
    """
    3D MAC staggered grid representation.
    
    Attributes:
        nx (int): Cell count in x-direction.
        ny (int): Cell count in y-direction.
        nz (int): Cell count in z-direction.
        dx (float): Uniform cell spacing.
        origin (Tuple[float, float, float]): World-space origin.
    """
    nx: int
    ny: int
    nz: int
    dx: float = 1.0
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def ndim(self) -> int:
        return 3

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)

    @property
    def domain_size(self) -> Tuple[float, float, float]:
        return (self.nx * self.dx, self.ny * self.dx, self.nz * self.dx)

    def face_shape(self, axis: int) -> Tuple[int, int, int]:
        """Returns the shape of the face-centered grid for a given axis (0=x, 1=y, 2=z)."""
        s = list(self.shape)
        s[axis] += 1
        return tuple(s)

    def zeros_faces(self, axis: int) -> np.ndarray:
        """Allocates a zero-initialized array for face-centered data."""
        return np.zeros(self.face_shape(axis), dtype=np.float64)

    def zeros_centers(self) -> np.ndarray:
        """Allocates a zero-initialized array for cell-centered data."""
        return np.zeros(self.shape, dtype=np.float64)

    def face_positions(self, axis: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes world-space coordinates of face centers for the specified axis.
        """
        shape = self.face_shape(axis)
        offsets = [0.5, 0.5, 0.5]
        offsets[axis] = 0.0

        arrays = []
        for d in range(3):
            arrays.append(np.arange(shape[d]) * self.dx + offsets[d] * self.dx)

        grids = np.meshgrid(*arrays, indexing='ij')
        return (
            grids[0] + self.origin[0],
            grids[1] + self.origin[1],
            grids[2] + self.origin[2],
        )

    def cell_center_positions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes world-space coordinates of cell centers.
        """
        arrays = []
        for d, n in enumerate(self.shape):
            arrays.append(np.arange(n) * self.dx + 0.5 * self.dx)

        grids = np.meshgrid(*arrays, indexing='ij')
        return (
            grids[0] + self.origin[0],
            grids[1] + self.origin[1],
            grids[2] + self.origin[2],
        )