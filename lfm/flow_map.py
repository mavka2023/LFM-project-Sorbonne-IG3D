"""
Flow Map Marching 
Implements the bidirectional flow map tracking that gives LFM its
ability to preserve vortical structures. A flow map Φ_{a,b} maps
initial positions to final positions over the interval [a,b], and
its Jacobian F_{a,b} tracks how infinitesimal volumes deform.

"""

import numpy as np
from typing import Tuple, Optional, List
from .grid import MACGrid2D, MACGrid3D
from .interpolation import (
    interp_mac_2d, interp_mac_2d_grad,
    interp_mac_3d, interp_mac_3d_grad,
)
class FlowMap2D:
    """
    Bidirectional flow maps for a 2D MAC staggered grid.

    """

    def __init__(self, grid: MACGrid2D):
        """Initialize flow maps to identity mapping."""
        self.grid = grid
        self.ndim = 2
        self.psi = [None, None]
        self.T = [None, None]
        self.reset()

    def reset(self):
        """Reset flow maps to identity: Ψ(x) = x, T = I (axis unit vector)."""
        for axis in range(2):
            pos_x, pos_y = self.grid.face_positions(axis)
            self.psi[axis] = np.stack([pos_x, pos_y], axis=-1)
            shape = self.grid.face_shape(axis)
            self.T[axis] = np.zeros(shape + (2,), dtype=np.float64)
            self.T[axis][..., axis] = 1.0


def rk4_march_2d(
    flow_map: FlowMap2D,
    grid: MACGrid2D,
    u_x: np.ndarray,
    u_y: np.ndarray,
    dt: float,
):
    """
    RK4 march the 2D flow map by one time step.

    """
    for axis in range(2):
        psi = flow_map.psi[axis]
        T = flow_map.T[axis]  
        # RK4 stage 1  
        px, py = psi[..., 0], psi[..., 1]
        (vx1, vy1), grad1 = interp_mac_2d_grad(grid, u_x, u_y, px, py)
    
        kT1_0 = grad1[0, 0] * T[..., 0] + grad1[0, 1] * T[..., 1]
        kT1_1 = grad1[1, 0] * T[..., 0] + grad1[1, 1] * T[..., 1]

        half_dt = 0.5 * dt

        # RK4 stage 2
        px2 = px - half_dt * vx1
        py2 = py - half_dt * vy1
        T2_0 = T[..., 0] - half_dt * kT1_0
        T2_1 = T[..., 1] - half_dt * kT1_1
        (vx2, vy2), grad2 = interp_mac_2d_grad(grid, u_x, u_y, px2, py2)
        kT2_0 = grad2[0, 0] * T2_0 + grad2[0, 1] * T2_1
        kT2_1 = grad2[1, 0] * T2_0 + grad2[1, 1] * T2_1

        # RK4 stage 3
        px3 = px - half_dt * vx2
        py3 = py - half_dt * vy2
        T3_0 = T[..., 0] - half_dt * kT2_0
        T3_1 = T[..., 1] - half_dt * kT2_1
        (vx3, vy3), grad3 = interp_mac_2d_grad(grid, u_x, u_y, px3, py3)
        kT3_0 = grad3[0, 0] * T3_0 + grad3[0, 1] * T3_1
        kT3_1 = grad3[1, 0] * T3_0 + grad3[1, 1] * T3_1

        # RK4 stage 4
        px4 = px - dt * vx3
        py4 = py - dt * vy3
        T4_0 = T[..., 0] - dt * kT3_0
        T4_1 = T[..., 1] - dt * kT3_1
        (vx4, vy4), grad4 = interp_mac_2d_grad(grid, u_x, u_y, px4, py4)
        kT4_0 = grad4[0, 0] * T4_0 + grad4[0, 1] * T4_1
        kT4_1 = grad4[1, 0] * T4_0 + grad4[1, 1] * T4_1

        # Combine
        sixth_dt = dt / 6.0
        flow_map.psi[axis][..., 0] = px - sixth_dt * (vx1 + 2*vx2 + 2*vx3 + vx4)
        flow_map.psi[axis][..., 1] = py - sixth_dt * (vy1 + 2*vy2 + 2*vy3 + vy4)
        flow_map.T[axis][..., 0] = T[..., 0] - sixth_dt * (kT1_0 + 2*kT2_0 + 2*kT3_0 + kT4_0)
        flow_map.T[axis][..., 1] = T[..., 1] - sixth_dt * (kT1_1 + 2*kT2_1 + 2*kT3_1 + kT4_1)


def pullback_2d(
    grid: MACGrid2D,
    flow_map: FlowMap2D,
    src_u_x: np.ndarray,
    src_u_y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pullback operation: reconstruct impulse from initial velocity using flow maps.

    """
    results = []
    for axis in range(2):
        psi = flow_map.psi[axis]
        T = flow_map.T[axis]
        px, py = psi[..., 0], psi[..., 1]
        vx, vy = interp_mac_2d(grid, src_u_x, src_u_y, px, py)
        result = T[..., 0] * vx + T[..., 1] * vy
        results.append(result)
    return results[0], results[1]


class FlowMap3D:
    """
    Bidirectional flow maps for a 3D MAC staggered grid.

    """

    def __init__(self, grid: MACGrid3D):
        self.grid = grid
        self.ndim = 3
        self.psi = [None, None, None]
        self.T = [None, None, None]
        self.reset()

    def reset(self):
        """Reset to identity mapping."""
        for axis in range(3):
            pos = self.grid.face_positions(axis)
            self.psi[axis] = np.stack(pos, axis=-1)
            shape = self.grid.face_shape(axis)
            self.T[axis] = np.zeros(shape + (3,), dtype=np.float64)
            self.T[axis][..., axis] = 1.0


def rk4_march_3d(
    flow_map: FlowMap3D,
    grid: MACGrid3D,
    u_x: np.ndarray,
    u_y: np.ndarray,
    u_z: np.ndarray,
    dt: float,
):
    """
    RK4 march the 3D flow map by one time step. Updates in-place.

    """
    for axis in range(3):
        psi = flow_map.psi[axis]
        T = flow_map.T[axis]

        px, py, pz = psi[..., 0], psi[..., 1], psi[..., 2]

        # Stage 1
        (vx1, vy1, vz1), g1 = interp_mac_3d_grad(grid, u_x, u_y, u_z, px, py, pz)
        kT1 = np.stack([
            g1[0, 0]*T[..., 0] + g1[0, 1]*T[..., 1] + g1[0, 2]*T[..., 2],
            g1[1, 0]*T[..., 0] + g1[1, 1]*T[..., 1] + g1[1, 2]*T[..., 2],
            g1[2, 0]*T[..., 0] + g1[2, 1]*T[..., 1] + g1[2, 2]*T[..., 2],
        ], axis=-1)

        h = 0.5 * dt
        # Stage 2
        p2 = psi - h * np.stack([vx1, vy1, vz1], axis=-1)
        T2 = T - h * kT1
        (vx2, vy2, vz2), g2 = interp_mac_3d_grad(grid, u_x, u_y, u_z,
                                                    p2[..., 0], p2[..., 1], p2[..., 2])
        kT2 = np.stack([
            g2[0, 0]*T2[..., 0] + g2[0, 1]*T2[..., 1] + g2[0, 2]*T2[..., 2],
            g2[1, 0]*T2[..., 0] + g2[1, 1]*T2[..., 1] + g2[1, 2]*T2[..., 2],
            g2[2, 0]*T2[..., 0] + g2[2, 1]*T2[..., 1] + g2[2, 2]*T2[..., 2],
        ], axis=-1)

        # Stage 3
        p3 = psi - h * np.stack([vx2, vy2, vz2], axis=-1)
        T3 = T - h * kT2
        (vx3, vy3, vz3), g3 = interp_mac_3d_grad(grid, u_x, u_y, u_z,
                                                    p3[..., 0], p3[..., 1], p3[..., 2])
        kT3 = np.stack([
            g3[0, 0]*T3[..., 0] + g3[0, 1]*T3[..., 1] + g3[0, 2]*T3[..., 2],
            g3[1, 0]*T3[..., 0] + g3[1, 1]*T3[..., 1] + g3[1, 2]*T3[..., 2],
            g3[2, 0]*T3[..., 0] + g3[2, 1]*T3[..., 1] + g3[2, 2]*T3[..., 2],
        ], axis=-1)

        # Stage 4
        p4 = psi - dt * np.stack([vx3, vy3, vz3], axis=-1)
        T4 = T - dt * kT3
        (vx4, vy4, vz4), g4 = interp_mac_3d_grad(grid, u_x, u_y, u_z,
                                                    p4[..., 0], p4[..., 1], p4[..., 2])
        kT4 = np.stack([
            g4[0, 0]*T4[..., 0] + g4[0, 1]*T4[..., 1] + g4[0, 2]*T4[..., 2],
            g4[1, 0]*T4[..., 0] + g4[1, 1]*T4[..., 1] + g4[1, 2]*T4[..., 2],
            g4[2, 0]*T4[..., 0] + g4[2, 1]*T4[..., 1] + g4[2, 2]*T4[..., 2],
        ], axis=-1)

        # Combine
        s = dt / 6.0
        vel_avg = np.stack([
            vx1 + 2*vx2 + 2*vx3 + vx4,
            vy1 + 2*vy2 + 2*vy3 + vy4,
            vz1 + 2*vz2 + 2*vz3 + vz4,
        ], axis=-1)
        flow_map.psi[axis] = psi - s * vel_avg
        flow_map.T[axis] = T - s * (kT1 + 2*kT2 + 2*kT3 + kT4)


def pullback_3d(
    grid: MACGrid3D,
    flow_map: FlowMap3D,
    src_u_x: np.ndarray,
    src_u_y: np.ndarray,
    src_u_z: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    3D pullback: m_a(x) = T_a · u(Ψ_a(x)).

    """
    results = []
    for axis in range(3):
        psi = flow_map.psi[axis]
        T = flow_map.T[axis]
        px, py, pz = psi[..., 0], psi[..., 1], psi[..., 2]
        vx, vy, vz = interp_mac_3d(grid, src_u_x, src_u_y, src_u_z, px, py, pz)
        result = T[..., 0] * vx + T[..., 1] * vy + T[..., 2] * vz
        results.append(result)
    return results[0], results[1], results[2]
