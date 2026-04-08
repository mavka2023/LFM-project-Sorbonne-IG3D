"""
Leapfrog Flow Maps (LFM) Solver — Main Algorithm Implementation.


The LFM solver uses a hybrid velocity-impulse scheme:
  - VELOCITY ADVECTION (cheap) for computing midpoint velocities during a
    reinitialization cycle, using the leapfrog method for second-order accuracy.
  - IMPULSE-BASED FLOW MAPS (accurate) at the end of each reinitialization
    cycle, using long-range backward maps with back-and-forth error compensation.

This combination reduces the computational workload of impulse-based methods
by ~76% while maintaining comparable vortex preservation.
"""

import numpy as np
from typing import Optional, Tuple, List, Callable
from .grid import MACGrid2D, MACGrid3D
from .boundary import BoundaryCondition2D, BoundaryCondition3D
from .advection import advect_rk2_2d, advect_rk2_3d, advect_center_rk2_2d, advect_center_rk2_3d
from .flow_map import (
    FlowMap2D, FlowMap3D,
    rk4_march_2d, rk4_march_3d,
    pullback_2d, pullback_3d,
)
from .projection import project_2d, project_3d


class LFMSolver:
    """
    Leapfrog Flow Maps solver for 2D and 3D incompressible fluid simulation.

    """

    def __init__(
        self,
        grid,
        reinit_every: int = 10,
        viscosity: float = 0.0,
        density: float = 1.0,
        use_bfecc_clamp: bool = True,
        poisson_tol: float = 1e-6,
        poisson_max_iter: int = 500,
    ):
        """
        Initialize the LFM solver.
        """
        self.grid = grid
        self.ndim = grid.ndim
        self.reinit_every = reinit_every
        self.viscosity = viscosity
        self.density = density
        self.use_bfecc_clamp = use_bfecc_clamp
        self.poisson_tol = poisson_tol
        self.poisson_max_iter = poisson_max_iter

        self.step_in_cycle = 0
        self.step_count = 0

        # Velocity fields
        if self.ndim == 2:
            self.u_x = grid.zeros_faces(0)
            self.u_y = grid.zeros_faces(1)
        else:
            self.u_x = grid.zeros_faces(0)
            self.u_y = grid.zeros_faces(1)
            self.u_z = grid.zeros_faces(2)

        self._init_u = None
        self._mid_u = []
        self.bc = None
        self._laplacian_cache = {}
        self._force_fn = None
        self._scalars = {}

    def set_initial_velocity(self, *velocity_components):
        """
        Set the initial velocity field.
        """
        if self.ndim == 2:
            self.u_x = velocity_components[0].copy()
            self.u_y = velocity_components[1].copy()
        else:
            self.u_x = velocity_components[0].copy()
            self.u_y = velocity_components[1].copy()
            self.u_z = velocity_components[2].copy()

    def set_boundary_conditions(self, bc):
        """
        Set boundary conditions. Invalidates the Laplacian cache.

        """
        self.bc = bc
        self._laplacian_cache = {}  # Force rebuild

    def set_force_function(self, fn: Callable):
        """
        Set an external force function.
        """
        self._force_fn = fn

    def add_scalar_field(self, name: str, initial_value: np.ndarray):
        """
        Register a scalar field for advection (e.g., smoke density).

        """
        self._scalars[name] = {
            'current': initial_value.copy(),
            'initial': initial_value.copy(), 
        }

    def get_scalar(self, name: str) -> np.ndarray:
        """Get the current value of a registered scalar field."""
        return self._scalars[name]['current']


    def _get_velocity(self):
        """Return velocity components as a tuple."""
        if self.ndim == 2:
            return (self.u_x, self.u_y)
        else:
            return (self.u_x, self.u_y, self.u_z)

    def _set_velocity(self, *components):
        """Set velocity components from a tuple."""
        self.u_x = components[0]
        self.u_y = components[1]
        if self.ndim == 3:
            self.u_z = components[2]

    def _copy_velocity(self, *components):
        """Copy velocity components and return as list."""
        return [c.copy() for c in components]

    def _project(self, u_x, u_y, u_z=None):
        """
        Run pressure projection in-place.

        """
        if self.ndim == 2:
            project_2d(self.grid, u_x, u_y, self.bc,
                       laplacian_cache=self._laplacian_cache,
                       tol=self.poisson_tol, max_iter=self.poisson_max_iter)
        else:
            project_3d(self.grid, u_x, u_y, u_z, self.bc,
                       laplacian_cache=self._laplacian_cache,
                       tol=self.poisson_tol, max_iter=self.poisson_max_iter)


    def step(self, dt: float):
        """
        Advance the simulation by one time step.
        """
        if self.bc is None:
            raise RuntimeError("Boundary conditions not set. Call set_boundary_conditions() first.")

        if self.step_in_cycle == 0:
            self._init_u = self._copy_velocity(*self._get_velocity())
            self._mid_u = []

        self._advance(dt)

        self.step_in_cycle += 1
        self.step_count += 1

        if self.step_in_cycle >= self.reinit_every:
            self._reinitialize(dt)
            self.step_in_cycle = 0

    def _advance(self, dt: float):
        """
        Compute the next midpoint velocity using leapfrog advection.
        """
        step = self.step_in_cycle

        if self.ndim == 2:
            self._advance_2d(dt, step)
        else:
            self._advance_3d(dt, step)

    def _advance_2d(self, dt: float, step: int):
        """2D leapfrog advection step."""
        if step == 0:
            # Bootstrap step 0: half-step advection
            mid_dt = 0.5 * dt
            src_x, src_y = self._init_u[0], self._init_u[1]
            adv_x, adv_y = src_x, src_y  # Advect using init velocity
        elif step == 1:
            # Bootstrap step 1: full-step advection
            mid_dt = dt
            src_x, src_y = self._mid_u[0][0], self._mid_u[0][1]
            adv_x, adv_y = self._mid_u[0][0], self._mid_u[0][1]
        else:
            # Leapfrog step : advect u_{k-3/2} by 2Δt using u_{k-1/2}
            mid_dt = 2.0 * dt
            src_x, src_y = self._mid_u[step - 2][0], self._mid_u[step - 2][1]
            adv_x, adv_y = self._mid_u[step - 1][0], self._mid_u[step - 1][1]

        # Semi-Lagrangian advection
        tmp_x = advect_rk2_2d(self.grid, src_x, adv_x, adv_y, mid_dt, axis=0)
        tmp_y = advect_rk2_2d(self.grid, src_y, adv_x, adv_y, mid_dt, axis=1)

        # Pressure projection
        self._project(tmp_x, tmp_y)
        self._mid_u.append((tmp_x.copy(), tmp_y.copy()))
        self.u_x = tmp_x
        self.u_y = tmp_y

    def _advance_3d(self, dt: float, step: int):
        """3D leapfrog advection step."""
        if step == 0:
            mid_dt = 0.5 * dt
            src = self._init_u
            adv = self._init_u
        elif step == 1:
            mid_dt = dt
            src = self._mid_u[0]
            adv = self._mid_u[0]
        else:
            mid_dt = 2.0 * dt
            src = self._mid_u[step - 2]
            adv = self._mid_u[step - 1]

        tmp_x = advect_rk2_3d(self.grid, src[0], adv[0], adv[1], adv[2], mid_dt, axis=0)
        tmp_y = advect_rk2_3d(self.grid, src[1], adv[0], adv[1], adv[2], mid_dt, axis=1)
        tmp_z = advect_rk2_3d(self.grid, src[2], adv[0], adv[1], adv[2], mid_dt, axis=2)

        self._project(tmp_x, tmp_y, tmp_z)

        self._mid_u.append((tmp_x.copy(), tmp_y.copy(), tmp_z.copy()))

        self.u_x = tmp_x
        self.u_y = tmp_y
        self.u_z = tmp_z

    def _reinitialize(self, dt: float):
        """
        Reinitialization at the end of a leapfrog cycle.

        """
        if self.ndim == 2:
            self._reinitialize_2d(dt)
        else:
            self._reinitialize_3d(dt)

    def _reinitialize_2d(self, dt: float):
        """2D reinitialization cycle."""
        grid = self.grid
        n = self.reinit_every

        psi = FlowMap2D(grid)  # Backward map: Ψ_{n,0}
        phi = FlowMap2D(grid)  # Forward map:  Φ_{0,n}

        for i in range(n - 1, -1, -1):
            mid_ux, mid_uy = self._mid_u[i]
            rk4_march_2d(psi, grid, mid_ux, mid_uy, dt)  # positive dt = backward

        for i in range(n):
            mid_ux, mid_uy = self._mid_u[i]
            rk4_march_2d(phi, grid, mid_ux, mid_uy, -dt)  # negative dt = forward

        init_ux, init_uy = self._init_u
        m_x, m_y = pullback_2d(grid, psi, init_ux, init_uy)
        u_hat_x, u_hat_y = pullback_2d(grid, phi, m_x, m_y)
        err_x = (u_hat_x - init_ux) / 2.0
        err_y = (u_hat_y - init_uy) / 2.0
        err_pb_x, err_pb_y = pullback_2d(grid, psi, err_x, err_y)
        m_x -= err_pb_x
        m_y -= err_pb_y

        if self.use_bfecc_clamp:
            m_x = self._bfecc_clamp_face(m_x, self.u_x)
            m_y = self._bfecc_clamp_face(m_y, self.u_y)

        self._project(m_x, m_y)
        self.u_x = m_x
        self.u_y = m_y
        self._reinit_scalars_2d(psi, phi)

    def _reinitialize_3d(self, dt: float):
        """3D reinitialization cycle."""
        grid = self.grid
        n = self.reinit_every

        psi = FlowMap3D(grid)
        phi = FlowMap3D(grid)

        # Backward march
        for i in range(n - 1, -1, -1):
            mid_ux, mid_uy, mid_uz = self._mid_u[i]
            rk4_march_3d(psi, grid, mid_ux, mid_uy, mid_uz, dt)

        # Forward march
        for i in range(n):
            mid_ux, mid_uy, mid_uz = self._mid_u[i]
            rk4_march_3d(phi, grid, mid_ux, mid_uy, mid_uz, -dt)

        # Pullback
        init_ux, init_uy, init_uz = self._init_u
        m_x, m_y, m_z = pullback_3d(grid, psi, init_ux, init_uy, init_uz)

        # Error estimation
        u_hat_x, u_hat_y, u_hat_z = pullback_3d(grid, phi, m_x, m_y, m_z)

        err_x = (u_hat_x - init_ux) / 2.0
        err_y = (u_hat_y - init_uy) / 2.0
        err_z = (u_hat_z - init_uz) / 2.0

        err_pb_x, err_pb_y, err_pb_z = pullback_3d(grid, psi, err_x, err_y, err_z)
        m_x -= err_pb_x
        m_y -= err_pb_y
        m_z -= err_pb_z

        if self.use_bfecc_clamp:
            m_x = self._bfecc_clamp_face(m_x, self.u_x)
            m_y = self._bfecc_clamp_face(m_y, self.u_y)
            m_z = self._bfecc_clamp_face(m_z, self.u_z)

        self._project(m_x, m_y, m_z)

        self.u_x = m_x
        self.u_y = m_y
        self.u_z = m_z

    def _bfecc_clamp_face(
        self,
        after_bfecc: np.ndarray,
        before_bfecc: np.ndarray,
    ) -> np.ndarray:
        """
        Back-and-Forth Error Compensation and Correction (BFECC) clamping.

        Clamps the BFECC-corrected values to the local min/max of the
        pre-correction field, preventing overshoot that can cause instability.
        """
        ndim = after_bfecc.ndim

        if ndim == 2:
            padded = np.pad(before_bfecc, 1, mode='edge')
            neighbors = [
                padded[:-2, 1:-1],  # left
                padded[2:, 1:-1],   # right
                padded[1:-1, :-2],  # bottom
                padded[1:-1, 2:],   # top
            ]
            local_min = np.minimum.reduce(neighbors)
            local_max = np.maximum.reduce(neighbors)
            return np.clip(after_bfecc, local_min, local_max)
        elif ndim == 3:
            padded = np.pad(before_bfecc, 1, mode='edge')
            neighbors = [
                padded[:-2, 1:-1, 1:-1],
                padded[2:, 1:-1, 1:-1],
                padded[1:-1, :-2, 1:-1],
                padded[1:-1, 2:, 1:-1],
                padded[1:-1, 1:-1, :-2],
                padded[1:-1, 1:-1, 2:],
            ]
            local_min = np.minimum.reduce(neighbors)
            local_max = np.maximum.reduce(neighbors)
            return np.clip(after_bfecc, local_min, local_max)
        else:
            return after_bfecc

    def _reinit_scalars_2d(self, psi: FlowMap2D, phi: FlowMap2D):
        """
        Advect registered scalar fields through flow maps during reinitialization.
        """
        from .interpolation import interp_center_2d

        for name, data in self._scalars.items():
            init_val = data['initial']

            psi_c = self._get_central_psi_2d(psi)

            px, py = psi_c[..., 0], psi_c[..., 1]
            advected = interp_center_2d(self.grid, init_val, px, py)

            phi_c = self._get_central_psi_2d(phi)
            fx, fy = phi_c[..., 0], phi_c[..., 1]
            round_trip = interp_center_2d(self.grid, advected, fx, fy)
            error = (round_trip - init_val) / 2.0

            err_pb = interp_center_2d(self.grid, error, px, py)
            result = advected - err_pb

            # Update
            data['current'] = result
            data['initial'] = result.copy()

    def _get_central_psi_2d(self, flow_map: FlowMap2D) -> np.ndarray:
        """
        Compute cell-centered positions from face-stored flow map.

        """
        grid = self.grid
        nx, ny = grid.nx, grid.ny
        psi_x_avg = (flow_map.psi[0][:-1, :, :] + flow_map.psi[0][1:, :, :]) / 2.0
        psi_y_avg = (flow_map.psi[1][:, :-1, :] + flow_map.psi[1][:, 1:, :]) / 2.0
        return (psi_x_avg + psi_y_avg) / 2.0

    def compute_vorticity_2d(self) -> np.ndarray:
        """
        Compute the vorticity ω = ∂v/∂x - ∂u/∂y for the current 2D velocity field.

        """
        dx = self.grid.dx
        nx, ny = self.grid.nx, self.grid.ny

        dvdx = (self.u_y[1:, 1:-1] - self.u_y[:-1, 1:-1]) / dx
        dudy = (self.u_x[1:-1, 1:] - self.u_x[1:-1, :-1]) / dx

        vort_interior = dvdx - dudy
        result = np.zeros((nx, ny), dtype=np.float64)
        result[:nx-1, :ny-1] = vort_interior
        return result

    def compute_kinetic_energy(self) -> float:
        """
        Compute the total kinetic energy  E = 0.5 * ρ * ∫|u|² dV.

        """
        dx = self.grid.dx
        if self.ndim == 2:
            ux_c = (self.u_x[:-1, :] + self.u_x[1:, :]) / 2.0
            uy_c = (self.u_y[:, :-1] + self.u_y[:, 1:]) / 2.0
            energy = 0.5 * self.density * np.sum(ux_c**2 + uy_c**2) * dx**2
        else:
            ux_c = (self.u_x[:-1, :, :] + self.u_x[1:, :, :]) / 2.0
            uy_c = (self.u_y[:, :-1, :] + self.u_y[:, 1:, :]) / 2.0
            uz_c = (self.u_z[:, :, :-1] + self.u_z[:, :, 1:]) / 2.0
            energy = 0.5 * self.density * np.sum(ux_c**2 + uy_c**2 + uz_c**2) * dx**3
        return energy

    def compute_max_divergence(self) -> float:
        """
        Compute the maximum absolute divergence of the current velocity field.

        """
        from .projection import calc_divergence_2d, calc_divergence_3d
        if self.ndim == 2:
            div = calc_divergence_2d(self.grid, self.u_x, self.u_y)
        else:
            div = calc_divergence_3d(self.grid, self.u_x, self.u_y, self.u_z)
        return np.max(np.abs(div))

    def compute_vorticity_3d(self) -> np.ndarray:
        """
        Compute vorticity magnitude |ω| = |∇ × u| for the 3D velocity field.

        """
        dx = self.grid.dx

        ux_c = (self.u_x[:-1, :, :] + self.u_x[1:, :, :]) / 2.0
        uy_c = (self.u_y[:, :-1, :] + self.u_y[:, 1:, :]) / 2.0
        uz_c = (self.u_z[:, :, :-1] + self.u_z[:, :, 1:]) / 2.0

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

