"""
Leapfrog Flow Maps (LFM)
Python Implementation of the Leapfrog Flow Maps algorithm for
incompressible fluid simulation, based on the paper:

    "Leapfrog Flow Maps for Real-Time Fluid Simulation"
    Yuchen Sun et al., ACM Trans. Graph., August 2025.

This package implements the core LFM algorithm including:
  - MAC staggered grid for 2D and 3D domains
  - Quadratic B-spline (N2) interpolation on staggered grids
  - RK2/RK4 flow map marching with Jacobian tracking
  - Leapfrog time integration for midpoint velocity computation
  - Impulse-based pullback with back-and-forth error compensation
  - Pressure Poisson projection via SciPy sparse CG solver
  - Viscosity via path integrals along forward flow maps
  - External force support
"""

__version__ = "1.0.0"

from .grid import MACGrid2D, MACGrid3D
from .lfm_solver import LFMSolver
