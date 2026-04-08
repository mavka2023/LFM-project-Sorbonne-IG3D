"""
2D Visualization Utilities for LFM Fluid Simulations

Provides matplotlib-based plotting functions.

All plots follow the paper's visual style with diverging colormaps for
vorticity and coolwarm for scalars.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import TwoSlopeNorm
from typing import Optional, Tuple, List
import os


def plot_vorticity(
    vorticity: np.ndarray,
    dx: float = 1.0,
    title: str = "Vorticity",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "RdBu_r",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot a 2D vorticity field as a heatmap.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if vmin is None or vmax is None:
        abs_max = np.max(np.abs(vorticity))
        vmin = -abs_max
        vmax = abs_max

    im = ax.imshow(
        vorticity.T, origin='lower', cmap=cmap,
        vmin=vmin, vmax=vmax,
        extent=[0, vorticity.shape[0] * dx, 0, vorticity.shape[1] * dx],
        aspect='equal', interpolation='bilinear',
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="ω")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def plot_velocity_magnitude(
    u_x: np.ndarray,
    u_y: np.ndarray,
    dx: float = 1.0,
    title: str = "Velocity Magnitude",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot the velocity magnitude |u| as a heatmap.

    """
    ux_c = (u_x[:-1, :] + u_x[1:, :]) / 2.0
    uy_c = (u_y[:, :-1] + u_y[:, 1:]) / 2.0
    mag = np.sqrt(ux_c**2 + uy_c**2)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(
        mag.T, origin='lower', cmap='viridis',
        extent=[0, mag.shape[0] * dx, 0, mag.shape[1] * dx],
        aspect='equal', interpolation='bilinear',
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="|u|")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def plot_quiver(
    u_x: np.ndarray,
    u_y: np.ndarray,
    dx: float = 1.0,
    stride: int = 4,
    title: str = "Velocity Field",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot a velocity quiver plot.

    """
    ux_c = (u_x[:-1, :] + u_x[1:, :]) / 2.0
    uy_c = (u_y[:, :-1] + u_y[:, 1:]) / 2.0
    nx, ny = ux_c.shape

    x = np.arange(nx) * dx + 0.5 * dx
    y = np.arange(ny) * dx + 0.5 * dx
    X, Y = np.meshgrid(x, y, indexing='ij')

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    mag = np.sqrt(ux_c**2 + uy_c**2)
    ax.quiver(
        X[::stride, ::stride], Y[::stride, ::stride],
        ux_c[::stride, ::stride], uy_c[::stride, ::stride],
        mag[::stride, ::stride], cmap='coolwarm', scale=None,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect('equal')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def create_animation(
    frames: List[np.ndarray],
    dx: float = 1.0,
    fps: int = 30,
    title: str = "Vorticity Evolution",
    cmap: str = "RdBu_r",
    save_path: str = "simulation.mp4",
    figsize: Tuple[int, int] = (8, 6),
):
    """
    Create an animation from a sequence of vorticity frames.

    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    abs_max = max(np.max(np.abs(f)) for f in frames if f.size > 0)
    if abs_max == 0:
        abs_max = 1.0

    im = ax.imshow(
        frames[0].T, origin='lower', cmap=cmap,
        vmin=-abs_max, vmax=abs_max,
        extent=[0, frames[0].shape[0] * dx, 0, frames[0].shape[1] * dx],
        aspect='equal', interpolation='bilinear',
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title_text = ax.set_title(f"{title} — Frame 0")
    plt.colorbar(im, ax=ax, label="ω")

    def update(frame_idx):
        im.set_data(frames[frame_idx].T)
        title_text.set_text(f"{title} — Frame {frame_idx}")
        return [im, title_text]

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 // fps, blit=True,
    )

    if save_path.endswith('.gif'):
        anim.save(save_path, writer='pillow', fps=fps)
    else:
        anim.save(save_path, writer='ffmpeg', fps=fps)

    plt.close()
    print(f"Animation saved to {save_path}")


def plot_scalar_field(
    field: np.ndarray,
    dx: float = 1.0,
    title: str = "Scalar Field",
    cmap: str = "inferno",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot a cell-centered scalar field (smoke density, temperature, etc.).

    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(
        field.T, origin='lower', cmap=cmap,
        vmin=vmin, vmax=vmax,
        extent=[0, field.shape[0] * dx, 0, field.shape[1] * dx],
        aspect='equal', interpolation='bilinear',
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
