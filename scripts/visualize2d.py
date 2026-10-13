"""Plots one frame of a 2d BubbleML file: the temperature, the signed distance
function, the velocity magnitude, and the divergence of the velocity, with the
liquid-vapor interface (dfun = 0) drawn on every panel. The velocity is either the
face velocities, whose divergence comes from each cell's four faces, or the
cell-centered velocity, whose divergence comes from central differences."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from boiling_data.boiling_data import BoilingSimulation
from boiling_data.bubbleml import read_bubbleml, read_bubbleml_num_timesteps

SCALAR_FIELDS = ("temperature", "dfun")
# the x and y velocity fields of each kind of velocity
VELOCITY_FIELDS = {"faces": ("velfacex", "velfacey"), "cells": ("velx", "vely")}
# the divergence spikes at the interface would wash out the rest of the plot
DIVERGENCE_PERCENTILE = 99.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="a 2d BubbleML .hdf5 file")
    parser.add_argument(
        "--frame",
        type=int,
        default=-1,
        help="frame to plot. Defaults to last frame. negative indexes from end of sim",
    )
    parser.add_argument(
        "--velocity",
        choices=sorted(VELOCITY_FIELDS),
        default="faces",
        help="plot the face velocities (velfacex, velfacey) or the cell-centered "
        "velocity (velx, vely)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="PNG to write; <file name>_frame<frame>.png by default",
    )
    return parser.parse_args()


def read_frame(path: Path, frame: int, velocity: str) -> tuple[BoilingSimulation, int]:
    """The frame of the plotted fields, and its index counted from the start."""
    num_timesteps = read_bubbleml_num_timesteps(path)
    index = frame % num_timesteps if -num_timesteps <= frame < num_timesteps else None
    if index is None:
        raise SystemExit(f"{path} has frames 0 to {num_timesteps - 1}, not {frame}")
    simulation = read_bubbleml(
        path,
        field_names=[*SCALAR_FIELDS, *VELOCITY_FIELDS[velocity]],
        frames=slice(index, index + 1),
    )
    return simulation, index


def face_velocity_magnitude(simulation: BoilingSimulation) -> np.ndarray:
    """At the cell centers, from the average of each cell's two faces per axis."""
    velfacex = simulation.field("velfacex").data[0]
    velfacey = simulation.field("velfacey").data[0]
    velx = 0.5 * (velfacex[:, :-1] + velfacex[:, 1:])
    vely = 0.5 * (velfacey[:-1, :] + velfacey[1:, :])
    magnitude: np.ndarray = np.hypot(velx, vely)
    return magnitude


def cell_velocity_magnitude(simulation: BoilingSimulation) -> np.ndarray:
    magnitude: np.ndarray = np.hypot(
        simulation.field("velx").data[0], simulation.field("vely").data[0]
    )
    return magnitude


def face_divergence(simulation: BoilingSimulation) -> np.ndarray:
    """The divergence of every cell from the velocities on its four faces, with the
    cell widths taken from the face coordinates."""
    velfacex, velfacey = simulation.field("velfacex"), simulation.field("velfacey")
    dx = np.diff(velfacex.grid_x)
    dy = np.diff(velfacey.grid_y)
    divergence: np.ndarray = (
        np.diff(velfacex.data[0], axis=1) / dx[None, :]
        + np.diff(velfacey.data[0], axis=0) / dy[:, None]
    )
    return divergence


def cell_divergence(simulation: BoilingSimulation) -> np.ndarray:
    """The divergence of the cell-centered velocity by central differences between
    neighboring cells, one-sided at the walls."""
    velx, vely = simulation.field("velx"), simulation.field("vely")
    divergence: np.ndarray = np.gradient(
        velx.data[0], velx.grid_x, axis=1
    ) + np.gradient(vely.data[0], vely.grid_y, axis=0)
    return divergence


# the magnitude and the divergence of each kind of velocity, and its name in titles
VELOCITY_PLOTS = {
    "faces": (face_velocity_magnitude, face_divergence, "face velocity"),
    "cells": (cell_velocity_magnitude, cell_divergence, "cell-centered velocity"),
}


def plot_panel(
    axes: Axes,
    simulation: BoilingSimulation,
    values: np.ndarray,
    title: str,
    cmap: str,
    limit: float | None = None,
) -> None:
    """limit: colors symmetric about zero, from -limit to limit."""
    cells = simulation.field("temperature")
    image = axes.pcolormesh(
        cells.grid_x,
        cells.grid_y,
        values,
        shading="nearest",
        cmap=cmap,
        vmin=None if limit is None else -limit,
        vmax=limit,
    )
    axes.contour(
        cells.grid_x,
        cells.grid_y,
        simulation.field("dfun").data[0],
        levels=[0.0],
        colors="black",
        linewidths=0.6,
    )
    axes.set_title(title, fontsize=10)
    axes.set_aspect("equal")
    axes.figure.colorbar(image, ax=axes, fraction=0.05)


def plot_frame(simulation: BoilingSimulation, velocity: str, title: str) -> Figure:
    magnitude, divergence_of, velocity_name = VELOCITY_PLOTS[velocity]
    divergence = divergence_of(simulation)
    limit = float(np.percentile(np.abs(divergence), DIVERGENCE_PERCENTILE)) or 1.0
    dfun = simulation.field("dfun").data[0]
    dfun_limit = float(np.abs(dfun).max()) or 1.0

    figure, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    plot_panel(
        axes[0, 0],
        simulation,
        simulation.field("temperature").data[0],
        "temperature",
        cmap="inferno",
    )
    plot_panel(
        axes[0, 1],
        simulation,
        dfun,
        "signed distance (dfun)",
        cmap="RdBu_r",
        limit=dfun_limit,
    )
    plot_panel(
        axes[1, 0],
        simulation,
        magnitude(simulation),
        f"{velocity_name} magnitude",
        cmap="viridis",
    )
    plot_panel(
        axes[1, 1],
        simulation,
        divergence,
        f"divergence of {velocity_name}\n(clipped at {DIVERGENCE_PERCENTILE:g}th "
        f"percentile, max |div| {np.abs(divergence).max():.3g})",
        cmap="RdBu_r",
        limit=limit,
    )
    figure.suptitle(title)
    return figure


def main() -> None:
    args = parse_args()
    simulation, index = read_frame(args.path, args.frame, args.velocity)
    output = args.output or Path(f"{args.path.stem}_frame{index}.png")
    title = f"{args.path.name}, frame {index}, t = {simulation.time[0]:.4g}"
    figure = plot_frame(simulation, args.velocity, title)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    print(f"frame {index} -> {output}")


if __name__ == "__main__":
    main()
