"""Dynamic mode decomposition of fields of a BubbleML file, plotting the singular
value spectrum, the DMD eigenvalues, and the dominant modes of each field.

Besides stored fields, it decomposes the derived vorticity and the heat flux from a
heater into the liquid cells directly above it."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from boiling_data.boiling_data import Field
from boiling_data.bubbleml import read_bubbleml
from boiling_data.dmd import DMD, compute_dmd, uniform_timestep, vorticity
from boiling_data.flashx.heat_flux import (
    HEAT_FLUX_FIELDS,
    HeaterHeatFlux,
    heater_heat_flux,
)

VORTICITY = "vorticity"
HEAT_FLUX = "heatflux"
DERIVED_FROM = {VORTICITY: ("velx", "vely"), HEAT_FLUX: HEAT_FLUX_FIELDS}

type Target = Field | HeaterHeatFlux


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="BubbleML .hdf5 file")
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["dfun", "temperature", VORTICITY],
        help=f"stored fields, {VORTICITY!r} derived from velx and vely, or "
        f"{HEAT_FLUX!r} in W/m^2 from the heater into the liquid above it",
    )
    parser.add_argument(
        "--heater", type=int, default=0, help=f"index of the heater for {HEAT_FLUX!r}"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=20, help="SVD truncation rank")
    parser.add_argument(
        "--num-modes", type=int, default=6, help="dominant modes to plot per field"
    )
    parser.add_argument("--start", type=int, default=None, help="first frame")
    parser.add_argument("--stop", type=int, default=None, help="frame to stop before")
    parser.add_argument("--stride", type=int, default=1, help="use every nth frame")
    parser.add_argument(
        "--subtract-mean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="decompose fluctuations about the temporal mean",
    )
    return parser.parse_args()


def read_targets(
    path: Path, names: list[str], frames: slice, heater: int
) -> tuple[dict[str, Target], float]:
    stored = [
        stored_name for name in names for stored_name in DERIVED_FROM.get(name, (name,))
    ]
    simulation = read_bubbleml(path, list(dict.fromkeys(stored)), frames=frames)
    targets: dict[str, Target] = {}
    for name in names:
        if name == VORTICITY:
            targets[name] = vorticity(
                simulation.field("velx"), simulation.field("vely")
            )
        elif name == HEAT_FLUX:
            targets[name] = heater_heat_flux(simulation, heater)
        else:
            targets[name] = simulation.field(name)
    return targets, uniform_timestep(simulation.time)


def target_data(target: Target) -> np.ndarray:
    return target.data if isinstance(target, Field) else target.flux


def dominant_modes(dmd: DMD, count: int) -> list[int]:
    """Of each complex-conjugate pair only the positive-frequency mode is kept,
    since the pair sums to a single real oscillation."""
    order = [int(k) for k in dmd.order_by_amplitude() if dmd.eigenvalues[k].imag >= 0]
    return order[:count]


def plot_spectrum(name: str, dmd: DMD) -> Figure:
    figure, (singular_axes, eigen_axes) = plt.subplots(1, 2, figsize=(11, 4.5))
    rank = len(dmd.eigenvalues)
    energy = dmd.singular_values / dmd.singular_values[0]
    singular_axes.semilogy(energy, "o-", markersize=3)
    singular_axes.axvline(rank - 0.5, color="gray", linestyle="--", label="rank")
    singular_axes.set(
        xlabel="index", ylabel="sigma / sigma_0", title="snapshot singular values"
    )
    singular_axes.legend()

    angle = np.linspace(0, 2 * np.pi, 400)
    eigen_axes.plot(np.cos(angle), np.sin(angle), color="gray", linewidth=0.8)
    weight = np.abs(dmd.amplitudes)
    scatter = eigen_axes.scatter(
        dmd.eigenvalues.real,
        dmd.eigenvalues.imag,
        c=np.log10(weight + np.finfo(float).tiny),
        cmap="viridis",
    )
    figure.colorbar(scatter, ax=eigen_axes, label="log10 |b|")
    eigen_axes.set(
        xlabel="Re lambda", ylabel="Im lambda", title="DMD eigenvalues", aspect="equal"
    )
    figure.suptitle(name)
    figure.tight_layout()
    return figure


def plot_mode(axes: Axes, target: Target, mode: np.ndarray, title: str) -> None:
    axes.set_title(title, fontsize=9)
    if isinstance(target, HeaterHeatFlux):
        axes.plot(target.x, mode.real, label="real")
        axes.plot(target.x, np.abs(mode), color="gray", linestyle="--", label="|mode|")
        axes.axhline(0, color="black", linewidth=0.5)
        axes.set_xlabel("x")
        axes.legend(fontsize=7)
        return
    real = mode.real
    limit = np.abs(real).max() or 1.0
    image = axes.pcolormesh(
        target.grid_x,
        target.grid_y,
        real,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        shading="nearest",
    )
    axes.set_aspect("equal")
    axes.set_xticks([])
    axes.set_yticks([])
    axes.figure.colorbar(image, ax=axes, fraction=0.05)


def plot_modes(name: str, target: Target, dmd: DMD, count: int) -> Figure:
    indices = dominant_modes(dmd, count)
    width, height = (4, 3.5) if isinstance(target, HeaterHeatFlux) else (2.6, 6)
    figure, axes = plt.subplots(
        1,
        len(indices),
        figsize=(max(width * len(indices), 6), height),
        squeeze=False,
    )
    omega = dmd.continuous_eigenvalues
    for axis, k in zip(axes[0], indices, strict=True):
        title = (
            f"mode {k}\nf={dmd.frequencies[k]:.3g}  growth={omega[k].real:.2g}\n"
            f"|b|={abs(dmd.amplitudes[k]):.2g}"
        )
        plot_mode(axis, target, dmd.modes[k], title)
    figure.suptitle(f"{name}: real part of dominant DMD modes")
    figure.tight_layout()
    return figure


def plot_heat_flux(heat_flux: HeaterHeatFlux) -> Figure:
    figure, axes = plt.subplots(figsize=(7, 4.5))
    frames = np.arange(len(heat_flux.flux))
    image = axes.pcolormesh(
        heat_flux.x, frames, heat_flux.flux, cmap="inferno", shading="nearest"
    )
    figure.colorbar(image, ax=axes, label="heat flux (W/m^2)")
    axes.set(
        xlabel="x", ylabel="frame", title="heat flux into the liquid above the heater"
    )
    figure.tight_layout()
    return figure


def relative_error(data: np.ndarray, dmd: DMD) -> float:
    reconstruction = dmd.reconstruct(len(data))
    return float(np.linalg.norm(reconstruction - data) / np.linalg.norm(data))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = slice(args.start, args.stop, args.stride)
    targets, dt = read_targets(args.path, args.fields, frames, args.heater)
    for name, target in targets.items():
        data = target_data(target)
        dmd = compute_dmd(data, dt, args.rank, subtract_mean=args.subtract_mean)
        figures = {
            "spectrum": plot_spectrum(name, dmd),
            "modes": plot_modes(name, target, dmd, args.num_modes),
        }
        if isinstance(target, HeaterHeatFlux):
            figures["signal"] = plot_heat_flux(target)
        for kind, figure in figures.items():
            figure.savefig(args.output_dir / f"{name}_{kind}.png", dpi=150)
            plt.close(figure)
        print(
            f"{name}: {len(data)} frames, rank {len(dmd.eigenvalues)}, "
            f"reconstruction error {relative_error(data, dmd):.3g}"
        )
    print(f"figures -> {args.output_dir}")


if __name__ == "__main__":
    main()
