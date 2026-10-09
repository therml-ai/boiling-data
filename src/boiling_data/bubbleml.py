import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from boiling_data.boiling_data import (
    BoilingSimulation,
    Field,
    FloatArray,
    SimulationParameters,
    parameter_units,
)

PARAMETERS_ATTRIBUTE = "parameters"
UNITS_ATTRIBUTE = "units"


def _faces(discretization: dict[str, Any], axis: str) -> FloatArray:
    num_cells = discretization[f"num_blocks_{axis}"] * discretization[f"n{axis}_block"]
    return np.linspace(
        discretization[f"{axis}_min"], discretization[f"{axis}_max"], num_cells + 1
    )


def _centers(faces: FloatArray) -> FloatArray:
    return 0.5 * (faces[1:] + faces[:-1])


def _grid(discretization: dict[str, Any]) -> dict[str, FloatArray]:
    x_faces, y_faces = _faces(discretization, "x"), _faces(discretization, "y")
    return {
        "x_faces": x_faces,
        "y_faces": y_faces,
        "x_centers": _centers(x_faces),
        "y_centers": _centers(y_faces),
    }


def _check_field_on_grid(name: str, field: Field, grid: dict[str, FloatArray]) -> None:
    for axis, coordinates in (("x", field.grid_x), ("y", field.grid_y)):
        on_grid = any(
            len(coordinates) == len(grid[key]) and np.allclose(coordinates, grid[key])
            for key in (f"{axis}_centers", f"{axis}_faces")
        )
        if not on_grid:
            raise ValueError(
                f"field {name!r} is not sampled at the cell centers or faces of "
                f"the grid described by the discretization parameters along {axis}"
            )


def write_bubbleml(simulation: BoilingSimulation, path: str | os.PathLike[str]) -> Path:
    """One HDF5 file: every field as a dataset, ``time``, the grid coordinates,
    and the grouped parameters and the units of the dimensional ones as JSON in
    root attributes."""
    path = Path(path)
    grid = _grid(simulation.parameters.discretization)
    for name, field in simulation.fields.items():
        _check_field_on_grid(name, field, grid)

    with h5py.File(path, "w") as handle:
        handle.attrs[PARAMETERS_ATTRIBUTE] = json.dumps(
            simulation.parameters.to_dict(), indent=4
        )
        handle.attrs[UNITS_ATTRIBUTE] = json.dumps(
            parameter_units(simulation.parameters), indent=4
        )
        for name, field in simulation.fields.items():
            handle.create_dataset(name, data=field.data)
        handle.create_dataset("time", data=simulation.time)
        for name, coordinates in grid.items():
            handle.create_dataset(name, data=coordinates)
    return path


def _axis_coordinates(
    name: str, num_samples: int, grid: dict[str, FloatArray], axis: str
) -> FloatArray:
    for key in (f"{axis}_centers", f"{axis}_faces"):
        if len(grid[key]) == num_samples:
            return grid[key]
    raise ValueError(
        f"field {name!r} has {num_samples} samples along {axis}, which is neither "
        f"the {len(grid[f'{axis}_centers'])} cells nor the "
        f"{len(grid[f'{axis}_faces'])} faces of the grid"
    )


def _read_parameters(handle: h5py.File, path: Path) -> SimulationParameters:
    """Files written before the parameters were embedded keep them in a JSON
    file next to the HDF5 with the same stem."""
    if PARAMETERS_ATTRIBUTE in handle.attrs:
        return SimulationParameters.from_dict(
            json.loads(handle.attrs[PARAMETERS_ATTRIBUTE])
        )
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        raise ValueError(
            f"{path} has no {PARAMETERS_ATTRIBUTE!r} attribute and no {sidecar.name}"
        )
    with open(sidecar, encoding="utf-8") as sidecar_handle:
        return SimulationParameters.from_dict(json.load(sidecar_handle))


def read_bubbleml(
    path: str | os.PathLike[str],
    field_names: Iterable[str] | None = None,
    frames: slice = slice(None),
) -> BoilingSimulation:
    """field_names and frames select what is read from disk, so a window of a
    large case can be loaded without reading all of it."""
    path = Path(path)
    fields = {}
    with h5py.File(path, "r") as handle:
        parameters = _read_parameters(handle, path)
        grid = _grid(parameters.discretization)
        stored = _stored_field_names(handle, grid)
        selected = stored if field_names is None else list(field_names)
        missing = sorted(set(selected) - set(stored))
        if missing:
            raise KeyError(f"{path} has no fields {missing}; it has {sorted(stored)}")
        for name in selected:
            data = handle[name][frames].astype(np.float64)
            fields[name] = Field(
                data,
                grid_x=_axis_coordinates(name, data.shape[-1], grid, "x"),
                grid_y=_axis_coordinates(name, data.shape[-2], grid, "y"),
            )
        time = _read_time(handle, parameters.discretization, stored[0])[frames]
    return BoilingSimulation(fields=fields, parameters=parameters, time=time)


def read_bubbleml_num_timesteps(path: str | os.PathLike[str]) -> int:
    path = Path(path)
    with h5py.File(path, "r") as handle:
        grid = _grid(_read_parameters(handle, path).discretization)
        return int(handle[_stored_field_names(handle, grid)[0]].shape[0])


def read_bubbleml_parameters(path: str | os.PathLike[str]) -> SimulationParameters:
    path = Path(path)
    with h5py.File(path, "r") as handle:
        return _read_parameters(handle, path)


def read_bubbleml_units(path: str | os.PathLike[str]) -> dict[str, dict[str, str]]:
    """Files written before units were recorded have none."""
    with h5py.File(path, "r") as handle:
        if UNITS_ATTRIBUTE not in handle.attrs:
            return {}
        units: dict[str, dict[str, str]] = json.loads(handle.attrs[UNITS_ATTRIBUTE])
        return units


def _stored_field_names(handle: h5py.File, grid: dict[str, FloatArray]) -> list[str]:
    return [name for name in handle if name != "time" and name not in grid]


def _read_time(
    handle: h5py.File, discretization: dict[str, Any], any_field: str
) -> FloatArray:
    if "time" in handle:
        time: FloatArray = handle["time"][()].astype(np.float64)
        return time
    return _uniform_time(discretization, int(handle[any_field].shape[0]))


def _uniform_time(discretization: dict[str, Any], num_timesteps: int) -> FloatArray:
    """Files written before ``time`` was stored hold frames at every plot
    interval from the start of the run."""
    t_initial, dt = float(discretization["t_initial"]), float(discretization["dt"])
    return t_initial + dt * np.arange(num_timesteps, dtype=np.float64)
