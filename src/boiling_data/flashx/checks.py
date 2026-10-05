from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from boiling_data.boiling_data import (
    BoilingSimulation,
    FloatArray,
    SimulationParameters,
)
from boiling_data.flashx.reader import (
    FIELD_NAMES,
    HEATER_SCALARS,
    NON_DIMENSIONAL_PARAMETERS,
)


class Severity(StrEnum):
    """error: the data is wrong or unusable. warning: it is suspicious, or
    incomplete for some uses (such as re-dimensionalizing)."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class Issue:
    severity: Severity
    check: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.check}: {self.message}"


REQUIRED_DISCRETIZATION = (
    "num_blocks_x",
    "num_blocks_y",
    "nx_block",
    "ny_block",
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "dt",
    "t_initial",
)
EXPECTED_PHYSICAL = (
    "wall_temp",
    "bulk_temp",
    "sat_temp",
    "length_scale",
    "velocity_scale",
    "time_scale",
    "rho_liquid",
    "thco_liquid",
)
REQUIRED_NON_DIMENSIONAL = tuple(NON_DIMENSIONAL_PARAMETERS.values())
TEMPERATURE = FIELD_NAMES["temp"]
VELFACEX, VELFACEY = FIELD_NAMES["fv_x"], FIELD_NAMES["fv_y"]
POSITIVE_PARAMETERS = {
    "physical": ("length_scale", "velocity_scale", "time_scale", "rho_liquid"),
    "non_dimensional": ("inv_reynolds", "prandtl", "cpgas", "mugas", "rhogas"),
}


def check_simulation(simulation: BoilingSimulation) -> list[Issue]:
    """Every check that the simulation has the fields and parameters for; a check
    whose fields are absent is skipped."""
    checks: list[Callable[[BoilingSimulation], list[Issue]]] = [
        lambda s: check_parameters(s.parameters),
        lambda s: check_heaters(s.parameters),
        check_time,
        check_finite,
        check_temperature_range,
        check_signed_distance,
        check_cell_face_velocity,
        check_divergence,
    ]
    return [issue for check in checks for issue in check(simulation)]


def check_parameters(parameters: SimulationParameters) -> list[Issue]:
    issues = [
        Issue(
            Severity.ERROR,
            "parameters",
            f"discretization is missing {name!r}, needed to build the grid",
        )
        for name in REQUIRED_DISCRETIZATION
        if name not in parameters.discretization
    ]
    for group, names, severity in (
        ("non_dimensional", REQUIRED_NON_DIMENSIONAL, Severity.ERROR),
        ("physical", EXPECTED_PHYSICAL, Severity.WARNING),
    ):
        present = getattr(parameters, group)
        missing = [name for name in names if name not in present]
        if missing:
            issues.append(
                Issue(severity, "parameters", f"{group} is missing {missing}")
            )
    for group, positive in POSITIVE_PARAMETERS.items():
        present = getattr(parameters, group)
        issues += [
            Issue(
                Severity.ERROR,
                "parameters",
                f"{group} {name} is {present[name]}, but must be positive",
            )
            for name in positive
            if name in present and not present[name] > 0
        ]
    return issues + _check_physical_consistency(parameters)


def _check_physical_consistency(parameters: SimulationParameters) -> list[Issue]:
    physical, issues = parameters.physical, []
    if {"wall_temp", "bulk_temp"} <= physical.keys() and not (
        physical["wall_temp"] > physical["bulk_temp"]
    ):
        issues.append(
            Issue(
                Severity.ERROR,
                "parameters",
                f"wall_temp {physical['wall_temp']} is not above bulk_temp "
                f"{physical['bulk_temp']}, so temperatures cannot be "
                "re-dimensionalized",
            )
        )
    if {"sat_temp", "bulk_temp"} <= physical.keys() and (
        physical["sat_temp"] < physical["bulk_temp"]
    ):
        issues.append(
            Issue(
                Severity.WARNING,
                "parameters",
                f"bulk_temp {physical['bulk_temp']} is above sat_temp "
                f"{physical['sat_temp']}: the bulk liquid is superheated",
            )
        )
    if {"length_scale", "velocity_scale", "time_scale"} <= physical.keys():
        expected = physical["length_scale"] / physical["velocity_scale"]
        if not np.isclose(physical["time_scale"], expected, rtol=1e-3):
            issues.append(
                Issue(
                    Severity.WARNING,
                    "parameters",
                    f"time_scale {physical['time_scale']} is not length_scale / "
                    f"velocity_scale = {expected}",
                )
            )
    if (
        {"sat_temp", "bulk_temp", "wall_temp"} <= physical.keys()
        and "tsat" in parameters.non_dimensional
        and physical["wall_temp"] > physical["bulk_temp"]
    ):
        expected = (physical["sat_temp"] - physical["bulk_temp"]) / (
            physical["wall_temp"] - physical["bulk_temp"]
        )
        tsat = parameters.non_dimensional["tsat"]
        if not np.isclose(tsat, expected, atol=1e-3):
            issues.append(
                Issue(
                    Severity.WARNING,
                    "parameters",
                    f"non-dimensional tsat {tsat} does not match the physical "
                    f"temperatures, which give {expected}",
                )
            )
    return issues


def check_heaters(parameters: SimulationParameters) -> list[Issue]:
    if not parameters.heaters:
        return [Issue(Severity.WARNING, "heaters", "the simulation has no heaters")]
    discretization, issues = parameters.discretization, []
    for index, heater in enumerate(parameters.heaters):
        missing = [name for name in HEATER_SCALARS if name not in heater]
        if missing:
            issues.append(
                Issue(Severity.ERROR, "heaters", f"heater {index} is missing {missing}")
            )
            continue
        if not heater["xMin"] < heater["xMax"]:
            issues.append(
                Issue(
                    Severity.ERROR,
                    "heaters",
                    f"heater {index} spans x in [{heater['xMin']}, {heater['xMax']}]",
                )
            )
        if {"x_min", "x_max"} <= discretization.keys() and (
            heater["xMax"] <= discretization["x_min"]
            or heater["xMin"] >= discretization["x_max"]
        ):
            issues.append(
                Issue(
                    Severity.ERROR,
                    "heaters",
                    f"heater {index} lies outside the domain along x",
                )
            )
        outside = [
            site
            for site in heater.get("nuc_sites_x", [])
            if not heater["xMin"] <= site <= heater["xMax"]
        ]
        if outside:
            issues.append(
                Issue(
                    Severity.WARNING,
                    "heaters",
                    f"heater {index} has nucleation sites off the heater at x = "
                    f"{outside}",
                )
            )
    return issues


def check_time(simulation: BoilingSimulation) -> list[Issue]:
    steps = np.diff(simulation.time)
    if steps.size == 0:
        return []
    if not (steps > 0).all():
        return [
            Issue(
                Severity.ERROR,
                "time",
                f"time is not strictly increasing at frames "
                f"{np.flatnonzero(steps <= 0).tolist()}",
            )
        ]
    if not np.allclose(steps, steps.mean(), rtol=1e-2, atol=0):
        return [
            Issue(
                Severity.WARNING,
                "time",
                f"frames are unevenly spaced: steps range over "
                f"[{steps.min()}, {steps.max()}]",
            )
        ]
    return []


def check_finite(simulation: BoilingSimulation) -> list[Issue]:
    issues = []
    for name, field in simulation.fields.items():
        bad = ~np.isfinite(field.data)
        if bad.any():
            frames = np.flatnonzero(bad.any(axis=tuple(range(1, bad.ndim))))
            issues.append(
                Issue(
                    Severity.ERROR,
                    "finite",
                    f"{name} has {int(bad.sum())} NaN or infinite values in frames "
                    f"{_abbreviate(frames)}",
                )
            )
    return issues


def check_temperature_range(
    simulation: BoilingSimulation, tolerance: float = 1e-2
) -> list[Issue]:
    """Non-dimensional temperature should stay between the bulk liquid (0) and the
    hottest heater wall; tolerance allows for over- and undershoot of the solver."""
    if TEMPERATURE not in simulation.fields:
        return []
    wall_temps = [
        heater["wallTemp"]
        for heater in simulation.parameters.heaters
        if "wallTemp" in heater
    ]
    upper = max(wall_temps, default=1.0)
    lower = min(0.0, simulation.parameters.non_dimensional.get("tsat", 0.0))
    data = simulation.field(TEMPERATURE).data
    # non-finite values are reported by check_finite
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return []
    low, high = float(finite.min()), float(finite.max())
    if low < lower - tolerance or high > upper + tolerance:
        return [
            Issue(
                Severity.WARNING,
                "temperature",
                f"temperature spans [{low:.4g}, {high:.4g}], beyond the bulk to "
                f"wall range [{lower}, {upper}]",
            )
        ]
    return []


def check_signed_distance(
    simulation: BoilingSimulation, band_cells: float = 4, tolerance: float = 0.1
) -> list[Issue]:
    """The level set is reinitialized to a signed distance, so |grad dfun| should
    be close to 1 near the interface, where it is used."""
    if "dfun" not in simulation.fields:
        return []
    field = simulation.field("dfun")
    spacing = min(np.diff(field.grid_x).min(), np.diff(field.grid_y).min())
    near = np.abs(field.data) < band_cells * spacing
    if not near.any():
        return []
    grad_y, grad_x = np.gradient(field.data, field.grid_y, field.grid_x, axis=(1, 2))
    median = float(np.median(np.hypot(grad_x, grad_y)[near]))
    if abs(median - 1) > tolerance:
        return [
            Issue(
                Severity.WARNING,
                "signed distance",
                f"median |grad dfun| near the interface is {median:.3g}, not 1: dfun "
                "may not be a signed distance",
            )
        ]
    return []


def check_cell_face_velocity(
    simulation: BoilingSimulation, tolerance: float = 1e-6
) -> list[Issue]:
    """Cell-centered velocities should be the average of the two faces of each
    cell; a mismatch suggests the staggered fields were shifted or mirrored."""
    issues = []
    for center, face, axis in (("velx", VELFACEX, -1), ("vely", VELFACEY, -2)):
        if not {center, face} <= simulation.fields.keys():
            continue
        faces = simulation.field(face).data
        averaged = 0.5 * (
            np.take(faces, np.arange(faces.shape[axis] - 1), axis=axis)
            + np.take(faces, np.arange(1, faces.shape[axis]), axis=axis)
        )
        cells = simulation.field(center).data
        if averaged.shape != cells.shape:
            issues.append(
                Issue(
                    Severity.ERROR,
                    "cell/face velocity",
                    f"{face} {faces.shape} does not surround the cells of {center} "
                    f"{cells.shape}",
                )
            )
            continue
        error = _relative_max(cells - averaged, faces)
        if error > tolerance:
            issues.append(
                Issue(
                    Severity.ERROR,
                    "cell/face velocity",
                    f"{center} differs from the average of {face} by up to "
                    f"{error:.3g} of the peak velocity",
                )
            )
    return issues


def check_divergence(
    simulation: BoilingSimulation, band_cells: float = 4, tolerance: float = 1e-6
) -> list[Issue]:
    """Discrete divergence of the face velocities, in units of the peak velocity
    per cell. Phase change makes the velocity diverge at the interface, so cells
    within band_cells of it (by dfun) are left out."""
    if not {VELFACEX, VELFACEY} <= simulation.fields.keys():
        return []
    velfacex, velfacey = simulation.field(VELFACEX), simulation.field(VELFACEY)
    dx, dy = np.diff(velfacex.grid_x), np.diff(velfacey.grid_y)
    divergence = (
        np.diff(velfacex.data, axis=-1) / dx
        + np.diff(velfacey.data, axis=-2) / dy[:, None]
    )
    if "dfun" in simulation.fields:
        sdf = simulation.field("dfun").data
        if sdf.shape != divergence.shape:
            return [
                Issue(
                    Severity.ERROR,
                    "divergence",
                    f"dfun {sdf.shape} is not on the cells between the faces "
                    f"{divergence.shape}",
                )
            ]
        away = np.abs(sdf) > band_cells * min(dx.min(), dy.min())
    else:
        away = np.ones(divergence.shape, dtype=bool)
    if not away.any():
        return []
    cell_size = min(dx.min(), dy.min())
    error = _relative_max(
        divergence[away] * cell_size,
        np.concatenate([velfacex.data.ravel(), velfacey.data.ravel()]),
    )
    if error > tolerance:
        frames = np.flatnonzero(
            (np.abs(divergence) * away).max(axis=(1, 2)) * cell_size
            > tolerance * _peak(velfacex.data, velfacey.data)
        )
        return [
            Issue(
                Severity.ERROR,
                "divergence",
                f"velocity divergence away from the interface reaches {error:.3g} "
                f"of the peak velocity per cell, in frames {_abbreviate(frames)}",
            )
        ]
    return []


def _peak(*arrays: FloatArray) -> float:
    return max(float(np.nanmax(np.abs(array))) for array in arrays) or 1.0


def _relative_max(difference: FloatArray, scale: FloatArray) -> float:
    return float(np.nanmax(np.abs(difference))) / _peak(scale)


def _abbreviate(indices: npt.NDArray[np.intp], limit: int = 10) -> str:
    shown = [int(index) for index in indices[:limit]]
    return f"{shown}{' ...' if len(indices) > limit else ''}"
