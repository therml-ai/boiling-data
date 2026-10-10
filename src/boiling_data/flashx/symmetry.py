from collections.abc import Iterable
from typing import Any

import numpy as np

from boiling_data.boiling_data import BoilingSimulation, Field, SimulationParameters

# The fields whose sign changes when the domain is flipped along an axis: the
# vector components along that axis, and the 2d vorticity omgm = dv/dx - du/dy,
# which changes sign under any single reflection.
NEGATED_BY_X_FLIP = frozenset({"velx", "normx", "velfacex", "omgm"})
NEGATED_BY_Y_FLIP = frozenset({"vely", "normy", "velfacey", "omgm"})


def _lies_on_axis(coordinate: float) -> bool:
    return bool(np.isclose(coordinate, 0.0))


def mirror_field_x(field: Field, negate: bool = False) -> Field:
    """Reflect a field about x = 0 into x in [-x_max, x_max]. A sample sitting
    on the axis (a face-centered field's first face) belongs to both halves and
    is kept once."""
    if not _lies_on_axis(field.grid_x[0]) and field.grid_x[0] < 0:
        raise ValueError("field already extends to x < 0")
    reflected = (
        slice(None, 0, -1) if _lies_on_axis(field.grid_x[0]) else slice(None, None, -1)
    )
    sign = -1.0 if negate else 1.0
    return Field(
        np.concatenate([sign * field.data[..., reflected], field.data], axis=-1),
        grid_x=np.concatenate([-field.grid_x[reflected], field.grid_x]),
        grid_y=field.grid_y,
        grid_z=field.grid_z,
    )


def _mirror_heater(heater: dict[str, Any]) -> dict[str, Any]:
    """Nucleation sites off the axis gain an image site; one on the axis is its
    own image."""
    mirrored = dict(heater)
    sites_x = heater.get("nuc_sites_x", [])
    off_axis = [index for index, x in enumerate(sites_x) if not _lies_on_axis(x)]
    for key in ("nuc_sites_x", "nuc_sites_y", "nuc_sites_z", "nuc_seed_radii"):
        if key not in heater:
            continue
        images = [heater[key][index] for index in reversed(off_axis)]
        if key == "nuc_sites_x":
            images = [-x for x in images]
        mirrored[key] = images + list(heater[key])
    return mirrored


def _mirror_parameters(parameters: SimulationParameters) -> SimulationParameters:
    discretization = dict(parameters.discretization)
    discretization["x_min"] = -discretization["x_max"]
    discretization["num_blocks_x"] = 2 * discretization["num_blocks_x"]
    discretization["mirrored_about_x"] = 0.0
    boundary = {side: dict(values) for side, values in parameters.boundary.items()}
    boundary["left"] = dict(boundary["right"])
    return SimulationParameters(
        physical=dict(parameters.physical),
        non_dimensional=dict(parameters.non_dimensional),
        discretization=discretization,
        heaters=[_mirror_heater(heater) for heater in parameters.heaters],
        boundary=boundary,
    )


def mirror_x(
    simulation: BoilingSimulation, negated_fields: Iterable[str] = NEGATED_BY_X_FLIP
) -> BoilingSimulation:
    """The full-domain simulation of a run that was symmetric about x = 0 and
    only computed x >= 0. Fields named in negated_fields flip sign across the axis."""
    discretization = simulation.parameters.discretization
    if not _lies_on_axis(discretization["x_min"]):
        raise ValueError(
            f"the symmetry axis x = 0 has to be the left boundary, but x_min is "
            f"{discretization['x_min']}"
        )
    left_type = simulation.parameters.boundary["left"].get("type", "")
    if "slip" not in left_type or "noslip" in left_type:
        raise ValueError(
            f"the left boundary type is {left_type!r}; a run symmetric about x = 0 "
            "has a slip wall there"
        )
    negated_fields = set(negated_fields)
    return BoilingSimulation(
        fields={
            name: mirror_field_x(field, negate=name in negated_fields)
            for name, field in simulation.fields.items()
        },
        parameters=_mirror_parameters(simulation.parameters),
        time=simulation.time,
    )
