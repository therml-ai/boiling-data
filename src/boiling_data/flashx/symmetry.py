from collections.abc import Iterable
from typing import Any

import numpy as np

from boiling_data.boiling_data import BoilingSimulation, Field, SimulationParameters

ODD_FIELDS = frozenset({"velx", "normx", "velfacex"})


def _lies_on_axis(coordinate: float) -> bool:
    return bool(np.isclose(coordinate, 0.0))


def mirror_field_x(field: Field, odd: bool = False) -> Field:
    """Reflect a field about x = 0 into x in [-x_max, x_max]. A sample sitting
    on the axis (a face-centered field's first face) belongs to both halves and
    is kept once."""
    if not _lies_on_axis(field.grid_x[0]) and field.grid_x[0] < 0:
        raise ValueError("field already extends to x < 0")
    reflected = (
        slice(None, 0, -1) if _lies_on_axis(field.grid_x[0]) else slice(None, None, -1)
    )
    sign = -1.0 if odd else 1.0
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
    discretization["xl_boundary_type"] = discretization["xr_boundary_type"]
    discretization["mirrored_about_x"] = 0.0
    return SimulationParameters(
        physical=dict(parameters.physical),
        non_dimensional=dict(parameters.non_dimensional),
        discretization=discretization,
        heaters=[_mirror_heater(heater) for heater in parameters.heaters],
    )


def mirror_x(
    simulation: BoilingSimulation, odd_fields: Iterable[str] = ODD_FIELDS
) -> BoilingSimulation:
    """The full-domain simulation of a run that was symmetric about x = 0 and
    only computed x >= 0. Fields named in odd_fields flip sign across the axis."""
    discretization = simulation.parameters.discretization
    if not _lies_on_axis(discretization["x_min"]):
        raise ValueError(
            f"the symmetry axis x = 0 has to be the left boundary, but x_min is "
            f"{discretization['x_min']}"
        )
    boundary = discretization["xl_boundary_type"]
    if "slip" not in boundary or "noslip" in boundary:
        raise ValueError(
            f"xl_boundary_type is {boundary!r}; a run symmetric about x = 0 has a "
            "slip wall there"
        )
    odd_fields = set(odd_fields)
    return BoilingSimulation(
        fields={
            name: mirror_field_x(field, odd=name in odd_fields)
            for name, field in simulation.fields.items()
        },
        parameters=_mirror_parameters(simulation.parameters),
        time=simulation.time,
    )
