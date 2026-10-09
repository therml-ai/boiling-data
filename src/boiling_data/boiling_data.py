import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from boiling_data.flashx.checks import Issue

FloatArray = npt.NDArray[np.float64]

BOUNDARY_SIDES = ("left", "right", "top", "bottom")
# Flash-X's runtime parameter for each side's boundary type, stored as its "type"
BOUNDARY_TYPE_KEYS = {
    "left": "xl_boundary_type",
    "right": "xr_boundary_type",
    "bottom": "yl_boundary_type",
    "top": "yr_boundary_type",
}
# how a heater drives the flow: a fixed wall temperature (wall_temp_fraction and
# wall_temp) or a fixed heat flux into the fluid (heat_flux, W/m^2)
CONSTANT_WALL_TEMP = "constant_wall_temp"
CONSTANT_HEAT_FLUX = "constant_heat_flux"
HEATER_TYPES = (CONSTANT_WALL_TEMP, CONSTANT_HEAT_FLUX)

# The unit of every dimensional parameter; all other parameters are
# non-dimensional, or not numbers (such as the fluid's name).
PHYSICAL_UNITS = {
    "bulk_temp": "degC",
    "sat_temp": "degC",
    "wall_temp_scale": "degC",
    "length_scale": "m",
    "velocity_scale": "m/s",
    "time_scale": "s",
    "rho_liquid": "kg/m^3",
    "thco_liquid": "W/(m K)",
}
HEATER_UNITS = {"wall_temp": "degC", "heat_flux": "W/m^2"}


def parameter_units(parameters: "SimulationParameters") -> dict[str, dict[str, str]]:
    """The unit of each dimensional physical parameter these parameters hold, and
    of every heater parameter: any heater can hold a wall temperature or a heat
    flux, whichever type the heaters here happen to be."""
    return {
        "physical": {
            name: unit
            for name, unit in PHYSICAL_UNITS.items()
            if name in parameters.physical
        },
        "heaters": dict(HEATER_UNITS),
    }


def heater_wall_temp(wall_temp_fraction: float, physical: dict[str, Any]) -> float:
    """A heater's wall temperature in C. Its fraction is (T_wall - T_bulk) /
    (wall_temp_scale - T_bulk), the scale every temperature is non-dimensionalized
    against, so several heaters can sit at different temperatures."""
    bulk_temp = float(physical["bulk_temp"])
    scale = float(physical["wall_temp_scale"])
    return bulk_temp + float(wall_temp_fraction) * (scale - bulk_temp)


@dataclass
class Field:
    """One time-dependent field laid out as [T, Y, X] (2d) or [T, Z, Y, X] (3d).

    Different fields (i.e., temp and velocity) from the same simulation need not
    share a shape.
    """

    data: FloatArray
    grid_x: FloatArray
    grid_y: FloatArray
    grid_z: FloatArray | None = None

    def __post_init__(self) -> None:
        expected: tuple[int, ...] = (len(self.grid_y), len(self.grid_x))
        if self.grid_z is not None:
            expected = (len(self.grid_z), *expected)
        if self.data.shape[1:] != expected:
            raise ValueError(
                f"field of shape {self.data.shape} does not match grid of shape "
                f"[T, {', '.join(str(n) for n in expected)}]"
            )

    @property
    def num_timesteps(self) -> int:
        return int(self.data.shape[0])

    def time_slice(self, start: int, stop: int | None = None) -> Self:
        if stop is None:
            stop = start + 1
        return type(self)(self.data[start:stop], self.grid_x, self.grid_y, self.grid_z)


@dataclass
class SimulationParameters:
    """Simulation parameters grouped by what they describe.

    physical: dimensional quantities (temperatures in C, length scale in m, ...)
        needed to re-dimensionalize the non-dimensional fields; wall_temp_scale is
        the temperature that non-dimensional temperatures are measured against.
    non_dimensional: the dimensionless groups the solver ran with (Reynolds,
        Prandtl, Stefan, gas/liquid property ratios, gravity, ...).
    discretization: the grid, the domain extent and the time stepping.
    heaters: List of dict, one dict per heater with its extent, contact angles,
        nucleation sites and type. A constant_wall_temp heater has its wall
        temperature as wall_temp_fraction, non-dimensional, and wall_temp in C, which
        is filled in from the fraction when physical has bulk_temp and
        wall_temp_scale. A constant_heat_flux heater has its heat_flux in W/m^2.
    boundary: one dict per side of the domain (left, right, top, bottom), holding
        its boundary condition's type; every side is always present, empty when
        nothing is recorded for it.
    """

    physical: dict[str, Any] = field(default_factory=dict)
    non_dimensional: dict[str, float] = field(default_factory=dict)
    discretization: dict[str, Any] = field(default_factory=dict)
    heaters: list[dict[str, Any]] = field(default_factory=list)
    boundary: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = sorted(set(self.boundary) - set(BOUNDARY_SIDES))
        if unknown:
            raise ValueError(
                f"boundary has unknown sides {unknown}; the sides are {BOUNDARY_SIDES}"
            )
        self.boundary = {
            side: dict(self.boundary.get(side, {})) for side in BOUNDARY_SIDES
        }
        for index, heater in enumerate(self.heaters):
            if heater.get("type", CONSTANT_WALL_TEMP) not in HEATER_TYPES:
                raise ValueError(
                    f"heater {index} has unknown type {heater['type']!r}; the types "
                    f"are {HEATER_TYPES}"
                )
        self.heaters = [self._with_wall_temp(heater) for heater in self.heaters]

    def _with_wall_temp(self, heater: dict[str, Any]) -> dict[str, Any]:
        heater = dict(heater)
        has_scale = {"bulk_temp", "wall_temp_scale"} <= self.physical.keys()
        holds_temperature = heater.get("type", CONSTANT_WALL_TEMP) == CONSTANT_WALL_TEMP
        if (
            has_scale
            and holds_temperature
            and "wall_temp_fraction" in heater
            and "wall_temp" not in heater
        ):
            heater["wall_temp"] = heater_wall_temp(
                heater["wall_temp_fraction"], self.physical
            )
        return heater

    def to_dict(self) -> dict[str, Any]:
        return {
            "physical": dict(self.physical),
            "non_dimensional": dict(self.non_dimensional),
            "discretization": dict(self.discretization),
            "heaters": [dict(heater) for heater in self.heaters],
            "boundary": {side: dict(values) for side, values in self.boundary.items()},
        }

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> Self:
        """Older records load too: boundary types kept in discretization move to
        their side, a physical wall_temp and heater wallTemp are read as
        wall_temp_scale and wall_temp_fraction, and heaters without a type held a
        constant wall temperature."""
        discretization = dict(record.get("discretization", {}))
        return cls(
            physical=_renamed(
                record.get("physical", {}), "wall_temp", "wall_temp_scale"
            ),
            non_dimensional=dict(record.get("non_dimensional", {})),
            discretization=discretization,
            heaters=[
                {"type": CONSTANT_WALL_TEMP}
                | _renamed(heater, "wallTemp", "wall_temp_fraction")
                for heater in record.get("heaters", [])
            ],
            boundary=_boundary_with_types_from(
                record.get("boundary", {}), discretization
            ),
        )


def _boundary_with_types_from(
    boundary: dict[str, dict[str, Any]], discretization: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Moves the *_boundary_type entries out of discretization into their side,
    unless the side already has a type."""
    sides = {side: dict(values) for side, values in boundary.items()}
    for side, key in BOUNDARY_TYPE_KEYS.items():
        if key in discretization:
            sides.setdefault(side, {}).setdefault("type", discretization.pop(key))
    return sides


def _renamed(values: dict[str, Any], old: str, new: str) -> dict[str, Any]:
    renamed = dict(values)
    if old in renamed and new not in renamed:
        renamed[new] = renamed.pop(old)
    return renamed


@dataclass
class BoilingSimulation:
    """
    This is a common in-memory format that every storage format is read into and
    written from. I.e., flash-x outputs can be read into this object, but so can
    bubbleml files.

    fields: keyed by field name (temperature, velx, velfacex, ...), each a
        Field with its own grid.
    parameters: the simulation parameters, grouped.
    time: the simulation time of every frame, shape [T].
    """

    fields: dict[str, Field]
    parameters: SimulationParameters
    time: FloatArray

    def __post_init__(self) -> None:
        for name, boiling_field in self.fields.items():
            if boiling_field.num_timesteps != len(self.time):
                raise ValueError(
                    f"field {name!r} has {boiling_field.num_timesteps} frames but "
                    f"time has {len(self.time)}"
                )

    @property
    def num_timesteps(self) -> int:
        return len(self.time)

    def field(self, name: str) -> Field:
        return self.fields[name]

    @classmethod
    def from_flashx(
        cls, directory: str | os.PathLike[str], mirror_symmetric: bool = False
    ) -> "BoilingSimulation":
        """mirror_symmetric: the run was symmetric about x = 0 and only computed
        the right half; reflect it into the full domain."""
        from boiling_data.flashx.reader import read_flashx

        return read_flashx(directory, mirror_symmetric=mirror_symmetric)

    @classmethod
    def from_bubbleml(cls, path: str | os.PathLike[str]) -> "BoilingSimulation":
        from boiling_data.bubbleml import read_bubbleml

        return read_bubbleml(path)

    def to_bubbleml(self, path: str | os.PathLike[str]) -> Path:
        from boiling_data.bubbleml import write_bubbleml

        return write_bubbleml(self, path)

    def check(self) -> list["Issue"]:
        """Problems found in the fields and parameters; see
        boiling_data.flashx.checks."""
        from boiling_data.flashx.checks import check_simulation

        return check_simulation(self)
