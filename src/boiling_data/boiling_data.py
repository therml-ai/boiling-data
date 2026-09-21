import os
from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


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
        needed to re-dimensionalize the non-dimensional fields.
    non_dimensional: the dimensionless groups the solver ran with (Reynolds,
        Prandtl, Stefan, gas/liquid property ratios, gravity, ...).
    discretization: the grid, the domain extent, the boundary conditions and the
        time stepping.
    heaters: List of dict, one dict per heater with its extent, contact angles, wall
        temperature and nucleation sites.
    """

    physical: dict[str, Any] = field(default_factory=dict)
    non_dimensional: dict[str, float] = field(default_factory=dict)
    discretization: dict[str, Any] = field(default_factory=dict)
    heaters: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "physical": dict(self.physical),
            "non_dimensional": dict(self.non_dimensional),
            "discretization": dict(self.discretization),
            "heaters": [dict(heater) for heater in self.heaters],
        }

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> Self:
        return cls(
            physical=dict(record.get("physical", {})),
            non_dimensional=dict(record.get("non_dimensional", {})),
            discretization=dict(record.get("discretization", {})),
            heaters=[dict(heater) for heater in record.get("heaters", [])],
        )


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
        from boiling_data.flashx import read_flashx

        return read_flashx(directory, mirror_symmetric=mirror_symmetric)
