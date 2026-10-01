from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from boiling_data.boiling_data import BoilingSimulation, FloatArray

HEAT_FLUX_FIELDS = ("temperature", "dfun")


def dimensional_temperature(
    temperature: FloatArray | float, physical: dict[str, Any]
) -> FloatArray:
    """Temperatures are stored as (T - T_bulk) / (T_wall - T_bulk); returns C."""
    bulk, wall = float(physical["bulk_temp"]), float(physical["wall_temp"])
    return np.asarray(bulk + np.asarray(temperature) * (wall - bulk), dtype=np.float64)


@dataclass
class HeaterHeatFlux:
    """Heat flux from a heater into the first row of cells above it.

    x: cell centers over the heater, in non-dimensional length.
    flux: [T, X] in W/m^2, positive from the heater into the fluid.
    liquid: [T, X] whether the cell was liquid.
    """

    x: FloatArray
    flux: FloatArray
    liquid: npt.NDArray[np.bool_]


def heater_heat_flux(
    simulation: BoilingSimulation, heater_index: int = 0
) -> HeaterHeatFlux:
    """Conduction q = -k dT/dy, with dT/dy the one-sided difference between the
    heater's wall temperature and the cell center, and k the liquid conductivity
    in liquid cells and the vapor's (thcogas times it) in vapor cells."""
    parameters = simulation.parameters
    physical = parameters.physical
    heater = parameters.heaters[heater_index]
    temperature, sdf = simulation.field("temperature"), simulation.field("dfun")
    if temperature.data.shape != sdf.data.shape:
        raise ValueError(
            f"temperature {temperature.data.shape} and dfun {sdf.data.shape} must "
            "be sampled on the same grid"
        )

    columns = (temperature.grid_x >= heater["xMin"]) & (
        temperature.grid_x <= heater["xMax"]
    )
    above = np.flatnonzero(temperature.grid_y > heater["yMax"])
    if not columns.any() or above.size == 0:
        raise ValueError(f"heater {heater_index} has no cells above it on the grid")
    row = int(above[0])

    distance = (temperature.grid_y[row] - heater["yMax"]) * physical["length_scale"]
    wall = dimensional_temperature(heater["wallTemp"], physical)
    cell = dimensional_temperature(temperature.data[:, row, columns], physical)
    liquid = sdf.data[:, row, columns] < 0
    thco_liquid = float(physical["thco_liquid"])
    thco_vapor = parameters.non_dimensional["thcogas"] * thco_liquid
    conductivity = np.where(liquid, thco_liquid, thco_vapor)
    flux = (conductivity * (wall - cell) / distance).astype(np.float64)
    return HeaterHeatFlux(x=temperature.grid_x[columns], flux=flux, liquid=liquid)
