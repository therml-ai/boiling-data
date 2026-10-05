import numpy as np
import pytest

from boiling_data.boiling_data import BoilingSimulation, Field, SimulationParameters
from boiling_data.flashx.heat_flux import dimensional_temperature, heater_heat_flux

PHYSICAL = {
    "wall_temp": 70.0,
    "bulk_temp": 50.0,
    "length_scale": 1e-3,
    "thco_liquid": 0.05,
}


def _simulation(temperature: np.ndarray, sdf: np.ndarray) -> BoilingSimulation:
    """Cells of width 0.5 over x in [0, 2], y in [0, 1], under a heater on the
    bottom wall spanning x in [0.5, 1.5]."""
    x = np.array([0.25, 0.75, 1.25, 1.75])
    y = np.array([0.25, 0.75])
    parameters = SimulationParameters(
        physical=PHYSICAL,
        non_dimensional={"thcogas": 0.2},
        heaters=[{"xMin": 0.5, "xMax": 1.5, "yMax": 0.0, "wallTemp": 1.0}],
    )
    return BoilingSimulation(
        fields={
            "temperature": Field(temperature, grid_x=x, grid_y=y),
            "dfun": Field(sdf, grid_x=x, grid_y=y),
        },
        parameters=parameters,
        time=np.arange(len(temperature), dtype=np.float64),
    )


def test_dimensional_temperature_spans_bulk_to_wall() -> None:
    np.testing.assert_allclose(
        dimensional_temperature(np.array([0.0, 0.5, 1.0]), PHYSICAL), [50, 60, 70]
    )


def test_heat_flux_conducts_through_liquid_and_vapor() -> None:
    temperature = np.full((2, 2, 4), 0.5)
    sdf = np.full((2, 2, 4), -1.0)
    sdf[1, 0, 2] = 1.0
    heat_flux = heater_heat_flux(_simulation(temperature, sdf))

    np.testing.assert_allclose(heat_flux.x, [0.75, 1.25])
    # 0.05 W/m/K * (70 C - 60 C) / (0.25 * 1 mm)
    liquid_flux = 0.05 * 10 / 0.25e-3
    vapor_flux = 0.2 * liquid_flux
    np.testing.assert_allclose(
        heat_flux.flux, [[liquid_flux] * 2, [liquid_flux, vapor_flux]]
    )
    np.testing.assert_array_equal(heat_flux.liquid, [[True, True], [True, False]])


def test_heater_without_cells_above_it_is_rejected() -> None:
    simulation = _simulation(np.zeros((1, 2, 4)), np.zeros((1, 2, 4)))
    simulation.parameters.heaters[0]["yMax"] = 1.0
    with pytest.raises(ValueError, match="no cells above"):
        heater_heat_flux(simulation)


def test_heat_flux_of_bubbleml_case(bubbleml_case: BoilingSimulation) -> None:
    heat_flux = heater_heat_flux(bubbleml_case)
    assert heat_flux.flux.shape == (bubbleml_case.num_timesteps, len(heat_flux.x))
    assert not heat_flux.liquid.all()
    assert (heat_flux.flux > 0).all()
