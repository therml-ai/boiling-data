import numpy as np
import pytest

from boiling_data.boiling_data import (
    BoilingSimulation,
    Field,
    SimulationParameters,
    parameter_units,
)


def _field(num_frames: int = 3, height: int = 4, width: int = 5) -> Field:
    return Field(
        np.zeros((num_frames, height, width)),
        grid_x=np.arange(width, dtype=np.float64),
        grid_y=np.arange(height, dtype=np.float64),
    )


def test_field_rejects_a_grid_that_does_not_match_its_data() -> None:
    with pytest.raises(ValueError, match="does not match grid"):
        Field(np.zeros((3, 4, 5)), grid_x=np.zeros(4), grid_y=np.zeros(4))


def test_field_time_slice_keeps_the_grid() -> None:
    sliced = _field().time_slice(1)
    assert sliced.data.shape == (1, 4, 5)
    np.testing.assert_array_equal(sliced.grid_x, _field().grid_x)


def test_simulation_rejects_fields_with_a_different_number_of_frames() -> None:
    with pytest.raises(ValueError, match="frames"):
        BoilingSimulation(
            {"temperature": _field(num_frames=3)},
            SimulationParameters(),
            time=np.zeros(2),
        )


def test_boundary_has_every_side_empty_by_default() -> None:
    assert SimulationParameters().boundary == {
        "left": {},
        "right": {},
        "top": {},
        "bottom": {},
    }


def test_boundary_round_trips_and_fills_missing_sides() -> None:
    parameters = SimulationParameters(boundary={"top": {"type": "outflow"}})
    assert parameters.boundary["top"] == {"type": "outflow"}
    assert parameters.boundary["left"] == {}
    assert SimulationParameters.from_dict(parameters.to_dict()) == parameters


def test_records_without_a_boundary_load_with_empty_sides() -> None:
    record = {"physical": {"wall_temp": 70.0}, "heaters": []}
    parameters = SimulationParameters.from_dict(record)
    assert set(parameters.boundary) == {"left", "right", "top", "bottom"}
    assert all(side == {} for side in parameters.boundary.values())


def test_unknown_boundary_side_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown sides"):
        SimulationParameters(boundary={"lft": {}})


def test_each_heater_gets_its_wall_temp_from_its_fraction() -> None:
    parameters = SimulationParameters(
        physical={"bulk_temp": 58.0, "wall_temp_scale": 70.0},
        heaters=[{"wall_temp_fraction": 1.0}, {"wall_temp_fraction": 0.5}],
    )
    assert [heater["wall_temp"] for heater in parameters.heaters] == [70.0, 64.0]


def test_given_heater_wall_temp_is_kept() -> None:
    parameters = SimulationParameters(
        physical={"bulk_temp": 58.0, "wall_temp_scale": 70.0},
        heaters=[{"wall_temp_fraction": 1.0, "wall_temp": 71.0}],
    )
    assert parameters.heaters[0]["wall_temp"] == 71.0


def test_heater_wall_temp_needs_the_scale() -> None:
    parameters = SimulationParameters(heaters=[{"wall_temp_fraction": 1.0}])
    assert "wall_temp" not in parameters.heaters[0]


def test_records_with_the_old_wall_temp_names_are_upgraded() -> None:
    record = {
        "physical": {"bulk_temp": 58.0, "wall_temp": 70.0},
        "heaters": [{"wallTemp": 0.5}],
    }
    parameters = SimulationParameters.from_dict(record)
    assert parameters.physical == {"bulk_temp": 58.0, "wall_temp_scale": 70.0}
    assert parameters.heaters == [
        {"type": "constant_wall_temp", "wall_temp_fraction": 0.5, "wall_temp": 64.0}
    ]


def test_constant_heat_flux_heater_keeps_its_flux_and_no_wall_temp() -> None:
    heater = {"type": "constant_heat_flux", "heat_flux": 5e4}
    parameters = SimulationParameters(
        physical={"bulk_temp": 58.0, "wall_temp_scale": 70.0}, heaters=[heater]
    )
    assert parameters.heaters == [heater]


def test_unknown_heater_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown type"):
        SimulationParameters(heaters=[{"type": "constant_wall_temperature"}])


def test_boundary_types_in_discretization_move_to_their_side() -> None:
    record = {
        "discretization": {"x_min": 0.0, "xl_boundary_type": "slip_ins"},
        "boundary": {"top": {"type": "outflow_ins"}},
    }
    parameters = SimulationParameters.from_dict(record)
    assert parameters.discretization == {"x_min": 0.0}
    assert parameters.boundary["left"] == {"type": "slip_ins"}
    assert parameters.boundary["top"] == {"type": "outflow_ins"}


def test_units_cover_the_physical_parameters_present_and_every_heater_one() -> None:
    parameters = SimulationParameters(
        physical={"fluid": "FC-72", "bulk_temp": 58.0, "wall_temp_scale": 70.0},
        non_dimensional={"stefan": 0.1},
        heaters=[{"type": "constant_heat_flux", "heat_flux": 5e4}],
    )
    assert parameter_units(parameters) == {
        "physical": {"bulk_temp": "degC", "wall_temp_scale": "degC"},
        "heaters": {"wall_temp": "degC", "heat_flux": "W/m^2"},
    }
