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


def _simulation(boiling_field: Field) -> BoilingSimulation:
    return BoilingSimulation(
        {"dfun": boiling_field},
        SimulationParameters(),
        time=np.zeros(boiling_field.num_timesteps),
    )


def test_simulation_gives_each_axis_grid_spacing_of_a_field() -> None:
    boiling_field = Field(
        np.zeros((1, 2, 4, 5)),
        grid_x=np.arange(5, dtype=np.float64) * np.float64(0.5),
        grid_y=np.arange(4, dtype=np.float64) * np.float64(0.25),
        grid_z=np.arange(2, dtype=np.float64) * np.float64(0.125),
    )
    simulation = _simulation(boiling_field)
    assert simulation.dx("dfun") == pytest.approx(0.5)
    assert simulation.dy("dfun") == pytest.approx(0.25)
    assert simulation.dz("dfun") == pytest.approx(0.125)


def test_a_2d_field_has_no_z_spacing() -> None:
    with pytest.raises(ValueError, match="2d"):
        _simulation(_field()).dz("dfun")


def test_a_nonuniform_grid_has_no_spacing() -> None:
    boiling_field = Field(
        np.zeros((1, 4, 3)),
        grid_x=np.array([0.0, 1.0, 3.0]),
        grid_y=np.arange(4, dtype=np.float64),
    )
    with pytest.raises(ValueError, match="nonuniform x grid"):
        _simulation(boiling_field).dx("dfun")


def test_a_single_point_axis_has_no_spacing() -> None:
    with pytest.raises(ValueError, match="one y point"):
        _simulation(_field(height=1)).dy("dfun")


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
