import copy

import numpy as np
import pytest

from boiling_data.boiling_data import BoilingSimulation
from boiling_data.flashx.checks import (
    Issue,
    Severity,
    check_divergence,
    check_simulation,
)


@pytest.fixture
def case(bubbleml_case: BoilingSimulation) -> BoilingSimulation:
    return copy.deepcopy(bubbleml_case)


def _found(issues: list[Issue], check: str, severity: Severity) -> bool:
    return any(issue.check == check and issue.severity == severity for issue in issues)


def test_clean_case_has_no_issues(bubbleml_case: BoilingSimulation) -> None:
    assert bubbleml_case.check() == []


def test_missing_and_nonsensical_parameters(case: BoilingSimulation) -> None:
    del case.parameters.physical["thco_liquid"]
    del case.parameters.discretization["dt"]
    del case.parameters.non_dimensional["stefan"]
    case.parameters.physical["wall_temp_scale"] = case.parameters.physical["bulk_temp"]
    case.parameters.non_dimensional["prandtl"] = 0.0
    messages = [str(issue) for issue in check_simulation(case)]
    assert any("[warning]" in m and "thco_liquid" in m for m in messages)
    assert any("[error]" in m and "stefan" in m for m in messages)
    assert any("[error]" in m and "'dt'" in m for m in messages)
    assert any("[error]" in m and "wall_temp_scale" in m for m in messages)
    assert any("[error]" in m and "prandtl" in m for m in messages)


def test_inconsistent_scales(case: BoilingSimulation) -> None:
    case.parameters.physical["time_scale"] *= 2
    case.parameters.non_dimensional["tsat"] = 0.5
    messages = [str(issue) for issue in case.check()]
    assert any("time_scale" in message for message in messages)
    assert any("tsat" in message for message in messages)


def test_heater_missing_a_scalar(case: BoilingSimulation) -> None:
    del case.parameters.heaters[0]["nucWaitTime"]
    messages = [str(issue) for issue in case.check()]
    assert any("[error]" in m and "nucWaitTime" in m for m in messages)


def test_heater_outside_domain(case: BoilingSimulation) -> None:
    case.parameters.heaters[0].update(xMin=5.0, xMax=6.0)
    assert _found(case.check(), "heaters", Severity.ERROR)


def test_time_out_of_order(case: BoilingSimulation) -> None:
    case.time = case.time[::-1].copy()
    assert _found(case.check(), "time", Severity.ERROR)


def test_nan_in_a_field(case: BoilingSimulation) -> None:
    case.field("pressure").data[1, 5, 5] = np.nan
    issues = [issue for issue in case.check() if issue.check == "finite"]
    assert len(issues) == 1
    assert "pressure" in issues[0].message and "[1]" in issues[0].message


def test_temperature_beyond_wall(case: BoilingSimulation) -> None:
    case.field("temperature").data[0, 100, 50] = 1.5
    assert _found(case.check(), "temperature", Severity.WARNING)


def test_dfun_that_is_not_a_distance(case: BoilingSimulation) -> None:
    case.field("dfun").data *= 3
    assert _found(case.check(), "signed distance", Severity.WARNING)


def test_shifted_cell_velocity(case: BoilingSimulation) -> None:
    velx = case.field("velx")
    velx.data[:] = np.roll(velx.data, 1, axis=-1)
    assert _found(case.check(), "cell/face velocity", Severity.ERROR)


def test_divergence_away_from_the_interface(case: BoilingSimulation) -> None:
    sdf = case.field("dfun").data
    frame, row, column = np.argwhere(sdf < -1.0)[0]
    case.field("velfacex").data[frame, row, column + 1] += 0.1
    issues = [issue for issue in check_divergence(case) if issue.severity == "error"]
    assert len(issues) == 1
    assert f"[{frame}]" in issues[0].message


def test_divergence_at_the_interface_is_allowed(case: BoilingSimulation) -> None:
    sdf = case.field("dfun").data
    frame, row, column = np.argwhere(np.abs(sdf) < 0.01)[0]
    case.field("velfacex").data[frame, row, column + 1] += 0.1
    assert check_divergence(case) == []


def test_heater_wall_temp_disagreeing_with_its_fraction(
    case: BoilingSimulation,
) -> None:
    case.parameters.heaters[0]["wall_temp"] += 5.0
    messages = [str(issue) for issue in case.check()]
    assert any("[warning]" in m and "wall_temp_fraction" in m for m in messages)


def test_heat_flux_heater_needs_its_flux(case: BoilingSimulation) -> None:
    heater = case.parameters.heaters[0]
    heater["type"] = "constant_heat_flux"
    messages = [str(issue) for issue in case.check()]
    assert any("[error]" in m and "heat_flux" in m for m in messages)
    heater["heat_flux"] = 5e4
    del heater["wall_temp_fraction"], heater["wall_temp"]
    assert not _found(case.check(), "heaters", Severity.ERROR)


def test_physical_parameter_without_a_unit(case: BoilingSimulation) -> None:
    case.parameters.physical["heater_power"] = 12.0
    messages = [str(issue) for issue in case.check()]
    assert any("[warning]" in m and "heater_power" in m for m in messages)
