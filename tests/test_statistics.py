from pathlib import Path

import numpy as np
import pytest

from boiling_data.boiling_data import BoilingSimulation, SimulationParameters
from boiling_data.statistics import (
    RunningMoments,
    bubbleml_parameter_ranges,
    field_moments,
    parameter_ranges,
    statistics_record,
)


def test_moments_merged_over_chunks_match_the_whole() -> None:
    values = np.random.default_rng(0).normal(3.0, 2.0, size=1000)
    moments = RunningMoments()
    for chunk in np.array_split(values, [10, 11, 400]):
        moments.update(chunk)
    assert moments.count == values.size
    assert moments.mean == pytest.approx(values.mean())
    assert moments.std == pytest.approx(values.std())


def test_empty_chunks_are_ignored() -> None:
    moments = RunningMoments()
    moments.update(np.array([1.0, 3.0]))
    moments.update(np.array([]))
    assert (moments.mean, moments.std) == (2.0, 1.0)


@pytest.mark.parametrize("frames_per_read", [1, 2, 64])
def test_field_moments_cover_every_frame_of_every_file(
    bubbleml_path: Path, bubbleml_case: BoilingSimulation, frames_per_read: int
) -> None:
    moments = field_moments(
        [bubbleml_path, bubbleml_path], ["temperature", "velfacex"], frames_per_read
    )
    for name in ("temperature", "velfacex"):
        data = bubbleml_case.field(name).data
        assert moments[name].count == 2 * data.size
        assert moments[name].mean == pytest.approx(data.mean())
        assert moments[name].std == pytest.approx(data.std())


def _parameters(stefan: float, wall_temps: list[float]) -> SimulationParameters:
    return SimulationParameters(
        non_dimensional={"stefan": stefan, "prandtl": 7.0},
        heaters=[
            {"wallTemp": wall_temp, "nuc_sites_x": [0.0, 1.0]}
            for wall_temp in wall_temps
        ],
    )


def test_parameter_ranges_span_every_simulation_and_heater() -> None:
    ranges = parameter_ranges([_parameters(0.5, [1.0]), _parameters(0.2, [2.0, 0.5])])
    assert set(ranges["non_dimensional"]) == {"stefan", "prandtl"}
    stefan = ranges["non_dimensional"]["stefan"]
    assert (stefan.minimum, stefan.maximum) == (0.2, 0.5)
    wall_temp = ranges["heater"]["wallTemp"]
    assert (wall_temp.minimum, wall_temp.maximum) == (0.5, 2.0)
    assert "nuc_sites_x" not in ranges["heater"]


def test_bubbleml_parameter_ranges(bubbleml_path: Path) -> None:
    ranges = bubbleml_parameter_ranges([bubbleml_path])
    stefan = ranges["non_dimensional"]["stefan"]
    assert (stefan.minimum, stefan.maximum) == (0.156, 0.156)
    assert ranges["heater"]["advAngle"].maximum == 45.0


def test_statistics_record_nests_fields_and_parameter_groups() -> None:
    moments = RunningMoments()
    moments.update(np.array([1.0, 3.0]))
    record = statistics_record(
        {"temperature": moments}, parameter_ranges([_parameters(0.5, [1.0])])
    )
    assert record == {
        "fields": {"temperature": {"mean": 2.0, "std": 1.0}},
        "parameters": {
            "non_dimensional": {
                "prandtl": {"min": 7.0, "max": 7.0},
                "stefan": {"min": 0.5, "max": 0.5},
            },
            "heater": {"wallTemp": {"min": 1.0, "max": 1.0}},
        },
    }
