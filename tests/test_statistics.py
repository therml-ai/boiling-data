from pathlib import Path
from typing import Any

import numpy as np
import pytest

from boiling_data.boiling_data import BoilingSimulation, SimulationParameters
from boiling_data.statistics import (
    RunningStatistics,
    bubbleml_config_statistics,
    config_statistics,
    dataset_statistics,
    field_statistics,
)


def test_statistics_merged_over_chunks_match_the_whole() -> None:
    values = np.random.default_rng(0).normal(3.0, 2.0, size=1000)
    statistics = RunningStatistics()
    for chunk in np.array_split(values, [10, 11, 400]):
        statistics.update(chunk)
    assert statistics.count == values.size
    assert statistics.mean == pytest.approx(values.mean())
    assert statistics.std == pytest.approx(values.std())
    assert (statistics.minimum, statistics.maximum) == (values.min(), values.max())


def test_empty_chunks_are_ignored() -> None:
    statistics = RunningStatistics()
    statistics.update(np.array([1.0, 3.0]))
    statistics.update(np.array([]))
    assert statistics.record() == {
        "count": 2,
        "mean": 2.0,
        "std": 1.0,
        "min": 1.0,
        "max": 3.0,
    }


@pytest.mark.parametrize("frames_per_read", [1, 2, 64])
def test_field_statistics_cover_every_frame_of_every_file(
    bubbleml_path: Path, bubbleml_case: BoilingSimulation, frames_per_read: int
) -> None:
    statistics = field_statistics(
        [bubbleml_path, bubbleml_path], ["temperature", "velfacex"], frames_per_read
    )
    for name in ("temperature", "velfacex"):
        data = bubbleml_case.field(name).data
        assert statistics[name]["count"] == 2 * data.size
        assert statistics[name]["mean"] == pytest.approx(data.mean())
        assert statistics[name]["std"] == pytest.approx(data.std())
        assert statistics[name]["min"] == pytest.approx(data.min())
        assert statistics[name]["max"] == pytest.approx(data.max())


def _config(stefan: float, wall_temp_fractions: list[float]) -> dict[str, Any]:
    return SimulationParameters(
        physical={"fluid": "FC-72", "bulk_temp": 58.0, "wall_temp_scale": 70.0},
        non_dimensional={"stefan": stefan, "prandtl": 7.0},
        heaters=[
            {"wall_temp_fraction": fraction, "nuc_sites_x": [0.0, 1.0]}
            for fraction in wall_temp_fractions
        ],
        boundary={"left": {"type": "slip_ins"}},
    ).to_dict()


def test_config_statistics_follow_the_config_groups_and_names() -> None:
    statistics = config_statistics([_config(0.5, [1.0]), _config(0.2, [0.5, 1.0])])
    assert set(statistics) == {"physical", "non_dimensional", "heaters"}
    assert set(statistics["physical"]) == {"bulk_temp", "wall_temp_scale"}
    stefan = statistics["non_dimensional"]["stefan"]
    assert (stefan["count"], stefan["min"], stefan["max"]) == (2, 0.2, 0.5)
    assert stefan["mean"] == pytest.approx(0.35)


def test_heaters_are_summarized_over_every_heater_of_every_config() -> None:
    statistics = config_statistics([_config(0.5, [1.0]), _config(0.2, [0.5, 1.0])])
    fraction = statistics["heaters"]["wall_temp_fraction"]
    assert (fraction["count"], fraction["min"], fraction["max"]) == (3, 0.5, 1.0)
    # the C wall temperature is filled in from each fraction
    assert statistics["heaters"]["wall_temp"]["min"] == pytest.approx(64.0)


def test_non_numeric_values_and_empty_groups_are_left_out() -> None:
    statistics = config_statistics([_config(0.5, [1.0])])
    assert "fluid" not in statistics["physical"]
    assert "nuc_sites_x" not in statistics["heaters"]
    assert "type" not in statistics["heaters"]
    assert "boundary" not in statistics


def test_a_parameter_only_some_configs_have_is_summarized_over_those() -> None:
    sparse = _config(0.5, [1.0])
    del sparse["non_dimensional"]["prandtl"]
    statistics = config_statistics([sparse, _config(0.2, [1.0])])
    assert statistics["non_dimensional"]["prandtl"]["count"] == 1
    assert statistics["non_dimensional"]["stefan"]["count"] == 2


def test_bubbleml_config_statistics(bubbleml_path: Path) -> None:
    statistics = bubbleml_config_statistics([bubbleml_path, bubbleml_path])
    stefan = statistics["non_dimensional"]["stefan"]
    assert (stefan["count"], stefan["mean"], stefan["std"]) == (2, 0.156, 0.0)
    assert statistics["discretization"]["num_blocks_x"]["max"] == 6
    assert statistics["heaters"]["advAngle"]["max"] == 45.0


def test_dataset_statistics_hold_fields_and_config_under_separate_keys(
    bubbleml_path: Path,
) -> None:
    statistics = dataset_statistics([bubbleml_path], ["temperature"], 64)
    assert set(statistics) == {"fields", "config"}
    assert set(statistics["fields"]) == {"temperature"}
    assert statistics["config"] == bubbleml_config_statistics([bubbleml_path])
