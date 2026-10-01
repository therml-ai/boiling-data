import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from boiling_data.boiling_data import FloatArray, SimulationParameters
from boiling_data.bubbleml import (
    read_bubbleml,
    read_bubbleml_num_timesteps,
    read_bubbleml_parameters,
)


@dataclass
class RunningMoments:
    """Mean and population standard deviation over values seen in chunks, merged
    with Chan et al.'s parallel update so no chunk needs to stay in memory."""

    count: int = 0
    mean: float = 0.0
    sum_squared_deviations: float = 0.0

    def update(self, values: FloatArray) -> None:
        chunk_count = values.size
        if chunk_count == 0:
            return
        chunk_mean = float(values.mean())
        chunk_squared_deviations = float(((values - chunk_mean) ** 2).sum())
        total = self.count + chunk_count
        delta = chunk_mean - self.mean
        self.mean += delta * chunk_count / total
        self.sum_squared_deviations += (
            chunk_squared_deviations + delta**2 * self.count * chunk_count / total
        )
        self.count = total

    @property
    def std(self) -> float:
        return float(np.sqrt(self.sum_squared_deviations / self.count))


@dataclass
class ParameterRange:
    minimum: float
    maximum: float

    def update(self, value: float) -> None:
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)


type ParameterRanges = dict[str, dict[str, ParameterRange]]


def field_moments(
    paths: Sequence[str | os.PathLike[str]],
    field_names: Sequence[str],
    frames_per_read: int,
) -> dict[str, RunningMoments]:
    """Moments of every named field over all frames and grid points of all files,
    reading frames_per_read frames of one field at a time to bound memory."""
    moments = {name: RunningMoments() for name in field_names}
    for path in paths:
        num_timesteps = read_bubbleml_num_timesteps(path)
        for start in range(0, num_timesteps, frames_per_read):
            for name in field_names:
                simulation = read_bubbleml(
                    path, [name], frames=slice(start, start + frames_per_read)
                )
                moments[name].update(simulation.field(name).data)
    return moments


def parameter_ranges(parameters: Iterable[SimulationParameters]) -> ParameterRanges:
    """Min / max of every numeric non-dimensional parameter and every numeric
    heater scalar, over all heaters of all simulations. A parameter only some
    simulations have is ranged over those."""
    ranges: ParameterRanges = {"non_dimensional": {}, "heater": {}}
    for simulation_parameters in parameters:
        _update_ranges(ranges["non_dimensional"], simulation_parameters.non_dimensional)
        for heater in simulation_parameters.heaters:
            _update_ranges(ranges["heater"], heater)
    return ranges


def bubbleml_parameter_ranges(
    paths: Sequence[str | os.PathLike[str]],
) -> ParameterRanges:
    return parameter_ranges(read_bubbleml_parameters(path) for path in paths)


def _update_ranges(ranges: dict[str, ParameterRange], values: dict[str, Any]) -> None:
    # lists such as nucleation sites vary in length and are not parameters to range
    for name, value in values.items():
        if not isinstance(value, Real):
            continue
        if name in ranges:
            ranges[name].update(float(value))
        else:
            ranges[name] = ParameterRange(float(value), float(value))


def statistics_record(
    moments: dict[str, RunningMoments], ranges: ParameterRanges
) -> dict[str, Any]:
    return {
        "fields": {
            name: {"mean": field.mean, "std": field.std}
            for name, field in moments.items()
        },
        "parameters": {
            group: {
                name: {"min": parameter.minimum, "max": parameter.maximum}
                for name, parameter in sorted(group_ranges.items())
            }
            for group, group_ranges in ranges.items()
        },
    }
