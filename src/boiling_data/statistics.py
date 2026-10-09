import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from boiling_data.boiling_data import FloatArray
from boiling_data.bubbleml import (
    read_bubbleml,
    read_bubbleml_num_timesteps,
    read_bubbleml_parameters,
)

type StatisticsTree = dict[str, "RunningStatistics | StatisticsTree"]


@dataclass
class RunningStatistics:
    """Count, mean, population standard deviation, min and max over values seen
    in chunks, merged with Chan et al.'s parallel update so no chunk needs to stay
    in memory."""

    count: int = 0
    mean: float = 0.0
    sum_squared_deviations: float = 0.0
    minimum: float = np.inf
    maximum: float = -np.inf

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
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))

    @property
    def std(self) -> float:
        return float(np.sqrt(self.sum_squared_deviations / self.count))

    def record(self) -> dict[str, float]:
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.minimum,
            "max": self.maximum,
        }


def field_statistics(
    paths: Sequence[str | os.PathLike[str]],
    field_names: Sequence[str],
    frames_per_read: int,
) -> dict[str, dict[str, float]]:
    """Statistics of every named field over all frames and grid points of all
    files, reading frames_per_read frames of one field at a time to bound memory."""
    statistics = {name: RunningStatistics() for name in field_names}
    for path in paths:
        num_timesteps = read_bubbleml_num_timesteps(path)
        for start in range(0, num_timesteps, frames_per_read):
            for name in field_names:
                simulation = read_bubbleml(
                    path, [name], frames=slice(start, start + frames_per_read)
                )
                statistics[name].update(simulation.field(name).data)
    return {name: field.record() for name, field in statistics.items()}


def config_statistics(configs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Statistics of every numeric config parameter over the configs, nested by
    the configs' own groups and key names. A list of dicts, such as heaters, is
    summarized as one dict over all of its entries; other lists, strings and
    booleans are not parameters to normalize, and groups left empty are dropped.
    A parameter only some configs have is summarized over those."""
    tree: StatisticsTree = {}
    for config in configs:
        _accumulate(tree, config)
    return _record(tree)


def bubbleml_config_statistics(
    paths: Sequence[str | os.PathLike[str]],
) -> dict[str, Any]:
    return config_statistics(read_bubbleml_parameters(path).to_dict() for path in paths)


def dataset_statistics(
    paths: Sequence[str | os.PathLike[str]],
    field_names: Sequence[str],
    frames_per_read: int,
) -> dict[str, Any]:
    """The field statistics under "fields" and the config statistics under
    "config", the record a normalizer reads."""
    return {
        "fields": field_statistics(paths, field_names, frames_per_read),
        "config": bubbleml_config_statistics(paths),
    }


def _accumulate(tree: StatisticsTree, values: dict[str, Any]) -> None:
    for name, value in values.items():
        if isinstance(value, dict):
            _accumulate(_subtree(tree, name), value)
        elif _is_list_of_dicts(value):
            for entry in value:
                _accumulate(_subtree(tree, name), entry)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            statistics = tree.setdefault(name, RunningStatistics())
            assert isinstance(statistics, RunningStatistics)
            statistics.update(np.array([float(value)]))


def _is_list_of_dicts(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(entry, dict) for entry in value)
    )


def _subtree(tree: StatisticsTree, name: str) -> StatisticsTree:
    subtree = tree.setdefault(name, {})
    assert isinstance(subtree, dict)
    return subtree


def _record(tree: StatisticsTree) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for name, node in tree.items():
        if isinstance(node, RunningStatistics):
            record[name] = node.record()
            continue
        nested = _record(node)
        if nested:
            record[name] = nested
    return record
