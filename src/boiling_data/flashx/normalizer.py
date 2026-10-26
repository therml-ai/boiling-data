import copy
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import torch
from torch import nn

from boiling_data.flashx.batch import FlashXBatch
from boiling_data.statistics import is_sequence_of_mappings

# (value, offset, scale) -> transformed value, for floats and tensors alike
type Transform = Callable[[Any, float, float], Any]
# a statistics entry -> the (offset, scale) that normalizes its values
type Scaling = Callable[[Mapping[str, float]], tuple[float, float]]
type ConfigScaling = Literal["standard", "min_max"]

# config groups describing the grid and time stepping rather than the physics, which
# FlashXBatch.dx, dy and dz read the spacing from
UNNORMALIZED_CONFIG_GROUPS = frozenset({"discretization"})


def _standardize(value: Any, mean: float, scale: float) -> Any:
    return (value - mean) / scale


def _unstandardize(value: Any, mean: float, scale: float) -> Any:
    return value * scale + mean


class NormalizerWrapper(nn.Module):
    """Normalizes fields and config parameters of a `FlashXBatch`:
        - fields normalized by dataset mean and std.
        - config parameters mapped from their dataset min and max to [-1, 1],
          or with ``config_scaling="standard"`` normalized by dataset mean and
          std.
        - values that are constant over a dataset only have mean subtracted.
        - the grid discretization config group is kept as it is, so a normalized
          batch's dx, dy and dz are the grid spacings.
        - the components of each of ``vector_fields``, e.g. ``("velx", "vely")``,
          share one scale, the root mean square of the components, and keep
          their zero, so linear operators such as the divergence commute with
          normalization.
    This can be used as a module: `NormalizerWrapper.from_json("stats.json", module)`
    to automatically normalize inputs and unnormalize outputs.
    The normalization statistics are saved with the module's state, so a checkpoint
    carries the normalization it was trained with.
    """

    def __init__(
        self,
        statistics: Mapping[str, Any],
        module: nn.Module | None = None,
        config_scaling: ConfigScaling = "min_max",
        vector_fields: Sequence[Sequence[str]] = (),
    ) -> None:
        super().__init__()
        self.statistics = _validated_statistics(statistics)
        self.module = module
        self.config_scaling = _validated_config_scaling(config_scaling)
        self.vector_fields = _validated_vector_fields(vector_fields, self.statistics)

    @classmethod
    def from_json(
        cls,
        path: str | os.PathLike[str],
        module: nn.Module | None = None,
        config_scaling: ConfigScaling = "min_max",
        vector_fields: Sequence[Sequence[str]] = (),
    ) -> "NormalizerWrapper":
        with open(path, encoding="utf-8") as handle:
            return cls(json.load(handle), module, config_scaling, vector_fields)

    def forward(self, batch: FlashXBatch) -> FlashXBatch:
        if self.module is None:
            raise RuntimeError(
                "the NormalizerWrapper has no module to wrap; use normalize and "
                "unnormalize directly"
            )
        prediction: FlashXBatch = self.module(self.normalize(batch))
        # the input's own configs, not the normalized ones transformed back, which
        # differ from them by rounding and would stop the prediction extending it
        return batch.with_fields(
            self._transform_fields(prediction.fields, _unstandardize)
        )

    def normalize(self, batch: FlashXBatch) -> FlashXBatch:
        if batch.is_normalized:
            raise ValueError("the batch is already normalized")
        return self._transform(batch, _standardize, is_normalized=True)

    def unnormalize(self, batch: FlashXBatch) -> FlashXBatch:
        if not batch.is_normalized:
            raise ValueError("the batch is not normalized")
        return self._transform(batch, _unstandardize, is_normalized=False)

    def normalize_field(self, name: str, values: Any) -> Any:
        """values of the field ``name``, a tensor or a float, normalized."""
        return _standardize(values, *self.field_offset_and_scale(name))

    def unnormalize_field(self, name: str, values: Any) -> Any:
        return _unstandardize(values, *self.field_offset_and_scale(name))

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "statistics": self.statistics,
            "config_scaling": self.config_scaling,
            "vector_fields": self.vector_fields,
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        self.statistics = _validated_statistics(state["statistics"])
        self.config_scaling = _validated_config_scaling(state["config_scaling"])
        self.vector_fields = _validated_vector_fields(
            state["vector_fields"], self.statistics
        )

    def _transform(
        self, batch: FlashXBatch, transform: Transform, is_normalized: bool
    ) -> FlashXBatch:
        fields = self._transform_fields(batch.fields, transform)
        scaling = CONFIG_SCALINGS[self.config_scaling]
        configs = [
            _transform_config_groups(
                config, self.statistics["config"], transform, scaling
            )
            for config in batch.configs
        ]
        return FlashXBatch(
            fields, configs, batch.config_keys, batch.device, is_normalized
        )

    def _transform_fields(
        self, fields: Mapping[str, torch.Tensor], transform: Transform
    ) -> dict[str, torch.Tensor]:
        return {
            name: transform(tensor, *self.field_offset_and_scale(name))
            for name, tensor in fields.items()
        }

    def field_offset_and_scale(self, name: str) -> tuple[float, float]:
        """The (offset, scale) a field is normalized by, (x - offset) / scale."""
        _require_field_statistics(self.statistics, name)
        for components in self.vector_fields:
            if name in components:
                return 0.0, _root_mean_square(
                    [self.statistics["fields"][component] for component in components]
                )
        return _mean_and_scale(self.statistics["fields"][name])


def _require_field_statistics(statistics: Mapping[str, Any], name: str) -> None:
    if name not in statistics["fields"]:
        raise KeyError(
            f"no statistics for field {name!r}; there are statistics for "
            f"{sorted(statistics['fields'])}"
        )


def _validated_vector_fields(
    vector_fields: Sequence[Sequence[str]], statistics: Mapping[str, Any]
) -> list[list[str]]:
    validated = [list(components) for components in vector_fields]
    names = [name for components in validated for name in components]
    if len(names) != len(set(names)):
        raise ValueError(f"a field is in more than one vector field: {validated}")
    for components in validated:
        if len(components) < 2:
            raise ValueError(f"a vector field needs two components, not {components}")
        for name in components:
            _require_field_statistics(statistics, name)
    return validated


def _root_mean_square(entries: Sequence[Mapping[str, float]]) -> float:
    """The root mean square of values spread over several components, from each
    component's mean and std: E[x^2] = std^2 + mean^2."""
    mean_square = sum(
        float(entry["std"]) ** 2 + float(entry["mean"]) ** 2 for entry in entries
    ) / len(entries)
    return math.sqrt(mean_square) if mean_square > 0 else 1.0


def _validated_statistics(statistics: Mapping[str, Any]) -> dict[str, Any]:
    missing = sorted({"fields", "config"} - set(statistics))
    if missing:
        raise ValueError(f"the statistics are missing {missing}")
    return {"fields": dict(statistics["fields"]), "config": dict(statistics["config"])}


def _validated_config_scaling(config_scaling: str) -> ConfigScaling:
    if config_scaling not in CONFIG_SCALINGS:
        raise ValueError(
            f"config_scaling must be one of {sorted(CONFIG_SCALINGS)}, not "
            f"{config_scaling!r}"
        )
    return config_scaling  # type: ignore[return-value]


def _mean_and_scale(entry: Mapping[str, float]) -> tuple[float, float]:
    std = float(entry["std"])
    return float(entry["mean"]), std if std > 0 else 1.0


def _center_and_half_range(entry: Mapping[str, float]) -> tuple[float, float]:
    """2 (x - min) / (max - min) - 1 is (x - center) / half range, which maps the
    dataset's range to [-1, 1]."""
    minimum, maximum = float(entry["min"]), float(entry["max"])
    half_range = (maximum - minimum) / 2
    return (minimum + maximum) / 2, half_range if half_range > 0 else 1.0


CONFIG_SCALINGS: dict[str, Scaling] = {
    "standard": _mean_and_scale,
    "min_max": _center_and_half_range,
}


def _transform_config_groups(
    config: Mapping[str, Any],
    statistics: Mapping[str, Any],
    transform: Transform,
    scaling: Scaling,
) -> dict[str, Any]:
    """config with every group but those in UNNORMALIZED_CONFIG_GROUPS transformed."""
    transformed = _transform_config(
        {
            name: value
            for name, value in config.items()
            if name not in UNNORMALIZED_CONFIG_GROUPS
        },
        statistics,
        transform,
        scaling,
        "",
    )
    kept = {
        name: copy.deepcopy(value)
        for name, value in config.items()
        if name in UNNORMALIZED_CONFIG_GROUPS
    }
    return {name: kept[name] if name in kept else transformed[name] for name in config}


def _transform_config(
    values: Mapping[str, Any],
    statistics: Mapping[str, Any],
    transform: Transform,
    scaling: Scaling,
    path: str,
) -> dict[str, Any]:
    """values with every number transformed by the statistics at its key, walking
    the config and its statistics side by side; a list of dicts, such as heaters,
    shares one set of statistics."""
    transformed: dict[str, Any] = {}
    for name, value in values.items():
        key = f"{path}{name}"
        if isinstance(value, Mapping):
            transformed[name] = _transform_config(
                value, statistics.get(name, {}), transform, scaling, f"{key}."
            )
        elif is_sequence_of_mappings(value):
            transformed[name] = [
                _transform_config(
                    entry, statistics.get(name, {}), transform, scaling, f"{key}."
                )
                for entry in value
            ]
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if name not in statistics:
                raise KeyError(f"no statistics for config parameter {key!r}")
            transformed[name] = float(
                transform(float(value), *scaling(statistics[name]))
            )
        else:
            transformed[name] = value
    return transformed
