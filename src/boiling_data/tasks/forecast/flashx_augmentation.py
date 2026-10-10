import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from boiling_data.flashx.batch import FlashXBatch
from boiling_data.flashx.symmetry import NEGATED_BY_X_FLIP, NEGATED_BY_Y_FLIP
from boiling_data.tasks.forecast.flashx_batch import FlashXForecastBatch

GRAVITY = ("gravx", "gravy", "gravz")


@dataclass(frozen=True)
class FlipAxis:
    """What a reflection along one axis of [..., Y, X] fields changes."""

    gravity: str
    dim: int
    negated_fields: frozenset[str]
    sides: tuple[str, str]
    heater_extent: tuple[str, str]
    heater_sites: str
    domain: tuple[str, str]


FLIP_AXES = (
    FlipAxis(
        "gravx",
        -1,
        NEGATED_BY_X_FLIP,
        ("left", "right"),
        ("xMin", "xMax"),
        "nuc_sites_x",
        ("x_min", "x_max"),
    ),
    FlipAxis(
        "gravy",
        -2,
        NEGATED_BY_Y_FLIP,
        ("bottom", "top"),
        ("yMin", "yMax"),
        "nuc_sites_y",
        ("y_min", "y_max"),
    ),
)


def flippable_axes(config: Mapping[str, Any]) -> list[FlipAxis]:
    non_dimensional = config.get("non_dimensional", {})
    gravity = {name: float(non_dimensional.get(name, 0.0)) for name in GRAVITY}
    if gravity["gravz"] != 0.0:
        raise ValueError(
            f"a sample has gravz = {gravity['gravz']}, gravity out of the 2d plane "
            "the fields lie in"
        )
    if not any(gravity.values()):
        raise ValueError(
            "a sample has no gravity, so there is no direction to choose the axes "
            "to flip from"
        )
    return [axis for axis in FLIP_AXES if gravity[axis.gravity] == 0.0]


class FlipAugmentation(nn.Module):
    def __init__(
        self, flip_probability: float = 0.5, generator: torch.Generator | None = None
    ) -> None:
        super().__init__()
        if not 0.0 <= flip_probability <= 1.0:
            raise ValueError(
                f"flip_probability must be in [0, 1], not {flip_probability}"
            )
        self.flip_probability = flip_probability
        self.generator = generator

    def forward(self, batch: FlashXForecastBatch) -> FlashXForecastBatch:
        if not self.training:
            return batch
        _require_two_dimensional(batch)
        flips = self._choose_flips(batch.input.configs)
        configs = [
            reflected_config(config, axes)
            for config, axes in zip(batch.input.configs, flips, strict=True)
        ]
        return FlashXForecastBatch(
            _flip_fields(batch.input, flips, configs),
            _flip_fields(batch.target, flips, configs),
        )

    def _choose_flips(
        self, configs: Sequence[Mapping[str, Any]]
    ) -> list[list[FlipAxis]]:
        draws = torch.rand(len(configs), len(FLIP_AXES), generator=self.generator)
        return [
            [
                axis
                for axis, draw in zip(FLIP_AXES, sample_draws.tolist(), strict=True)
                if axis in flippable_axes(config) and draw < self.flip_probability
            ]
            for config, sample_draws in zip(configs, draws, strict=True)
        ]


class GaussianNoiseAugmentation(nn.Module):
    def __init__(
        self,
        min_std: float,
        max_std: float,
        fields: Sequence[str] | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if not 0 < min_std <= max_std:
            raise ValueError(
                f"the noise stds need 0 < min_std <= max_std, not min_std={min_std} "
                f"and max_std={max_std}"
            )
        self.min_std = min_std
        self.max_std = max_std
        self.fields = None if fields is None else list(fields)
        self.generator = generator

    def forward(self, batch: FlashXForecastBatch) -> FlashXForecastBatch:
        if not self.training:
            return batch
        names = list(batch.input.fields) if self.fields is None else self.fields
        stds = self.sample_stds(batch.input.batch_size, batch.input.device)
        fields = dict(batch.input.fields)
        for name in names:
            fields[name] = fields[name] + self._noise(fields[name], stds)
        return FlashXForecastBatch(batch.input.with_fields(fields), batch.target)

    def sample_stds(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """One std per sample, log-uniform in [min_std, max_std]."""
        uniform = torch.rand(
            batch_size, generator=self.generator, device=self._draw_device(device)
        )
        log_min, log_max = math.log(self.min_std), math.log(self.max_std)
        return torch.exp(log_min + uniform * (log_max - log_min)).to(device)

    def _noise(self, field: torch.Tensor, stds: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(
            field.shape,
            generator=self.generator,
            dtype=field.dtype,
            device=self._draw_device(field.device),
        ).to(field.device)
        return noise * stds.to(field.dtype).view(-1, *([1] * (field.ndim - 1)))

    def _draw_device(self, device: torch.device) -> torch.device:
        # a generator can only draw on its own device, which need not be the batch's
        return device if self.generator is None else self.generator.device


class ForecastAugmentation(nn.Module):
    """Applies augmentations, such as FlipAugmentation, one after another to whole
    FlashXForecastBatches while training. In eval mode batches pass through unchanged.
    """

    def __init__(self, augmentations: Sequence[nn.Module] = ()) -> None:
        super().__init__()
        # a ModuleList, so train() and eval() reach every augmentation
        self.augmentations = nn.ModuleList(augmentations)

    def forward(self, batch: FlashXForecastBatch) -> FlashXForecastBatch:
        if not self.training:
            return batch
        for augmentation in self.augmentations:
            batch = augmentation(batch)
        return batch


def _require_two_dimensional(batch: FlashXForecastBatch) -> None:
    for windows in (batch.input, batch.target):
        for name, tensor in windows.fields.items():
            if tensor.ndim != 4:
                raise ValueError(
                    f"field {name!r} of shape {tuple(tensor.shape)} is not a 2d "
                    "[batch, time, y, x] field, the only kind FlipAugmentation "
                    "flips"
                )


def _flip_fields(
    windows: FlashXBatch,
    flips: Sequence[Sequence[FlipAxis]],
    configs: Sequence[Mapping[str, Any]],
) -> FlashXBatch:
    fields = dict(windows.fields)
    for axis in FLIP_AXES:
        selected = torch.tensor([axis in axes for axes in flips], device=windows.device)
        if not selected.any():
            continue
        for name, tensor in fields.items():
            flipped = torch.flip(tensor, dims=[axis.dim])
            if name in axis.negated_fields:
                flipped = -flipped
            mask = selected.view(-1, *([1] * (tensor.ndim - 1)))
            fields[name] = torch.where(mask, flipped, tensor)
    return FlashXBatch(
        fields, configs, windows.config_keys, windows.device, windows.is_normalized
    )


def reflected_config(
    config: Mapping[str, Any], axes: Sequence[FlipAxis]
) -> dict[str, Any]:
    """The config of a sample mirrored along each of axes."""
    reflected: dict[str, Any] = _thawed(config)
    for axis in axes:
        low, high = (float(reflected["discretization"][key]) for key in axis.domain)
        boundary = reflected.get("boundary", {})
        first, second = axis.sides
        boundary[first], boundary[second] = (
            boundary.get(second, {}),
            boundary.get(first, {}),
        )
        for heater in reflected.get("heaters", []):
            _reflect_heater(heater, axis, low, high)
    return reflected


def _reflect_heater(
    heater: dict[str, Any], axis: FlipAxis, low: float, high: float
) -> None:
    lower, upper = axis.heater_extent
    if lower in heater and upper in heater:
        heater[lower], heater[upper] = (
            low + high - heater[upper],
            low + high - heater[lower],
        )
    if axis.heater_sites in heater:
        heater[axis.heater_sites] = [
            low + high - site for site in heater[axis.heater_sites]
        ]


def _thawed(value: Any) -> Any:
    """A mutable copy of a possibly frozen config."""
    if isinstance(value, Mapping):
        return {key: _thawed(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thawed(item) for item in value]
    return value
