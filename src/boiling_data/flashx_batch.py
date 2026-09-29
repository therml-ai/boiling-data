from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from boiling_data.flashx import FIELD_NAMES

CPU = torch.device("cpu")
# the last axis of a field is x and the one before it y
FACE_FIELD_AXES = {FIELD_NAMES["fv_x"]: -1, FIELD_NAMES["fv_y"]: -2}
NON_DIMENSIONAL_GROUP = "non_dimensional"
HEATERS_GROUP = "heaters"


@dataclass
class FlashXSample:
    """One Flash-X sample: unbatched field tensors and its raw config dict."""

    fields: dict[str, torch.Tensor]
    config: dict[str, Any]


@dataclass
class FlashXBatch:
    """A batch of Flash-X samples.

    fields: keyed by field name, each a tensor whose leading dimension is the batch.
    configs: the raw config dict of every sample in the batch, in batch order.
    device: where the fields are moved to and the config tensor is created.
    """

    fields: dict[str, torch.Tensor]
    configs: list[dict[str, Any]]
    device: torch.device = CPU

    def __post_init__(self) -> None:
        self.fields = {
            name: tensor.to(self.device) for name, tensor in self.fields.items()
        }
        for name, tensor in self.fields.items():
            if tensor.shape[0] != self.batch_size:
                raise ValueError(
                    f"field {name!r} has batch size {tensor.shape[0]} but there are "
                    f"{self.batch_size} configs"
                )

    @property
    def batch_size(self) -> int:
        return len(self.configs)

    def to(self, device: torch.device, non_blocking: bool = False) -> "FlashXBatch":
        """non_blocking only overlaps the copy with compute when the fields are in
        pinned memory, e.g. from a DataLoader with ``pin_memory=True``."""
        fields = {
            name: tensor.to(device, non_blocking=non_blocking)
            for name, tensor in self.fields.items()
        }
        return FlashXBatch(fields, list(self.configs), device)

    def pin_memory(self) -> "FlashXBatch":
        """Called by a DataLoader with ``pin_memory=True``, which otherwise passes
        objects it does not recognize through unpinned."""
        fields = {name: tensor.pin_memory() for name, tensor in self.fields.items()}
        return FlashXBatch(fields, list(self.configs), self.device)

    def config_tensor(
        self,
        non_dimensional: Sequence[str],
        heater: Sequence[str] = (),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """[batch_size, len(non_dimensional) + len(heater)] tensor of every config's
        named non-dimensional parameters followed by its named heater parameters,
        each in the order given. Heater parameters need a config with one heater."""
        require_non_empty_sequence("non_dimensional", non_dimensional)
        if isinstance(heater, str):
            raise ValueError(f"heater must be a sequence of names, not {heater!r}")
        values = [
            [
                _parameter(
                    config.get(NON_DIMENSIONAL_GROUP, {}), name, "non-dimensional"
                )
                for name in non_dimensional
            ]
            + [_parameter(_only_heater(config), name, "heater") for name in heater]
            for config in self.configs
        ]
        return torch.tensor(values, dtype=dtype, device=self.device).reshape(
            self.batch_size, len(non_dimensional) + len(heater)
        )

    def stacked_fields(self, names: Sequence[str]) -> torch.Tensor:
        """[batch_size, len(names), ...] tensor of the named fields as channels, in
        the order given. The fields must share a shape, so cell-centered and face
        fields cannot be mixed; see ``stack_field_cells``."""
        self._require_fields(names)
        return _stack_channels([(name, self.fields[name]) for name in names])

    def stack_field_cells(self, names: Sequence[str]) -> torch.Tensor:
        """[batch_size, num_channels, T, Y, X] tensor of the named fields on the
        cells. A cell-centered field is one channel; a face field is two, the low
        then the high face of every cell (left/right for velfacex, bottom/top for
        velfacey), so velfacex and velfacey give each cell its four face
        velocities."""
        self._require_fields(names)
        channels: list[tuple[str, torch.Tensor]] = []
        for name in names:
            tensor = self.fields[name]
            axis = FACE_FIELD_AXES.get(name)
            if axis is None:
                channels.append((name, tensor))
                continue
            num_cells = tensor.shape[axis] - 1
            channels.append((f"{name} low faces", tensor.narrow(axis, 0, num_cells)))
            channels.append((f"{name} high faces", tensor.narrow(axis, 1, num_cells)))
        return _stack_channels(channels)

    def _require_fields(self, names: Sequence[str]) -> None:
        require_non_empty_sequence("names", names)
        missing = [name for name in names if name not in self.fields]
        if missing:
            raise KeyError(
                f"batch has no fields {missing}; it has {sorted(self.fields)}"
            )


@dataclass
class FlashXForecastBatch:
    """A batch of input windows and the target windows that follow them."""

    input: FlashXBatch
    target: FlashXBatch

    @property
    def batch_size(self) -> int:
        return self.input.batch_size

    def to(
        self, device: torch.device, non_blocking: bool = False
    ) -> "FlashXForecastBatch":
        return FlashXForecastBatch(
            self.input.to(device, non_blocking),
            self.target.to(device, non_blocking),
        )

    def pin_memory(self) -> "FlashXForecastBatch":
        return FlashXForecastBatch(self.input.pin_memory(), self.target.pin_memory())


type FlashXForecastSample = tuple[FlashXSample, FlashXSample]


def flashx_collater(
    samples: Sequence[FlashXForecastSample], device: torch.device = CPU
) -> FlashXForecastBatch:
    """Collate (input, target) sample pairs into one batch of inputs and one of
    targets. Usable as a DataLoader ``collate_fn``; bind ``device`` with
    ``functools.partial``."""
    if not samples:
        raise ValueError("cannot collate an empty list of samples")
    return FlashXForecastBatch(
        _collate_samples([input_sample for input_sample, _ in samples], device),
        _collate_samples([target_sample for _, target_sample in samples], device),
    )


def _collate_samples(
    samples: Sequence[FlashXSample], device: torch.device
) -> FlashXBatch:
    names = samples[0].fields.keys()
    for index, sample in enumerate(samples):
        if sample.fields.keys() != names:
            raise ValueError(
                f"sample {index} has fields {sorted(sample.fields)} but sample 0 "
                f"has {sorted(names)}"
            )
    fields = {
        name: torch.stack([sample.fields[name] for sample in samples]) for name in names
    }
    return FlashXBatch(fields, [sample.config for sample in samples], device)


def _stack_channels(channels: list[tuple[str, torch.Tensor]]) -> torch.Tensor:
    shapes = {label: tuple(tensor.shape) for label, tensor in channels}
    if len(set(shapes.values())) > 1:
        raise ValueError(f"cannot stack fields of different shapes: {shapes}")
    return torch.stack([tensor for _, tensor in channels], dim=1)


def require_non_empty_sequence(argument: str, values: Sequence[str]) -> None:
    # a bare string is a Sequence[str] too, and would be read one character per entry
    if isinstance(values, str) or not values:
        raise ValueError(f"{argument} must be a non-empty sequence, not {values!r}")


def _only_heater(config: dict[str, Any]) -> dict[str, Any]:
    heaters = config.get(HEATERS_GROUP, [])
    if len(heaters) != 1:
        raise ValueError(
            f"heater parameters need a config with one heater, not {len(heaters)}"
        )
    heater: dict[str, Any] = heaters[0]
    return heater


def _parameter(parameters: dict[str, Any], name: str, kind: str) -> float:
    if name not in parameters:
        raise KeyError(
            f"config has no {kind} parameter {name!r}; it has {sorted(parameters)}"
        )
    return float(parameters[name])
