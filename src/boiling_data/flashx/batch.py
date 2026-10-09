from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from boiling_data.flashx.reader import FIELD_NAMES

CPU = torch.device("cpu")
# the last axis of a field is x and the one before it y
FACE_FIELD_AXES = {FIELD_NAMES["fv_x"]: -1, FIELD_NAMES["fv_y"]: -2}
HEATERS_GROUP = "heaters"
# the config groups a config tensor can read; heaters reads the sample's one heater
CONFIG_GROUPS = ("physical", "non_dimensional", "discretization", HEATERS_GROUP)

type ConfigKeys = Mapping[str, Sequence[str]]


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
    config_keys: the parameters the config tensor holds, by config group, e.g.
        ``{"non_dimensional": ["stefan"], "heaters": ["wall_temp_fraction"]}``;
        its columns follow the groups and then the names in the order given.
    device: where the fields are moved to and the config tensor is created.
    """

    fields: dict[str, torch.Tensor]
    configs: list[dict[str, Any]]
    config_keys: ConfigKeys = field(default_factory=dict)
    device: torch.device = CPU
    _config_rows: list[list[float]] = field(init=False, repr=False)

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
        self.config_keys = _validated_config_keys(self.config_keys)
        _check_configs_have_keys(self.configs, self.config_keys)
        self._config_rows = [
            _config_row(config, self.config_keys) for config in self.configs
        ]

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
        return FlashXBatch(fields, list(self.configs), self.config_keys, device)

    def pin_memory(self) -> "FlashXBatch":
        """Called by a DataLoader with ``pin_memory=True``, which otherwise passes
        objects it does not recognize through unpinned."""
        fields = {name: tensor.pin_memory() for name, tensor in self.fields.items()}
        return FlashXBatch(fields, list(self.configs), self.config_keys, self.device)

    def config_tensor(self) -> torch.Tensor:
        """[batch_size, number of config keys] float32 tensor of every config's
        parameters named in config_keys."""
        if not self.config_keys:
            raise ValueError(
                "the batch has no config_keys; pass them to FlashXBatch or "
                "flashx_collater to build a config tensor"
            )
        width = sum(len(names) for names in self.config_keys.values())
        return torch.tensor(
            self._config_rows, dtype=torch.float32, device=self.device
        ).reshape(self.batch_size, width)

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
    samples: Sequence[FlashXForecastSample],
    config_keys: ConfigKeys | None = None,
    device: torch.device = CPU,
) -> FlashXForecastBatch:
    """Collate (input, target) sample pairs into one batch of inputs and one of
    targets, both with config_keys. Usable as a DataLoader ``collate_fn``; bind
    ``config_keys`` and ``device`` with ``functools.partial``."""
    if not samples:
        raise ValueError("cannot collate an empty list of samples")
    config_keys = config_keys or {}
    return FlashXForecastBatch(
        _collate_samples(
            [input_sample for input_sample, _ in samples], config_keys, device
        ),
        _collate_samples(
            [target_sample for _, target_sample in samples], config_keys, device
        ),
    )


def _collate_samples(
    samples: Sequence[FlashXSample], config_keys: ConfigKeys, device: torch.device
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
    return FlashXBatch(
        fields, [sample.config for sample in samples], config_keys, device
    )


def _stack_channels(channels: list[tuple[str, torch.Tensor]]) -> torch.Tensor:
    shapes = {label: tuple(tensor.shape) for label, tensor in channels}
    if len(set(shapes.values())) > 1:
        raise ValueError(f"cannot stack fields of different shapes: {shapes}")
    return torch.stack([tensor for _, tensor in channels], dim=1)


def require_non_empty_sequence(argument: str, values: Sequence[str]) -> None:
    # a bare string is a Sequence[str] too, and would be read one character per entry
    if isinstance(values, str) or not values:
        raise ValueError(f"{argument} must be a non-empty sequence, not {values!r}")


def _validated_config_keys(config_keys: ConfigKeys) -> dict[str, list[str]]:
    unknown = sorted(set(config_keys) - set(CONFIG_GROUPS))
    if unknown:
        raise ValueError(
            f"config_keys has unknown groups {unknown}; the groups are {CONFIG_GROUPS}"
        )
    for group, names in config_keys.items():
        require_non_empty_sequence(f"config_keys[{group!r}]", names)
    return {group: list(names) for group, names in config_keys.items()}


def _check_configs_have_keys(
    configs: Sequence[dict[str, Any]], config_keys: ConfigKeys
) -> None:
    """Every config holds every key, so the batch fails when it is built, naming
    all that is missing from every sample rather than the first gap found."""
    missing: dict[int, list[str]] = {}
    for index, config in enumerate(configs):
        for group, names in config_keys.items():
            if group == HEATERS_GROUP:
                heaters = config.get(HEATERS_GROUP, [])
                if len(heaters) != 1:
                    raise ValueError(
                        f"sample {index}: heater parameters need a config with one "
                        f"heater, not {len(heaters)}"
                    )
                parameters = heaters[0]
            else:
                parameters = config.get(group, {})
            missing.setdefault(index, []).extend(
                f"{group}.{name}" for name in names if name not in parameters
            )
    missing = {index: keys for index, keys in missing.items() if keys}
    if missing:
        raise KeyError(f"configs are missing config keys, by sample: {missing}")


def _config_row(config: dict[str, Any], config_keys: ConfigKeys) -> list[float]:
    row = []
    for group, names in config_keys.items():
        if group == HEATERS_GROUP:
            parameters, kind = _only_heater(config), "heater"
        else:
            parameters, kind = config.get(group, {}), group
        row += [_parameter(parameters, name, kind) for name in names]
    return row


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
