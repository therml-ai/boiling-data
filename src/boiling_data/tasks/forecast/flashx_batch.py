from collections.abc import Sequence
from dataclasses import dataclass

import torch

from boiling_data.flashx.batch import (
    ConfigKeys,
    FlashXBatch,
    FlashXSample,
    collate_samples,
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
    device: torch.device | None = None,
) -> FlashXForecastBatch:
    """Collate (input, target) sample pairs into one batch of inputs and one of
    targets, both with config_keys, on device (the CPU by default). Usable as a
    DataLoader ``collate_fn``; bind ``config_keys`` and ``device`` with
    ``functools.partial``."""
    if not samples:
        raise ValueError("cannot collate an empty list of samples")
    config_keys = config_keys or {}
    device = torch.device("cpu") if device is None else device
    return FlashXForecastBatch(
        collate_samples(
            [input_sample for input_sample, _ in samples], config_keys, device
        ),
        collate_samples(
            [target_sample for _, target_sample in samples], config_keys, device
        ),
    )
