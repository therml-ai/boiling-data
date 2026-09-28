import bisect
import itertools
import os
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import Dataset

from boiling_data.bubbleml import read_bubbleml, read_bubbleml_num_timesteps
from boiling_data.flashx_batch import FlashXSample, require_non_empty_sequence


class FlashXDataset(Dataset[FlashXSample]):
    """Every window of ``timesteps_per_sample`` consecutive frames of every BubbleML
    file, as a sliding window with stride one.

    Each item reads only its window from disk, so files are never held in memory
    and the dataset is safe to use from DataLoader worker processes.
    """

    def __init__(
        self,
        paths: Sequence[str | os.PathLike[str]],
        field_names: Sequence[str],
        timesteps_per_sample: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not paths:
            raise ValueError("paths must name at least one BubbleML file")
        require_non_empty_sequence("field_names", field_names)
        if timesteps_per_sample < 1:
            raise ValueError(
                f"timesteps_per_sample must be positive, not {timesteps_per_sample}"
            )
        self.paths = [Path(path) for path in paths]
        self.field_names = list(field_names)
        self.timesteps_per_sample = timesteps_per_sample
        self.dtype = dtype
        samples_per_file = [
            max(read_bubbleml_num_timesteps(path) - timesteps_per_sample + 1, 0)
            for path in self.paths
        ]
        self._file_offsets = list(itertools.accumulate(samples_per_file, initial=0))

    def __len__(self) -> int:
        return self._file_offsets[-1]

    def __getitem__(self, index: int) -> FlashXSample:
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} is out of range for {len(self)} samples")
        file_index = bisect.bisect_right(self._file_offsets, index) - 1
        start = index - self._file_offsets[file_index]
        simulation = read_bubbleml(
            self.paths[file_index],
            field_names=self.field_names,
            frames=slice(start, start + self.timesteps_per_sample),
        )
        return FlashXSample(
            fields={
                name: torch.from_numpy(field.data).to(self.dtype)
                for name, field in simulation.fields.items()
            },
            config=simulation.parameters.to_dict(),
        )
