import bisect
import itertools
import os
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import Dataset

from boiling_data.bubbleml import read_bubbleml, read_bubbleml_num_timesteps
from boiling_data.flashx_batch import (
    FlashXForecastSample,
    FlashXSample,
    require_non_empty_sequence,
)


class FlashXDataset(Dataset[FlashXForecastSample]):
    """Every window of ``input_timesteps`` consecutive frames of every BubbleML file,
    paired with the ``target_timesteps`` frames that follow it, as a sliding window
    with stride one.

    Each item reads only its window from disk, so files are never held in memory
    and the dataset is safe to use from DataLoader worker processes.
    """

    def __init__(
        self,
        paths: Sequence[str | os.PathLike[str]],
        field_names: Sequence[str],
        input_timesteps: int,
        target_timesteps: int,
        start_frame: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """start_frame: windows begin at or after this frame of every file, e.g. to
        skip the start-up transient of each run."""
        if not paths:
            raise ValueError("paths must name at least one BubbleML file")
        require_non_empty_sequence("field_names", field_names)
        for argument, timesteps in (
            ("input_timesteps", input_timesteps),
            ("target_timesteps", target_timesteps),
        ):
            if timesteps < 1:
                raise ValueError(f"{argument} must be positive, not {timesteps}")
        self.paths = [Path(path) for path in paths]
        self.field_names = list(field_names)
        if start_frame < 0:
            raise ValueError(f"start_frame must be non-negative, not {start_frame}")
        self.input_timesteps = input_timesteps
        self.target_timesteps = target_timesteps
        self.start_frame = start_frame
        self.dtype = dtype
        samples_per_file = [
            max(
                read_bubbleml_num_timesteps(path) - start_frame - self._window + 1,
                0,
            )
            for path in self.paths
        ]
        self._file_offsets = list(itertools.accumulate(samples_per_file, initial=0))

    def __len__(self) -> int:
        return self._file_offsets[-1]

    @property
    def _window(self) -> int:
        return self.input_timesteps + self.target_timesteps

    def __getitem__(self, index: int) -> FlashXForecastSample:
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} is out of range for {len(self)} samples")
        file_index = bisect.bisect_right(self._file_offsets, index) - 1
        start = self.start_frame + index - self._file_offsets[file_index]
        # one read covers the input and target frames, which are contiguous
        simulation = read_bubbleml(
            self.paths[file_index],
            field_names=self.field_names,
            frames=slice(start, start + self._window),
        )
        fields = {
            name: torch.from_numpy(field.data).to(self.dtype)
            for name, field in simulation.fields.items()
        }
        config = simulation.parameters.to_dict()
        return (
            FlashXSample(
                {name: data[: self.input_timesteps] for name, data in fields.items()},
                config,
            ),
            FlashXSample(
                {name: data[self.input_timesteps :] for name, data in fields.items()},
                config,
            ),
        )
