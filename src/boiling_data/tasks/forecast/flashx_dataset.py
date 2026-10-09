import bisect
import itertools
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from boiling_data.bubbleml import read_bubbleml, read_bubbleml_num_timesteps
from boiling_data.flashx.batch import FlashXSample, require_non_empty_sequence
from boiling_data.frozen import freeze
from boiling_data.tasks.forecast.flashx_batch import FlashXForecastSample

type Window = tuple[dict[str, torch.Tensor], dict[str, Any]]


class _FlashXForecastWindows(Dataset[FlashXForecastSample]):
    """Every window of ``input_timesteps`` consecutive frames of every BubbleML file,
    paired with the ``target_timesteps`` frames that follow it, as a sliding window
    whose starts are ``stride`` frames apart. Subclasses decide where a window's
    frames come from."""

    def __init__(
        self,
        paths: Sequence[str | os.PathLike[str]],
        field_names: Sequence[str],
        input_timesteps: int,
        target_timesteps: int,
        start_frame: int = 0,
        stride: int = 1,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """start_frame: windows begin at or after this frame of every file, e.g. to
        skip the start-up transient of each run.
        stride: frames between the starts of consecutive windows of a file; above
        one, neighbouring samples share fewer frames and so are less correlated.
        The frames within a window stay consecutive."""
        if not paths:
            raise ValueError("paths must name at least one BubbleML file")
        require_non_empty_sequence("field_names", field_names)
        for argument, timesteps in (
            ("input_timesteps", input_timesteps),
            ("target_timesteps", target_timesteps),
        ):
            if timesteps < 1:
                raise ValueError(f"{argument} must be positive, not {timesteps}")
        if start_frame < 0:
            raise ValueError(f"start_frame must be non-negative, not {start_frame}")
        if stride < 1:
            raise ValueError(f"stride must be positive, not {stride}")
        self.paths = [Path(path) for path in paths]
        self.field_names = list(field_names)
        self.input_timesteps = input_timesteps
        self.target_timesteps = target_timesteps
        self.start_frame = start_frame
        self.stride = stride
        self.dtype = dtype
        self._windows_per_file = [
            self._num_windows(read_bubbleml_num_timesteps(path)) for path in self.paths
        ]
        self._file_offsets = list(
            itertools.accumulate(self._windows_per_file, initial=0)
        )

    def __len__(self) -> int:
        return self._file_offsets[-1]

    @property
    def _window(self) -> int:
        return self.input_timesteps + self.target_timesteps

    def _num_windows(self, num_timesteps: int) -> int:
        spare_frames = num_timesteps - self.start_frame - self._window
        return spare_frames // self.stride + 1 if spare_frames >= 0 else 0

    def __getitem__(self, index: int) -> FlashXForecastSample:
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} is out of range for {len(self)} samples")
        file_index = bisect.bisect_right(self._file_offsets, index) - 1
        start = (
            self.start_frame + (index - self._file_offsets[file_index]) * self.stride
        )
        fields, config = self._read_window(file_index, start)
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

    def _read_window(self, file_index: int, start: int) -> Window:
        """The window's frames from start of every field, and the file's config."""
        raise NotImplementedError


class FlashXForecastDataset(_FlashXForecastWindows):
    """Forecast windows read from disk as they are indexed: each item reads only its
    window, so files are never held in memory and the dataset is safe to use from
    DataLoader worker processes."""

    def _read_window(self, file_index: int, start: int) -> Window:
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
        return fields, simulation.parameters.to_dict()


class FlashXInMemoryForecastDataset(_FlashXForecastWindows):
    """The same windows as FlashXForecastDataset, from every file loaded into CPU
    memory once, when the dataset is built: each file's fields from start_frame on,
    as dtype, and its config, frozen since every sample of the file shares it. Files
    too short for a window are not loaded. Items are views of the loaded tensors,
    so indexing copies nothing."""

    def __init__(
        self,
        paths: Sequence[str | os.PathLike[str]],
        field_names: Sequence[str],
        input_timesteps: int,
        target_timesteps: int,
        start_frame: int = 0,
        stride: int = 1,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            paths,
            field_names,
            input_timesteps,
            target_timesteps,
            start_frame,
            stride,
            dtype,
        )
        self._loaded: dict[int, Window] = {
            file_index: self._load(path)
            for file_index, path in enumerate(self.paths)
            if self._windows_per_file[file_index] > 0
        }

    def _load(self, path: Path) -> Window:
        simulation = read_bubbleml(
            path, field_names=self.field_names, frames=slice(self.start_frame, None)
        )
        fields = {
            name: torch.from_numpy(field.data).to(self.dtype)
            for name, field in simulation.fields.items()
        }
        config: dict[str, Any] = freeze(simulation.parameters.to_dict())
        return fields, config

    def _read_window(self, file_index: int, start: int) -> Window:
        fields, config = self._loaded[file_index]
        offset = start - self.start_frame
        window = slice(offset, offset + self._window)
        return {name: data[window] for name, data in fields.items()}, config
