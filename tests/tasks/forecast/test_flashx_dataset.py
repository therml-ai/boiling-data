from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from boiling_data.boiling_data import BoilingSimulation, Field, SimulationParameters
from boiling_data.bubbleml import read_bubbleml
from boiling_data.frozen import freeze
from boiling_data.tasks.forecast import flashx_dataset
from boiling_data.tasks.forecast.flashx_batch import flashx_collater
from boiling_data.tasks.forecast.flashx_dataset import (
    FlashXForecastDataset,
    FlashXInMemoryForecastDataset,
)

FIELD_NAMES = ["temperature", "velfacex"]


def _dataset(
    paths: list[Path], input_timesteps: int = 1, target_timesteps: int = 1
) -> FlashXForecastDataset:
    return FlashXForecastDataset(paths, FIELD_NAMES, input_timesteps, target_timesteps)


def _from_file(case: BoilingSimulation, name: str, frames: slice) -> torch.Tensor:
    return torch.from_numpy(case.field(name).data[frames]).float()


def test_every_sliding_window_of_every_file_is_a_sample(bubbleml_path: Path) -> None:
    assert len(_dataset([bubbleml_path])) == 2
    assert len(_dataset([bubbleml_path, bubbleml_path])) == 4
    assert len(_dataset([bubbleml_path], input_timesteps=2)) == 1
    assert len(_dataset([bubbleml_path], input_timesteps=2, target_timesteps=2)) == 0


def test_target_holds_the_frames_after_the_input(
    bubbleml_path: Path, bubbleml_case: BoilingSimulation
) -> None:
    input_sample, target_sample = _dataset(
        [bubbleml_path], input_timesteps=2, target_timesteps=1
    )[0]
    assert set(input_sample.fields) == set(target_sample.fields) == set(FIELD_NAMES)
    assert input_sample.fields["temperature"].dtype == torch.float32
    assert input_sample.fields["velfacex"].shape == (2, 288, 97)
    assert target_sample.fields["velfacex"].shape == (1, 288, 97)
    torch.testing.assert_close(
        input_sample.fields["temperature"],
        _from_file(bubbleml_case, "temperature", slice(0, 2)),
    )
    torch.testing.assert_close(
        target_sample.fields["temperature"],
        _from_file(bubbleml_case, "temperature", slice(2, 3)),
    )
    assert input_sample.config == target_sample.config
    assert input_sample.config == bubbleml_case.parameters.to_dict()


def test_windows_start_at_the_start_frame(
    bubbleml_path: Path, bubbleml_case: BoilingSimulation
) -> None:
    dataset = FlashXForecastDataset([bubbleml_path], FIELD_NAMES, 1, 1, start_frame=1)
    assert len(dataset) == 1
    input_sample, _ = dataset[0]
    torch.testing.assert_close(
        input_sample.fields["temperature"],
        _from_file(bubbleml_case, "temperature", slice(1, 2)),
    )


def test_indices_past_the_first_file_read_the_next_file(bubbleml_path: Path) -> None:
    dataset = _dataset([bubbleml_path, bubbleml_path])
    torch.testing.assert_close(
        dataset[3][1].fields["temperature"], dataset[1][1].fields["temperature"]
    )


def test_out_of_range_index_raises(bubbleml_path: Path) -> None:
    with pytest.raises(IndexError):
        _dataset([bubbleml_path])[2]


def test_missing_field_raises(bubbleml_path: Path) -> None:
    dataset = FlashXForecastDataset([bubbleml_path], ["temperature", "velz"], 1, 1)
    with pytest.raises(KeyError, match="velz"):
        dataset[0]


@pytest.mark.parametrize(
    ("field_names", "input_timesteps", "target_timesteps", "message"),
    [
        ([], 1, 1, "non-empty sequence"),
        ("temperature", 1, 1, "non-empty sequence"),
        (FIELD_NAMES, 0, 1, "input_timesteps"),
        (FIELD_NAMES, 1, 0, "target_timesteps"),
    ],
)
def test_invalid_arguments_raise(
    bubbleml_path: Path,
    field_names: list[str],
    input_timesteps: int,
    target_timesteps: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        FlashXForecastDataset(
            [bubbleml_path], field_names, input_timesteps, target_timesteps
        )


def test_negative_start_frame_raises(bubbleml_path: Path) -> None:
    with pytest.raises(ValueError, match="start_frame"):
        FlashXForecastDataset([bubbleml_path], FIELD_NAMES, 1, 1, start_frame=-1)


def test_data_loader_batches_inputs_and_targets_of_the_case(
    bubbleml_path: Path, bubbleml_case: BoilingSimulation
) -> None:
    field_names = ["temperature", "dfun", "velfacex", "velfacey"]
    dataset = FlashXForecastDataset([bubbleml_path], field_names, 1, 1)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=partial(
            flashx_collater,
            config_keys={"non_dimensional": ["stefan", "prandtl"]},
            device=torch.device("cpu"),
        ),
    )

    (batch,) = list(loader)

    assert batch.batch_size == 2
    for windows, frames in ((batch.input, slice(0, 2)), (batch.target, slice(1, 3))):
        stacked = windows.stack_field_cells(field_names)
        assert stacked.shape == (2, 1, 288, 96, 6)
        velfacex = _from_file(bubbleml_case, "velfacex", frames)
        velfacey = _from_file(bubbleml_case, "velfacey", frames)
        expected_channels = [
            _from_file(bubbleml_case, "temperature", frames),
            _from_file(bubbleml_case, "dfun", frames),
            velfacex[..., :-1],
            velfacex[..., 1:],
            velfacey[..., :-1, :],
            velfacey[..., 1:, :],
        ]
        for channel, expected in enumerate(expected_channels):
            torch.testing.assert_close(stacked[:, 0, ..., channel], expected)
    torch.testing.assert_close(
        batch.input.config_tensor(),
        torch.tensor([[0.156, 7.35]] * 2),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="DataLoader pins memory only for CUDA"
)
def test_data_loader_pins_batches_for_a_non_blocking_copy(bubbleml_path: Path) -> None:
    loader = DataLoader(
        _dataset([bubbleml_path]),
        batch_size=2,
        collate_fn=flashx_collater,
        pin_memory=True,
    )
    (batch,) = list(loader)
    for windows in (batch.input, batch.target):
        assert all(tensor.is_pinned() for tensor in windows.fields.values())
    on_gpu = batch.to(torch.device("cuda"), non_blocking=True)
    assert on_gpu.input.fields["temperature"].is_cuda
    assert on_gpu.target.fields["temperature"].is_cuda


@pytest.fixture
def two_lengths(bubbleml_path: Path, tmp_path: Path) -> list[Path]:
    """The three-frame test case and a two-frame copy of it."""
    short = read_bubbleml(bubbleml_path, frames=slice(0, 2))
    return [bubbleml_path, short.to_bubbleml(tmp_path / "short.hdf5")]


@pytest.mark.parametrize(
    ("input_timesteps", "target_timesteps", "start_frame"),
    [(1, 1, 0), (2, 1, 0), (1, 1, 1), (1, 2, 0)],
)
def test_in_memory_items_match_the_items_read_from_disk(
    two_lengths: list[Path],
    input_timesteps: int,
    target_timesteps: int,
    start_frame: int,
) -> None:
    arguments = (
        two_lengths,
        FIELD_NAMES,
        input_timesteps,
        target_timesteps,
        start_frame,
    )
    on_disk = FlashXForecastDataset(*arguments)
    in_memory = FlashXInMemoryForecastDataset(*arguments)
    assert len(in_memory) == len(on_disk)
    for index in range(len(on_disk)):
        for expected, actual in zip(on_disk[index], in_memory[index], strict=True):
            # the in-memory configs are frozen: their lists are tuples
            assert actual.config == freeze(expected.config)
            for name in FIELD_NAMES:
                torch.testing.assert_close(actual.fields[name], expected.fields[name])


def _count_reads(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    reads: list[Path] = []

    def counting_read(path: Path, *args: object, **kwargs: object) -> BoilingSimulation:
        reads.append(Path(path))
        return read_bubbleml(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(flashx_dataset, "read_bubbleml", counting_read)
    return reads


def test_files_are_read_once_when_the_dataset_is_built(
    two_lengths: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    reads = _count_reads(monkeypatch)
    dataset = FlashXInMemoryForecastDataset(two_lengths, FIELD_NAMES, 1, 1)
    assert reads == two_lengths
    for index in range(len(dataset)):
        dataset[index]
    assert reads == two_lengths


def test_files_too_short_for_a_window_are_not_loaded(
    two_lengths: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    reads = _count_reads(monkeypatch)
    dataset = FlashXInMemoryForecastDataset(two_lengths, FIELD_NAMES, 2, 1)
    assert reads == two_lengths[:1]
    assert len(dataset) == 1


def test_in_memory_configs_cannot_be_changed(bubbleml_path: Path) -> None:
    input_sample, _ = FlashXInMemoryForecastDataset([bubbleml_path], FIELD_NAMES, 1, 1)[
        0
    ]
    with pytest.raises(TypeError, match="cannot be changed"):
        input_sample.config["physical"]["bulk_temp"] = 0.0


def test_in_memory_fields_have_the_requested_dtype(bubbleml_path: Path) -> None:
    dataset = FlashXInMemoryForecastDataset(
        [bubbleml_path], FIELD_NAMES, 1, 1, dtype=torch.float64
    )
    input_sample, target_sample = dataset[0]
    assert input_sample.fields["temperature"].dtype == torch.float64
    assert target_sample.fields["velfacex"].dtype == torch.float64


def test_in_memory_dataset_batches_like_the_disk_dataset(bubbleml_path: Path) -> None:
    config_keys = {"non_dimensional": ["stefan"]}
    loaders = [
        DataLoader(
            dataset_class([bubbleml_path], FIELD_NAMES, 1, 1),
            batch_size=2,
            collate_fn=partial(flashx_collater, config_keys=config_keys),
        )
        for dataset_class in (FlashXForecastDataset, FlashXInMemoryForecastDataset)
    ]
    on_disk, in_memory = (next(iter(loader)) for loader in loaders)
    torch.testing.assert_close(
        in_memory.input.fields["temperature"], on_disk.input.fields["temperature"]
    )
    torch.testing.assert_close(
        in_memory.target.config_tensor(), on_disk.target.config_tensor()
    )


@pytest.fixture
def ten_frames(tmp_path: Path) -> Path:
    """Ten frames whose every value is the frame index."""
    frames = np.arange(10, dtype=np.float64)[:, None, None] * np.ones((1, 2, 2))
    centers, faces = np.array([0.25, 0.75]), np.array([0.0, 0.5, 1.0])
    simulation = BoilingSimulation(
        fields={"temperature": Field(frames, grid_x=centers, grid_y=centers)},
        parameters=SimulationParameters(
            discretization={
                "num_blocks_x": 1,
                "num_blocks_y": 1,
                "nx_block": 2,
                "ny_block": 2,
                "x_min": faces[0],
                "x_max": faces[-1],
                "y_min": faces[0],
                "y_max": faces[-1],
            }
        ),
        time=np.arange(10, dtype=np.float64),
    )
    return simulation.to_bubbleml(tmp_path / "ten-frames.hdf5")


def _window_starts(dataset: FlashXForecastDataset) -> list[float]:
    return [
        float(dataset[index][0].fields["temperature"][0, 0, 0])
        for index in range(len(dataset))
    ]


@pytest.mark.parametrize(
    ("stride", "start_frame", "expected_starts"),
    [
        (1, 0, [0, 1, 2, 3, 4, 5, 6, 7, 8]),
        (3, 0, [0, 3, 6]),
        (3, 1, [1, 4, 7]),
        (8, 0, [0, 8]),
        (9, 0, [0]),
    ],
)
def test_windows_start_stride_frames_apart(
    ten_frames: Path, stride: int, start_frame: int, expected_starts: list[int]
) -> None:
    dataset = FlashXForecastDataset(
        [ten_frames], ["temperature"], 1, 1, start_frame=start_frame, stride=stride
    )
    assert _window_starts(dataset) == expected_starts


def test_frames_within_a_window_stay_consecutive(ten_frames: Path) -> None:
    history, future = FlashXForecastDataset(
        [ten_frames], ["temperature"], 3, 2, stride=4
    )[1]
    assert history.fields["temperature"][:, 0, 0].tolist() == [4.0, 5.0, 6.0]
    assert future.fields["temperature"][:, 0, 0].tolist() == [7.0, 8.0]


def test_stride_counts_windows_across_files(ten_frames: Path) -> None:
    dataset = FlashXForecastDataset(
        [ten_frames, ten_frames], ["temperature"], 1, 1, stride=3
    )
    assert len(dataset) == 6
    assert _window_starts(dataset) == [0, 3, 6, 0, 3, 6]


@pytest.mark.parametrize("stride", [1, 2, 4])
def test_in_memory_dataset_strides_like_the_disk_dataset(
    ten_frames: Path, stride: int
) -> None:
    arguments = ([ten_frames], ["temperature"], 2, 1)
    on_disk = FlashXForecastDataset(*arguments, start_frame=1, stride=stride)
    in_memory = FlashXInMemoryForecastDataset(*arguments, start_frame=1, stride=stride)
    assert len(in_memory) == len(on_disk)
    for index in range(len(on_disk)):
        torch.testing.assert_close(
            in_memory[index][1].fields["temperature"],
            on_disk[index][1].fields["temperature"],
        )


@pytest.mark.parametrize("stride", [0, -1])
def test_stride_must_be_positive(ten_frames: Path, stride: int) -> None:
    with pytest.raises(ValueError, match="stride must be positive"):
        FlashXForecastDataset([ten_frames], ["temperature"], 1, 1, stride=stride)
