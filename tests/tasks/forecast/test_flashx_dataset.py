from functools import partial
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from boiling_data.boiling_data import BoilingSimulation
from boiling_data.tasks.forecast.flashx_batch import flashx_collater
from boiling_data.tasks.forecast.flashx_dataset import FlashXForecastDataset

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
