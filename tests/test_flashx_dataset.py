from functools import partial
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from boiling_data.boiling_data import BoilingSimulation
from boiling_data.flashx_batch import flashx_collater
from boiling_data.flashx_dataset import FlashXDataset

FIELD_NAMES = ["temperature", "velfacex"]


def _dataset(paths: list[Path], timesteps_per_sample: int = 2) -> FlashXDataset:
    return FlashXDataset(paths, FIELD_NAMES, timesteps_per_sample)


def test_every_sliding_window_of_every_file_is_a_sample(bubbleml_path: Path) -> None:
    assert len(_dataset([bubbleml_path])) == 2
    assert len(_dataset([bubbleml_path, bubbleml_path])) == 4
    assert len(_dataset([bubbleml_path], timesteps_per_sample=3)) == 1
    assert len(_dataset([bubbleml_path], timesteps_per_sample=4)) == 0


def test_sample_holds_the_window_of_the_named_fields(
    bubbleml_path: Path, bubbleml_case: BoilingSimulation
) -> None:
    sample = _dataset([bubbleml_path])[1]
    assert set(sample.fields) == set(FIELD_NAMES)
    assert sample.fields["temperature"].dtype == torch.float32
    assert sample.fields["velfacex"].shape == (2, 288, 97)
    expected = torch.from_numpy(bubbleml_case.field("temperature").data[1:3])
    torch.testing.assert_close(sample.fields["temperature"], expected.float())
    assert sample.config == bubbleml_case.parameters.to_dict()


def test_indices_past_the_first_file_read_the_next_file(bubbleml_path: Path) -> None:
    dataset = _dataset([bubbleml_path, bubbleml_path])
    torch.testing.assert_close(
        dataset[3].fields["temperature"], dataset[1].fields["temperature"]
    )


def test_out_of_range_index_raises(bubbleml_path: Path) -> None:
    with pytest.raises(IndexError):
        _dataset([bubbleml_path])[2]


def test_missing_field_raises(bubbleml_path: Path) -> None:
    dataset = FlashXDataset([bubbleml_path], ["temperature", "velz"], 1)
    with pytest.raises(KeyError, match="velz"):
        dataset[0]


@pytest.mark.parametrize(
    ("field_names", "timesteps_per_sample"),
    [([], 1), ("temperature", 1), (FIELD_NAMES, 0)],
)
def test_invalid_arguments_raise(
    bubbleml_path: Path, field_names: list[str], timesteps_per_sample: int
) -> None:
    with pytest.raises(ValueError):
        FlashXDataset([bubbleml_path], field_names, timesteps_per_sample)


def test_data_loader_batches_every_frame_of_the_case_in_order(
    bubbleml_path: Path, bubbleml_case: BoilingSimulation
) -> None:
    field_names = ["temperature", "dfun", "velfacex", "velfacey"]
    dataset = FlashXDataset([bubbleml_path], field_names, timesteps_per_sample=1)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=partial(flashx_collater, device=torch.device("cpu")),
    )

    batches = list(loader)

    assert [batch.batch_size for batch in batches] == [2, 1]
    stacked = torch.cat([batch.stack_field_cells(field_names) for batch in batches])
    assert stacked.shape == (3, 6, 1, 288, 96)

    def from_file(name: str) -> torch.Tensor:
        return torch.from_numpy(bubbleml_case.field(name).data).float()

    velfacex, velfacey = from_file("velfacex"), from_file("velfacey")
    expected_channels = [
        from_file("temperature"),
        from_file("dfun"),
        velfacex[..., :-1],
        velfacex[..., 1:],
        velfacey[..., :-1, :],
        velfacey[..., 1:, :],
    ]
    for channel, expected in enumerate(expected_channels):
        torch.testing.assert_close(stacked[:, channel, 0], expected)
    configs = torch.cat(
        [batch.config_tensor(["stefan", "prandtl"]) for batch in batches]
    )
    torch.testing.assert_close(configs, torch.tensor([[0.156, 7.35]] * 3))


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
    assert all(tensor.is_pinned() for tensor in batch.fields.values())
    on_gpu = batch.to(torch.device("cuda"), non_blocking=True)
    assert on_gpu.fields["temperature"].is_cuda
