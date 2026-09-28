from typing import Any

import pytest
import torch

from boiling_data.boiling_data import BoilingSimulation
from boiling_data.flashx_batch import FlashXBatch, FlashXSample, flashx_collater


def _config(wall_temp: float, stefan: float) -> dict[str, Any]:
    return {
        "non_dimensional": {"stefan": stefan, "prandtl": 7.0},
        "discretization": {"geometry": "cartesian", "num_blocks_x": 6},
        "heaters": [{"wallTemp": wall_temp, "nuc_sites_x": [0.0, 0.5]}],
    }


def _batch(configs: list[dict[str, Any]]) -> FlashXBatch:
    return FlashXBatch({"temperature": torch.zeros(len(configs), 2, 4, 3)}, configs)


def test_batch_rejects_fields_whose_batch_size_differs_from_the_configs() -> None:
    with pytest.raises(ValueError, match="batch size"):
        FlashXBatch({"temperature": torch.zeros(3, 2, 4, 3)}, [_config(1.0, 0.5)])


def test_config_tensor_has_one_row_per_sample_in_the_order_given() -> None:
    batch = _batch([_config(1.0, 0.5), _config(2.0, 0.25)])
    tensor = batch.config_tensor(["stefan", "prandtl"], dtype=torch.float64)
    expected = torch.tensor([[0.5, 7.0], [0.25, 7.0]], dtype=torch.float64)
    torch.testing.assert_close(tensor, expected)


@pytest.mark.parametrize("names", [[], "stefan"])
def test_config_tensor_requires_a_non_empty_sequence_of_names(names: Any) -> None:
    with pytest.raises(ValueError, match="non-empty sequence"):
        _batch([_config(1.0, 0.5)]).config_tensor(names)


@pytest.mark.parametrize("name", ["reynolds", "num_blocks_x"])
def test_config_tensor_only_reads_non_dimensional_parameters(name: str) -> None:
    with pytest.raises(KeyError, match="no non-dimensional parameter"):
        _batch([_config(1.0, 0.5)]).config_tensor([name])


def test_config_tensor_from_simulation_parameters(
    bubbleml_case: BoilingSimulation,
) -> None:
    batch = _batch([bubbleml_case.parameters.to_dict()])
    tensor = batch.config_tensor(["stefan", "prandtl"])
    torch.testing.assert_close(tensor, torch.tensor([[0.156, 7.35]]))


def test_fields_and_config_tensor_are_on_the_batch_device() -> None:
    batch = FlashXBatch(
        {"temperature": torch.zeros(1, 2, 4, 3)},
        [_config(1.0, 0.5)],
        device=torch.device("meta"),
    )
    assert batch.device == torch.device("meta")
    assert batch.fields["temperature"].device == batch.device
    assert batch.config_tensor(["stefan"]).device == batch.device


def test_device_defaults_to_cpu() -> None:
    assert _batch([_config(1.0, 0.5)]).device == torch.device("cpu")


def _batch_with_fields() -> FlashXBatch:
    return FlashXBatch(
        {
            "temperature": torch.full((2, 3, 4, 5), 1.0),
            "pressure": torch.full((2, 3, 4, 5), 2.0),
            "velfacex": torch.zeros(2, 3, 4, 6),
        },
        [_config(1.0, 0.5), _config(2.0, 0.25)],
    )


def test_stacked_fields_are_channels_after_the_batch_in_the_given_order() -> None:
    stacked = _batch_with_fields().stacked_fields(["pressure", "temperature"])
    assert stacked.shape == (2, 2, 3, 4, 5)
    torch.testing.assert_close(stacked[:, 0], torch.full((2, 3, 4, 5), 2.0))
    torch.testing.assert_close(stacked[:, 1], torch.full((2, 3, 4, 5), 1.0))


def test_stacked_fields_rejects_fields_of_different_shapes() -> None:
    with pytest.raises(ValueError, match="different shapes"):
        _batch_with_fields().stacked_fields(["temperature", "velfacex"])


def test_stacked_fields_rejects_a_missing_field() -> None:
    with pytest.raises(KeyError, match="massflux"):
        _batch_with_fields().stacked_fields(["temperature", "massflux"])


@pytest.mark.parametrize("names", [[], "temperature"])
def test_stacked_fields_requires_a_non_empty_sequence_of_names(names: Any) -> None:
    with pytest.raises(ValueError, match="non-empty sequence"):
        _batch_with_fields().stacked_fields(names)


def _staggered_batch() -> FlashXBatch:
    """Two samples, three frames, on a grid of 4 x 5 cells."""
    return FlashXBatch(
        {
            "temperature": torch.rand(2, 3, 4, 5),
            "velfacex": torch.arange(6.0).expand(2, 3, 4, 6),
            "velfacey": torch.arange(5.0).reshape(5, 1).expand(2, 3, 5, 5),
        },
        [_config(1.0, 0.5), _config(2.0, 0.25)],
    )


def test_stack_field_cells_gives_every_cell_its_four_face_velocities() -> None:
    batch = _staggered_batch()
    stacked = batch.stack_field_cells(["temperature", "velfacex", "velfacey"])
    assert stacked.shape == (2, 5, 3, 4, 5)
    torch.testing.assert_close(stacked[:, 0], batch.fields["temperature"])
    left, right, bottom, top = stacked[0, 1:, 0].unbind(0)
    torch.testing.assert_close(left[0], torch.arange(5.0))
    torch.testing.assert_close(right[0], torch.arange(1.0, 6.0))
    torch.testing.assert_close(bottom[:, 0], torch.arange(4.0))
    torch.testing.assert_close(top[:, 0], torch.arange(1.0, 5.0))


def test_stack_field_cells_keeps_the_order_given() -> None:
    batch = _staggered_batch()
    stacked = batch.stack_field_cells(["velfacey", "temperature"])
    assert stacked.shape == (2, 3, 3, 4, 5)
    torch.testing.assert_close(stacked[:, 2], batch.fields["temperature"])


def test_stack_field_cells_rejects_a_face_field_off_the_cell_grid() -> None:
    batch = _staggered_batch()
    batch.fields["velfacey"] = torch.zeros(2, 3, 6, 5)
    with pytest.raises(ValueError, match="different shapes"):
        batch.stack_field_cells(["temperature", "velfacey"])


def test_stack_field_cells_rejects_a_missing_field() -> None:
    with pytest.raises(KeyError, match="dfun"):
        _staggered_batch().stack_field_cells(["temperature", "dfun"])


# DataLoader only pins for CUDA, and torch < 2.14 cannot pin when MPS is present
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="pinning memory is only used with CUDA"
)


def test_to_moves_the_fields_to_the_device_and_keeps_the_configs() -> None:
    batch = _batch_with_fields()
    moved = batch.to(torch.device("meta"))
    assert moved.device == torch.device("meta")
    assert all(tensor.is_meta for tensor in moved.fields.values())
    assert moved.configs == batch.configs
    assert batch.fields["temperature"].device == torch.device("cpu")


@requires_cuda
def test_pin_memory_pins_every_field() -> None:
    batch = _batch_with_fields()
    pinned = batch.pin_memory()
    assert all(tensor.is_pinned() for tensor in pinned.fields.values())
    assert not batch.fields["temperature"].is_pinned()
    assert pinned.device == batch.device


def _sample(wall_temp: float) -> FlashXSample:
    return FlashXSample(
        {
            "temperature": torch.full((2, 4, 3), wall_temp),
            "velfacex": torch.zeros(2, 4, 4),
        },
        _config(wall_temp, 0.5),
    )


def test_collater_stacks_fields_along_a_new_batch_dimension() -> None:
    batch = flashx_collater([_sample(1.0), _sample(2.0)])
    assert batch.batch_size == 2
    assert batch.fields["temperature"].shape == (2, 2, 4, 3)
    assert batch.fields["velfacex"].shape == (2, 2, 4, 4)
    torch.testing.assert_close(
        batch.fields["temperature"][1], torch.full((2, 4, 3), 2.0)
    )
    assert [config["heaters"][0]["wallTemp"] for config in batch.configs] == [1.0, 2.0]


def test_collater_moves_the_batch_to_the_device() -> None:
    batch = flashx_collater([_sample(1.0)], device=torch.device("meta"))
    assert batch.fields["temperature"].device == torch.device("meta")


def test_collater_rejects_samples_with_different_fields() -> None:
    mismatched = FlashXSample({"temperature": torch.zeros(2, 4, 3)}, _config(2.0, 0.5))
    with pytest.raises(ValueError, match="fields"):
        flashx_collater([_sample(1.0), mismatched])


def test_collater_rejects_an_empty_list() -> None:
    with pytest.raises(ValueError, match="empty"):
        flashx_collater([])
