import dataclasses
import pickle
from typing import Any

import pytest
import torch

from boiling_data.boiling_data import BoilingSimulation
from boiling_data.flashx.batch import (
    FlashXBatch,
    unstack_field_cells,
)


def _config(wall_temp: float, stefan: float) -> dict[str, Any]:
    return {
        "non_dimensional": {"stefan": stefan, "prandtl": 7.0},
        "discretization": {"geometry": "cartesian", "num_blocks_x": 6},
        "heaters": [{"wallTemp": wall_temp, "nuc_sites_x": [0.0, 0.5]}],
    }


NON_DIMENSIONAL_KEYS = {"non_dimensional": ["stefan", "prandtl"]}


def _batch(
    configs: list[dict[str, Any]], config_keys: dict[str, list[str]] | None = None
) -> FlashXBatch:
    return FlashXBatch(
        {"temperature": torch.zeros(len(configs), 2, 4, 3)},
        configs,
        config_keys or {},
    )


def test_batch_rejects_fields_whose_batch_size_differs_from_the_configs() -> None:
    with pytest.raises(ValueError, match="batch size"):
        FlashXBatch({"temperature": torch.zeros(3, 2, 4, 3)}, [_config(1.0, 0.5)])


def test_config_tensor_has_one_row_per_sample_in_the_key_order() -> None:
    batch = _batch([_config(1.0, 0.5), _config(2.0, 0.25)], NON_DIMENSIONAL_KEYS)
    tensor = batch.config_tensor()
    assert tensor.dtype == torch.float32
    torch.testing.assert_close(tensor, torch.tensor([[0.5, 7.0], [0.25, 7.0]]))


def test_heater_parameters_follow_the_groups_in_order() -> None:
    keys = {"non_dimensional": ["stefan"], "heaters": ["wallTemp"]}
    batch = _batch([_config(1.0, 0.5), _config(2.0, 0.25)], keys)
    torch.testing.assert_close(
        batch.config_tensor(), torch.tensor([[0.5, 1.0], [0.25, 2.0]])
    )


def test_config_keys_can_read_any_numeric_group() -> None:
    keys = {"discretization": ["num_blocks_x"], "non_dimensional": ["stefan"]}
    batch = _batch([_config(1.0, 0.5)], keys)
    torch.testing.assert_close(batch.config_tensor(), torch.tensor([[6.0, 0.5]]))


@pytest.mark.parametrize("names", [[], "stefan"])
def test_config_keys_need_a_non_empty_sequence_of_names(names: Any) -> None:
    with pytest.raises(ValueError, match="non-empty sequence"):
        _batch([_config(1.0, 0.5)], {"non_dimensional": names})


def test_config_keys_reject_an_unknown_group() -> None:
    with pytest.raises(ValueError, match="unknown groups"):
        _batch([_config(1.0, 0.5)], {"fluid": ["stefan"]})


def test_missing_parameter_fails_when_the_batch_is_built() -> None:
    with pytest.raises(KeyError, match=r"\{0: \['non_dimensional.reynolds'\]\}"):
        _batch([_config(1.0, 0.5)], {"non_dimensional": ["reynolds"]})
    with pytest.raises(KeyError, match=r"\{0: \['heaters.advAngle'\]\}"):
        _batch([_config(1.0, 0.5)], {"heaters": ["advAngle"]})


def test_every_sample_is_checked_and_every_gap_reported() -> None:
    gappy = _config(2.0, 0.25)
    del gappy["non_dimensional"]["prandtl"]
    del gappy["heaters"][0]["wallTemp"]
    keys = {"non_dimensional": ["stefan", "prandtl"], "heaters": ["wallTemp"]}
    with pytest.raises(KeyError) as error:
        _batch([_config(1.0, 0.5), gappy], keys)
    assert "{1: ['non_dimensional.prandtl', 'heaters.wallTemp']}" in str(error.value)


def test_heater_parameters_need_exactly_one_heater() -> None:
    config = _config(1.0, 0.5)
    config["heaters"].append(dict(config["heaters"][0]))
    with pytest.raises(ValueError, match="one heater, not 2"):
        _batch([config], {"heaters": ["wallTemp"]})


def test_config_tensor_needs_config_keys() -> None:
    with pytest.raises(ValueError, match="no config_keys"):
        _batch([_config(1.0, 0.5)]).config_tensor()


def test_config_tensor_from_simulation_parameters(
    bubbleml_case: BoilingSimulation,
) -> None:
    keys = {
        "non_dimensional": ["stefan", "prandtl"],
        "heaters": ["wall_temp_fraction", "xMax"],
    }
    batch = _batch([bubbleml_case.parameters.to_dict()], keys)
    torch.testing.assert_close(
        batch.config_tensor(), torch.tensor([[0.156, 7.35, 1.0, 2.0]])
    )


def test_fields_and_config_tensor_are_on_the_batch_device() -> None:
    batch = FlashXBatch(
        {"temperature": torch.zeros(1, 2, 4, 3)},
        [_config(1.0, 0.5)],
        NON_DIMENSIONAL_KEYS,
        device=torch.device("meta"),
    )
    assert batch.device == torch.device("meta")
    assert batch.fields["temperature"].device == batch.device
    assert batch.config_tensor().device == batch.device


def test_config_keys_survive_moving_the_batch() -> None:
    batch = _batch([_config(1.0, 0.5)], NON_DIMENSIONAL_KEYS)
    moved = batch.to(torch.device("meta"))
    assert moved.config_keys == {"non_dimensional": ("stefan", "prandtl")}
    assert moved.config_tensor().shape == (1, 2)


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


def test_stacked_fields_are_the_last_dimension_in_the_given_order() -> None:
    stacked = _batch_with_fields().stacked_fields(["pressure", "temperature"])
    assert stacked.shape == (2, 3, 4, 5, 2)
    torch.testing.assert_close(stacked[..., 0], torch.full((2, 3, 4, 5), 2.0))
    torch.testing.assert_close(stacked[..., 1], torch.full((2, 3, 4, 5), 1.0))


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
    assert stacked.shape == (2, 3, 4, 5, 5)
    torch.testing.assert_close(stacked[..., 0], batch.fields["temperature"])
    left, right, bottom, top = stacked[0, 0, ..., 1:].unbind(-1)
    torch.testing.assert_close(left[0], torch.arange(5.0))
    torch.testing.assert_close(right[0], torch.arange(1.0, 6.0))
    torch.testing.assert_close(bottom[:, 0], torch.arange(4.0))
    torch.testing.assert_close(top[:, 0], torch.arange(1.0, 5.0))


def test_stack_field_cells_keeps_the_order_given() -> None:
    batch = _staggered_batch()
    stacked = batch.stack_field_cells(["velfacey", "temperature"])
    assert stacked.shape == (2, 3, 4, 5, 3)
    torch.testing.assert_close(stacked[..., 2], batch.fields["temperature"])


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


def test_parameter_is_one_value_per_sample_on_the_batch_device() -> None:
    batch = FlashXBatch(
        {"temperature": torch.zeros(2, 2, 4, 3)},
        [_config(1.0, 0.5), _config(2.0, 0.25)],
        device=torch.device("meta"),
    )
    stefan = batch.parameter("non_dimensional", "stefan")
    assert stefan.shape == (2,)
    assert stefan.dtype == torch.float32
    assert stefan.device == batch.device


def test_parameter_reads_any_group_whether_or_not_in_config_keys() -> None:
    batch = _batch([_config(1.0, 0.5), _config(2.0, 0.25)], NON_DIMENSIONAL_KEYS)
    torch.testing.assert_close(
        batch.parameter("heaters", "wallTemp"), torch.tensor([1.0, 2.0])
    )
    torch.testing.assert_close(
        batch.parameter("discretization", "num_blocks_x"), torch.tensor([6.0, 6.0])
    )


def test_parameter_is_built_once_and_reused() -> None:
    batch = _batch([_config(1.0, 0.5)])
    assert batch.parameter("non_dimensional", "stefan") is batch.parameter(
        "non_dimensional", "stefan"
    )


def test_parameter_missing_from_a_sample_names_the_sample() -> None:
    gappy = _config(2.0, 0.25)
    del gappy["non_dimensional"]["prandtl"]
    with pytest.raises(KeyError, match=r"\{1: \['non_dimensional.prandtl'\]\}"):
        _batch([_config(1.0, 0.5), gappy]).parameter("non_dimensional", "prandtl")


def test_parameter_must_be_a_number() -> None:
    with pytest.raises(TypeError, match="'geometry' is 'cartesian', not a number"):
        _batch([_config(1.0, 0.5)]).parameter("discretization", "geometry")


def test_with_fields_keeps_the_samples_and_settings() -> None:
    batch = _batch([_config(1.0, 0.5), _config(2.0, 0.25)], NON_DIMENSIONAL_KEYS)
    predicted = batch.with_fields({"dfun": torch.ones(2, 1, 4, 3)})
    assert set(predicted.fields) == {"dfun"}
    assert predicted.configs == batch.configs
    assert predicted.config_keys == batch.config_keys
    torch.testing.assert_close(predicted.config_tensor(), batch.config_tensor())
    with pytest.raises(ValueError, match="batch size"):
        batch.with_fields({"dfun": torch.ones(3, 1, 4, 3)})


def _frames(first_frame: int, num_timesteps: int) -> FlashXBatch:
    """Two samples whose temperature is the frame index at every point."""
    frames = torch.arange(first_frame, first_frame + num_timesteps, dtype=torch.float32)
    temperature = frames[None, :, None, None].expand(2, num_timesteps, 4, 3)
    return _batch([_config(1.0, 0.5), _config(2.0, 0.25)]).with_fields(
        {"temperature": temperature}
    )


def _frame_indices(batch: FlashXBatch) -> list[float]:
    return batch.fields["temperature"][0, :, 0, 0].tolist()


def test_first_and_last_keep_the_ends_of_the_time_axis() -> None:
    batch = _frames(0, 5)
    assert batch.num_timesteps == 5
    assert _frame_indices(batch.head_time_window(2)) == [0.0, 1.0]
    assert _frame_indices(batch.tail_time_window(2)) == [3.0, 4.0]
    assert batch.tail_time_window(2).configs == batch.configs


@pytest.mark.parametrize("num_timesteps", [0, 6])
def test_window_must_fit_the_time_axis(num_timesteps: int) -> None:
    with pytest.raises(ValueError, match=r"must be in \[1, 5\]"):
        _frames(0, 5).tail_time_window(num_timesteps)


def test_extend_appends_frames_as_one_rollout_step() -> None:
    history = _frames(0, 3)
    prediction = _frames(3, 2)
    assert _frame_indices(history.extend(prediction)) == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert _frame_indices(history.extend(prediction).tail_time_window(3)) == [
        2.0,
        3.0,
        4.0,
    ]


def test_extend_needs_the_same_fields_and_samples() -> None:
    history = _frames(0, 3)
    with pytest.raises(ValueError, match="cannot extend fields"):
        history.extend(history.with_fields({"dfun": torch.zeros(2, 1, 4, 3)}))
    other_samples = FlashXBatch(
        dict(history.fields), [_config(1.0, 0.5), _config(9.0, 0.1)]
    )
    with pytest.raises(ValueError, match="different samples"):
        history.extend(other_samples)


def test_fields_must_share_one_number_of_timesteps() -> None:
    mismatched = {
        "temperature": torch.zeros(1, 2, 4, 3),
        "dfun": torch.zeros(1, 3, 4, 3),
    }
    with pytest.raises(ValueError, match="one number of timesteps"):
        FlashXBatch(mismatched, [_config(1.0, 0.5)])
    with pytest.raises(ValueError, match="one number of timesteps"):
        _batch([_config(1.0, 0.5)]).with_fields(mismatched)


def test_fields_need_a_time_axis() -> None:
    with pytest.raises(ValueError, match="no time axis"):
        FlashXBatch({"temperature": torch.zeros(1)}, [_config(1.0, 0.5)])


def test_num_timesteps_of_a_batch_without_fields() -> None:
    with pytest.raises(ValueError, match="no fields"):
        _ = FlashXBatch({}, [_config(1.0, 0.5)]).num_timesteps


def test_from_stacked_field_cells_inverts_stack_field_cells() -> None:
    batch = _staggered_batch()
    names = ["temperature", "velfacex", "velfacey"]
    restored = batch.from_stacked_field_cells(batch.stack_field_cells(names), names)
    assert restored.configs == batch.configs
    for name in names:
        torch.testing.assert_close(restored.fields[name], batch.fields[name])


def test_face_fields_are_rebuilt_on_their_own_grids() -> None:
    names = ["temperature", "velfacex", "velfacey"]
    fields = unstack_field_cells(torch.zeros(2, 3, 4, 5, 5), names)
    assert fields["temperature"].shape == (2, 3, 4, 5)
    assert fields["velfacex"].shape == (2, 3, 4, 6)
    assert fields["velfacey"].shape == (2, 3, 5, 5)


def test_the_two_predictions_of_an_interior_face_are_averaged() -> None:
    # low faces 0, 1, 2 and high faces 10, 11, 12 for one row of three cells
    stacked = torch.stack(
        (torch.tensor([0.0, 1.0, 2.0]), torch.tensor([10.0, 11.0, 12.0])), dim=-1
    ).reshape(1, 1, 1, 3, 2)
    faces = unstack_field_cells(stacked, ["velfacex"])["velfacex"]
    torch.testing.assert_close(faces.flatten(), torch.tensor([0.0, 5.5, 6.5, 12.0]))


def test_unstack_needs_the_channels_the_names_take() -> None:
    with pytest.raises(ValueError, match="take 3 channels, but the tensor has 2"):
        unstack_field_cells(torch.zeros(1, 1, 4, 5, 2), ["temperature", "velfacex"])


def test_configs_cannot_be_changed_after_the_batch_is_built() -> None:
    config = _config(1.0, 0.5)
    batch = _batch([config], NON_DIMENSIONAL_KEYS)
    with pytest.raises(TypeError, match="cannot be changed"):
        batch.configs[0]["non_dimensional"]["stefan"] = 9.0
    with pytest.raises(AttributeError):
        batch.configs[0]["heaters"][0]["nuc_sites_x"].append(2.0)
    # the batch holds its own frozen copy, so the dict it was built from is unaffected
    config["non_dimensional"]["stefan"] = 9.0
    torch.testing.assert_close(batch.config_tensor(), torch.tensor([[0.5, 7.0]]))


def test_batch_attributes_cannot_be_reassigned() -> None:
    batch = _batch([_config(1.0, 0.5)], NON_DIMENSIONAL_KEYS)
    with pytest.raises(dataclasses.FrozenInstanceError):
        batch.configs = []  # type: ignore[misc]
    with pytest.raises(TypeError, match="cannot be changed"):
        batch.config_keys["heaters"] = ["wallTemp"]  # type: ignore[index]


def test_batches_survive_pickling_for_data_loader_workers() -> None:
    batch = _batch([_config(1.0, 0.5)], NON_DIMENSIONAL_KEYS)
    restored = pickle.loads(pickle.dumps(batch))
    assert restored.configs == batch.configs
    torch.testing.assert_close(restored.config_tensor(), batch.config_tensor())
