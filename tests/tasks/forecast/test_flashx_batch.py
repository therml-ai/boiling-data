from typing import Any

import pytest
import torch

from boiling_data.flashx.batch import FlashXSample
from boiling_data.tasks.forecast.flashx_batch import flashx_collater

NON_DIMENSIONAL_KEYS = {"non_dimensional": ["stefan", "prandtl"]}


def _config(wall_temp: float, stefan: float) -> dict[str, Any]:
    return {
        "non_dimensional": {"stefan": stefan, "prandtl": 7.0},
        "discretization": {"geometry": "cartesian", "num_blocks_x": 6},
        "heaters": [{"wallTemp": wall_temp, "nuc_sites_x": [0.0, 0.5]}],
    }


def _sample(wall_temp: float, num_timesteps: int = 2) -> FlashXSample:
    return FlashXSample(
        {
            "temperature": torch.full((num_timesteps, 4, 3), wall_temp),
            "velfacex": torch.zeros(num_timesteps, 4, 4),
        },
        _config(wall_temp, 0.5),
    )


def _pair(wall_temp: float) -> tuple[FlashXSample, FlashXSample]:
    return _sample(wall_temp), _sample(wall_temp + 10.0, num_timesteps=1)


def test_collater_stacks_inputs_and_targets_along_a_new_batch_dimension() -> None:
    batch = flashx_collater([_pair(1.0), _pair(2.0)])
    assert batch.batch_size == batch.input.batch_size == batch.target.batch_size == 2
    assert batch.input.fields["temperature"].shape == (2, 2, 4, 3)
    assert batch.target.fields["velfacex"].shape == (2, 1, 4, 4)
    torch.testing.assert_close(
        batch.input.fields["temperature"][1], torch.full((2, 4, 3), 2.0)
    )
    torch.testing.assert_close(
        batch.target.fields["temperature"][1], torch.full((1, 4, 3), 12.0)
    )
    wall_temps = [config["heaters"][0]["wallTemp"] for config in batch.input.configs]
    assert wall_temps == [1.0, 2.0]


def test_collater_moves_the_batch_to_the_device() -> None:
    batch = flashx_collater([_pair(1.0)], device=torch.device("meta"))
    assert batch.input.fields["temperature"].is_meta
    assert batch.target.fields["temperature"].is_meta


def test_forecast_batch_to_moves_inputs_and_targets() -> None:
    moved = flashx_collater([_pair(1.0)]).to(torch.device("meta"))
    assert moved.input.device == moved.target.device == torch.device("meta")


def test_collater_rejects_samples_with_different_fields() -> None:
    mismatched = FlashXSample({"temperature": torch.zeros(2, 4, 3)}, _config(2.0, 0.5))
    with pytest.raises(ValueError, match="fields"):
        flashx_collater([_pair(1.0), (mismatched, _sample(2.0))])


def test_collater_rejects_an_empty_list() -> None:
    with pytest.raises(ValueError, match="empty"):
        flashx_collater([])


def test_collater_gives_input_and_target_the_config_keys() -> None:
    batch = flashx_collater([_pair(1.0), _pair(2.0)], config_keys=NON_DIMENSIONAL_KEYS)
    for windows in (batch.input, batch.target):
        assert windows.config_keys == {"non_dimensional": ("stefan", "prandtl")}
        assert windows.config_tensor().shape == (2, 2)
