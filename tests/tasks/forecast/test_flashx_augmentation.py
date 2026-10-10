from typing import Any

import pytest
import torch
from torch import nn

from boiling_data.flashx.batch import FlashXBatch
from boiling_data.tasks.forecast.flashx_augmentation import (
    FlipAugmentation,
    ForecastAugmentation,
)
from boiling_data.tasks.forecast.flashx_batch import FlashXForecastBatch

HEIGHT, WIDTH = 3, 4
CONFIG_KEYS = {"heaters": ["xMin", "yMax"]}


def _config(
    gravx: float = 0.0, gravy: float = -1.0, gravz: float = 0.0
) -> dict[str, Any]:
    return {
        "non_dimensional": {"gravx": gravx, "gravy": gravy, "gravz": gravz},
        "discretization": {"x_min": 0.0, "x_max": 4.0, "y_min": 0.0, "y_max": 3.0},
        "boundary": {
            "left": {"type": "slip_ins"},
            "right": {"type": "noslip_ins"},
            "bottom": {"type": "noslip_ins"},
            "top": {"type": "outflow_ins"},
        },
        "heaters": [
            {
                "xMin": 0.5,
                "xMax": 1.5,
                "yMin": 0.0,
                "yMax": 0.1,
                "nuc_sites_x": [1.0],
                "nuc_sites_y": [0.0],
            }
        ],
    }


def _ramp(
    batch_size: int, timesteps: int, height: int, width: int, dim: int
) -> torch.Tensor:
    """Values equal to the index along dim, -1 for x or -2 for y."""
    shape = (batch_size, timesteps, height, width)
    index = torch.arange(shape[dim], dtype=torch.float32)
    view = [1, 1, 1, 1]
    view[dim] = shape[dim]
    return index.view(view).expand(shape).clone()


def _windows(configs: list[dict[str, Any]], timesteps: int) -> FlashXBatch:
    size = len(configs)
    return FlashXBatch(
        {
            "temperature": _ramp(size, timesteps, HEIGHT, WIDTH, -1),
            "velx": _ramp(size, timesteps, HEIGHT, WIDTH, -1),
            "vely": _ramp(size, timesteps, HEIGHT, WIDTH, -2),
            "omgm": _ramp(size, timesteps, HEIGHT, WIDTH, -1),
            "velfacex": _ramp(size, timesteps, HEIGHT, WIDTH + 1, -1),
            "velfacey": _ramp(size, timesteps, HEIGHT + 1, WIDTH, -2),
        },
        configs,
        CONFIG_KEYS,
    )


def _batch(*configs: dict[str, Any]) -> FlashXForecastBatch:
    return FlashXForecastBatch(_windows(list(configs), 2), _windows(list(configs), 1))


def _always_flip() -> FlipAugmentation:
    return FlipAugmentation(flip_probability=1.0)


def test_eval_mode_leaves_batches_unchanged() -> None:
    augmentation = _always_flip().eval()
    batch = _batch(_config())
    assert augmentation(batch) is batch


def test_gravity_along_y_flips_horizontally() -> None:
    batch = _batch(_config(gravy=-1.0))
    flipped = _always_flip()(batch)
    for windows, original in (
        (flipped.input, batch.input),
        (flipped.target, batch.target),
    ):
        fields, before = windows.fields, original.fields
        torch.testing.assert_close(
            fields["temperature"], before["temperature"].flip(-1)
        )
        torch.testing.assert_close(fields["velx"], -before["velx"].flip(-1))
        torch.testing.assert_close(fields["velfacex"], -before["velfacex"].flip(-1))
        torch.testing.assert_close(fields["omgm"], -before["omgm"].flip(-1))
        torch.testing.assert_close(fields["vely"], before["vely"].flip(-1))
        torch.testing.assert_close(fields["velfacey"], before["velfacey"].flip(-1))


def test_horizontal_flip_mirrors_the_config() -> None:
    config = _always_flip()(_batch(_config(gravy=-1.0))).input.configs[0]
    assert config["boundary"]["left"] == {"type": "noslip_ins"}
    assert config["boundary"]["right"] == {"type": "slip_ins"}
    assert config["boundary"]["top"] == {"type": "outflow_ins"}
    heater = config["heaters"][0]
    assert (heater["xMin"], heater["xMax"]) == (2.5, 3.5)
    assert heater["nuc_sites_x"] == (3.0,)
    assert (heater["yMin"], heater["yMax"]) == (0.0, 0.1)


def test_gravity_along_x_flips_vertically() -> None:
    batch = _batch(_config(gravx=1.0, gravy=0.0))
    flipped = _always_flip()(batch)
    fields, before = flipped.input.fields, batch.input.fields
    torch.testing.assert_close(fields["vely"], -before["vely"].flip(-2))
    torch.testing.assert_close(fields["velfacey"], -before["velfacey"].flip(-2))
    torch.testing.assert_close(fields["omgm"], -before["omgm"].flip(-2))
    torch.testing.assert_close(fields["velx"], before["velx"].flip(-2))
    config = flipped.input.configs[0]
    assert config["boundary"]["bottom"] == {"type": "outflow_ins"}
    assert config["boundary"]["top"] == {"type": "noslip_ins"}
    heater = config["heaters"][0]
    assert heater["yMin"] == pytest.approx(2.9)
    assert heater["yMax"] == 3.0
    assert heater["nuc_sites_y"] == (3.0,)


def test_conditioning_describes_the_flipped_sample() -> None:
    flipped = _always_flip()(_batch(_config(gravy=-1.0)))
    torch.testing.assert_close(
        flipped.input.config_tensor(), torch.tensor([[2.5, 0.1]])
    )
    torch.testing.assert_close(
        flipped.target.config_tensor(), torch.tensor([[2.5, 0.1]])
    )


def test_oblique_gravity_is_never_flipped() -> None:
    batch = _batch(_config(gravx=0.5, gravy=-1.0))
    flipped = _always_flip()(batch)
    torch.testing.assert_close(flipped.input.fields["velx"], batch.input.fields["velx"])
    assert flipped.input.configs == batch.input.configs


@pytest.mark.parametrize(
    ("gravity", "message"),
    [({"gravz": -1.0}, "out of the 2d plane"), ({"gravy": 0.0}, "no gravity")],
)
def test_gravity_out_of_the_plane_or_absent_is_an_error(
    gravity: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _always_flip()(_batch(_config(**gravity)))


def test_three_dimensional_fields_are_an_error() -> None:
    batch = _batch(_config())
    volume = batch.input.with_fields(
        {"temperature": torch.zeros(1, 2, 2, HEIGHT, WIDTH)}
    )
    with pytest.raises(ValueError, match="not a 2d"):
        _always_flip()(FlashXForecastBatch(volume, volume))


def test_each_sample_is_flipped_on_its_own() -> None:
    batch = _batch(_config(gravy=-1.0), _config(gravx=0.5, gravy=-1.0))
    flipped = _always_flip()(batch)
    velx, before = flipped.input.fields["velx"], batch.input.fields["velx"]
    torch.testing.assert_close(velx[0], -before[0].flip(-1))
    torch.testing.assert_close(velx[1], before[1])


def test_a_probability_of_zero_flips_nothing() -> None:
    batch = _batch(_config())
    flipped = FlipAugmentation(flip_probability=0.0)(batch)
    torch.testing.assert_close(flipped.input.fields["velx"], batch.input.fields["velx"])
    assert flipped.input.configs == batch.input.configs


def test_flipping_twice_restores_the_batch() -> None:
    batch = _batch(_config())
    twice = _always_flip()(_always_flip()(batch))
    for name, tensor in batch.target.fields.items():
        torch.testing.assert_close(twice.target.fields[name], tensor)
    assert twice.input.configs == batch.input.configs


def test_a_seeded_generator_makes_the_flips_reproducible() -> None:
    batch = _batch(*[_config() for _ in range(8)])
    results = [
        FlipAugmentation(generator=torch.Generator().manual_seed(7))(batch)
        for _ in range(2)
    ]
    torch.testing.assert_close(
        results[0].input.fields["velx"], results[1].input.fields["velx"]
    )


def test_flip_probability_must_be_a_probability() -> None:
    with pytest.raises(ValueError, match=r"flip_probability must be in \[0, 1\]"):
        FlipAugmentation(flip_probability=1.5)


class _Recorder(nn.Module):
    """Records the order augmentations run in, passing batches through."""

    def __init__(self, name: str, calls: list[str]) -> None:
        super().__init__()
        self.name, self.calls = name, calls

    def forward(self, batch: FlashXForecastBatch) -> FlashXForecastBatch:
        self.calls.append(self.name)
        return batch


def test_forecast_augmentation_applies_its_augmentations_in_order() -> None:
    calls: list[str] = []
    augmentation = ForecastAugmentation(
        [_Recorder("first", calls), _Recorder("second", calls)]
    )
    augmentation(_batch(_config()))
    assert calls == ["first", "second"]


def test_forecast_augmentation_runs_the_flip() -> None:
    batch = _batch(_config(gravy=-1.0))
    augmented = ForecastAugmentation([_always_flip()])(batch)
    torch.testing.assert_close(
        augmented.input.fields["velx"], -batch.input.fields["velx"].flip(-1)
    )


def test_forecast_augmentation_in_eval_mode_runs_nothing() -> None:
    calls: list[str] = []
    augmentation = ForecastAugmentation([_Recorder("flip", calls), _always_flip()])
    augmentation.eval()
    batch = _batch(_config())
    assert augmentation(batch) is batch
    assert calls == []
    assert not any(child.training for child in augmentation.augmentations)


def test_forecast_augmentation_without_augmentations_changes_nothing() -> None:
    batch = _batch(_config())
    assert ForecastAugmentation()(batch) is batch
