import copy
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from boiling_data.flashx.batch import FlashXBatch
from boiling_data.flashx.normalizer import NormalizerWrapper
from boiling_data.statistics import dataset_statistics


def _entry(mean: float, std: float) -> dict[str, float]:
    return {"count": 2, "mean": mean, "std": std, "min": mean - std, "max": mean + std}


STATISTICS: dict[str, Any] = {
    "fields": {"temperature": _entry(1.0, 2.0)},
    "config": {
        "physical": {"bulk_temp": _entry(50.0, 10.0)},
        "non_dimensional": {"stefan": _entry(0.5, 0.25), "gravy": _entry(-1.0, 0.0)},
        "heaters": {"wall_temp": _entry(70.0, 5.0)},
    },
}


def _config(bulk_temp: float, wall_temps: list[float]) -> dict[str, Any]:
    return {
        "physical": {"fluid": "FC-72", "bulk_temp": bulk_temp},
        "non_dimensional": {"stefan": 0.75, "gravy": -1.0},
        "heaters": [
            {"type": "constant_wall_temp", "wall_temp": temp, "nuc_sites_x": [0.0]}
            for temp in wall_temps
        ],
    }


def _batch(temperature: float = 5.0) -> FlashXBatch:
    return FlashXBatch(
        {"temperature": torch.full((2, 3, 4, 5), temperature)},
        [_config(60.0, [75.0]), _config(40.0, [65.0])],
        {"non_dimensional": ["stefan"], "heaters": ["wall_temp"]},
    )


def test_fields_are_standardized_by_their_own_statistics() -> None:
    normalized = NormalizerWrapper(STATISTICS).normalize(_batch(temperature=5.0))
    torch.testing.assert_close(
        normalized.fields["temperature"], torch.full((2, 3, 4, 5), 2.0)
    )


def test_config_values_are_standardized_at_the_same_group_and_key() -> None:
    normalized = NormalizerWrapper(STATISTICS, config_scaling="standard").normalize(
        _batch()
    )
    config = normalized.configs[0]
    assert config["physical"]["bulk_temp"] == pytest.approx(1.0)
    assert config["non_dimensional"]["stefan"] == pytest.approx(1.0)
    assert config["heaters"][0]["wall_temp"] == pytest.approx(1.0)
    assert normalized.configs[1]["heaters"][0]["wall_temp"] == pytest.approx(-1.0)


def test_normalized_batches_give_normalized_conditioning_and_parameters() -> None:
    normalized = NormalizerWrapper(STATISTICS, config_scaling="standard").normalize(
        _batch()
    )
    torch.testing.assert_close(
        normalized.config_tensor(), torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    )
    torch.testing.assert_close(
        normalized.parameter("physical", "bulk_temp"), torch.tensor([1.0, -1.0])
    )


def test_values_without_a_numeric_meaning_pass_through() -> None:
    config = NormalizerWrapper(STATISTICS).normalize(_batch()).configs[0]
    assert config["physical"]["fluid"] == "FC-72"
    assert config["heaters"][0]["type"] == "constant_wall_temp"
    assert config["heaters"][0]["nuc_sites_x"] == (0.0,)


def test_a_constant_value_only_has_its_mean_subtracted() -> None:
    normalized = NormalizerWrapper(STATISTICS, config_scaling="standard").normalize(
        _batch()
    )
    assert normalized.configs[0]["non_dimensional"]["gravy"] == 0.0


def test_unnormalize_inverts_normalize() -> None:
    normalizer = NormalizerWrapper(STATISTICS, config_scaling="standard")
    batch = _batch()
    restored = normalizer.unnormalize(normalizer.normalize(batch))
    torch.testing.assert_close(
        restored.fields["temperature"], batch.fields["temperature"]
    )
    for original, round_trip in zip(batch.configs, restored.configs, strict=True):
        assert round_trip["physical"]["bulk_temp"] == pytest.approx(
            original["physical"]["bulk_temp"]
        )
        assert round_trip["heaters"][0]["wall_temp"] == pytest.approx(
            original["heaters"][0]["wall_temp"]
        )
    torch.testing.assert_close(restored.config_tensor(), batch.config_tensor())


def test_normalize_and_unnormalize_set_is_normalized() -> None:
    normalizer = NormalizerWrapper(STATISTICS)
    normalized = normalizer.normalize(_batch())
    assert normalized.is_normalized
    assert not normalizer.unnormalize(normalized).is_normalized


def test_a_batch_is_normalized_at_most_once() -> None:
    normalizer = NormalizerWrapper(STATISTICS)
    with pytest.raises(ValueError, match="already normalized"):
        normalizer.normalize(normalizer.normalize(_batch()))
    with pytest.raises(ValueError, match="is not normalized"):
        normalizer.unnormalize(_batch())


def test_missing_statistics_are_reported() -> None:
    normalizer = NormalizerWrapper(STATISTICS)
    with pytest.raises(KeyError, match="no statistics for field 'dfun'"):
        normalizer.normalize(_batch().with_fields({"dfun": torch.zeros(2, 3, 4, 5)}))
    batch = FlashXBatch(
        {"temperature": torch.zeros(1, 1, 2, 2)},
        [{"physical": {"bulk_temp": 58.0, "rho_liquid": 1620.0}}],
    )
    with pytest.raises(KeyError, match=r"'physical\.rho_liquid'"):
        normalizer.normalize(batch)


def test_statistics_need_fields_and_config() -> None:
    with pytest.raises(ValueError, match=r"missing \['config'\]"):
        NormalizerWrapper({"fields": {}})


class _Recorder(nn.Module):
    """Returns its input, recording what it was given."""

    def forward(self, batch: FlashXBatch) -> FlashXBatch:
        self.seen = batch
        return batch


def test_forward_runs_the_module_on_normalized_batches() -> None:
    recorder = _Recorder()
    wrapper = NormalizerWrapper(STATISTICS, recorder)
    batch = _batch(temperature=5.0)
    output = wrapper(batch)
    torch.testing.assert_close(
        recorder.seen.fields["temperature"], torch.full((2, 3, 4, 5), 2.0)
    )
    torch.testing.assert_close(
        output.fields["temperature"], batch.fields["temperature"]
    )
    assert recorder.seen.is_normalized
    assert not output.is_normalized


def test_forward_needs_a_module() -> None:
    with pytest.raises(RuntimeError, match="no module"):
        NormalizerWrapper(STATISTICS)(_batch())


def test_statistics_are_saved_with_the_state_dict() -> None:
    state = NormalizerWrapper(STATISTICS).state_dict()
    restored = NormalizerWrapper({"fields": {}, "config": {}})
    restored.load_state_dict(state)
    assert restored.statistics == STATISTICS


def test_statistics_file_from_a_dataset_normalizes_its_batches(
    bubbleml_path: Path, tmp_path: Path
) -> None:
    path = tmp_path / "statistics.json"
    statistics = dataset_statistics([bubbleml_path], ["temperature"], 64)
    path.write_text(json.dumps(statistics))
    normalizer = NormalizerWrapper.from_json(path)
    batch = FlashXBatch(
        {"temperature": torch.zeros(1, 1, 2, 2)},
        [{"physical": {"bulk_temp": 58.0}, "non_dimensional": {"stefan": 0.156}}],
    )
    normalized = normalizer.normalize(batch)
    mean = statistics["fields"]["temperature"]["mean"]
    std = statistics["fields"]["temperature"]["std"]
    torch.testing.assert_close(
        normalized.fields["temperature"], torch.full((1, 1, 2, 2), -mean / std)
    )
    assert normalized.configs[0]["non_dimensional"]["stefan"] == 0.0


def _min_max_statistics() -> dict[str, Any]:
    statistics = copy.deepcopy(STATISTICS)
    statistics["config"]["physical"]["bulk_temp"] = {
        "count": 3,
        "mean": 50.0,
        "std": 10.0,
        "min": 40.0,
        "max": 80.0,
    }
    return statistics


def test_min_max_scaling_maps_the_config_range_to_minus_one_one() -> None:
    normalizer = NormalizerWrapper(_min_max_statistics(), config_scaling="min_max")
    batch = FlashXBatch(
        {"temperature": torch.zeros(3, 1, 2, 2)},
        [_config(temp, [70.0]) for temp in (40.0, 60.0, 80.0)],
    )
    normalized = normalizer.normalize(batch)
    torch.testing.assert_close(
        normalized.parameter("physical", "bulk_temp"), torch.tensor([-1.0, 0.0, 1.0])
    )


def test_min_max_scaling_leaves_fields_standardized() -> None:
    normalizer = NormalizerWrapper(_min_max_statistics(), config_scaling="min_max")
    normalized = normalizer.normalize(_batch(temperature=5.0))
    torch.testing.assert_close(
        normalized.fields["temperature"], torch.full((2, 3, 4, 5), 2.0)
    )


def test_min_max_scaling_maps_a_constant_value_to_zero() -> None:
    normalizer = NormalizerWrapper(STATISTICS, config_scaling="min_max")
    normalized = normalizer.normalize(_batch())
    assert normalized.configs[0]["non_dimensional"]["gravy"] == 0.0


def test_min_max_scaling_round_trips() -> None:
    normalizer = NormalizerWrapper(_min_max_statistics(), config_scaling="min_max")
    batch = _batch()
    restored = normalizer.unnormalize(normalizer.normalize(batch))
    torch.testing.assert_close(
        restored.parameter("physical", "bulk_temp"),
        batch.parameter("physical", "bulk_temp"),
    )
    torch.testing.assert_close(restored.config_tensor(), batch.config_tensor())


def test_unknown_config_scaling_is_rejected() -> None:
    with pytest.raises(ValueError, match="config_scaling must be one of"):
        NormalizerWrapper(STATISTICS, config_scaling="robust")  # type: ignore[arg-type]


def test_config_scaling_is_saved_with_the_state_dict() -> None:
    state = NormalizerWrapper(STATISTICS, config_scaling="standard").state_dict()
    restored = NormalizerWrapper(STATISTICS)
    restored.load_state_dict(state)
    assert restored.config_scaling == "standard"


def test_config_scaling_defaults_to_min_max() -> None:
    assert NormalizerWrapper(STATISTICS).config_scaling == "min_max"


def test_forward_keeps_the_input_configs_so_rollouts_can_extend() -> None:
    # stefan = 0.072 over [0.023, 0.506] comes back from min-max scaling as
    # 0.07200000000000001, so transforming the configs back cannot be relied on
    statistics = copy.deepcopy(STATISTICS)
    statistics["config"]["non_dimensional"]["stefan"] = {
        "count": 2,
        "mean": 0.3,
        "std": 0.2,
        "min": 0.023,
        "max": 0.506,
    }
    config = _config(60.0, [75.0])
    config["non_dimensional"]["stefan"] = 0.072
    history = FlashXBatch(
        {"temperature": torch.zeros(1, 2, 2, 2)},
        [config],
        {"non_dimensional": ["stefan"]},
    )
    wrapper = NormalizerWrapper(statistics, _Recorder(), config_scaling="min_max")
    round_trip = wrapper.unnormalize(wrapper.normalize(history))
    assert round_trip.configs[0]["non_dimensional"]["stefan"] != 0.072

    prediction = wrapper(history)
    assert prediction.configs == history.configs
    extended = history.extend(prediction).tail_time_window(2)
    assert extended.configs[0]["non_dimensional"]["stefan"] == 0.072
