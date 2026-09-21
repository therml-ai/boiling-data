import numpy as np
import pytest

from boiling_data.boiling_data import BoilingSimulation, Field, SimulationParameters


def _field(num_frames: int = 3, height: int = 4, width: int = 5) -> Field:
    return Field(
        np.zeros((num_frames, height, width)),
        grid_x=np.arange(width, dtype=np.float64),
        grid_y=np.arange(height, dtype=np.float64),
    )


def test_field_rejects_a_grid_that_does_not_match_its_data() -> None:
    with pytest.raises(ValueError, match="does not match grid"):
        Field(np.zeros((3, 4, 5)), grid_x=np.zeros(4), grid_y=np.zeros(4))


def test_field_time_slice_keeps_the_grid() -> None:
    sliced = _field().time_slice(1)
    assert sliced.data.shape == (1, 4, 5)
    np.testing.assert_array_equal(sliced.grid_x, _field().grid_x)


def test_simulation_rejects_fields_with_a_different_number_of_frames() -> None:
    with pytest.raises(ValueError, match="frames"):
        BoilingSimulation(
            {"temperature": _field(num_frames=3)},
            SimulationParameters(),
            time=np.zeros(2),
        )
