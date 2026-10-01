import numpy as np
import pytest

from boiling_data.boiling_data import BoilingSimulation, Field, FloatArray
from boiling_data.dmd import compute_dmd, uniform_timestep, vorticity


def _travelling_wave(
    num_frames: int, dt: float, frequency: float, decay: float = 0.1
) -> FloatArray:
    """A decaying oscillation of two fixed spatial patterns, exactly rank 2."""
    x = np.linspace(0, 1, 12)
    y = np.linspace(0, 1, 8)
    pattern_a = np.outer(np.sin(np.pi * y), np.cos(2 * np.pi * x))
    pattern_b = np.outer(np.cos(np.pi * y), np.sin(2 * np.pi * x))
    t = dt * np.arange(num_frames)[:, None, None]
    phase = 2 * np.pi * frequency * t
    wave: FloatArray = np.exp(-decay * t) * (
        np.cos(phase) * pattern_a + np.sin(phase) * pattern_b
    )
    return wave


def test_dmd_recovers_frequency_and_growth_of_an_oscillation() -> None:
    data = _travelling_wave(num_frames=40, dt=0.1, frequency=0.7)
    dmd = compute_dmd(data, dt=0.1, rank=10, subtract_mean=False)
    assert len(dmd.eigenvalues) == 2
    assert dmd.modes.shape == (2, 8, 12)
    np.testing.assert_allclose(dmd.frequencies, [0.7, 0.7])
    np.testing.assert_allclose(dmd.continuous_eigenvalues.real, [-0.1, -0.1])
    np.testing.assert_allclose(dmd.reconstruct(40), data, atol=1e-10)


def test_mean_is_subtracted_by_default_and_restored_on_reconstruction() -> None:
    # two whole undamped periods, so the temporal mean is exactly the offset
    data = _travelling_wave(num_frames=40, dt=0.1, frequency=0.5, decay=0.0) + 3.0
    dmd = compute_dmd(data, dt=0.1, rank=10)
    assert dmd.mean is not None
    np.testing.assert_allclose(dmd.mean, 3.0, atol=1e-12)
    np.testing.assert_allclose(dmd.reconstruct(40), data, atol=1e-6)


def test_uniform_timestep_tolerates_rounding_but_rejects_uneven_frames() -> None:
    assert uniform_timestep(np.array([0.0, 0.5, 1.0])) == 0.5
    rounded = np.array([0.0, 0.4999000000166, 1.0])
    assert uniform_timestep(rounded) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="uniformly spaced"):
        uniform_timestep(np.array([0.0, 0.5, 1.5]))


def test_vorticity_of_solid_body_rotation_is_twice_the_rate() -> None:
    x = np.linspace(-1, 1, 9)
    y = np.linspace(-1, 1, 7)
    grid_x, grid_y = np.meshgrid(x, y)
    velx = Field(-grid_y[None], grid_x=x, grid_y=y)
    vely = Field(grid_x[None], grid_x=x, grid_y=y)
    np.testing.assert_allclose(vorticity(velx, vely).data, 2.0)


def test_dmd_runs_on_bubbleml_fields(bubbleml_case: BoilingSimulation) -> None:
    field = vorticity(bubbleml_case.field("velx"), bubbleml_case.field("vely"))
    dmd = compute_dmd(field.data, uniform_timestep(bubbleml_case.time), rank=5)
    assert dmd.modes.shape[1:] == field.data.shape[1:]
    assert len(dmd.eigenvalues) <= bubbleml_case.num_timesteps - 1
