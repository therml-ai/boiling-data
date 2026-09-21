import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from boiling_data.boiling_data import BoilingSimulation, Field
from boiling_data.bubbleml import PARAMETERS_ATTRIBUTE, read_bubbleml, write_bubbleml

NUM_FRAMES, HEIGHT, WIDTH = 3, 288, 96
GRID_DATASETS = {"time", "x_centers", "y_centers", "x_faces", "y_faces"}


def test_file_holds_fields_time_grid_and_parameters(
    bubbleml_path: Path, bubbleml_case: BoilingSimulation
) -> None:
    with h5py.File(bubbleml_path, "r") as handle:
        assert set(handle) == set(bubbleml_case.fields) | GRID_DATASETS
        assert (
            json.loads(handle.attrs[PARAMETERS_ATTRIBUTE])
            == bubbleml_case.parameters.to_dict()
        )


def test_fields_are_read_with_their_grids(bubbleml_case: BoilingSimulation) -> None:
    temperature = bubbleml_case.field("temperature")
    velfacex = bubbleml_case.field("velfacex")
    velfacey = bubbleml_case.field("velfacey")
    assert temperature.data.shape == (NUM_FRAMES, HEIGHT, WIDTH)
    assert velfacex.data.shape == (NUM_FRAMES, HEIGHT, WIDTH + 1)
    assert velfacey.data.shape == (NUM_FRAMES, HEIGHT + 1, WIDTH)
    assert (velfacex.grid_x[0], velfacex.grid_x[-1]) == (0.0, 2.0)
    assert (velfacey.grid_y[0], velfacey.grid_y[-1]) == (0.0, 6.0)
    np.testing.assert_allclose(
        temperature.grid_x, 0.5 * (velfacex.grid_x[1:] + velfacex.grid_x[:-1])
    )
    np.testing.assert_array_equal(temperature.grid_y, velfacex.grid_y)


def test_time_and_parameters(bubbleml_case: BoilingSimulation) -> None:
    np.testing.assert_allclose(bubbleml_case.time, [50.0, 52.5, 55.0], atol=1e-8)
    parameters = bubbleml_case.parameters
    assert parameters.physical["wall_temp"] == 70.0
    assert parameters.non_dimensional["stefan"] == 0.156
    assert parameters.discretization["num_blocks_x"] == 6
    (heater,) = parameters.heaters
    assert heater["advAngle"] == 45.0
    assert heater["nuc_sites_x"] == [0.0]


def test_write_then_read_gives_the_same_simulation(
    bubbleml_case: BoilingSimulation, tmp_path: Path
) -> None:
    restored = read_bubbleml(write_bubbleml(bubbleml_case, tmp_path / "copy.hdf5"))
    assert restored.parameters == bubbleml_case.parameters
    np.testing.assert_array_equal(restored.time, bubbleml_case.time)
    assert set(restored.fields) == set(bubbleml_case.fields)
    for name, field in bubbleml_case.fields.items():
        np.testing.assert_array_equal(restored.field(name).data, field.data)
        np.testing.assert_array_equal(restored.field(name).grid_x, field.grid_x)
        np.testing.assert_array_equal(restored.field(name).grid_y, field.grid_y)


def test_read_then_write_reproduces_the_file(
    bubbleml_path: Path, bubbleml_case: BoilingSimulation, tmp_path: Path
) -> None:
    rewritten = write_bubbleml(bubbleml_case, tmp_path / "copy.hdf5")
    with h5py.File(bubbleml_path, "r") as original, h5py.File(rewritten, "r") as copy:
        assert copy.attrs[PARAMETERS_ATTRIBUTE] == original.attrs[PARAMETERS_ATTRIBUTE]
        assert set(copy) == set(original)
        for name in original:
            assert copy[name].dtype == original[name].dtype
            np.testing.assert_array_equal(copy[name][()], original[name][()])


def test_methods_on_the_simulation(bubbleml_path: Path, tmp_path: Path) -> None:
    case = BoilingSimulation.from_bubbleml(bubbleml_path)
    restored = BoilingSimulation.from_bubbleml(case.to_bubbleml(tmp_path / "case.hdf5"))
    np.testing.assert_array_equal(restored.field("dfun").data, case.field("dfun").data)


def test_older_file_reads_parameters_from_a_sidecar_json(
    bubbleml_path: Path, bubbleml_case: BoilingSimulation, tmp_path: Path
) -> None:
    older = Path(shutil.copy(bubbleml_path, tmp_path / "case.hdf5"))
    with h5py.File(older, "r+") as handle:
        del handle.attrs[PARAMETERS_ATTRIBUTE]
    with pytest.raises(ValueError, match=r"no .*parameters"):
        read_bubbleml(older)
    with open(older.with_suffix(".json"), "w", encoding="utf-8") as handle:
        json.dump(bubbleml_case.parameters.to_dict(), handle)
    assert read_bubbleml(older).parameters == bubbleml_case.parameters


def test_file_without_time_gets_uniform_plot_times(
    bubbleml_path: Path, tmp_path: Path
) -> None:
    older = Path(shutil.copy(bubbleml_path, tmp_path / "case.hdf5"))
    with h5py.File(older, "r+") as handle:
        del handle["time"]
    np.testing.assert_array_equal(read_bubbleml(older).time, [0.0, 0.5, 1.0])


def test_field_off_the_grid_is_rejected(
    bubbleml_case: BoilingSimulation, tmp_path: Path
) -> None:
    temperature = bubbleml_case.field("temperature")
    off_grid = Field(temperature.data, temperature.grid_x + 0.1, temperature.grid_y)
    simulation = BoilingSimulation(
        {"temperature": off_grid}, bubbleml_case.parameters, bubbleml_case.time
    )
    with pytest.raises(ValueError, match="cell centers or faces"):
        write_bubbleml(simulation, tmp_path / "case.hdf5")


def test_dataset_of_unknown_extent_is_rejected(
    bubbleml_path: Path, tmp_path: Path
) -> None:
    broken = Path(shutil.copy(bubbleml_path, tmp_path / "case.hdf5"))
    with h5py.File(broken, "r+") as handle:
        handle["bad"] = np.zeros((NUM_FRAMES, HEIGHT, 50))
    with pytest.raises(ValueError, match="neither"):
        read_bubbleml(broken)


def test_flashx_sample_converts_to_the_same_layout(
    half_domain: BoilingSimulation, bubbleml_path: Path, tmp_path: Path
) -> None:
    converted = write_bubbleml(half_domain, tmp_path / "converted.hdf5")
    with h5py.File(bubbleml_path, "r") as sample, h5py.File(converted, "r") as copy:
        assert set(copy) == set(sample)
        for name in sample:
            assert copy[name].shape == sample[name].shape
            assert copy[name].dtype == sample[name].dtype
