import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from boiling_data.boiling_data import BoilingSimulation, SimulationParameters
from boiling_data.flashx import read_flashx

NUM_FRAMES, HEIGHT, WIDTH = 3, 288, 96
NX_BLOCK, NY_BLOCK = 16, 16


def test_all_stored_fields_are_loaded_with_clear_names(
    half_domain: BoilingSimulation,
) -> None:
    assert set(half_domain.fields) == {
        "pressure",
        "velx",
        "vely",
        "dfun",
        "temperature",
        "massflux",
        "rhoc",
        "curv",
        "normx",
        "normy",
        "omgm",
        "velfacex",
        "velfacey",
    }


def test_time_is_read_from_every_frame(half_domain: BoilingSimulation) -> None:
    # the solver accumulates 1e-4 steps, so plot times carry round-off
    np.testing.assert_allclose(half_domain.time, [50.0, 50.5, 51.0], atol=1e-8)
    assert half_domain.num_timesteps == NUM_FRAMES


@pytest.mark.parametrize(
    ("name", "shape"),
    [
        ("temperature", (NUM_FRAMES, HEIGHT, WIDTH)),
        ("velfacex", (NUM_FRAMES, HEIGHT, WIDTH + 1)),
        ("velfacey", (NUM_FRAMES, HEIGHT + 1, WIDTH)),
    ],
)
def test_field_shapes(
    half_domain: BoilingSimulation, name: str, shape: tuple[int, int, int]
) -> None:
    assert half_domain.field(name).data.shape == shape


def test_grids_span_the_domain(half_domain: BoilingSimulation) -> None:
    x_faces = half_domain.field("velfacex").grid_x
    y_faces = half_domain.field("velfacey").grid_y
    assert (x_faces[0], x_faces[-1]) == (0.0, 2.0)
    assert (y_faces[0], y_faces[-1]) == (0.0, 6.0)
    np.testing.assert_allclose(
        half_domain.field("temperature").grid_x, 0.5 * (x_faces[1:] + x_faces[:-1])
    )
    np.testing.assert_allclose(
        half_domain.field("temperature").grid_y, 0.5 * (y_faces[1:] + y_faces[:-1])
    )


@pytest.mark.parametrize("axis", [-1, -2])
def test_blocks_are_placed_in_order(half_domain: BoilingSimulation, axis: int) -> None:
    # a misplaced block shows up as a jump across a block seam far larger than
    # any jump inside a block
    jumps = np.abs(np.diff(half_domain.field("temperature").data, axis=axis))
    seams = np.take(jumps, np.arange(NX_BLOCK - 1, jumps.shape[axis], NX_BLOCK), axis)
    inside = np.delete(jumps, np.s_[NX_BLOCK - 1 :: NX_BLOCK], axis=axis)
    assert seams.max() <= inside.max()


def test_face_velocity_is_stitched_divergence_free(
    half_domain: BoilingSimulation,
) -> None:
    # the solver projects the face velocity to be divergence free away from the
    # interface, which only holds on the assembled grid if every shared face
    # was placed once and consistently
    velfacex = half_domain.field("velfacex")
    velfacey = half_domain.field("velfacey")
    dx = velfacex.grid_x[1] - velfacex.grid_x[0]
    dy = velfacey.grid_y[1] - velfacey.grid_y[0]
    divergence = (velfacex.data[..., 1:] - velfacex.data[..., :-1]) / dx + (
        velfacey.data[:, 1:] - velfacey.data[:, :-1]
    ) / dy
    liquid = half_domain.field("dfun").data < -1.0
    assert np.abs(divergence[liquid]).max() < 1e-10


def test_slip_wall_on_the_axis_has_no_normal_velocity(
    half_domain: BoilingSimulation,
) -> None:
    assert (half_domain.field("velfacex").data[..., 0] == 0.0).all()


def test_frames_are_in_time_order(half_domain: BoilingSimulation) -> None:
    gas_cells = (half_domain.field("dfun").data > 0).sum(axis=(1, 2))
    assert (np.diff(gas_cells) > 0).all(), "the bubble grows over these frames"


def test_physical_parameters_come_from_parameters_json(
    half_domain: BoilingSimulation,
) -> None:
    physical = half_domain.parameters.physical
    assert physical["fluid"] == "FC-72"
    assert physical["wall_temp"] == 70.0
    assert physical["sat_temp"] == 58.0
    assert physical["bulk_temp"] == 58.0
    assert physical["length_scale"] == 7e-4


def test_non_dimensional_parameters(half_domain: BoilingSimulation) -> None:
    assert half_domain.parameters.non_dimensional == {
        "gravx": 0.0,
        "gravy": -1.0,
        "gravz": 0.0,
        "inv_reynolds": 0.004315,
        "inflow_velscale": 1.0,
        "prandtl": 7.35,
        "stefan": 0.156,
        "tsat": 0.0,
        "inv_weber": 1.0,
        "cpgas": 0.7997,
        "mugas": 0.02816,
        "rhogas": 0.008687,
        "thcogas": 0.209,
    }


def test_discretization_parameters(half_domain: BoilingSimulation) -> None:
    assert half_domain.parameters.discretization == {
        "geometry": "cartesian",
        "xl_boundary_type": "slip_ins",
        "xr_boundary_type": "noslip_ins",
        "yl_boundary_type": "noslip_ins",
        "yr_boundary_type": "outflow_ins",
        "num_blocks_x": 6,
        "num_blocks_y": 18,
        "nx_block": NX_BLOCK,
        "ny_block": NY_BLOCK,
        "dt": 0.5,
        "t_initial": 0.0,
        "t_final": 250.0,
        "x_min": 0.0,
        "x_max": 2.0,
        "y_min": 0.0,
        "y_max": 6.0,
    }


def test_missing_gravity_components_default_to_zero(
    flashx_directory: Path, tmp_path: Path
) -> None:
    (plotfile,) = _copy_frames(flashx_directory, tmp_path, count=1)
    with h5py.File(plotfile, "r+") as frame:
        table = frame["real runtime parameters"][()]
        names = np.char.strip(table["name"].astype(str))
        del frame["real runtime parameters"]
        frame["real runtime parameters"] = table[names != "ins_gravz"]
    gravity = read_flashx(tmp_path).parameters.non_dimensional
    assert (gravity["gravx"], gravity["gravy"], gravity["gravz"]) == (0.0, -1.0, 0.0)


def test_run_without_heater_files_has_no_heaters(
    half_domain: BoilingSimulation,
) -> None:
    assert half_domain.parameters.heaters == []


def test_parameters_round_trip_through_a_dict(half_domain: BoilingSimulation) -> None:
    record = half_domain.parameters.to_dict()
    assert set(record) == {"physical", "non_dimensional", "discretization", "heaters"}
    assert SimulationParameters.from_dict(record) == half_domain.parameters


def _copy_frames(source: Path, destination: Path, count: int) -> list[Path]:
    plotfiles = sorted(source.glob("*_hdf5_plt_cnt_*"))[:count]
    return [Path(shutil.copy(plotfile, destination)) for plotfile in plotfiles]


def test_missing_parameters_json_gives_empty_physical_parameters(
    flashx_directory: Path, tmp_path: Path
) -> None:
    _copy_frames(flashx_directory, tmp_path, count=1)
    assert read_flashx(tmp_path).parameters.physical == {}


def test_refined_blocks_are_rejected(flashx_directory: Path, tmp_path: Path) -> None:
    (plotfile,) = _copy_frames(flashx_directory, tmp_path, count=1)
    with h5py.File(plotfile, "r+") as frame:
        frame["refine level"][0] = 2
    with pytest.raises(ValueError, match="refined"):
        read_flashx(tmp_path)


def test_frames_from_different_runs_are_rejected(
    flashx_directory: Path, tmp_path: Path
) -> None:
    _, second = _copy_frames(flashx_directory, tmp_path, count=2)
    with h5py.File(second, "r+") as frame:
        del frame["fv_x"]
    with pytest.raises(ValueError, match="missing"):
        read_flashx(tmp_path)


def test_directory_without_plotfiles_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_flashx(tmp_path)
