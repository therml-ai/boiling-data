import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

from boiling_data.boiling_data import (
    BoilingSimulation,
    Field,
    FloatArray,
    SimulationParameters,
)
from boiling_data.symmetry import mirror_x

PLOTFILE_PATTERN = re.compile(r"_hdf5_plt_cnt_\d+$")
HEATER_PATTERN = re.compile(r"_hdf5_htr_\d+$")

# Flash-X names its stored variables with four characters. these are
# converted to clearer names.
FIELD_NAMES = {
    "temp": "temperature",
    "pres": "pressure",
    "mflx": "massflux",
    "nrmx": "normx",
    "nrmy": "normy",
    "fv_x": "velfacex",
    "fv_y": "velfacey",
}
FACE_FIELDS = ("fv_x", "fv_y")

NON_DIMENSIONAL_PARAMETERS = {
    "ins_gravx": "gravx",
    "ins_gravy": "gravy",
    "ins_gravz": "gravz",
    "ins_invreynolds": "inv_reynolds",
    "ins_inflowvelscale": "inflow_velscale",
    "ht_prandtl": "prandtl",
    "mph_stefan": "stefan",
    "mph_tsat": "tsat",
    "mph_invweber": "inv_weber",
    "mph_cpgas": "cpgas",
    "mph_mugas": "mugas",
    "mph_rhogas": "rhogas",
    "mph_thcogas": "thcogas",
}

HEATER_SCALARS = (
    "xMin",
    "xMax",
    "yMin",
    "yMax",
    "zMin",
    "zMax",
    "wallTemp",
    "advAngle",
    "rcdAngle",
    "velContact",
    "nucWaitTime",
)


def _runtime_parameters(frame: h5py.File, kind: str) -> dict[str, Any]:
    """Flash-X stores each parameter table as (name, value) records with
    space-padded names; string values are space-padded too."""
    table = {}
    for name, value in frame[f"{kind} runtime parameters"][()]:
        key = name.decode("utf-8").strip()
        table[key] = (
            value.decode("utf-8").strip() if isinstance(value, bytes) else value
        )
    return table


def _scalars(frame: h5py.File, kind: str) -> dict[str, Any]:
    return {
        name.decode("utf-8").strip(): value
        for name, value in frame[f"{kind} scalars"][()]
    }


@dataclass(frozen=True)
class BlockLayout:
    """Where every block of a uniformly refined 2d grid sits in the global array."""

    num_blocks_x: int
    num_blocks_y: int
    nx_block: int
    ny_block: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    order: npt.NDArray[np.intp]  # block index in row-major (y, x) order

    @property
    def shape(self) -> tuple[int, int]:
        return self.num_blocks_y * self.ny_block, self.num_blocks_x * self.nx_block

    @property
    def x_faces(self) -> FloatArray:
        return np.linspace(self.x_min, self.x_max, self.shape[1] + 1)

    @property
    def y_faces(self) -> FloatArray:
        return np.linspace(self.y_min, self.y_max, self.shape[0] + 1)

    @property
    def x_centers(self) -> FloatArray:
        faces = self.x_faces
        return 0.5 * (faces[1:] + faces[:-1])

    @property
    def y_centers(self) -> FloatArray:
        faces = self.y_faces
        return 0.5 * (faces[1:] + faces[:-1])

    def assemble_cells(self, blocks: FloatArray) -> FloatArray:
        """[num_blocks, ny_block, nx_block] -> [Ny, Nx]."""
        tiles = blocks[self.order].reshape(
            self.num_blocks_y, self.num_blocks_x, self.ny_block, self.nx_block
        )
        assembled: FloatArray = tiles.transpose(0, 2, 1, 3).reshape(self.shape)
        return assembled

    def assemble_x_faces(self, blocks: FloatArray) -> FloatArray:
        """[num_blocks, ny_block, nx_block + 1] -> [Ny, Nx + 1]. Neighbouring
        blocks both store their shared face, so all but the last block column
        drop their high face."""
        interior = self.assemble_cells(blocks[:, :, : self.nx_block])
        last_column = blocks[self.order].reshape(
            self.num_blocks_y, self.num_blocks_x, self.ny_block, self.nx_block + 1
        )[:, -1, :, -1]
        return np.concatenate([interior, last_column.reshape(-1, 1)], axis=1)

    def assemble_y_faces(self, blocks: FloatArray) -> FloatArray:
        """[num_blocks, ny_block + 1, nx_block] -> [Ny + 1, Nx]."""
        interior = self.assemble_cells(blocks[:, : self.ny_block, :])
        last_row = blocks[self.order].reshape(
            self.num_blocks_y, self.num_blocks_x, self.ny_block + 1, self.nx_block
        )[-1, :, -1, :]
        return np.concatenate([interior, last_row.reshape(1, -1)], axis=0)


def block_layout(frame: h5py.File) -> BlockLayout:
    integer_scalars = _scalars(frame, "integer")
    if int(integer_scalars["dimensionality"]) != 2:
        raise NotImplementedError("only 2d Flash-X output is supported")

    levels = frame["refine level"][()]
    if not (levels == 1).all():
        raise ValueError(
            f"refined blocks (levels {sorted(set(levels.tolist()))}) cannot be "
            "placed on a single uniform grid"
        )

    integer_parameters = _runtime_parameters(frame, "integer")
    nx_block, ny_block = int(integer_scalars["nxb"]), int(integer_scalars["nyb"])
    num_blocks_x = int(integer_parameters["nblockx"])
    num_blocks_y = int(integer_parameters["nblocky"])

    bounding_box = frame["bounding box"][()].astype(np.float64)
    x_min, x_max = bounding_box[:, 0, :].min(), bounding_box[:, 0, :].max()
    y_min, y_max = bounding_box[:, 1, :].min(), bounding_box[:, 1, :].max()
    block_width = (x_max - x_min) / num_blocks_x
    block_height = (y_max - y_min) / num_blocks_y
    column = np.rint((bounding_box[:, 0, 0] - x_min) / block_width).astype(int)
    row = np.rint((bounding_box[:, 1, 0] - y_min) / block_height).astype(int)

    order = np.argsort(row * num_blocks_x + column)
    if len(order) != num_blocks_x * num_blocks_y:
        raise ValueError(
            f"{len(order)} blocks in the file but the grid is "
            f"{num_blocks_x} x {num_blocks_y} blocks"
        )
    return BlockLayout(
        num_blocks_x,
        num_blocks_y,
        nx_block,
        ny_block,
        float(x_min),
        float(x_max),
        float(y_min),
        float(y_max),
        order,
    )


def stored_variables(frame: h5py.File) -> list[str]:
    """Cell-centered variables named in the file, then the face velocities."""
    names = [name.decode("utf-8").strip() for (name,) in frame["unknown names"][()]]
    return names + [face for face in FACE_FIELDS if face in frame]


def _field_shape(
    name: str, num_frames: int, layout: BlockLayout
) -> tuple[int, int, int]:
    height, width = layout.shape
    if name == "fv_x":
        return num_frames, height, width + 1
    if name == "fv_y":
        return num_frames, height + 1, width
    return num_frames, height, width


def read_fields(plotfiles: list[Path], layout: BlockLayout) -> dict[str, Field]:
    with h5py.File(plotfiles[0], "r") as first:
        variables = stored_variables(first)

    data = {
        name: np.empty(_field_shape(name, len(plotfiles), layout), dtype=np.float64)
        for name in variables
    }

    for frame_index, plotfile in enumerate(plotfiles):
        with h5py.File(plotfile, "r") as frame:
            missing = [name for name in variables if name not in frame]
            if missing:
                raise ValueError(
                    f"{plotfile} is missing {missing}; the directory may hold "
                    "output from more than one run"
                )
            for name in variables:
                blocks = frame[name][:, 0].astype(np.float64)
                if name == "fv_x":
                    data[name][frame_index] = layout.assemble_x_faces(blocks)
                elif name == "fv_y":
                    data[name][frame_index] = layout.assemble_y_faces(blocks)
                else:
                    data[name][frame_index] = layout.assemble_cells(blocks)

    grids = {
        "fv_x": (layout.x_faces, layout.y_centers),
        "fv_y": (layout.x_centers, layout.y_faces),
    }
    fields = {}
    for name, array in data.items():
        grid_x, grid_y = grids.get(name, (layout.x_centers, layout.y_centers))
        fields[FIELD_NAMES.get(name, name)] = Field(array, grid_x, grid_y)
    return fields


def read_time(plotfiles: list[Path]) -> FloatArray:
    times = []
    for plotfile in plotfiles:
        with h5py.File(plotfile, "r") as frame:
            times.append(float(_scalars(frame, "real")["time"]))
    return np.array(times, dtype=np.float64)


def read_discretization(frame: h5py.File, layout: BlockLayout) -> dict[str, Any]:
    strings = _runtime_parameters(frame, "string")
    reals = _runtime_parameters(frame, "real")
    return {
        "geometry": strings["geometry"],
        "xl_boundary_type": strings["xl_boundary_type"],
        "xr_boundary_type": strings["xr_boundary_type"],
        "yl_boundary_type": strings["yl_boundary_type"],
        "yr_boundary_type": strings["yr_boundary_type"],
        "num_blocks_x": layout.num_blocks_x,
        "num_blocks_y": layout.num_blocks_y,
        "nx_block": layout.nx_block,
        "ny_block": layout.ny_block,
        "dt": float(reals["plotfileintervaltime"]),
        "t_initial": float(reals["tinitial"]),
        "t_final": float(reals["tmax"]),
        "x_min": layout.x_min,
        "x_max": layout.x_max,
        "y_min": layout.y_min,
        "y_max": layout.y_max,
    }


def read_non_dimensional(frame: h5py.File) -> dict[str, float]:
    """A gravity component the run did not set is zero."""
    reals = _runtime_parameters(frame, "real")
    parameters = {
        name: float(reals[flashx_name])
        for flashx_name, name in NON_DIMENSIONAL_PARAMETERS.items()
        if flashx_name in reals
    }
    for component in ("gravx", "gravy", "gravz"):
        parameters.setdefault(component, 0.0)
    return parameters


def read_physical(directory: Path) -> dict[str, Any]:
    """The Physical table of parameters.json, when the case has one. Cases
    written before bulk_temp was recorded are saturated."""
    case_json = directory / "parameters.json"
    if not case_json.exists():
        return {}
    with open(case_json, encoding="utf-8") as handle:
        physical: dict[str, Any] = dict(json.load(handle).get("Physical", {}))
    if "bulk_temp" not in physical and "sat_temp" in physical:
        physical["bulk_temp"] = physical["sat_temp"]
    return physical


def read_heater(heater_file: Path) -> dict[str, Any]:
    with h5py.File(heater_file, "r") as heater:
        record: dict[str, Any] = {
            name: float(heater["heater"][name][()].ravel()[0])
            for name in HEATER_SCALARS
            if name in heater["heater"]
        }
        record["nuc_seed_radii"] = heater["init"]["radii"][()].ravel().tolist()
        record["nuc_sites_x"] = heater["site"]["x"][()].ravel().tolist()
        record["nuc_sites_y"] = heater["site"]["y"][()].ravel().tolist()
        if "z" in heater["site"]:
            record["nuc_sites_z"] = heater["site"]["z"][()].ravel().tolist()
    return record


def _matching_files(directory: Path, pattern: re.Pattern[str]) -> list[Path]:
    return sorted(path for path in directory.iterdir() if pattern.search(path.name))


def read_flashx(
    directory: str | os.PathLike[str], mirror_symmetric: bool = False
) -> BoilingSimulation:
    directory = Path(directory)
    plotfiles = _matching_files(directory, PLOTFILE_PATTERN)
    if not plotfiles:
        raise FileNotFoundError(f"no *_hdf5_plt_cnt_* plotfiles in {directory}")

    with h5py.File(plotfiles[0], "r") as first:
        layout = block_layout(first)
        discretization = read_discretization(first, layout)
        non_dimensional = read_non_dimensional(first)

    parameters = SimulationParameters(
        physical=read_physical(directory),
        non_dimensional=non_dimensional,
        discretization=discretization,
        heaters=[
            read_heater(path) for path in _matching_files(directory, HEATER_PATTERN)
        ],
    )
    simulation = BoilingSimulation(
        fields=read_fields(plotfiles, layout),
        parameters=parameters,
        time=read_time(plotfiles),
    )
    return mirror_x(simulation) if mirror_symmetric else simulation
