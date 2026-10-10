from pathlib import Path
from typing import Any

import numpy as np
import pytest

from boiling_data.boiling_data import BoilingSimulation, Field, SimulationParameters
from boiling_data.flashx.symmetry import mirror_field_x, mirror_x

WIDTH = 96


@pytest.fixture(scope="module")
def full_domain(half_domain: BoilingSimulation) -> BoilingSimulation:
    return mirror_x(half_domain)


def _half_parameters(
    xl_boundary_type: str = "slip_ins", heaters: list[dict[str, Any]] | None = None
) -> SimulationParameters:
    return SimulationParameters(
        discretization={"x_min": 0.0, "x_max": 2.0, "num_blocks_x": 6},
        heaters=heaters or [],
        boundary={
            "left": {"type": xl_boundary_type},
            "right": {"type": "noslip_ins"},
        },
    )


def _centered(width: int) -> Field:
    grid_x = np.arange(width, dtype=np.float64) + 0.5
    return Field(grid_x[None, None, :] * np.ones((1, 2, 1)), grid_x, np.arange(2.0))


def test_even_reflection_of_a_cell_centered_field() -> None:
    mirrored = mirror_field_x(_centered(3))
    np.testing.assert_array_equal(mirrored.grid_x, [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    np.testing.assert_array_equal(mirrored.data[0, 0], [2.5, 1.5, 0.5, 0.5, 1.5, 2.5])


def test_negated_reflection_flips_sign() -> None:
    mirrored = mirror_field_x(_centered(3), negate=True)
    np.testing.assert_array_equal(
        mirrored.data[0, 0], [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    )


def test_face_on_the_axis_is_kept_once() -> None:
    grid_x = np.arange(4.0)
    faces = Field(grid_x[None, None, :] * np.ones((1, 2, 1)), grid_x, np.arange(2.0))
    mirrored = mirror_field_x(faces, negate=True)
    np.testing.assert_array_equal(mirrored.grid_x, [-3, -2, -1, 0, 1, 2, 3])
    np.testing.assert_array_equal(mirrored.data[0, 0], [-3, -2, -1, 0, 1, 2, 3])


def test_mirrored_fields_double_in_width(full_domain: BoilingSimulation) -> None:
    assert full_domain.field("temperature").data.shape[-1] == 2 * WIDTH
    assert full_domain.field("velfacex").data.shape[-1] == 2 * WIDTH + 1
    assert full_domain.field("velfacey").data.shape[-1] == 2 * WIDTH
    x_faces = full_domain.field("velfacex").grid_x
    assert (x_faces[0], x_faces[WIDTH], x_faces[-1]) == (-2.0, 0.0, 2.0)


def test_x_components_and_vorticity_are_negated_and_the_rest_mirrored(
    full_domain: BoilingSimulation,
) -> None:
    for name in ("temperature", "dfun", "vely", "velfacey"):
        data = full_domain.field(name).data
        np.testing.assert_array_equal(data, data[..., ::-1])
    for name in ("velx", "normx", "velfacex", "omgm"):
        data = full_domain.field(name).data
        np.testing.assert_array_equal(data, -data[..., ::-1])


def test_mirrored_vorticity_agrees_with_the_velocity_on_both_halves(
    full_domain: BoilingSimulation,
) -> None:
    velx, vely, omgm = (full_domain.field(name) for name in ("velx", "vely", "omgm"))
    vorticity = np.gradient(vely.data, vely.grid_x, axis=-1) - np.gradient(
        velx.data, velx.grid_y, axis=-2
    )
    left = velx.grid_x < 0
    for half in (left, ~left):
        correlation = np.corrcoef(
            omgm.data[..., half].ravel(), vorticity[..., half].ravel()
        )[0, 1]
        assert correlation > 0.5


def test_right_half_is_unchanged(
    half_domain: BoilingSimulation, full_domain: BoilingSimulation
) -> None:
    for name, half in half_domain.fields.items():
        width = half.data.shape[-1]
        np.testing.assert_array_equal(
            full_domain.field(name).data[..., -width:], half.data
        )
        np.testing.assert_array_equal(
            full_domain.field(name).grid_x[-width:], half.grid_x
        )


def test_discretization_describes_the_full_domain(
    full_domain: BoilingSimulation,
) -> None:
    discretization = full_domain.parameters.discretization
    assert discretization["x_min"] == -2.0
    assert discretization["num_blocks_x"] == 12
    assert full_domain.parameters.boundary["left"]["type"] == "noslip_ins"
    assert discretization["mirrored_about_x"] == 0.0


def test_off_axis_nucleation_sites_gain_an_image() -> None:
    parameters = mirror_x(
        BoilingSimulation(
            fields={},
            parameters=_half_parameters(
                heaters=[
                    {
                        "nuc_sites_x": [0.0, 0.5],
                        "nuc_sites_y": [0.0, 0.0],
                        "nuc_seed_radii": [0.15, 0.2],
                    }
                ]
            ),
            time=np.zeros(1),
        )
    ).parameters
    heater = parameters.heaters[0]
    assert heater["nuc_sites_x"] == [-0.5, 0.0, 0.5]
    assert heater["nuc_sites_y"] == [0.0, 0.0, 0.0]
    assert heater["nuc_seed_radii"] == [0.2, 0.15, 0.2]


def test_from_flashx_can_mirror_directly(flashx_directory: Path) -> None:
    full = BoilingSimulation.from_flashx(flashx_directory, mirror_symmetric=True)
    assert full.field("temperature").data.shape[-1] == 2 * WIDTH


def test_mirroring_a_full_domain_is_rejected(full_domain: BoilingSimulation) -> None:
    with pytest.raises(ValueError, match="x_min"):
        mirror_x(full_domain)


def test_mirroring_needs_a_slip_wall_on_the_axis() -> None:
    simulation = BoilingSimulation(
        fields={},
        parameters=_half_parameters(xl_boundary_type="noslip_ins"),
        time=np.zeros(1),
    )
    with pytest.raises(ValueError, match="slip"):
        mirror_x(simulation)


def test_mirrored_left_boundary_is_the_right_one() -> None:
    half = BoilingSimulation(
        fields={},
        parameters=SimulationParameters(
            discretization=_half_parameters().discretization,
            boundary={"left": {"type": "slip"}, "right": {"type": "noslip"}},
        ),
        time=np.zeros(1),
    )
    boundary = mirror_x(half).parameters.boundary
    assert boundary["left"] == boundary["right"] == {"type": "noslip"}
    assert half.parameters.boundary["left"] == {"type": "slip"}
