from pathlib import Path

import pytest

from boiling_data.boiling_data import BoilingSimulation
from boiling_data.bubbleml import read_bubbleml
from boiling_data.flashx import read_flashx

DATA = Path(__file__).parent / "data"
FLASHX_DATA = DATA / "flashx"
BUBBLEML_CASE = DATA / "bubbleml" / "case-twall-70.hdf5"


@pytest.fixture(scope="session")
def flashx_directory() -> Path:
    return FLASHX_DATA


@pytest.fixture(scope="session")
def half_domain(flashx_directory: Path) -> BoilingSimulation:
    """Three frames of a pool boiling case run on the right half of a domain
    symmetric about x = 0."""
    return read_flashx(flashx_directory)


@pytest.fixture(scope="session")
def bubbleml_path() -> Path:
    return BUBBLEML_CASE


@pytest.fixture(scope="session")
def bubbleml_case(bubbleml_path: Path) -> BoilingSimulation:
    """Three frames of the same pool boiling case, converted with
    scripts/flashx_to_bubbleml.py from a run that also wrote a heater file."""
    return read_bubbleml(bubbleml_path)
