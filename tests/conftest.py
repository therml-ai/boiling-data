from pathlib import Path

import pytest

from boiling_data.boiling_data import BoilingSimulation
from boiling_data.flashx import read_flashx

FLASHX_DATA = Path(__file__).parent / "data" / "flashx"


@pytest.fixture(scope="session")
def flashx_directory() -> Path:
    return FLASHX_DATA


@pytest.fixture(scope="session")
def half_domain(flashx_directory: Path) -> BoilingSimulation:
    """Three frames of a pool boiling case run on the right half of a domain
    symmetric about x = 0."""
    return read_flashx(flashx_directory)
