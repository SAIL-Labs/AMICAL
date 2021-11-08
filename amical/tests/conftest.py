import pathlib

import pytest


@pytest.fixture(scope="session")
def global_datadir():
    return pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def cli_datadir():
    return pathlib.Path(__file__).parent / "cli_data"
