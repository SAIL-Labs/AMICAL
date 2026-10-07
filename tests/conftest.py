import pathlib

import matplotlib
import pytest
from matplotlib import pyplot as plt


def pytest_configure(config):
    matplotlib.use("Agg")


@pytest.fixture(scope="session")
def global_datadir():
    return pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def cli_datadir():
    return pathlib.Path(__file__).parent / "cli_data"


@pytest.fixture()
def close_figures():
    plt.close("all")
    yield
    plt.close("all")
