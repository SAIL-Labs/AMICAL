from pathlib import Path

import pytest
from schema import Schema
from numpy import ndarray
from amical import load, loadc

TEST_DIR = Path(__file__).parent
TEST_DATA_DIR = TEST_DIR / "data"
example_oifits = TEST_DATA_DIR / "test.oifits"

schema = Schema(
    {
        "target": str,
        "calib": str,
        "seeing": float,
        "mjd": float,
        "wl": ndarray,
        "e_wl": ndarray,
        "vis2": ndarray,
        "e_vis2": ndarray,
        "u": ndarray,
        "v": ndarray,
        "bl": ndarray,
        "flag_vis": ndarray,
        "cp": ndarray,
        "e_cp": ndarray,
        "u1": ndarray,
        "v1": ndarray,
        "u2": ndarray,
        "v2": ndarray,
        "bl_cp": ndarray,
        "flag_cp": ndarray,
    }
)

@pytest.mark.parametrize("load_fun", [load, loadc])
def test_load(load_fun):
    data = load_fun(example_oifits)
    schema.validate(data)
