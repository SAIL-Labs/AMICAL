from pathlib import Path

import munch
import pytest

from amical import load, loadc
from amical.getInfosObs import GetPixelSize

TEST_DIR = Path(__file__).parent
TEST_DATA_DIR = TEST_DIR / "data"
example_oifits = TEST_DATA_DIR / "test.oifits"


@pytest.mark.parametrize("filepath", [example_oifits, str(example_oifits), example_oifits.with_suffix("")])
def test_load_file(filepath):
    s = load(filepath)
    assert isinstance(s, dict)


@pytest.mark.parametrize("filepath", [example_oifits, str(example_oifits), example_oifits.with_suffix("")])
def test_loadc_file(filepath):
    s = loadc(filepath)
    assert isinstance(s, munch.Munch)


@pytest.mark.parametrize("ins", ['NIRISS', 'SPHERE', 'VAMPIRES'])
def test_getPixel(ins):
    p = GetPixelSize(ins)
    assert isinstance(p, float)
