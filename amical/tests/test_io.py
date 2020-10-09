from pathlib import Path

import munch
import numpy as np
from numpy.testing import assert_approx_equal, assert_allclose
import pytest
from astropy.io import fits

import amical
from amical import load, loadc
from amical.get_infos_obs import get_pixel_size

TEST_DIR = Path(__file__).parent
TEST_DATA_DIR = TEST_DIR / "data"
example_oifits = TEST_DATA_DIR / "test.oifits"
example_fits = TEST_DATA_DIR / "test.fits"
save_v2_gauss = TEST_DATA_DIR / 'save_results_v2_example_gauss.fits'
save_cp_gauss = TEST_DATA_DIR / 'save_results_cp_example_gauss.fits'
save_cp_fft = TEST_DATA_DIR / 'save_results_cp_example_fft.fits'


def test_load_file():
    s = load(example_oifits)
    assert isinstance(s, dict)


@pytest.mark.slow
def test_extraction():
    with fits.open(example_fits) as fh:
        cube = fh[0].data

    method = ['fft', 'gauss', 'square']
    for m in method:
        params_ami = {"peakmethod": m,
                      "bs_multi_tri": False,
                      "maskname": "g7",
                      "fw_splodge": 0.7,
                      }
        bs = amical.extract_bs(cube, example_fits, targetname='test',
                               **params_ami, display=False)
        assert isinstance(bs, munch.Munch)


@pytest.mark.slow
def test_calibration():
    with fits.open(example_fits) as fh:
        cube = fh[0].data

    params_ami = {"peakmethod": 'fft',
                  "bs_multi_tri": False,
                  "maskname": "g7",
                  "fw_splodge": 0.7,
                  }
    bs = amical.extract_bs(cube, example_fits, targetname='test',
                           **params_ami, display=False)
    cal = amical.calibrate(bs, bs)
    assert isinstance(cal, munch.Munch)


@pytest.mark.slow
def test_show():
    with fits.open(example_fits) as fh:
        cube = fh[0].data
    params_ami = {"peakmethod": 'fft',
                  "bs_multi_tri": False,
                  "maskname": "g7",
                  "fw_splodge": 0.7,
                  }
    bs = amical.extract_bs(cube, example_fits, targetname='test',
                           **params_ami, display=False)
    cal = amical.calibrate(bs, bs)
    amical.show(cal)


def test_save():
    with fits.open(example_fits) as fh:
        cube = fh[0].data
    params_ami = {"peakmethod": 'fft',
                  "bs_multi_tri": False,
                  "maskname": "g7",
                  "fw_splodge": 0.7,
                  }
    bs = amical.extract_bs(cube, example_fits, targetname='test',
                           **params_ami, display=False)
    cal = amical.calibrate(bs, bs)
    assert isinstance(cal, munch.Munch)

    dic, savefile = amical.save(cal, oifits_file='test.oifits', fake_obj=True)
    v2 = dic['OI_VIS2']['VIS2DATA']
    cp = dic['OI_T3']['T3PHI']

    assert isinstance(dic, dict)
    assert isinstance(savefile, str)
    assert(isinstance(v2, np.ndarray))
    assert(isinstance(cp, np.ndarray))
    assert(len(v2) == 21)
    assert(len(cp) == 35)


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_candid():

    param_candid = {'rmin': 20,
                    'rmax': 250,
                    'step': 100,
                    'ncore': 1
                    }
    fit1 = amical.candid_grid(example_oifits, **param_candid)
    assert isinstance(fit1, dict)


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_candid_multiproc():

    param_candid = {'rmin': 20,
                    'rmax': 250,
                    'step': 100,
                    'ncore': 4
                    }
    fit1 = amical.candid_grid(example_oifits, **param_candid)
    assert isinstance(fit1, dict)


@pytest.mark.slow
def test_pymask():
    fit1 = amical.pymask_grid(str(example_oifits))
    assert isinstance(fit1, dict)
    fit2 = amical.pymask_grid([example_oifits])
    assert isinstance(fit2, dict)


def test_loadc_file():
    s = loadc(example_oifits)
    assert isinstance(s, munch.Munch)


@pytest.mark.parametrize("ins", ['NIRISS', 'SPHERE', 'VAMPIRES'])
def test_getPixel(ins):
    p = get_pixel_size(ins)
    assert isinstance(p, float)
