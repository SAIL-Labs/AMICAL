from pathlib import Path

import munch
import numpy as np
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


@pytest.mark.parametrize("filepath", [example_oifits])
def test_load_file(filepath):
    s = load(filepath)
    assert isinstance(s, dict)


@pytest.mark.parametrize("filepath", [example_fits])
def test_open_fits(filepath):
    hdu = fits.open(filepath)
    cube = hdu[0].data
    hdu.close()
    assert isinstance(cube, np.ndarray)


@pytest.mark.slow
@pytest.mark.parametrize("filepath", [example_fits])
def test_extraction(filepath):
    hdu = fits.open(filepath)
    cube = hdu[0].data
    hdu.close()
    method = ['fft', 'gauss', 'square']
    for m in method:
        params_ami = {"peakmethod": m,
                      "bs_multi_tri": False,
                      "maskname": "g7",
                      "fw_splodge": 0.7,
                      }
        bs = amical.extract_bs(cube, filepath, targetname='test',
                               **params_ami, display=False)
        assert isinstance(bs, munch.Munch)


@pytest.mark.slow
@pytest.mark.parametrize("filepath", [example_fits])
def test_calibration(filepath):
    hdu = fits.open(filepath)
    cube = hdu[0].data
    hdu.close()
    params_ami = {"peakmethod": 'fft',
                  "bs_multi_tri": False,
                  "maskname": "g7",
                  "fw_splodge": 0.7,
                  }
    bs = amical.extract_bs(cube, filepath, targetname='test',
                           **params_ami, display=False)
    cal = amical.calibrate(bs, bs)
    assert isinstance(cal, munch.Munch)


@pytest.mark.parametrize("filepath", [example_fits])
@pytest.mark.parametrize("savepath", [save_v2_gauss])
def test_bs_v2(filepath, savepath):
    hdu = fits.open(filepath)
    cube = hdu[0].data
    hdu.close()

    params_ami = {"peakmethod": 'gauss',
                  "bs_multi_tri": False,
                  "maskname": "g7",
                  "fw_splodge": 0.7,
                  }
    bs = amical.extract_bs(cube, filepath, targetname='test',
                           **params_ami, display=False)

    res_v2 = np.array([bs.vis2, bs.e_vis2])
    _saved_bs_v2 = fits.open(savepath)[0].data

    assert (len(_saved_bs_v2[0]) == len(res_v2[0]))
    assert (list(_saved_bs_v2[0]) == list(res_v2[0]))
    assert (list(_saved_bs_v2[1]) == list(res_v2[1]))


@pytest.mark.slow
@pytest.mark.parametrize("filepath", [example_fits])
def test_show(filepath):
    hdu = fits.open(filepath)
    cube = hdu[0].data
    hdu.close()
    params_ami = {"peakmethod": 'fft',
                  "bs_multi_tri": False,
                  "maskname": "g7",
                  "fw_splodge": 0.7,
                  }
    bs = amical.extract_bs(cube, filepath, targetname='test',
                           **params_ami, display=False)
    cal = amical.calibrate(bs, bs)
    amical.show(cal)


@pytest.mark.parametrize("filepath", [example_fits])
def test_save(filepath):
    hdu = fits.open(filepath)
    cube = hdu[0].data
    hdu.close()
    params_ami = {"peakmethod": 'fft',
                  "bs_multi_tri": False,
                  "maskname": "g7",
                  "fw_splodge": 0.7,
                  }
    bs = amical.extract_bs(cube, filepath, targetname='test',
                           **params_ami, display=False)
    cal = amical.calibrate(bs, bs)
    assert isinstance(cal, munch.Munch)

    dic, savefile = amical.save(cal, oifits_file='test.oifits', fake_obj=True)
    v2 = dic['OI_VIS2']['VIS2DATA']
    cp = dic['OI_T3']['T3PHI']

    expected_v2 = 0.9999591731687147
    expected_cp = 0.
    assert isinstance(dic, dict)
    assert isinstance(savefile, str)
    assert(isinstance(v2, np.ndarray))
    assert(isinstance(cp, np.ndarray))
    assert(len(v2) == 21)
    assert(len(cp) == 35)
    assert(v2[0] == expected_v2)
    assert(cp[0] == expected_cp)


@pytest.mark.slow
@pytest.mark.parametrize("filepath", [example_oifits])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_candid(filepath):

    param_candid = {'rmin': 20,
                    'rmax': 250,
                    'step': 100,
                    'ncore': 1
                    }
    fit1 = amical.candid_grid(filepath, **param_candid)
    assert isinstance(fit1, dict)


@pytest.mark.slow
@pytest.mark.parametrize("filepath", [example_oifits])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_candid_multiproc(filepath):

    param_candid = {'rmin': 20,
                    'rmax': 250,
                    'step': 100,
                    'ncore': 4
                    }
    fit1 = amical.candid_grid(filepath, **param_candid)
    assert isinstance(fit1, dict)


@pytest.mark.slow
@pytest.mark.parametrize("filepath", [example_oifits])
def test_pymask(filepath):
    fit1 = amical.pymask_grid(str(filepath))
    assert isinstance(fit1, dict)
    fit2 = amical.pymask_grid([filepath])
    assert isinstance(fit2, dict)


@pytest.mark.parametrize("filepath", [example_oifits])
def test_loadc_file(filepath):
    s = loadc(filepath)
    assert isinstance(s, munch.Munch)


@pytest.mark.parametrize("ins", ['NIRISS', 'SPHERE', 'VAMPIRES'])
def test_getPixel(ins):
    p = get_pixel_size(ins)
    assert isinstance(p, float)
