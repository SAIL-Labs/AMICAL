import munch
import numpy as np
import pytest
from astropy.io import fits

import amical
from amical import load
from amical import loadc
from amical.externals import pymask
from amical.get_infos_obs import get_pixel_size


@pytest.fixture()
def example_oifits(global_datadir):
    return global_datadir / "test.oifits"


@pytest.fixture()
def example_oifits_no_date_obs(global_datadir):
    return global_datadir / "test_no_date_obs.oifits"


def test_load_file(example_oifits):
    s = load(example_oifits)
    assert isinstance(s, dict)


peakmethods = ["fft", "gauss", "square"]


@pytest.fixture(name="bss", scope="session")
def example_bss(global_datadir):
    fits_file = global_datadir / "test.fits"
    with fits.open(fits_file) as fh:
        cube = fh[0].data
    bss = {}
    for peakmethod in peakmethods:
        bss[peakmethod] = amical.extract_bs(
            cube,
            fits_file,
            targetname="test",
            bs_multi_tri=False,
            maskname="g7",
            fw_splodge=0.7,
            display=False,
            peakmethod=peakmethod,
        )
    return bss


@pytest.fixture(name="cal", scope="session")
def example_cal_fft(bss):
    bs = bss["fft"]
    return amical.calibrate(bs, bs)


@pytest.mark.slow
def test_extraction(bss):
    assert isinstance(bss["gauss"], munch.Munch)


@pytest.mark.slow
def test_calibration(cal):
    assert isinstance(cal, munch.Munch)


@pytest.mark.slow
def test_show(cal):
    amical.show(cal)


def test_save_cal(cal, tmpdir):

    dic, savefile = amical.save(
        cal, oifits_file="test_cal.oifits", datadir=tmpdir, fake_obj=True
    )

    assert isinstance(dic, dict)
    assert isinstance(savefile, str)

    hdr = fits.getheader(savefile)
    v2 = dic["OI_VIS2"]["VIS2DATA"]
    cp = dic["OI_T3"]["T3PHI"]

    assert isinstance(v2, np.ndarray)
    assert isinstance(cp, np.ndarray)
    assert len(v2) == 21
    assert len(cp) == 35
    assert hdr["ORIGIN"] == "Sydney University"


def test_save_raw(bss, tmpdir):
    bs = bss["fft"]

    dic, savefile = amical.save(
        bs, oifits_file="test_raw.oifits", datadir=tmpdir, fake_obj=True
    )

    assert isinstance(dic, dict)
    assert isinstance(savefile, str)

    hdr = fits.getheader(savefile)
    v2 = dic["OI_VIS2"]["VIS2DATA"]
    cp = dic["OI_T3"]["T3PHI"]

    assert isinstance(v2, np.ndarray)
    assert isinstance(cp, np.ndarray)
    assert len(v2) == 21
    assert len(cp) == 35
    assert hdr["OBJECT"] == hdr["CALIB"]


def test_origin_type(cal, tmpdir):

    og = [50.0]
    with pytest.raises(TypeError, match="origin should be a str or None"):
        amical.save(
            cal,
            oifits_file="test_origin.oifits",
            datadir=tmpdir,
            fake_obj=True,
            origin=og,
        )


def test_save_origin(cal, tmpdir):

    og = "allo"
    _dic, savefile = amical.save(
        cal,
        oifits_file="test_origin.oifits",
        datadir=tmpdir,
        fake_obj=True,
        origin=og,
    )

    hdr = fits.getheader(savefile)

    assert hdr["ORIGIN"] == og


@pytest.mark.slow
@pytest.mark.parametrize("ncore", [1, 2, 4])
def test_candid_grid(example_oifits, ncore):

    param_candid = {"rmin": 20, "rmax": 250, "step": 100, "ncore": ncore}
    fit1 = amical.candid_grid(example_oifits, **param_candid)
    assert isinstance(fit1, dict)


@pytest.mark.slow
def test_pymask(example_oifits):
    fit1 = amical.pymask_grid(str(example_oifits))
    assert isinstance(fit1, dict)
    fit2 = amical.pymask_grid([example_oifits])
    assert isinstance(fit2, dict)


def test_pymask_oifits_no_date_obs(example_oifits_no_date_obs):
    o = pymask.oifits.open(str(example_oifits_no_date_obs))
    assert isinstance(o, pymask.oifits.oifits)


def test_loadc_file(example_oifits):
    s = loadc(example_oifits)
    assert isinstance(s, munch.Munch)


@pytest.mark.parametrize("ins", ["NIRISS", "SPHERE", "VAMPIRES"])
def test_getPixel(ins):
    p = get_pixel_size(ins)
    assert isinstance(p, float)
