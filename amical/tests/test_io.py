import munch
import numpy as np
import pytest
from astropy.io import fits
from matplotlib import pyplot as plt

import amical
from amical import load
from amical import loadc
from amical.get_infos_obs import get_pixel_size


@pytest.fixture()
def close_figures():
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture()
def example_oifits(global_datadir):
    return global_datadir / "test.oifits"


def test_load_file(example_oifits):
    s = load(example_oifits)
    assert isinstance(s, dict)


peakmethods = ["fft", "gauss", "square"]


@pytest.mark.parametrize("peakmethod", ["fft", "gauss", "square", "unique"])
def test_extract(peakmethod, global_datadir):
    fits_file = global_datadir / "test.fits"
    with fits.open(fits_file) as fh:
        cube = fh[0].data
    bs = amical.extract_bs(
        cube,
        fits_file,
        targetname="test",
        bs_multi_tri=False,
        maskname="g7",
        fw_splodge=0.7,
        display=False,
        peakmethod=peakmethod,
    )
    bs_keys = list(bs.keys())
    assert isinstance(bs, munch.Munch)
    assert len(bs_keys) == 13


def test_extract_multitri(global_datadir):
    fits_file = global_datadir / "test.fits"
    with fits.open(fits_file) as fh:
        cube = fh[0].data
    bs = amical.extract_bs(
        cube,
        fits_file,
        targetname="test",
        bs_multi_tri=True,
        maskname="g7",
        fw_splodge=0.7,
        display=False,
        peakmethod="fft",
    )
    bs_keys = list(bs.keys())
    assert isinstance(bs, munch.Munch)
    assert len(bs_keys) == 13


@pytest.fixture(name="cal", scope="session")
def example_cal_fft(global_datadir):
    fits_file = global_datadir / "test.fits"
    with fits.open(fits_file) as fh:
        cube = fh[0].data
    bs = amical.extract_bs(
        cube,
        fits_file,
        targetname="test",
        bs_multi_tri=False,
        maskname="g7",
        fw_splodge=0.7,
        display=False,
        peakmethod="fft",
    )
    return amical.calibrate(bs, bs)


def test_cal_atmcorr(global_datadir):
    fits_file = global_datadir / "test.fits"
    with fits.open(fits_file) as fh:
        cube = fh[0].data
    bs = amical.extract_bs(
        cube,
        fits_file,
        targetname="test",
        bs_multi_tri=False,
        maskname="g7",
        fw_splodge=0.7,
        display=False,
        peakmethod="fft",
    )

    cal = amical.calibrate(bs, bs, apply_atmcorr=True)
    assert isinstance(cal, munch.Munch)


def test_cal_phscorr(global_datadir):
    fits_file = global_datadir / "test.fits"
    with fits.open(fits_file) as fh:
        cube = fh[0].data
    bs = amical.extract_bs(
        cube,
        fits_file,
        targetname="test",
        bs_multi_tri=False,
        maskname="g7",
        fw_splodge=0.7,
        display=False,
        peakmethod="fft",
    )
    cal = amical.calibrate(bs, bs, apply_atmcorr=True)
    assert isinstance(cal, munch.Munch)


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


def test_save_raw(global_datadir, tmpdir):
    fits_file = global_datadir / "test.fits"
    with fits.open(fits_file) as fh:
        cube = fh[0].data
    bs = amical.extract_bs(
        cube,
        fits_file,
        targetname="test",
        bs_multi_tri=False,
        maskname="g7",
        fw_splodge=0.7,
        display=False,
        peakmethod="fft",
    )

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


def test_loadc_file(example_oifits):
    s = loadc(example_oifits)
    assert isinstance(s, munch.Munch)


@pytest.mark.parametrize("ins", ["NIRISS", "SPHERE", "VAMPIRES"])
def test_getPixel(ins):
    p = get_pixel_size(ins)
    assert isinstance(p, float)
