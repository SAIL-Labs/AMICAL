import re

import numpy as np
import pytest
from astropy.io import fits
from matplotlib import pyplot as plt

import amical
from amical import load, loadc
from amical.externals.munch import Munch
from amical.get_infos_obs import get_pixel_size

# import numpy as np

# from amical.mf_pipeline.ami_function import find_bad_BL_BS
# from amical.mf_pipeline.ami_function import find_bad_holes

# The test data has no SCI header and does not need one: ignore warning
pytestmark = pytest.mark.filterwarnings(
    "ignore: No SCI header for NIRISS. No PA correction will be applied."
)


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


@pytest.fixture()
def ifs_clean_param():
    clean_param = {
        "isz": 149,
        "r1": 70,
        "dr": 2,
        "apod": True,
        "window": 65,
        "f_kernel": 3,
    }
    return clean_param


@pytest.fixture()
def ifs_ami_param():
    return {
        "peakmethod": "fft",
        "bs_multi_tri": False,
        "maskname": "g7",
        "fw_splodge": 0.7,
        "filtname": "YJ",
    }


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
    assert isinstance(bs, Munch)
    assert len(bs_keys) == 13


@pytest.mark.usefixtures("close_figures")
def test_extract_multitri(global_datadir, tmp_path):
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
        display=True,
        expert_plot=True,
        naive_err=True,
        verbose=True,
        save_to=str(tmp_path),
        peakmethod="fft",
    )
    bs_keys = list(bs.keys())
    assert isinstance(bs, Munch)
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
    assert isinstance(cal, Munch)


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
    assert isinstance(cal, Munch)


def test_calibration(cal):
    assert isinstance(cal, Munch)


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
    assert hdr["TELESCOP"] == cal.raw_t.infos.telescop


def test_save_cal_1hole(cal, tmpdir):
    dic, savefile = amical.save(
        cal,
        oifits_file="test_cal.oifits",
        datadir=tmpdir,
        fake_obj=True,
        ind_hole=0,
    )

    assert isinstance(dic, dict)
    assert isinstance(savefile, str)

    hdr = fits.getheader(savefile)
    v2 = dic["OI_VIS2"]["VIS2DATA"]
    cp = dic["OI_T3"]["T3PHI"]

    assert isinstance(v2, np.ndarray)
    assert isinstance(cp, np.ndarray)
    assert len(v2) == 21
    assert len(cp) == 15
    assert hdr["ORIGIN"] == "Sydney University"


# def test_save_raw(global_datadir, tmpdir):
#     fits_file = global_datadir / "test.fits"
#     with fits.open(fits_file) as fh:
#         cube = fh[0].data
#     bs = amical.extract_bs(
#         cube,
#         fits_file,
#         targetname="WR104",
#         bs_multi_tri=False,
#         maskname="g7",
#         fw_splodge=0.7,
#         display=False,
#         peakmethod="fft",
#     )

#     dic, savefile = amical.save(
#         bs, oifits_file="test_raw.oifits", datadir=tmpdir, fake_obj=False
#     )

#     assert isinstance(dic, dict)
#     assert isinstance(savefile, str)

#     hdr = fits.getheader(savefile)
#     v2 = dic["OI_VIS2"]["VIS2DATA"]
#     cp = dic["OI_T3"]["T3PHI"]

#     assert isinstance(v2, np.ndarray)
#     assert isinstance(cp, np.ndarray)
#     assert len(v2) == 21
#     assert len(cp) == 35
#     assert hdr["OBJECT"] == hdr["CALIB"]


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
    assert isinstance(s, Munch)


@pytest.mark.parametrize("ins", ["NIRISS", "SPHERE", "VAMPIRES"])
def test_getPixel(ins):
    p = get_pixel_size(ins)
    assert isinstance(p, float)


def test_quiet_mode(global_datadir, capsys):
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
        verbose=False,
    )
    captured = capsys.readouterr()
    bs_keys = list(bs.keys())
    assert isinstance(bs, Munch)
    assert len(bs_keys) == 13
    assert captured.out.strip() == ""
    assert captured.err == ""


# @pytest.mark.usefixtures("close_figures")
# def test_bad_holes(global_datadir):
#     fits_file = global_datadir / "test.fits"
#     with fits.open(fits_file) as fh:
#         cube = fh[0].data

#     bs = amical.extract_bs(
#         cube,
#         fits_file,
#         targetname="test",
#         bs_multi_tri=False,
#         maskname="g7",
#         fw_splodge=0.7,
#         display=False,
#         peakmethod="fft",
#     )

#     bad_hole = find_bad_holes(bs, display=True, verbose=True)
#     index_onebad = find_bad_BL_BS([0], bs)
#     index_twobad = find_bad_BL_BS([0, 3], bs)
#     index_nobad = find_bad_BL_BS(bad_hole, bs)

#     n_holes = bs.mask.n_holes
#     n_bl_good = n_holes * (n_holes - 1) / 2.0
#     n_holes -= 1
#     n_bl_onebad = n_holes * (n_holes - 1) / 2.0
#     n_holes -= 1
#     n_bl_twobad = n_holes * (n_holes - 1) / 2.0

#     assert n_bl_good == len(index_nobad[2])
#     assert n_bl_onebad == len(index_onebad[2])
#     assert n_bl_twobad == len(index_twobad[2])


def test_extract_ifs(global_datadir, ifs_clean_param):
    fits_file = global_datadir / "test_ifs.fits"

    spectral_channel = 0
    cube = amical.select_clean_data(fits_file, **ifs_clean_param, i_wl=spectral_channel)

    bs = amical.extract_bs(
        cube,
        fits_file,
        i_wl=0,
        targetname="test",
        bs_multi_tri=False,
        maskname="g7",
        filtname="YH",
        fw_splodge=0.7,
        display=False,
    )
    assert isinstance(bs, Munch)
    assert len(bs) == 13


def test_extract_ifs_missing_iwl(global_datadir):
    fits_file = global_datadir / "test_ifs.fits"

    with fits.open(fits_file) as hdu:
        cube_dirty = hdu[0].data

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Your file seems to be obtained with an IFU instrument: spectral channel "
            "index `i_wl` must be specified."
        ),
    ):
        amical.extract_bs(
            cube_dirty,
            fits_file,
            targetname="test",
            bs_multi_tri=False,
            maskname="g7",
            filtname="YH",
            fw_splodge=0.7,
            display=False,
        )


@pytest.mark.usefixtures("close_figures")
def test_full_process_ifs(tmpdir, global_datadir, ifs_clean_param, ifs_ami_param):
    fits_file = global_datadir / "test_ifs.fits"
    list_index_ifu = [0, 10, 20]
    l_cal = []
    for i_wl in list_index_ifu:
        cube = amical.select_clean_data(fits_file, **ifs_clean_param, i_wl=i_wl)
        bs = amical.extract_bs(
            cube,
            fits_file,
            targetname="fake",
            **ifs_ami_param,
            display=False,
            i_wl=i_wl,
        )

        cal = amical.oifits.wrap_raw(bs)
        l_cal.append(cal)
    assert len(l_cal) == len(list_index_ifu)
    amical.show(l_cal)
    assert plt.gcf().number == 1

    amical.save(
        l_cal,
        oifits_file="fake_ifs.oifits",
        datadir=tmpdir,
        fake_obj=True,
        pa=bs.infos.pa,
    )
