"""Regression tests for small bugs in the cleaning and extraction steps."""

import warnings
from itertools import combinations

import numpy as np
import pytest
from astropy.io import fits

import amical
from amical import data_processing
from amical.data_processing import (
    _apply_edge_correction,
    clean_data,
    select_clean_data,
    select_data,
    show_clean_params,
)
from amical.externals.munch import munchify
from amical.get_infos_obs import get_ifu_table, get_pixel_size
from amical.mf_pipeline.ami_function import (
    compute_index_mask,
    find_bad_BL_BS,
    make_mf,
)
from amical.mf_pipeline.bispect import _compute_v2_quantities
from amical.mf_pipeline.idl_function import regress_noc


def test_unknown_instrument_pixel_size(capsys):
    assert np.isnan(get_pixel_size("UNKNOWN"))
    mf = make_mf("g7", "UNKNOWN", "F430M", 64, display=False)
    assert mf is None
    assert "Pixel size unknown" in capsys.readouterr().err


def test_clean_data_integer_cube():
    cube = np.full((2, 20, 20), 10, dtype=np.uint16)
    bad_map = np.zeros((20, 20))
    bad_map[5, 5] = 1
    cleaned = clean_data(cube, edge=2, bad_map=bad_map, sky=False, apod=False)
    assert cleaned.dtype.kind == "f"
    assert cleaned.shape == cube.shape
    # The input cube is not modified in place by the edge correction
    assert np.all(cube == 10)


def test_v2_blocks_do_not_overlap():
    rng = np.random.default_rng(0)
    n_ps, n_bl, n_blocks = 12, 3, 4
    v2_arr = rng.normal(size=(n_ps, n_bl))
    res = _compute_v2_quantities(v2_arr, np.ones(n_bl), n_blocks)
    diff = v2_arr.reshape(n_blocks, n_ps // n_blocks, n_bl).mean(axis=1)
    diff -= v2_arr.mean(axis=0)
    expected = diff.T @ diff / (n_blocks - 1) / n_blocks
    np.testing.assert_allclose(res["v2_cov"], expected)


def test_find_bad_BL_BS_all_bad_baselines():
    n_holes = 7
    bs = munchify({"mask": compute_index_mask(n_holes)})
    bad_holes = [0, 3]
    bad_bl, bad_bs, good_bl, good_bs = find_bad_BL_BS(bad_holes, bs)
    n_good = n_holes - len(bad_holes)
    assert len(good_bl) == len(list(combinations(range(n_good), 2)))
    assert len(good_bs) == len(list(combinations(range(n_good), 3)))
    assert len(bad_bs) == len(np.unique(bad_bs))


def test_select_data_clip_removes_off_centre_frames():
    n_frames, npix = 10, 16
    cube = np.array([np.full((npix, npix), 1 + 0.01 * i) for i in range(n_frames)])
    # Strong checkerboard: |FFT| peaks away from zero, so the frame is flagged
    checker = (-1.0) ** np.add.outer(np.arange(npix), np.arange(npix))
    i_flag = 8
    cube[i_flag] += 5 * checker
    out = select_data(cube, clip=True, verbose=False, display=False)
    # Frames 4-9 pass the flux clipping; frame 8 must also go
    assert len(out) == 5
    assert all(np.ptp(frame) == 0 for frame in out)


def test_edge_correction_full_edge():
    img = _apply_edge_correction(np.ones((10, 10)), edge=2)
    assert np.all(img[-2:, :] == 0)
    assert np.all(img[:, -2:] == 0)
    assert np.all(img[2:-2, 2:-2] == 1)


@pytest.mark.usefixtures("close_figures")
def test_show_clean_params_no_sky(global_datadir):
    fig = show_clean_params(global_datadir / "test.fits", isz=None, r1=None)
    assert fig is not None


def test_select_clean_data_passes_mask(global_datadir, monkeypatch):
    seen = {}

    class _Stop(Exception):
        pass

    def fake_show(*args, **kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(data_processing, "show_clean_params", fake_show)
    mask = np.zeros((81, 81))
    with pytest.raises(_Stop):
        select_clean_data(global_datadir / "test.fits", mask=mask, display=True)
    assert seen["mask"] is mask


def test_select_clean_data_ifu_channel_out_of_range(global_datadir):
    fits_file = global_datadir / "test_ifs.fits"
    naxis4 = fits.getheader(fits_file)["NAXIS4"]
    with pytest.raises(ValueError, match="do not exist"):
        select_clean_data(fits_file, i_wl=naxis4, r1=20, dr=2)


def test_regress_noc_square_and_mismatch():
    x = np.array([[1.0, 1.0], [0.0, 1.0]])
    y = np.array([1.0, 2.0])
    reg = regress_noc(x, y, np.ones(2))
    assert np.isnan(reg.MSE)
    with pytest.raises(ValueError, match="Incompatible"):
        regress_noc(x, y, np.ones(3))


@pytest.mark.usefixtures("close_figures")
def test_ifu_get_lambda_range_rounding():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        from amical import ifu

    wl = ifu.get_lambda([0, 5])
    assert not np.allclose(wl, np.round(wl))
    np.testing.assert_allclose(wl, np.round(wl, 2))


@pytest.mark.usefixtures("close_figures")
def test_get_ifu_table_int_display():
    get_ifu_table(5, display=True)


@pytest.mark.usefixtures("close_figures")
@pytest.mark.filterwarnings("ignore:No SCI header:RuntimeWarning")
def test_extract_bs_save_to_without_display(global_datadir, tmp_path):
    fits_file = global_datadir / "test.fits"
    with fits.open(fits_file) as fh:
        cube = fh[0].data
    save_to = tmp_path / "figs"
    with pytest.warns(UserWarning, match="only saved when display=True"):
        bs = amical.extract_bs(
            cube,
            fits_file,
            targetname="test",
            maskname="g7",
            peakmethod="fft",
            display=False,
            save_to=str(save_to),
        )
    assert bs is not None
    assert not save_to.exists()


def test_closing_triangle_order_large_mask():
    """Triangles keep (i, j, k) order even for hole indices >= 8."""
    from amical.mf_pipeline.ami_function import compute_index_mask
    from amical.mf_pipeline.bispect import _format_closing_triangle

    index_mask = compute_index_mask(18)
    tri = np.array(_format_closing_triangle(index_mask))
    assert (np.diff(tri, axis=1) > 0).all()


def test_clean_data_remove_bad_false_leaves_bad_pixels():
    data = np.random.default_rng(0).random((3, 20, 20))
    bad_map = np.zeros((20, 20))
    bad_map[5, 7] = 1
    data[:, 5, 7] = 1e5
    kw = {"sky": False, "apod": False, "bad_map": bad_map}
    fixed = clean_data(data, **kw)
    kept = clean_data(data, remove_bad=False, **kw)
    assert np.all(fixed[:, 5, 7] < 1)
    assert np.all(kept[:, 5, 7] == 1e5)


def test_select_clean_data_passes_remove_bad(global_datadir, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        data_processing, "clean_data", lambda *a, **k: seen.update(k) or None
    )
    select_clean_data(global_datadir / "test.fits", r1=30, dr=5, remove_bad=False)
    assert seen["remove_bad"] is False


def test_compute_phs_error_does_not_modify_fitmat():
    from amical.mf_pipeline.bispect import _compute_phs_error

    index_mask = compute_index_mask(7)
    n_holes, n_bl = 7, index_mask.n_baselines
    rng = np.random.default_rng(1)
    fitmat = np.zeros((n_holes, n_bl + 1))
    for j in range(n_bl):
        fitmat[index_mask.bl2h_ix[0, j], j] = 1.0
        fitmat[index_mask.bl2h_ix[1, j], j] = -1.0
    fitmat[0, n_bl] = 1.0
    before = fitmat.copy()
    phs = np.zeros((2, 5, n_bl), dtype=[("value", float), ("err", float)])
    phs["value"] = rng.normal(size=phs.shape)
    phs["err"] = rng.uniform(0.5, 1, size=phs.shape)
    _compute_phs_error({"phs": phs}, fitmat, index_mask, 64)
    np.testing.assert_array_equal(fitmat, before)


def test_calc_weight_reg_degenerate_frame_warns():
    from amical.mf_pipeline.bispect import _calc_weight_reg

    x = np.eye(3)
    y = np.ones((2, 3))
    w = np.ones((2, 3))
    w[1] = -1.0  # negative weights give a negative covariance diagonal
    with pytest.warns(RuntimeWarning, match="negative covariance"):
        ph, err = _calc_weight_reg(x, y, w)
    np.testing.assert_array_equal(ph[1], 0)
    np.testing.assert_array_equal(err[1], 0)
    assert np.all(ph[0] == 1)
