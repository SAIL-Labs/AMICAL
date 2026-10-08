import numpy as np
import pytest
from astropy.io import fits

from amical.externals.munch import Munch, munchify
from amical.mf_pipeline.ami_function import compute_index_mask, make_mf
from amical.mf_pipeline.bispect import _add_infos_header


@pytest.fixture()
def commentary_infos():
    # Add hdr to infos placeholders for everything but hdr
    mf = Munch(pixelSize=1.0)

    # SimulatedData avoids requiring extra keys in infos
    infos = Munch(orig="SimulatedData", instrument="unknown")

    # Create a fits header with commentary card
    hdr = fits.Header()
    hdr["HISTORY"] = "History is a commentary card"

    return _add_infos_header(infos, hdr, mf, 1.0, "afilename", "amaskname", 1)


def test_add_infos_simulated():
    # Ensure that keys are passed to infos for simulated data, but only when available

    # Create a fits header with two keywords that are usually passed to infos
    hdr = fits.Header()
    hdr["DATE-OBS"] = "2021-06-23"
    hdr["TELESCOP"] = "FAKE-TEL"

    # SimulatedData avoids requiring extra keys in infos
    infos = Munch(orig="SimulatedData", instrument="unknown")

    # Add hdr to infos placeholders for everything but hdr
    mf = Munch(pixelSize=1.0)
    infos = _add_infos_header(infos, hdr, mf, 1.0, "afilename", "amaskname", 1)

    # Check that we kept required keys
    assert infos["date-obs"] == hdr["DATE-OBS"]
    assert infos["telescop"] == hdr["TELESCOP"]

    # Keys that are not in hdr should not be in infos or hdr
    assert "observer" not in infos
    assert "observer" not in infos.hdr


@pytest.mark.filterwarnings("ignore: Commentary cards")
def test_add_infos_header_commentary(commentary_infos):
    # Make sure that _add_infos_header handles _HeaderCommentaryCards from astropy

    # Convert everything to munch object
    munchify(commentary_infos)


def test_commentary_infos_keep(commentary_infos):
    assert "HISTORY" in commentary_infos.hdr


def test_no_commentary_warning_astropy_version():
    # Add hdr to infos placeholders for everything but hdr
    mf = Munch(pixelSize=1.0)

    # SimulatedData avoids requiring extra keys in infos
    infos = Munch(orig="SimulatedData", instrument="unknown")

    # Create a fits header with commentary card
    hdr = fits.Header()
    hdr["HISTORY"] = "History is a commentary card"

    infos = _add_infos_header(infos, hdr, mf, 1.0, "afilename", "amaskname", 1)


@pytest.mark.parametrize("n_holes", [3, 4, 7, 9, 12])
def test_compute_index_mask_bscov(n_holes):
    # Pairs of bispectra (i < j, row-major) sharing at least one baseline.
    index_mask = compute_index_mask(n_holes)
    bs2bl_ix = index_mask.bs2bl_ix
    expected = [
        (i, j)
        for i in range(index_mask.n_bispect)
        for j in range(i + 1, index_mask.n_bispect)
        if set(bs2bl_ix[:, i]) & set(bs2bl_ix[:, j])
    ]
    expected = np.array(expected, dtype=int).reshape(-1, 2).T

    assert index_mask.n_cov == expected.shape[1]
    np.testing.assert_array_equal(index_mask.bscov2bs_ix, expected)


def test_make_mf_index_mask_mismatch():
    with pytest.raises(ValueError, match="index_mask is for 9 holes"):
        make_mf(
            "g7",
            "NIRISS",
            "F430M",
            80,
            display=False,
            index_mask=compute_index_mask(9),
        )


def test_regress_noc_batched():
    from amical.mf_pipeline.idl_function import regress_noc

    rng = np.random.default_rng(1)
    x = rng.normal(size=(5, 12))
    y = rng.normal(size=(4, 12))
    weights = rng.uniform(1, 2, size=(4, 12))

    batched = regress_noc(x, y, weights)
    for i in range(4):
        single = regress_noc(x, y[i], weights[i])
        for key in single:
            np.testing.assert_allclose(batched[key][i], single[key], rtol=1e-12)

    # Exactly determined fit: no degrees of freedom for the MSE.
    assert np.isnan(regress_noc(x[:, :5], y[0, :5], weights[0, :5]).MSE)


def test_compute_complex_bs_chunks_and_dark(global_datadir, monkeypatch):
    from amical.mf_pipeline import bispect
    from amical.mf_pipeline.ami_function import give_peak_info2d

    with fits.open(global_datadir / "test.fits") as fh:
        cube = fh[0].data[:7]
    ft_arr, n_ps, npix = bispect._construct_ft_arr(cube)
    index_mask = compute_index_mask(7)
    mf = make_mf("g7", "NIRISS", "F430M", npix, display=False)
    fringe_peak = give_peak_info2d(mf, index_mask.n_baselines, npix, npix)
    dark_ps = np.random.default_rng(2).uniform(0, 1e3, size=(n_ps, npix, npix))

    def run():
        return bispect._compute_complex_bs(
            ft_arr, index_mask, fringe_peak, mf, dark_ps=dark_ps, verbose=False
        )

    whole = run()
    monkeypatch.setattr(bispect, "_CHUNK_PIXELS", 3 * npix**2)  # chunks of 3 frames
    chunked = run()

    for key in ["vis_arr", "phs"]:
        for field in whole[key].dtype.names:
            np.testing.assert_array_equal(chunked[key][field], whole[key][field])
    for key in ["bs_arr", "fluxes"]:
        np.testing.assert_array_equal(chunked[key], whole[key])
    np.testing.assert_allclose(chunked["ps"], whole["ps"], rtol=1e-12)
    np.testing.assert_allclose(chunked["dps"], whole["dps"], rtol=1e-12)

    # The returned dark calibration is that of the last frame.
    last_dark = [
        np.sum(
            peak[:, 2].astype(float) ** 2
            * dark_ps[-1][tuple(peak[:, :2].T.astype(int))]
        )
        for peak in fringe_peak
    ]
    np.testing.assert_allclose(whole["calib_v2"]["dark"], last_dark, rtol=1e-12)


def test_bs_multi_triangle_matches_complex_bs(global_datadir):
    from amical.mf_pipeline import bispect
    from amical.mf_pipeline.ami_function import (
        bs_multi_triangle,
        give_peak_info2d,
        tri_pix,
    )

    with fits.open(global_datadir / "test.fits") as fh:
        cube = fh[0].data[:3]
    ft_arr, n_ps, npix = bispect._construct_ft_arr(cube)
    index_mask = compute_index_mask(7)
    mf = make_mf("g7", "NIRISS", "F430M", npix, display=False)
    fringe_peak = give_peak_info2d(mf, index_mask.n_baselines, npix, npix)
    sampledisk_r = np.min(np.hypot(mf.u, mf.v)) / 2 / mf.wl * mf.pixelSize * npix
    closing_tri_pix = tri_pix(npix, 0.7 * sampledisk_r, display=False, verbose=False)

    complex_bs = bispect._compute_complex_bs(
        ft_arr,
        index_mask,
        fringe_peak,
        mf,
        closing_tri_pix=closing_tri_pix,
        bs_multi_tri=True,
        verbose=False,
    )
    bs_arr = np.zeros((n_ps, index_mask.n_bispect), dtype=complex)
    for i in range(n_ps):
        bs_arr = bs_multi_triangle(
            i, bs_arr, ft_arr[i], index_mask.bs2bl_ix, mf, closing_tri_pix
        )
    np.testing.assert_array_equal(complex_bs["bs_arr"], bs_arr)
