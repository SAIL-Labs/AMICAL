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
