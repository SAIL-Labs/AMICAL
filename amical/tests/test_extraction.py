import astropy
import munch
import pytest
from astropy.io import fits
from packaging.version import Version

from amical.mf_pipeline.bispect import _add_infos_header


@pytest.fixture()
def infos():
    # SimulatedData avoids requiring extra keys in infos
    return munch.Munch(orig="SimulatedData", instrument="unknown")


@pytest.fixture()
def commentary_hdr():
    # Create a fits header with commentary card
    hdr = fits.Header()
    hdr["HISTORY"] = "History is a commentary card"
    return hdr


@pytest.fixture()
def astropy_versions():

    astropy_version = astropy.__version__
    working_version = "5.0rc1"

    return astropy_version, working_version


@pytest.fixture
def commentary_infos(infos, commentary_hdr):

    # Add hdr to infos placeholders for everything but hdr
    mf = munch.Munch(pixelSize=1.0)

    return _add_infos_header(
        infos, commentary_hdr, mf, 1.0, "afilename", "amaskname", 1
    )


def test_add_infos_simulated(infos):
    # Ensure that keys are passed to infos for simulated data, but only when available

    # Create a fits header with two keywords that are usually passed to infos
    hdr = fits.Header()
    hdr["DATE-OBS"] = "2021-06-23"
    hdr["TELESCOP"] = "FAKE-TEL"

    # Add hdr to infos placeholders for everything but hdr
    mf = munch.Munch(pixelSize=1.0)
    infos = _add_infos_header(infos, hdr, mf, 1.0, "afilename", "amaskname", 1)

    assert infos["date-obs"] == hdr["DATE-OBS"]
    assert infos["telescop"] == hdr["TELESCOP"]
    assert "observer" not in infos  # Keys that are not in hdr should not be in infos
    assert "observer" not in infos.hdr  # Keys that are not in hdr should still not be


def test_add_infos_header_commentary(commentary_infos):
    # Make sure that _add_infos_header handles _HeaderCommentaryCards from astropy

    # Convert everything to munch object
    munch.munchify(commentary_infos)


def test_commentary_infos_keep(commentary_infos, astropy_versions):
    # Check that commentary cards are removed or kept depending on astropy version

    astropy_version, working_version = astropy_versions

    if Version(astropy_version) < Version(working_version):
        assert "HISTORY" not in commentary_infos.hdr
    else:
        assert "HISTORY" in commentary_infos.hdr


def test_astropy_version_warning(infos, commentary_hdr, astropy_versions, capfd):
    # Test that AMICAL warns about astropy < 5.0 removing commentary cards

    astropy_version, working_version = astropy_versions

    # Add hdr to infos placeholders for everything but hdr
    mf = munch.Munch(pixelSize=1.0)

    infos = _add_infos_header(
        infos, commentary_hdr, mf, 1.0, "afilename", "amaskname", 1
    )
    captured = capfd.readouterr()

    if Version(astropy_version) < Version(working_version):
        # NOTE: Adding colors codes because output with cprint has them
        msg = (
            "\x1b[32mCommentary cards are removed from the header with astropy"
            f" version < {working_version}. Your astropy version is"
            f" {astropy_version}\x1b[0m\n"
        )
        assert captured.out == msg
    else:
        assert not captured.out
