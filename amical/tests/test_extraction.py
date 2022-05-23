import astropy
import munch
import pytest
from astropy.io import fits
from packaging.version import Version

from amical.mf_pipeline.bispect import _add_infos_header


# Astropy versions for
ASTROPY_VERSION = Version(astropy.__version__)


@pytest.fixture()
def commentary_infos():

    # Add hdr to infos placeholders for everything but hdr
    mf = munch.Munch(pixelSize=1.0)

    # SimulatedData avoids requiring extra keys in infos
    infos = munch.Munch(orig="SimulatedData", instrument="unknown")

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
    infos = munch.Munch(orig="SimulatedData", instrument="unknown")

    # Add hdr to infos placeholders for everything but hdr
    mf = munch.Munch(pixelSize=1.0)
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
    munch.munchify(commentary_infos)


@pytest.mark.xfail(
    ASTROPY_VERSION < Version("5.0rc"),
    reason="Munch cannot handle commentary cards for Astropy < 5.0",
)
def test_commentary_infos_keep(commentary_infos):
    assert "HISTORY" in commentary_infos.hdr


@pytest.mark.skipif(
    ASTROPY_VERSION < Version("5.0"),
    reason="Astropy < 5.0 raised a warning for commentary cards",
)
def test_no_commentary_warning_astropy_version():

    # Add hdr to infos placeholders for everything but hdr
    mf = munch.Munch(pixelSize=1.0)

    # SimulatedData avoids requiring extra keys in infos
    infos = munch.Munch(orig="SimulatedData", instrument="unknown")

    # Create a fits header with commentary card
    hdr = fits.Header()
    hdr["HISTORY"] = "History is a commentary card"

    infos = _add_infos_header(infos, hdr, mf, 1.0, "afilename", "amaskname", 1)
