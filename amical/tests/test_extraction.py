import munch
from astropy.io import fits

from amical.mf_pipeline.bispect import _add_infos_header


def test_add_infos_simulated():
    # Ensure that keys are passed to infos for simulated data, but only when available

    # Create a fits header with two keywords that are usually passed to infos
    hdr = fits.Header()
    hdr["DATE-OBS"] = "2021-06-23"
    hdr["TELESCOP"] = "FAKE-TEL"

    # This test is for simulated data only
    infos = munch.Munch(orig="SimulatedData", instrument="unknown")

    # Add hdr to infos placeholders for everything but hdr
    mf = munch.Munch(pixelSize=1.0)
    infos = _add_infos_header(infos, hdr, mf, 1.0, "afilename", "amaskname", 1)

    assert infos["DATE-OBS"] == hdr["DATE-OBS"]
    assert infos["TELESCOP"] == hdr["TELESCOP"]
    assert "OBSERVER" not in infos  # Keys that are not in hdr should not be in infos
