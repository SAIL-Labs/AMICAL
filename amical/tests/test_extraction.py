import munch
from munch import munchify as dict2class
from astropy.io import fits

from amical.mf_pipeline.bispect import _add_infos_header


def test_add_infos_header_commentary():
    # Make sure that _add_infos_header handles _HeaderCommentaryCards from astropy

    # Create a fits header with commentary card
    hdr = fits.Header()
    hdr["HISTORY"] = "History is a commentary card"

    # SimulatedData avoids requiring extra keys in infos
    infos = munch.Munch(orig="SimulatedData", instrument="unknown")

    # Add hdr to infos placeholders for everything but hdr
    mf = munch.Munch(pixelSize=1.0)
    infos = _add_infos_header(infos, hdr, mf, 1.0, "afilename", "amaskname", 1)

    # Convert everything to munch object
    dict2class(infos)
