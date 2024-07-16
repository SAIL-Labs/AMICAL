import os

from astropy.io import fits
from matplotlib import pyplot as plt

import amical

plt.close("all")

datadir = "NRM_DATA/"

# This example comes with 2 NIRISS simulated dataset representing
# a relativily faint binary star (dm = 6) @ 4.3 μm :
# - < λ/D: sep = 147.7, theta = 46.6 #
# - > λ/D: sep = 302.0, theta = 260.9

sep = 147.7  # binary separation [mas]
theta = 46.6  # position angle (pa) [deg]
dm = 6.0  # contrast ratio [mag]

file_t = os.path.join(
    datadir,
    f"t_binary_s={sep:2.1f}mas_mag=6.0_dm={dm:2.1f}_posang={theta:2.1f}__F430M_81_flat_x11__00.fits",
)
file_c = os.path.join(
    datadir,
    f"c_binary_s={sep:2.1f}mas_mag=6.0_dm={dm:2.1f}_posang={theta:2.1f}__F430M_81_flat_x11__00.fits",
)

# Firslty, open fits files as cube: _t is for target (astronomical scene) and _c for calibrator (point source)
hdu = fits.open(file_t)
cube_t = hdu[0].data
hdu.close()

hdu = fits.open(file_c)
cube_c = hdu[0].data
hdu.close()

# ----------------------------------
# Additionnal cleaning step is required for real NIRISS data
# or MIRAGE data (bad pixel, centering, and background).
# Check example_SPHERE.py for more details.
# ----------------------------------

#  AMI parameters (refer to the docstrings of `extract_bs` for details)
params_ami = {
    "peakmethod": "fft",
    "bs_multi_tri": False,
    "maskname": "g7",
    "fw_splodge": 0.7,
}

# Extract raw complex observables for the target and the calibrator:
# It's the core of the pipeline (amical/mf_pipeline/bispect.py)
bs_t = amical.extract_bs(
    cube_t, file_t, targetname="fakebinary", **params_ami, display=True
)
bs_c = amical.extract_bs(
    cube_c, file_c, targetname="fakepsf", **params_ami, display=False
)

# Calibrate the raw data to get calibrated V2 and CP.
# bs_c can be a single calibrator result or a list of calibrators.
# (see amical/calibration.py for details).
cal = amical.calibrate(bs_t, bs_c)

# Display and save the results as oifits
amical.show(cal)
dic = amical.save(cal, oifits_file="example_fakebinary_NIRISS.oifits", fake_obj=True)

plt.show(block=True)
