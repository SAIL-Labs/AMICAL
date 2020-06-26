from astropy.io import fits
from matplotlib import pyplot as plt

import miamis

plt.close("all")

datadir = 'Simulated_NRMdata/'

# This example comes with 2 NIRISS simulated dataset representing
# a relativily faint binary star (dm = 6) @ 4.3 μm :
# - < λ/D: sep = 147.7, theta = 46.6 #
# - > λ/D: sep = 302.0, theta = 260.9

sep = 302.0  # binary separation [mas]
theta = 260.9  # position angle (pa) [deg]
dm = 6.0  # contrast ratio [mag]

file_t = datadir + \
    't_binary_s=%2.1fmas_mag=6.0_dm=%2.1f_posang=%2.1f__F430M_81_flat_x11__00.fits' % (
        sep, dm, theta)
file_c = datadir + \
    'c_binary_s=%2.1fmas_mag=6.0_dm=%2.1f_posang=%2.1f__F430M_81_flat_x11__00.fits' % (
        sep, dm, theta)

# Firslty, open fits files as cube: _t is for target (astronomical scene) and _c for calibrator (point source)
hdu = fits.open(file_t)
cube_t = hdu[0].data
hdu.close()

hdu = fits.open(file_c)
cube_c = hdu[0].data
hdu.close()

# ----------------------------------
# Additionnal cleaning step is required here for
# groundbased observations.
# ----------------------------------

#  AMI parameters (refer to the docstrings of `extract_bs_mf` for details)
params_ami = {"peakmethod": 'fft',
              "bs_MultiTri": False,
              "maskname": "g7",
              "fw_splodge": 0.7,
              }

# Extract raw complex observables for the target and the calibrator:
# It's the core of the pipeline (miamis/mf_pipeline/bispect.py)
bs_t = miamis.extract_bs_mf(cube_t, file_t, targetname='fakebinary',
                            **params_ami, display=True)
bs_c = miamis.extract_bs_mf(cube_c, file_c, targetname='fakepsf',
                            **params_ami, display=False)

# Calibrate the raw data to get get calibrated V2 and CP
# bs_c can be a single calibrator result or a list of calibrator.
# (see miamis/core.py for details).
cal = miamis.calibrate(bs_t, bs_c)

# Display and save the results as oifits
miamis.show(cal, true_flag_t3=False, cmax=180)
s = miamis.save(cal, fake_obj=True, verbose=False)

# We perform some analysis of the extracted V2 and CP using
# CANDID package (developped by A. Merand and A. Gallenne).
fit = miamis.fit_binary(s[1], step=50, verbose=False)

plt.show(block=True)
