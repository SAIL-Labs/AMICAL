from astropy.io import fits
from matplotlib import pyplot as plt

import miamis
from miamis.dataProcessing import selectCleanData
from miamis.tools import checkSeeingCond, plotSeeingCond

plt.close("all")

datadir = 'TestSPHEREData/'


file_t = datadir + 'HD142527_IRD_SCIENCE_DBI_LEFT_CUBE.fits'
file_c = datadir + 'HD142695_IRD_SCIENCE_DBI_LEFT_CUBE.fits'


# ----------------------------------
# Cleaning step
# ----------------------------------
cube_t = selectCleanData(file_t, clip=True,
                         corr_ghost=False,
                         display=True)[0]
cube_c = selectCleanData(file_c, clip=True,
                         corr_ghost=False,
                         display=True)[0]

#  AMI parameters (refer to the docstrings of `extract_bs_mf` for details)
params_ami = {"peakmethod": 'gauss',
              "bs_MultiTri": False,
              "maskname": "g7",
              "fw_splodge": 0.7,
              "filtname": 'K1'
              }


# # Extract raw complex observables for the target and the calibrator:
# # It's the core of the pipeline (miamis/mf_pipeline/bispect.py)
bs_t = miamis.extract_bs_mf(cube_t, file_t, targetname='HD142527',
                            **params_ami, display=True)
bs_c = miamis.extract_bs_mf(cube_c, file_c, targetname='HD142695',
                            **params_ami, display=False)

cond_t = checkSeeingCond([bs_t])
cond_c = checkSeeingCond([bs_c])

pa = cond_t['pa'][0]


# Calibrate the raw data to get get calibrated V2 and CP
# bs_c can be a single calibrator result or a list of calibrator.
# (see miamis/core.py for details).
cal = miamis.calibrate(bs_t, bs_c)

# Display and save the results as oifits
miamis.show(cal, true_flag_t3=False, cmax=180, pa=pa)
s = miamis.save(cal, fake_obj=True, verbose=False, pa=pa)

# We perform some analysis of the extracted V2 and CP using
# CANDID package (developped by A. Merand and A. Gallenne).

# WARNING: CANDID uses multiprocessing to compute the grid, and
# it appeared to be instable in the last version of OSX catalina+
# So we imposed ncore=1 by default (no multiproc), you can
# try to increase ncore option in fit_binary but it could crash
# depending on your system (tested on OSX-mojave).
fit = miamis.fit_binary(s[1], step=20, rmax=150, diam=20, doNotFit=[])

plt.show(block=True)
