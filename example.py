#!/usr/bin/env python
from matplotlib import pyplot as plt

import miamis

plt.close("all")

datadir = 'Simulated_NRMdata/'

file_t = datadir + 't_binary_120mas_dm=5.0_mag=5.0_posang=45.0__F430M_81_flat_x11__00.fits'
file_c = datadir + 'c_binary_120mas_dm=5.0_mag=5.0_posang=45.0__F430M_81_flat_x11__00.fits'

maskname = "g7"

params_ami = {"peakmethod": True,
              "bs_MultiTri": True,
              "naive_err": False,
              "n_blocks": 0,
              }

params_data = {"clean": False,  # if clean=True, crop, substract sky and apply apodisation. If simulated NIRISS, do not clean.
               "isz": 150,  # cropped image size (if clean).
               "r1": 60,  # Radius to compute sky (if clean).
               "dr": 10,  # Outer radius to compute sky (if clean).
               "checkrad": False}  # If True, do not perform reduction but check the cropped and rings position (if clean).

bs_t = miamis.extract_bs_mf(file_t, maskname, targetname='fakebinary',
                            **params_ami, **params_data, display=False)

bs_c = miamis.extract_bs_mf(file_c, maskname, targetname='fakepsf',
                            **params_ami, **params_data, display=False)

cal = miamis.calibrate(bs_t, bs_c)

miamis.show(cal, true_flag_t3=False)
miamis.save(cal, fake_obj=True, verbose=False)


plt.show()
