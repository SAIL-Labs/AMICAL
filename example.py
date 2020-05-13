#!/usr/bin/env python
from astropy.io import fits
from matplotlib import pyplot as plt

import miamis

plt.close("all")

datadir = 'Simulated_NRMdata/'

file_t = datadir + 't_binary_120mas_dm=5.0_mag=5.0_posang=45.0__F430M_81_flat_x11__00.fits'
file_c = datadir + 'c_binary_120mas_dm=5.0_mag=5.0_posang=45.0__F430M_81_flat_x11__00.fits'

hdu = fits.open(file_t)
cube_t = hdu[0].data
hdu.close()

hdu = fits.open(file_c)
cube_c = hdu[0].data
hdu.close()

params_ami = {"peakmethod": True,
              "bs_MultiTri": True,
              "naive_err": False,
              "n_blocks": 0,
              "maskname": "g7"
              }

bs_t = miamis.extract_bs_mf(cube_t, file_t, targetname='fakebinary',
                            **params_ami, display=False)

bs_c = miamis.extract_bs_mf(cube_c, file_c, targetname='fakepsf',
                            **params_ami, display=True)

cal = miamis.calibrate(bs_t, bs_c)

miamis.show(cal, true_flag_t3=False)
miamis.save(cal, fake_obj=True, verbose=True)

plt.show()
