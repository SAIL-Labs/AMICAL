from matplotlib import pyplot as plt
import os
import amical

plt.close("all")

datadir = 'TestSPHEREData/'

file_t = os.path.join(datadir, 'HD142527_IRD_SCIENCE_DBI_LEFT_CUBE.fits')
file_c = os.path.join(datadir, 'HD142695_IRD_SCIENCE_DBI_LEFT_CUBE.fits')

# ----------------------------------
# Cleaning step
# ----------------------------------

clean_param = {'isz': 149,
               'r1': 70,
               'dr': 2,
               'apod': True,
               'window': 65,
               'f_kernel': 3
               }

amical.check_data_params(file_t, **clean_param)
cube_t = amical.select_clean_data(file_t, clip=True,
                                  **clean_param, display=True)

cube_c = amical.select_clean_data(file_c, clip=True,
                                  **clean_param, display=True)

#  AMI parameters (refer to the docstrings of `extract_bs` for details)
params_ami = {"peakmethod": 'fft',
              "bs_multi_tri": False,
              "maskname": "g7",
              "fw_splodge": 0.7,
              "filtname": 'K1'
              }


# # Extract raw complex observables for the target and the calibrator:
# # It's the core of the pipeline (amical/mf_pipeline/bispect.py)
bs_t = amical.extract_bs(cube_t, file_t, targetname='HD142527',
                         **params_ami, display=True)
bs_c = amical.extract_bs(cube_c, file_c, targetname='HD142695',
                         **params_ami, display=False)

# (from amical.tools import check_seeing_cond, plot_seeing_cond)
# In case of multiple files for a same target, you can
# check the seeing condition and select only the good ones.
# cond_t = check_seeing_cond([bs_t])
# cond_c = check_seeing_cond([bs_c])
# plot_seeing_cond([cond_t, cond_c], lim_seeing=1)


# Calibrate the raw data to get get calibrated V2 and CP
# bs_c can be a single calibrator result or a list of calibrator.
# (see amical/core.py for details).
cal = amical.calibrate(bs_t, bs_c)


# Display and save the results as oifits
amical.show(cal, true_flag_t3=False, cmax=180, pa=0)  # bs_t.infos.pa)
# amical.save(cal, oifits_file='example_HD142527_SPHERE.oifits',
#             fake_obj=True, verbose=False, pa=bs_t.infos.pa)

plt.show(block=True)
