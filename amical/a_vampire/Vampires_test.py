import shutil
import matplotlib.pyplot as plt
import amical
import os
import numpy as np
from scipy import io
from matplotlib.colors import PowerNorm
from astropy.io import fits
from diff_cal_AMICAL_VAMPIRES import diff_cal_AMICAL_VAMPIRES
from astropy.io import fits
plt.ion()

# set whether to create and format dark frames or not - only needs to be done once before the newly formatted dark frames can be used
makedarks = False
# set whether to use AMICAL to extract polarised visibilities from raw data - only needs to be done once before you can then use these files with the diff_cal_AMICAL_VAMPIRES library
amicalextract = True
# set whether to use file strings or headers to extract parameters for calibration. Use headers if the headers contain all information
useheaders = False  # must be False as still in development

print("DO NOT RUN THIS CODE UNLESS YOU ARE LUCI - IT IS NOT FINISHED AND IT WILL SAVE FILES INTO MY (LUCI) DIRECTORIES")

bootstrap = 100  # the number of times we bootstrap
starname = 'muCep'  # star name
starcode = 'muCep_20201207'  # star code
condition = '_darkframe_skysub'  # conditions used to extract data
extranotes = ''  # '_bias0p5sub' #_bias0p1max' Ignore this, keep as ''


# location of this script - a copy will be saved with your extracted quantities
codedir = '/suphys/llil9854/code/a_vampire_code/AMICAL/amical/a_vampire'

# location where we will store images of the results
imagedir = '/import/morgana2/snert/lucinda/AMICAL_VAMPIRES/Results/results_2/images_results/'

# location of raw input data
datadir = '/import/morgana2/snert/VAMPIRESData/202012/20201207/preproc'

# location of results - ie, extracted quantities that will come out of AMICAL, and go into library diff_cal_AMICAL_VAMPIRES
resultsdir = '/import/morgana2/snert/lucinda/AMICAL_VAMPIRES/Results/results_2' + '/' + starcode + condition

# location of the RAW dark frame files
darkframefiles = '/Volumes/snert/lucinda/AMICAL_VAMPIRES/Results/results_2/muCep_20201207_darkframe_skysub/dark_frames_muCep_20201209/darks_256_10ms_em300_20201209_Open_Mirror_0.npz'

# location of where you would like to put PROCESSED dark frame files - that will be inputted into AMICAL.
darkframedir = '/import/morgana2/snert/lucinda/AMICAL_VAMPIRES/Results/results_2' + '/' + starcode + condition + '/dark_frames_' + 'muCep_20201209'

# location for each camera's dark frame file - the above path with different endings
camera1dark_path = darkframedir + '/camera1dark.fits'
camera2dark_path = darkframedir + '/camera2dark.fits'

# list of all files in our datadirectory
datadir_AMICAL = datadir
paths_AMICAL = os.listdir(datadir_AMICAL + '/')

# specify which .fits files meet criteria relevant for this star - you will need to adapt this depending on the nature of your data and what you want to process
paths_AMICAL = [a for a in paths_AMICAL if a.endswith('.fits') and a.startswith('muCep_01_20201207_750-50_18holeNudged_')]


if makedarks:

    # this currently assumes that dark frame files are .npz format, but this will not always be the case and we can add in different formats in due course.
    npzfile = np.load(darkframefiles)
    camera1dark = npzfile.f.finalDarks[:, :, 0]
    hdu1 = fits.PrimaryHDU(camera1dark)
    hdu1_l = fits.HDUList([hdu1])
    hdu1_l.writeto(camera1dark_path)

    camera2dark = npzfile.f.finalDarks[:, :, 1]
    hdu2 = fits.PrimaryHDU(camera2dark)
    hdu2_l = fits.HDUList([hdu2])
    hdu2_l.writeto(camera2dark_path)

if amicalextract:

    for i in range(0, len(paths_AMICAL)):

        temp = paths_AMICAL[i]
        file_t = os.path.join(datadir_AMICAL, temp)

        clean_param = {'isz': 220,  # Size of the final image (cropped)
                       'r1': 95,  # Radius to compute the background (r1 to r1+dr)
                       'dr': 2,  # not for VAMPIRES - check with plots
                       'apod': True,  # Apodisation to go smoothly to zero at the edge - could do?
                       'window': 80,  # FWHM of the super-gaussian to be applied as apod
                       'f_kernel': 3  # gaussian convolution to ensure the good centering
                       }

        params_ami = {"peakmethod": 'fft',  # CHECK method - try SQUARE - check this?
                      "bs_multi_tri": False,  # compute different triangles for closure phases
                      "maskname": "g18",  # CHECK mask - fact that there is a missing hole?
                      "instrum": 'VAMPIRES',  # instrument
                      'fw_splodge': 0.7,
                      'hole_diam': 0.162,
                      'cutoff': 8e-2,  # CHECK
                      'n_wl': 3,
                      'filtname': '750-50'
                      }

        # this useheaders method is still incomplete
        if useheaders:
            hdul = fits.open(file_t)
            hdr = hdul[0].header
            camera = int(hdr['U_CAMERA'])
            print(camera)

        # this is extracting which camera we are using from the file names
        camera = temp[-8:-7]
        tempx = temp[-12:-9]
        tempi = int(tempx)
        elsetemp = temp[-9:-5]
        camera = int(camera)

        if camera == 1:
            # clean data
            cube_t = amical.select_clean_data(file_t, clip=False, **clean_param, display=False,
                                              darkfile=camera1dark_path, sky=False)
            # extract data
            bs_t = amical.extract_bs(cube_t, file_t, targetname=starcode, **params_ami, display=False,
                                     compute_cp_cov=False, theta_detector=83., scaling_uv=0.94, fliplr=False)

        elif camera == 2:
            # clean data
            cube_t = amical.select_clean_data(file_t, clip=False, **clean_param, display=False,
                                              darkfile=camera2dark_path, sky=False)
            # extract data
            bs_t = amical.extract_bs(cube_t, file_t, targetname=starcode, **params_ami, display=False,
                                     compute_cp_cov=False, theta_detector=97, scaling_uv=0.94, fliplr=True)

        # construct path for the extracted file
        path = resultsdir + '/' + str(tempi) + elsetemp + '_bs_t'

        # save extracted file
        amical.save_bs_hdf5(bs_t, path)


### Now we use the diff_cal_AMICAL_VAMPIRES library

# create an object for the star
AMICAL_obj = diff_cal_AMICAL_VAMPIRES(starcode, paths_AMICAL, datadir_AMICAL, codedir, resultsdir, 'AMICAL', bootstrap)

# compute differential polarisation, bootstrap argument will perform bootstrapping to produce the error terms which we need
AMICAL_obj.diff_pol_global(bootstrap)

# plot the result and save the image
AMICAL_obj.plot_bootstrap_diff_pol_cal(condition=condition, imagedir=imagedir, extranotes=extranotes)

# save copy of file used to run script
shutil.copyfile(codedir + '/Vampires_test.py', resultsdir)