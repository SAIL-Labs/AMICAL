import shutil
import matplotlib.pyplot as plt
import amical
import os
import numpy as np
from scipy import io
from matplotlib.colors import PowerNorm
from astropy.io import fits
from diff_cal_AMICAL_VAMPIRES import diff_cal_AMICAL_VAMPIRES
from diff_cal_tools import makeVampDarks

from astropy.io import fits
plt.ion()
from os import path

useheaders = False                                                                      # MUST BE FALSE RIGHT NOW set whether to use file strings or headers to extract parameters for calibration. Use headers if the headers contain all information

########################   SET THESE PARAMETERS

project_folder = '/import/morgana2/snert/lucinda/amical_vampires_project'               # user pathname on morgana2 - location of desired project folder
datadir = '/import/morgana2/snert/VAMPIRESData/202012/20201207/preproc'                 # location of raw input data
starcode = 'muCep_20201207_TEST'                                                             # star code
bootstrap = 100                                                                         # number of times you would like to bootstrap
darkdataPath = '/import/pendragon1/snert/VAMPIRES/VAMPIRESData_201705/20170502/' ##### THIS IS WRONG!
darkfilePref = 'darks_256_10ms_em300_20170502_750-50_Mirror_0'                          # start of the dark files
nSubFiles = 16 * 2                                                                      # number of sub files used for dark files
rawfilePref = 'muCep_01_20201207_750-50_18holeNudged_'                                  # raw file prefix
#########################

if path.exists(project_folder + '/') == False:
    os.mkdir(project_folder + '/')

if path.exists(project_folder + '/results/') == False:
    os.mkdir(project_folder + '/results/')

if path.exists(project_folder + '/results/image_results/') == False:
    os.mkdir(project_folder + '/results/image_results/')

codedir = os.getcwd()                                                                                                        # location of this script - a copy will be saved with your extracted quantities in resultsdir
imagedir = project_folder + '/results/image_results'                                                                        # location where we will store images of the results
resultsdir = project_folder + '/results/' + starcode                                                                        # location of results - ie, extracted quantities that will come out of AMICAL, and go into library diff_cal_AMICAL_VAMPIRES

darkframedir = resultsdir + '/dark_frames'                                                                                  # location of where you would like to put PROCESSED dark frame files - that will be inputted into AMICAL.
camera1dark_path = darkframedir + '/camera1dark.fits'                                                                       # location for each camera's dark frame file - the above path with different endings
camera2dark_path = darkframedir + '/camera2dark.fits'                                                                       # location for each camera's dark frame file - the above path with different endings

datadir_AMICAL = datadir                                                                                                    # list of all files in our datadirectory
paths_AMICAL = os.listdir(datadir_AMICAL + '/')
paths_AMICAL = [a for a in paths_AMICAL if a.endswith('.fits') and a.startswith(rawfilePref)]                               # specify which .fits files meet criteria relevant for this star - you will need to adapt this depending on the nature of your data and what you want to process



if path.exists(resultsdir) == False:

    os.mkdir(resultsdir)
    os.mkdir(darkframedir)

    print('Creating Dark Frames for {}........................................'.format(starcode))
    makeVampDarks(darkdataPath, darkfilePref, nSubFiles, darkframedir, camera1dark_path, camera2dark_path)
    print('Extracting Visibilities using AMICAL for {}........................................'.format(starcode))

    for i in range(0, len(paths_AMICAL)):

        temp = paths_AMICAL[i]
        file_t = os.path.join(datadir_AMICAL, temp)
        print(temp)

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

            cube_t = amical.select_clean_data(file_t, clip=False, **clean_param, display=False,
                                              darkfile=camera1dark_path, sky=False)

            bs_t = amical.extract_bs(cube_t, file_t, targetname=starcode, **params_ami, display=False,
                                     compute_cp_cov=False, theta_detector=83., scaling_uv=0.94, fliplr=False)

        elif camera == 2:

            cube_t = amical.select_clean_data(file_t, clip=False, **clean_param, display=False,
                                              darkfile=camera2dark_path, sky=False)

            bs_t = amical.extract_bs(cube_t, file_t, targetname=starcode, **params_ami, display=False,
                                     compute_cp_cov=False, theta_detector=97, scaling_uv=0.94, fliplr=True)

        # construct path for the extracted file
        path = resultsdir + '/' + str(tempi) + elsetemp + '_bs_t'
        #
        # print(path)
        # print(bs_t)

        # save extracted file
        amical.save_bs_hdf5(bs_t, path) # cannot change this?


### Now we use the diff_cal_AMICAL_VAMPIRES library

print('Making a diff_cal object for {}........................................'.format(starcode))
AMICAL_obj = diff_cal_AMICAL_VAMPIRES(starcode, paths_AMICAL, datadir_AMICAL, codedir, resultsdir, 'AMICAL', bootstrap)

# compute differential polarisation, bootstrap argument will perform bootstrapping to produce the error terms which we need
AMICAL_obj.diff_pol_global(bootstrap)

# plot the result and save the image
AMICAL_obj.plot_bootstrap_diff_pol_cal(imagedir=imagedir)

# save copy of file used to run script
shutil.copyfile(codedir + '/Vampires_test.py', resultsdir + '/Vampires_test.py')