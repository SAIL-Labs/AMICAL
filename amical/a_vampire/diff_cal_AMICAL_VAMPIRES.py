import os
import amical
import matplotlib.pyplot as plt
from scipy import io
from matplotlib.colors import PowerNorm

plt.ion()
import numpy as np
import pylab as plt # cant find
from astropy.io import fits


import math # cant find
import os
import amical

import h5py as h5py
from astropy.io import fits
from matplotlib import pyplot as plt



class diff_cal_AMICAL_VAMPIRES:

    def __init__(self, starcode, paths, datadir, codedir, resultsdir,
                 type_extract, bootstrap):  # instance variable unique to each instances

        self.bootstrap = bootstrap
        self.starcode = starcode
        self.paths = paths
        self.datadir = datadir
        self.codedir = codedir
        self.resultsdir = resultsdir
        self.type_extract = type_extract

        self.qpol_vis2 = None
        self.qpol_vis2_e = None
        self.qpol_cp = None
        self.qpol_cp_e = None
        self.upol_vis2 = None
        self.upol_vis2_e = None
        self.upol_cp = None
        self.upol_cp_e = None
        self.e_wl = None

        self.hwp0_A1_mean = None
        self.hwp0_A2_mean = None
        self.hwp0_B1_mean = None
        self.hwp0_B2_mean = None
        self.hwp45_A1_mean = None
        self.hwp45_A2_mean = None
        self.hwp45_B1_mean = None
        self.hwp45_B2_mean = None
        self.hwp225_A1_mean = None
        self.hwp225_A2_mean = None
        self.hwp225_B1_mean = None
        self.hwp225_B2_mean = None
        self.hwp675_A1_mean = None
        self.hwp675_A2_mean = None
        self.hwp675_B1_mean = None
        self.hwp675_B2_mean = None

        if self.type_extract == 'AMICAL':

            self.numobs = 153  # 136 #153

            self.bootstrap_vis_q = np.zeros((self.bootstrap, self.numobs))
            self.bootstrap_vis_u = np.zeros((self.bootstrap, self.numobs))

            rawviscptemp = os.listdir(self.resultsdir + '/')
            self.rawviscp = [a for a in rawviscptemp if a.endswith('.h5') and not (a.startswith(
                '._'))]  # and a.startswith('0_') or a.startswith('1_') or a.startswith('2_') or a.startswith('3_')]
            #print(self.resultsdir)
            #print(self.rawviscp[0])
            examplefile_AMICAL = amical.load_bs_hdf5(self.resultsdir + '/' + self.rawviscp[0])

            self.bl_cp = examplefile_AMICAL.bl_cp
            self.u = examplefile_AMICAL.u
            self.v = examplefile_AMICAL.v  # check these are all universal - idl compute u and v around pixel directly - cannot place u point on pixel
            self.azimuth = np.arctan(self.v / self.u)
            self.bl = np.sqrt(self.u ** 2 + self.v ** 2)  # example.bl - this should be the same
            self.wl = examplefile_AMICAL.wl

            self.hwp0s = np.arange(0, len(self.rawviscp) / 4, 4).tolist()
            self.hwp225s = np.arange(1, len(self.rawviscp) / 4, 4).tolist()
            self.hwp45s = np.arange(2, len(self.rawviscp) / 4, 4).tolist()
            self.hwp675s = np.arange(3, len(self.rawviscp) / 4, 4).tolist()
            self.hwp0_A1 = np.zeros((1, self.numobs))  # MUST CUT OUT THE FIRST ROW.
            self.hwp0_A2 = np.zeros((1, self.numobs))
            self.hwp0_B1 = np.zeros((1, self.numobs))
            self.hwp0_B2 = np.zeros((1, self.numobs))

            self.hwp225_A1 = np.zeros((1, self.numobs))
            self.hwp225_A2 = np.zeros((1, self.numobs))
            self.hwp225_B1 = np.zeros((1, self.numobs))
            self.hwp225_B2 = np.zeros((1, self.numobs))

            self.hwp45_A1 = np.zeros((1, self.numobs))
            self.hwp45_A2 = np.zeros((1, self.numobs))
            self.hwp45_B1 = np.zeros((1, self.numobs))
            self.hwp45_B2 = np.zeros((1, self.numobs))

            self.hwp675_A1 = np.zeros((1, self.numobs))
            self.hwp675_A2 = np.zeros((1, self.numobs))
            self.hwp675_B1 = np.zeros((1, self.numobs))
            self.hwp675_B2 = np.zeros((1, self.numobs))




        elif self.type_extract == 'IDL':

            self.numobs = 153
            rawviscptemp = os.listdir(self.resultsdir + '/')
            self.rawviscp = [a for a in rawviscptemp if a.endswith('.idlvar') and a.startswith(
                'bs_')]  # and "00000" in a or "00001" in a or "00002" in a or "00003" in a]
            examplefile_IDL = io.readsav(self.resultsdir + '/' + self.rawviscp[2], python_dict=False, verbose=False)
            print(self.rawviscp[2])

            self.u = examplefile_IDL.u  # universal
            self.v = examplefile_IDL.v  # universal
            self.azimuth = np.arctan(self.v / self.u)  # universal   ### IMPLEMENT THIS arctan(v/u)
            self.bl = np.sqrt(self.u ** 2 + self.v ** 2)  # examplefile.bl     # universal
            self.wl = 7.5e-7
            self.bl = self.bl * self.wl

            self.hwp0s = np.arange(0, len(self.rawviscp) / 4, 4).tolist()
            self.hwp225s = np.arange(1, len(self.rawviscp) / 4, 4).tolist()
            self.hwp45s = np.arange(2, len(self.rawviscp) / 4, 4).tolist()
            self.hwp675s = np.arange(3, len(self.rawviscp) / 4, 4).tolist()
            self.hwp0_A1 = np.zeros((1, self.numobs))  # MUST CUT OUT THE FIRST ROW.
            self.hwp0_A2 = np.zeros((1, self.numobs))
            self.hwp0_B1 = np.zeros((1, self.numobs))
            self.hwp0_B2 = np.zeros((1, self.numobs))

            self.hwp225_A1 = np.zeros((1, self.numobs))
            self.hwp225_A2 = np.zeros((1, self.numobs))
            self.hwp225_B1 = np.zeros((1, self.numobs))
            self.hwp225_B2 = np.zeros((1, self.numobs))

            self.hwp45_A1 = np.zeros((1, self.numobs))
            self.hwp45_A2 = np.zeros((1, self.numobs))
            self.hwp45_B1 = np.zeros((1, self.numobs))
            self.hwp45_B2 = np.zeros((1, self.numobs))

            self.hwp675_A1 = np.zeros((1, self.numobs))
            self.hwp675_A2 = np.zeros((1, self.numobs))
            self.hwp675_B1 = np.zeros((1, self.numobs))
            self.hwp675_B2 = np.zeros((1, self.numobs))

    def diff_pol_global(self, bootstrap=0):

        #print('Running the diff_pol_global function')

        if self.type_extract == 'AMICAL':
            print('Computing Interferometric Visibilities.......')
            for i in range(len(self.rawviscp)):
                #print('Extracting file ' + str(i) + ' of ' + str(len(self.rawviscp)))
                filename = self.rawviscp[i]
                index = filename[-15:-12]
                indexint = int(index)
                filetest = amical.load_bs_hdf5(self.resultsdir + '/' + filename)
                visibilities = filetest['matrix'].v2_arr  # may not need to be transposed.
                testazimuth = np.arctan(filetest['v'] / filetest['u'])  # also need to check between amical and idl

                # this would be where you edit the visibilities

                # visibilities = visibilities - 0.5 #0.1*visibilities.max() #-  0.5#*visibilities.max();

                self.diff_cal_all(indexint, filename, visibilities)


        elif self.type_extract == 'IDL':
            print("IDL")
            # plt.figure()
            # plt.plot(self.u, self.v, 'x', label = 'original')
            # plt.plot(-self.u, -self.v, 'x', label = 'original')

            for i in range(len(self.rawviscp)):
                filename = self.rawviscp[i]
                index = filename[-16:-11]
                indexint = int(index)
                filetest = io.readsav(self.resultsdir + '/' + filename, python_dict=False, verbose=False)
                visibilities = filetest['v2_all'].T

                # visibilities = visibilities*self.wl
                # visibilities = visibilities/

                self.diff_cal_all(indexint, filename,
                                  visibilities)  # get this to return the visibilities just for each file?? how does this work...
                self.diff_cal_zero_negative_average()

        ##################################
        for bootnum in range(bootstrap):
            self.bootstrap_diffcal(bootnum)

        return

    def diff_cal_all(self, indexint, filename, visibilities):  ########## IN USE

        # print('Running the diff_cal_all function')

        if indexint in self.hwp0s:  # if this is a 0 hwp file
            if filename.endswith('_1_A.idlvar') or filename.endswith('_1_A_bs_t.h5'):  # save
                self.hwp0_A1 = np.vstack((self.hwp0_A1, visibilities))
            elif filename.endswith('_2_A.idlvar') or filename.endswith('_2_A_bs_t.h5'):
                self.hwp0_A2 = np.vstack((self.hwp0_A2, visibilities))
            elif filename.endswith('_1_B.idlvar') or filename.endswith('_1_B_bs_t.h5'):
                self.hwp0_B1 = np.vstack((self.hwp0_B1, visibilities))
            elif filename.endswith('_2_B.idlvar') or filename.endswith(
                    '_2_B_bs_t.h5'):  # need to add OR with the AMICAL ending
                self.hwp0_B2 = np.vstack((self.hwp0_B2, visibilities))

        elif indexint in self.hwp225s:  # if this is a 0 hwp file
            if filename.endswith('_1_A.idlvar') or filename.endswith('_1_A_bs_t.h5'):  # save
                self.hwp225_A1 = np.vstack((self.hwp225_A1, visibilities))
            elif filename.endswith('_2_A.idlvar') or filename.endswith('_2_A_bs_t.h5'):
                self.hwp225_A2 = np.vstack((self.hwp225_A2, visibilities))
            elif filename.endswith('_1_B.idlvar') or filename.endswith('_1_B_bs_t.h5'):
                self.hwp225_B1 = np.vstack((self.hwp225_B1, visibilities))
            elif filename.endswith('_2_B.idlvar') or filename.endswith('_2_B_bs_t.h5'):
                self.hwp225_B2 = np.vstack((self.hwp225_B2, visibilities))

        elif indexint in self.hwp45s:  # if this is a 0 hwp file
            if filename.endswith('_1_A.idlvar') or filename.endswith('_1_A_bs_t.h5'):  # save
                self.hwp45_A1 = np.vstack((self.hwp45_A1, visibilities))
            elif filename.endswith('_2_A.idlvar') or filename.endswith('_2_A_bs_t.h5'):
                self.hwp45_A2 = np.vstack((self.hwp45_A2, visibilities))
            elif filename.endswith('_1_B.idlvar') or filename.endswith('_1_B_bs_t.h5'):
                self.hwp45_B1 = np.vstack((self.hwp45_B1, visibilities))
            elif filename.endswith('_2_B.idlvar') or filename.endswith('_2_B_bs_t.h5'):
                self.hwp45_B2 = np.vstack((self.hwp45_B2, visibilities))

        elif indexint in self.hwp675s:  # if this is a 0 hwp file
            if filename.endswith('_1_A.idlvar') or filename.endswith('_1_A_bs_t.h5'):  # save
                self.hwp675_A1 = np.vstack((self.hwp675_A1, visibilities))
            elif filename.endswith('_2_A.idlvar') or filename.endswith('_2_A_bs_t.h5'):
                self.hwp675_A2 = np.vstack((self.hwp675_A2, visibilities))
            elif filename.endswith('_1_B.idlvar') or filename.endswith('_1_B_bs_t.h5'):
                self.hwp675_B1 = np.vstack((self.hwp675_B1, visibilities))
            elif filename.endswith('_2_B.idlvar') or filename.endswith('_2_B_bs_t.h5'):
                self.hwp675_B2 = np.vstack((self.hwp675_B2, visibilities))

        return

    def diff_cal_zero_negative_average(self):

        # this step MUST be done to remove a row of zeros.
        self.hwp0_A1 = self.hwp0_A1[1:, :]
        self.hwp0_A2 = self.hwp0_A2[1:, :]
        self.hwp0_B1 = self.hwp0_B1[1:, :]
        self.hwp0_B2 = self.hwp0_B2[1:, :]

        self.hwp225_A1 = self.hwp225_A1[1:, :]
        self.hwp225_A2 = self.hwp225_A2[1:, :]
        self.hwp225_B1 = self.hwp225_B1[1:, :]
        self.hwp225_B2 = self.hwp225_B2[1:, :]

        self.hwp45_A1 = self.hwp45_A1[1:, :]
        self.hwp45_A2 = self.hwp45_A2[1:, :]
        self.hwp45_B1 = self.hwp45_B1[1:, :]
        self.hwp45_B2 = self.hwp45_B2[1:, :]

        self.hwp675_A1 = self.hwp675_A1[1:, :]
        self.hwp675_A2 = self.hwp675_A2[1:, :]
        self.hwp675_B1 = self.hwp675_B1[1:, :]
        self.hwp675_B2 = self.hwp675_B2[1:, :]

        self.hwp0_A1[np.where(self.hwp0_A1 <= 0)] = 1e-19
        self.hwp0_A2[np.where(self.hwp0_A2 <= 0)] = 1e-19
        self.hwp0_B1[np.where(self.hwp0_B1 <= 0)] = 1e-19
        self.hwp0_B2[np.where(self.hwp0_B2 <= 0)] = 1e-19

        self.hwp225_A1[np.where(self.hwp225_A1 <= 0)] = 1e-19
        self.hwp225_A2[np.where(self.hwp225_A2 <= 0)] = 1e-19
        self.hwp225_B1[np.where(self.hwp225_B1 <= 0)] = 1e-19
        self.hwp225_B2[np.where(self.hwp225_B2 <= 0)] = 1e-19

        self.hwp45_A1[np.where(self.hwp45_A1 <= 0)] = 1e-19
        self.hwp45_A2[np.where(self.hwp45_A2 <= 0)] = 1e-19
        self.hwp45_B1[np.where(self.hwp45_B1 <= 0)] = 1e-19
        self.hwp45_B2[np.where(self.hwp45_B2 <= 0)] = 1e-19

        self.hwp675_A1[np.where(self.hwp675_A1 <= 0)] = 1e-19
        self.hwp675_A2[np.where(self.hwp675_A2 <= 0)] = 1e-19
        self.hwp675_B1[np.where(self.hwp675_B1 <= 0)] = 1e-19
        self.hwp675_B2[np.where(self.hwp675_B2 <= 0)] = 1e-19

        return

        ####################################################################################

    def bootstrap_diffcal(self, bootnum):

        randsamp_indx = np.random.choice(np.arange(len(self.hwp0_A1)), len(self.hwp0_A1))
        randsamp = self.hwp0_A1[randsamp_indx, :]
        self.hwp0_A1_mean = np.mean(randsamp, 0)  # mean over measurement

        randsamp_indx = np.random.choice(np.arange(len(self.hwp0_A2)), len(self.hwp0_A2))
        randsamp = self.hwp0_A2[randsamp_indx, :]
        self.hwp0_A2_mean = np.mean(randsamp, 0)

        randsamp_indx = np.random.choice(np.arange(len(self.hwp0_B1)), len(self.hwp0_B1))
        randsamp = self.hwp0_B1[randsamp_indx, :]
        self.hwp0_B1_mean = np.mean(randsamp, 0)

        randsamp_indx = np.random.choice(np.arange(len(self.hwp0_B2)), len(self.hwp0_B2))
        randsamp = self.hwp0_B2[randsamp_indx, :]
        self.hwp0_B2_mean = np.mean(randsamp, 0)

        #########

        randsamp_indx = np.random.choice(np.arange(len(self.hwp45_A1)), len(self.hwp45_A1))
        randsamp = self.hwp45_A1[randsamp_indx, :]
        self.hwp45_A1_mean = np.mean(randsamp, 0)

        randsamp_indx = np.random.choice(np.arange(len(self.hwp45_A2)), len(self.hwp45_A2))
        randsamp = self.hwp45_A2[randsamp_indx, :]
        self.hwp45_A2_mean = np.mean(randsamp, 0)

        randsamp_indx = np.random.choice(np.arange(len(self.hwp45_B1)), len(self.hwp45_B1))
        randsamp = self.hwp45_B1[randsamp_indx, :]
        self.hwp45_B1_mean = np.mean(randsamp, 0)

        randsamp_indx = np.random.choice(np.arange(len(self.hwp45_B2)), len(self.hwp45_B2))
        randsamp = self.hwp45_B2[randsamp_indx, :]
        self.hwp45_B2_mean = np.mean(randsamp, 0)

        ##############

        randsamp_indx = np.random.choice(np.arange(len(self.hwp225_A1)), len(self.hwp225_A1))
        randsamp = self.hwp225_A1[randsamp_indx, :]
        self.hwp225_A1_mean = np.mean(randsamp, 0)

        randsamp_indx = np.random.choice(np.arange(len(self.hwp225_A2)), len(self.hwp225_A2))
        randsamp = self.hwp225_A2[randsamp_indx, :]
        self.hwp225_A2_mean = np.mean(randsamp, 0)

        randsamp_indx = np.random.choice(np.arange(len(self.hwp225_B1)), len(self.hwp225_B1))
        randsamp = self.hwp225_B1[randsamp_indx, :]
        self.hwp225_B1_mean = np.mean(randsamp, 0)

        randsamp_indx = np.random.choice(np.arange(len(self.hwp225_B2)), len(self.hwp225_B2))
        randsamp = self.hwp225_B2[randsamp_indx, :]
        self.hwp225_B2_mean = np.mean(randsamp, 0)

        ###############

        randsamp_indx = np.random.choice(np.arange(len(self.hwp675_A1)), len(self.hwp675_A1))
        randsamp = self.hwp675_A1[randsamp_indx, :]
        self.hwp675_A1_mean = np.mean(randsamp, 0)

        randsamp_indx = np.random.choice(np.arange(len(self.hwp675_A2)), len(self.hwp675_A2))
        randsamp = self.hwp675_A2[randsamp_indx, :]
        self.hwp675_A2_mean = np.mean(randsamp, 0)

        randsamp_indx = np.random.choice(np.arange(len(self.hwp675_B1)), len(self.hwp675_B1))
        randsamp = self.hwp675_B1[randsamp_indx, :]
        self.hwp675_B1_mean = np.mean(randsamp, 0)

        randsamp_indx = np.random.choice(np.arange(len(self.hwp675_B2)), len(self.hwp675_B2))
        randsamp = self.hwp675_B2[randsamp_indx, :]
        self.hwp675_B2_mean = np.mean(randsamp, 0)

        #########################################################


        hwp0_1 = self.hwp0_A1_mean / self.hwp0_A2_mean #H/V
        hwp0_2 = self.hwp0_B1_mean / self.hwp0_B2_mean #V/H
        hwp0_3 = np.sqrt(hwp0_1 / hwp0_2) #H/V

        hwp45_1 = self.hwp45_A1_mean / self.hwp45_A2_mean # V/H
        hwp45_2 = self.hwp45_B1_mean / self.hwp45_B2_mean # H/V
        hwp45_3 = np.sqrt(hwp45_1 / hwp45_2) #V/H

        final_q = np.sqrt(hwp45_3 / hwp0_3) #H/V
        # but then we want actualy final quantities to be square root of that
        final_q = np.sqrt(final_q)


        hwp225_1 = self.hwp225_A1_mean / self.hwp225_A2_mean # V/H
        hwp225_2 = self.hwp225_B1_mean / self.hwp225_B2_mean # H/V
        hwp225_3 = np.sqrt(hwp225_1 / hwp225_2) # V/H

        hwp675_1 = self.hwp675_A1_mean / self.hwp675_A2_mean # H/V
        hwp675_2 = self.hwp675_B1_mean / self.hwp675_B2_mean # V/H
        hwp675_3 = np.sqrt(hwp675_1 / hwp675_2) # H/V

        final_u = np.sqrt(hwp675_3 / hwp225_3) # V/H
        # but then we want actual final quantities to be square root of that
        final_u = np.sqrt(final_u)

        self.qpol_vis2 = final_q
        self.upol_vis2 = final_u

        self.bootstrap_vis_q[bootnum, :] = final_q
        self.bootstrap_vis_u[bootnum, :] = final_u

        return

    def plot_bootstrap_diff_pol_cal(self, imagedir = ' '):

        fig, ax = plt.subplots(2, 1)
        fig.set_size_inches(15, 10)
        # fig.tight_layout()

        space = 100

        averageq = np.mean(self.bootstrap_vis_q, 0)
        errorq = np.std(self.bootstrap_vis_q, 0)

        averageu = np.mean(self.bootstrap_vis_u, 0)
        erroru = np.std(self.bootstrap_vis_u, 0)

        im1 = ax[0].scatter(self.azimuth, averageq, c=abs(self.bl), cmap='jet', marker='x',
                            label='Range = ' + str(round((np.max(averageq) - np.min(averageq)), 5)))
        fig.colorbar(im1, ax=ax[0])
        ax[0].set_title(
            'Differential Visibilities (Stokes Q polarization) vs Azimuth - ' + self.starcode + '_' + self.type_extract)
        ax[0].set_xlabel('Azimuth Angle')
        ax[0].set_ylabel('Differential Visibilities')
        ax[0].plot(self.azimuth, np.mean(averageq) * np.ones((len(self.azimuth))), linestyle='dashed', color='black',
                   label='Mean ' + str(round(np.mean(averageq), 5)))
        ax[0].plot(np.max(averageq), 'r^', label='Max = ' + str(round(np.max(averageq), 5)))
        ax[0].plot(np.min(averageq), 'ko', label='Min = ' + str(round(np.min(averageq), 5)))
        ax[0].legend(loc='upper right')
        ax[0].errorbar(self.azimuth, averageq, yerr=errorq, linestyle="None", ecolor='lightgrey')
        # ax[0].set_xlim([-1.5*np.pi, 1.5*np.pi])
        # ax[0].set_ylim(([ymin, ymax]))

        im2 = ax[1].scatter(self.azimuth, averageu, c=abs(self.bl), cmap='jet', marker='x',
                            label='Range = ' + str(np.round((np.max(averageu) - np.min(averageu)), 5)))
        fig.colorbar(im2, ax=ax[1])
        ax[1].set_title(
            'Differential Visibilities (Stokes U polarization) vs Azimuth - ' + self.starcode + '_' + self.type_extract)
        ax[1].set_xlabel('Azimuth Angle')
        ax[1].set_ylabel('Differential Visibilities')
        ax[1].plot(self.azimuth, np.mean(averageu) * np.ones((len(self.azimuth))), linestyle='dashed', color='black',
                   label='Mean ' + str(round(np.mean(averageu), 5)))
        ax[1].plot(np.max(averageu), 'r^', label='Max = ' + str(round(np.max(averageu), 5)))
        ax[1].plot(np.min(averageu), 'ko', label='Min = ' + str(round(np.min(averageu), 5)))
        ax[1].legend(loc='upper right')  #
        ax[1].errorbar(self.azimuth, averageu, yerr=erroru, linestyle="None", ecolor='lightgrey')

        plt.show()
        plt.savefig(imagedir + '/' + self.starcode + '_' + '_diff_vis_bootstrap.pdf')
        print("saved image")


    def plot_diff_pol_cal(self, offsetq=0, offsetu=0, condition=' ',
                          imagedir='/import/tintagel3/snert/lucinda/AMICAL_VAMPIRES/Results/results_2', extranotes=''):

        fig, ax = plt.subplots(2, 1)
        fig.set_size_inches(15, 10)
        # fig.tight_layout()

        space = 100

        im1 = ax[0].scatter(self.azimuth, self.qpol_vis2, c=abs(self.bl), cmap='jet', marker='x',
                            label='Range = ' + str(round((np.max(self.qpol_vis2) - np.min(self.qpol_vis2)), 5)))
        fig.colorbar(im1, ax=ax[0])
        ax[0].set_title(
            'Differential Visibilities (Stokes Q polarization) vs Azimuth - ' + self.starcode + '_' + self.type_extract + condition)
        ax[0].set_xlabel('Azimuth Angle')
        ax[0].set_ylabel('Differential Visibilities')
        ax[0].plot(self.azimuth, np.mean(self.qpol_vis2) * np.ones((len(self.azimuth))), linestyle='dashed',
                   color='black', label='Mean ' + str(round(np.mean(self.qpol_vis2), 5)))
        ax[0].plot(np.max(self.qpol_vis2), 'r^', label='Max = ' + str(round(np.max(self.qpol_vis2), 5)))
        ax[0].plot(np.min(self.qpol_vis2), 'ko', label='Min = ' + str(round(np.min(self.qpol_vis2), 5)))
        ax[0].legend(loc='upper right')
        # ax[0, 0].errorbar(self.azimuth, self.qpol_vis2, yerr=self.qpol_vis2_e, linestyle="None", ecolor = 'lightgrey')
        # ax[0].set_xlim([-1.5*np.pi, 1.5*np.pi])
        # ax[0].set_ylim(([ymin, ymax]))

        im2 = ax[1].scatter(self.azimuth, self.upol_vis2, c=abs(self.bl), cmap='jet', marker='x',
                            label='Range = ' + str(np.round((np.max(self.upol_vis2) - np.min(self.upol_vis2)), 5)))
        fig.colorbar(im2, ax=ax[1])
        ax[1].set_title(
            'Differential Visibilities (Stokes U polarization) vs Azimuth - ' + self.starcode + '_' + self.type_extract + condition)
        ax[1].set_xlabel('Azimuth Angle')
        ax[1].set_ylabel('Differential Visibilities')
        ax[1].plot(self.azimuth, np.mean(self.upol_vis2) * np.ones((len(self.azimuth))), linestyle='dashed',
                   color='black', label='Mean ' + str(round(np.mean(self.upol_vis2), 5)))
        ax[1].plot(np.max(self.upol_vis2), 'r^', label='Max = ' + str(round(np.max(self.upol_vis2), 5)))
        ax[1].plot(np.min(self.upol_vis2), 'ko', label='Min = ' + str(round(np.min(self.upol_vis2), 5)))
        ax[1].legend(loc='upper right')  #

        # ax[1, 0].errorbar(self.azimuth, self.upol_vis2, yerr=self.upol_vis2_e, linestyle="None", ecolor = 'lightgrey')
        # ax[1].set_xlim([-1.5*np.pi, 1.5*np.pi])
        # ax[1].set_ylim([ymin, ymax])

        plt.savefig(imagedir + self.starcode + '_' + condition + extranotes + '_diff_vis.pdf')
        print("saved image")

        # im3 = ax[0, 1].hist(self.qpol_cp, bins=50)
        # ax[0, 1].set_title('Differential Closure Phases (Stokes Q Polarization)')
        # ax[0, 1].set_xlabel('Differential Closure Phases')
        # ax[0, 1].set_ylabel('Quantity')
        # ax[0, 1].set_xlim([-80, 80])

        # im4 = ax[1, 1].hist(self.upol_cp, bins=50)
        # ax[1, 1].set_title('Differential Closure Phases (Stokes U Polarization)')
        # ax[1, 1].set_xlabel('Differential Closure Phases')
        # ax[1, 1].set_ylabel('Quantity')
        # ax[1, 1].set_xlim([-80, 80])

    def demonstrate_calibration(self, filenumber1, filenumber2):

        rawviscp = os.listdir(self.resultsdir + '/')
        res_1 = [k for k in rawviscp if k.startswith(str(filenumber1) + '_') and k.endswith('.h5')]

        A1_file1 = [a for a in res_1 if a.endswith('_1_A_bs_t.h5')]
        A1_file1 = amical.load_bs_hdf5(self.resultsdir + '/' + A1_file1[0])

        A2_file1 = [a for a in res_1 if a.endswith('_2_A_bs_t.h5')]
        A2_file1 = amical.load_bs_hdf5(self.resultsdir + '/' + A2_file1[0])

        B1_file1 = [a for a in res_1 if a.endswith('_1_B_bs_t.h5')]
        B1_file1 = amical.load_bs_hdf5(self.resultsdir + '/' + B1_file1[0])

        B2_file1 = [a for a in res_1 if a.endswith('_2_B_bs_t.h5')]
        B2_file1 = amical.load_bs_hdf5(self.resultsdir + '/' + B2_file1[0])

        # WOLLASTON PRISM ONLY

        FLConly = A1_file1['vis2'] / B1_file1['vis2']
        wollastononly = A1_file1['vis2'] / A2_file1['vis2']
        FLCandwoll = FLConly / wollastononly

        res_2 = [k for k in rawviscp if k.startswith(str(filenumber2) + '_') and k.endswith('.h5')]

        A1_file2 = [a for a in res_2 if a.endswith('_1_A_bs_t.h5')]
        A1_file2 = amical.load_bs_hdf5(self.resultsdir + '/' + A1_file2[0])

        A2_file2 = [a for a in res_2 if a.endswith('_2_A_bs_t.h5')]
        A2_file2 = amical.load_bs_hdf5(self.resultsdir + '/' + A2_file2[0])

        B1_file2 = [a for a in res_2 if a.endswith('_1_B_bs_t.h5')]
        B1_file2 = amical.load_bs_hdf5(self.resultsdir + '/' + B1_file2[0])

        B2_file2 = [a for a in res_2 if a.endswith('_2_B_bs_t.h5')]
        B2_file2 = amical.load_bs_hdf5(self.resultsdir + '/' + B2_file2[0])

        vis2_file2, vis2_e_file2 = cal_vis(A1_file2, A2_file2, B1_file2, B2_file2)

        allcall = FLCandwoll / vis2_file2

        fig, ax = plt.subplots(4, 1)
        fig.tight_layout()

        im1 = ax[0].scatter(self.azimuth, wollastononly, c=abs(self.bl), cmap='jet')
        fig.colorbar(im1, ax=ax[0])
        ax[0].set_title('Wollaston Prism only')
        ax[0].set_ylabel('Differential Visibilities')
        # ax[0, 0].errorbar(self.azimuth, self.qpol_vis2, yerr=self.qpol_vis2_e, linestyle="None", ecolor = 'lightgrey')
        ax[0].set_xlabel('Azimuth Angle')

        im2 = ax[1].scatter(self.azimuth, FLConly, c=abs(self.bl), cmap='jet')
        fig.colorbar(im2, ax=ax[1])
        ax[1].set_title('FLC only')
        ax[1].set_ylabel('Differential Visibilities')
        # ax[0, 0].errorbar(self.azimuth, self.qpol_vis2, yerr=self.qpol_vis2_e, linestyle="None", ecolor = 'lightgrey')
        ax[1].set_xlabel('Azimuth Angle')

        im3 = ax[2].scatter(self.azimuth, FLCandwoll, c=abs(self.bl), cmap='jet')
        fig.colorbar(im3, ax=ax[2])
        ax[2].set_title('FLC and Wollaston')
        ax[2].set_ylabel('Differential Visibilities')
        # ax[0, 0].errorbar(self.azimuth, self.qpol_vis2, yerr=self.qpol_vis2_e, linestyle="None", ecolor = 'lightgrey')
        ax[2].set_xlabel('Azimuth Angle')

        im4 = ax[3].scatter(self.azimuth, allcall, c=abs(self.bl), cmap='jet')
        fig.colorbar(im3, ax=ax[3])
        ax[3].set_title('FLC and Wollaston and HWP')
        ax[3].set_ylabel('Differential Visibilities')
        # ax[0, 0].errorbar(self.azimuth, self.qpol_vis2, yerr=self.qpol_vis2_e, linestyle="None", ecolor = 'lightgrey')
        ax[3].set_xlabel('Azimuth Angle')

        return

    def errors_sum(errora, errorb):
        ## assumes files are independent and have different sources of error
        abserrorc = np.sqrt((errora) ** 2 + (errorb) ** 2)

        return abserrorc

    def calc_errors(ob1, ob2, property):

        if property == 'vis2':
            ## assumes files are independent and have different sources of error
            abserror = np.sqrt(
                (ob1['e_' + property] / ob1[property]) ** 2 + (ob2['e_' + property] / ob2[property]) ** 2) * (
                                   ob1[property] / ob2[property])

        if property == 'cp':
            abserror = ob1['e_' + property] + ob2['e_' + property]

        return abserror

    def perform_diff_cal(self, indx1, files1, indx2, files2):

        print(indx1)

        if indx1 % 2 == 0 and indx2 % 2 == 0:

            remainder = 0
            HWP1 = '0'
            HWP2 = '45'

        elif indx1 % 2 != 0 and indx2 % 2 != 0:

            remainder = 1
            HWP1 = '22.5'
            HWP2 = '67.5'

        else:
            print('ERROR - the two files have incompatible HWP')

        res_1 = [k for k in files1 if k.startswith(str(indx1) + '_')]  # redundant

        A1_file1 = [a for a in res_1 if a.endswith('_1_A_bs_t.h5')]
        A1_file1 = amical.load_bs_hdf5(self.resultsdir + '/' + A1_file1[0])

        if A1_file1['infos'].hdr['HWP'] != HWP1 and A1_file1['infos'].hdr['HWP'] != HWP2:
            print('ERROR - WRONG HALF WAVE PLATE DETECTED')
            print('Should be HWP: ' + str(HWP1) + ' ' + str(HWP2))
            print('HWP is ' + str(A1_file1['infos'].hdr['HWP']) + ' and ' + str(A1_file1['infos'].hdr['HWP']))

        A2_file1 = [a for a in res_1 if a.endswith('_2_A_bs_t.h5')]
        A2_file1 = amical.load_bs_hdf5(self.resultsdir + '/' + A2_file1[0])
        if A2_file1['infos'].hdr['HWP'] != HWP1 and A2_file1['infos'].hdr['HWP'] != HWP2:
            print('ERROR - WRONG HALF WAVE PLATE DETECTED')
            print('Should be HWP: ' + str(HWP1) + ' ' + str(HWP2))
            print('HWP is ' + str(A2_file1['infos'].hdr['HWP']) + ' and ' + str(A2_file1['infos'].hdr['HWP']))

        B1_file1 = [a for a in res_1 if a.endswith('_1_B_bs_t.h5')]
        B1_file1 = amical.load_bs_hdf5(self.resultsdir + '/' + B1_file1[0])
        if B1_file1['infos'].hdr['HWP'] != HWP1 and B1_file1['infos'].hdr['HWP'] != HWP2:
            print('ERROR - WRONG HALF WAVE PLATE DETECTED')
            print('Should be HWP: ' + str(HWP1) + ' ' + str(HWP2))
            print('HWP is ' + str(B1_file1['infos'].hdr['HWP']) + ' and ' + str(B1_file1['infos'].hdr['HWP']))

        B2_file1 = [a for a in res_1 if a.endswith('_2_B_bs_t.h5')]
        B2_file1 = amical.load_bs_hdf5(self.resultsdir + '/' + B2_file1[0])
        if B2_file1['infos'].hdr['HWP'] != HWP1 and B2_file1['infos'].hdr['HWP'] != HWP2:
            print('ERROR - WRONG HALF WAVE PLATE DETECTED')
            print('Should be HWP: ' + str(HWP1) + ' ' + str(HWP2))
            print('HWP is ' + str(B2_file1['infos'].hdr['HWP']) + ' and ' + str(B2_file1['infos'].hdr['HWP']))

        vis2_file1, vis2_e_file1 = cal_vis(A1_file1, A2_file1, B1_file1, B2_file1)
        cp_file1, cp_e_file1 = cal_cp(A1_file1, A2_file1, B1_file1, B2_file1)

        ######################################################################
        ######################################################################

        res_2 = [k for k in files2 if k.startswith(str(indx2) + '_')]

        A1_file2 = [a for a in res_2 if a.endswith('_1_A_bs_t.h5')]
        A1_file2 = amical.load_bs_hdf5(self.resultsdir + '/' + A1_file2[0])
        if A1_file2['infos'].hdr['HWP'] != HWP1 and A1_file2['infos'].hdr['HWP'] != HWP2:
            print('ERROR - WRONG HALF WAVE PLATE DETECTED')
            print('Should be HWP: ' + str(HWP1) + ' ' + str(HWP2))
            print('HWP is ' + str(A1_file2['infos'].hdr['HWP']) + ' and ' + str(A1_file2['infos'].hdr['HWP']))

        A2_file2 = [a for a in res_2 if a.endswith('_2_A_bs_t.h5')]
        A2_file2 = amical.load_bs_hdf5(self.resultsdir + '/' + A2_file2[0])
        if A2_file2['infos'].hdr['HWP'] != HWP1 and A2_file2['infos'].hdr['HWP'] != HWP2:
            print('ERROR - WRONG HALF WAVE PLATE DETECTED')
            print('Should be HWP: ' + str(HWP1) + ' ' + str(HWP2))
            print('HWP is ' + str(A2_file2['infos'].hdr['HWP']) + ' and ' + str(A2_file2['infos'].hdr['HWP']))

        B1_file2 = [a for a in res_2 if a.endswith('_1_B_bs_t.h5')]
        B1_file2 = amical.load_bs_hdf5(self.resultsdir + '/' + B1_file2[0])
        if B1_file2['infos'].hdr['HWP'] != HWP1 and B1_file2['infos'].hdr['HWP'] != HWP2:
            print('ERROR - WRONG HALF WAVE PLATE DETECTED')
            print('Should be HWP: ' + str(HWP1) + ' ' + str(HWP2))
            print('HWP is ' + str(B1_file2['infos'].hdr['HWP']) + ' and ' + str(B1_file2['infos'].hdr['HWP']))

        B2_file2 = [a for a in res_2 if a.endswith('_2_B_bs_t.h5')]
        B2_file2 = amical.load_bs_hdf5(self.resultsdir + '/' + B2_file2[0])
        if B2_file2['infos'].hdr['HWP'] != HWP1 and B2_file2['infos'].hdr['HWP'] != HWP2:
            print('ERROR - WRONG HALF WAVE PLATE DETECTED')
            print('Should be HWP: ' + str(HWP1) + ' ' + str(HWP2))
            print('HWP is ' + str(B2_file2['infos'].hdr['HWP']) + ' and ' + str(B2_file2['infos'].hdr['HWP']))

        vis2_file2, vis2_e_file2 = cal_vis(A1_file2, A2_file2, B1_file2, B2_file2)
        cp_file2, cp_e_file2 = cal_cp(A1_file2, A2_file2, B1_file2, B2_file2)

        finalvis2 = np.sqrt(vis2_file1 / vis2_file2)
        finalvis2_e = np.sqrt((vis2_e_file1 / vis2_file1) ** 2 + (vis2_e_file2 / vis2_file2) ** 2) * finalvis2
        finalcp = cp_file1 - cp_file2
        finalcp_e = np.sqrt((cp_e_file1) ** 2 + (cp_e_file2) ** 2)

        return finalvis2, finalvis2_e, finalcp, finalcp_e

    def cal_vis(A1, A2, B1, B2):

        div1 = A1['vis2'] / A2['vis2']
        div1_e = calc_errors(A1, A2, 'vis2')

        div2 = B1['vis2'] / B2['vis2']
        div2_e = calc_errors(B1, B2, 'vis2')

        div3 = div1 / div2
        div3_e = np.sqrt((div1_e / div1) ** 2 + (div2_e / div2) ** 2) * div3

        return div3, div3_e

    def cal_cp(A1, A2, B1, B2):

        sub1 = A1['cp'] - A2['cp']
        sub1_e = calc_errors(A1, A2, 'cp')

        sub2 = B1['cp'] - B2['cp']
        sub2_e = calc_errors(B1, B2, 'cp')

        sub3 = sub1 - sub2
        sub3_e = np.sqrt((sub2_e) ** 2 + (sub1_e) ** 2)

        return sub3, sub3_e

    def diff_cal_all(self, indexint, filename, visibilities):  ########## IN USE

        if indexint in self.hwp0s:  # if this is a 0 hwp file
            if filename.endswith('_1_A.idlvar') or filename.endswith('_1_A_bs_t.h5'):  # save
                self.hwp0_A1 = np.vstack((self.hwp0_A1, visibilities))
            elif filename.endswith('_2_A.idlvar') or filename.endswith('_2_A_bs_t.h5'):
                self.hwp0_A2 = np.vstack((self.hwp0_A2, visibilities))
            elif filename.endswith('_1_B.idlvar') or filename.endswith('_1_B_bs_t.h5'):
                self.hwp0_B1 = np.vstack((self.hwp0_B1, visibilities))
            elif filename.endswith('_2_B.idlvar') or filename.endswith(
                    '_2_B_bs_t.h5'):  # need to add OR with the AMICAL ending
                self.hwp0_B2 = np.vstack((self.hwp0_B2, visibilities))

        elif indexint in self.hwp225s:  # if this is a 0 hwp file
            if filename.endswith('_1_A.idlvar') or filename.endswith('_1_A_bs_t.h5'):  # save
                self.hwp225_A1 = np.vstack((self.hwp225_A1, visibilities))
            elif filename.endswith('_2_A.idlvar') or filename.endswith('_2_A_bs_t.h5'):
                self.hwp225_A2 = np.vstack((self.hwp225_A2, visibilities))
            elif filename.endswith('_1_B.idlvar') or filename.endswith('_1_B_bs_t.h5'):
                self.hwp225_B1 = np.vstack((self.hwp225_B1, visibilities))
            elif filename.endswith('_2_B.idlvar') or filename.endswith('_2_B_bs_t.h5'):
                self.hwp225_B2 = np.vstack((self.hwp225_B2, visibilities))

        elif indexint in self.hwp45s:  # if this is a 0 hwp file
            if filename.endswith('_1_A.idlvar') or filename.endswith('_1_A_bs_t.h5'):  # save
                self.hwp45_A1 = np.vstack((self.hwp45_A1, visibilities))
            elif filename.endswith('_2_A.idlvar') or filename.endswith('_2_A_bs_t.h5'):
                self.hwp45_A2 = np.vstack((self.hwp45_A2, visibilities))
            elif filename.endswith('_1_B.idlvar') or filename.endswith('_1_B_bs_t.h5'):
                self.hwp45_B1 = np.vstack((self.hwp45_B1, visibilities))
            elif filename.endswith('_2_B.idlvar') or filename.endswith('_2_B_bs_t.h5'):
                self.hwp45_B2 = np.vstack((self.hwp45_B2, visibilities))

        elif indexint in self.hwp675s:  # if this is a 0 hwp file
            if filename.endswith('_1_A.idlvar') or filename.endswith('_1_A_bs_t.h5'):  # save
                self.hwp675_A1 = np.vstack((self.hwp675_A1, visibilities))
            elif filename.endswith('_2_A.idlvar') or filename.endswith('_2_A_bs_t.h5'):
                self.hwp675_A2 = np.vstack((self.hwp675_A2, visibilities))
            elif filename.endswith('_1_B.idlvar') or filename.endswith('_1_B_bs_t.h5'):
                self.hwp675_B1 = np.vstack((self.hwp675_B1, visibilities))
            elif filename.endswith('_2_B.idlvar') or filename.endswith('_2_B_bs_t.h5'):
                self.hwp675_B2 = np.vstack((self.hwp675_B2, visibilities))

        return


