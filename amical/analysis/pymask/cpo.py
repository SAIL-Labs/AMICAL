import glob
import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.io.idl import readsav
from termcolor import cprint

# import pymask.oifits
from . import oifits

from .cp_tools import project_cps


'''------------------------------------------------------------------------
cpo.py - Python class for manipulating oifits format closure phase data.
------------------------------------------------------------------------'''


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    green = "\x1b[32;5m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# create logger with 'spam_application'
log = logging.getLogger("PYMASK")
log.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

log.addHandler(ch)
# log.debug("A quirky message only developers care about")
# log.info("Curious users might want to know this")
# log.warn("Something is wrong and any user should be informed")
# log.error("Serious stuff, this is red for a reason")
# log.critical("OH NO everything is on fire")


class cpo():
    ''' Class used to manipulate multiple closure phase datasets'''

    def __init__(self, oifits):
        # Default instantiation.
        if type(oifits) == str:
            oifits = [oifits]
        if isinstance(oifits, str):
            self.extract_from_oifits(oifits)
        elif isinstance(oifits, list):
            self.extract_multi_oifits(oifits)
        else:
            log.error('inputs must be an oifits file or a list of oifits files.')

    def extract_from_oifits(self, filename):
        '''Extract closure phase data from an oifits file.'''

        data = oifits.open(filename)
        self.name = ''

        self.ndata = len(data.t3)

        for j in data.wavelength:
            wavel = data.wavelength[j].eff_wave
            self.wavel = wavel

        self.nwavs = len(self.wavel)

        self.target = data.target[0].target

        t3data = []
        t3err = []
        self.u = np.zeros((self.ndata, 3))
        self.v = np.zeros((self.ndata, 3))

        for j, t3 in enumerate(data.t3):
            t3data.append(t3.t3phi)
            t3err.append(t3.t3phierr)
            self.u[j, :] = [t3.u1coord, t3.u2coord, -(t3.u1coord+t3.u2coord)]
            self.v[j, :] = [t3.v1coord, t3.v2coord, -(t3.v1coord+t3.v2coord)]

        self.t3data = np.array(t3data)
        self.t3err = np.array(t3err)

        # Also load the v2
        nv2 = len(data.vis2)
        vis2data = []
        vis2err = []
        self.v2_u = np.zeros((nv2))
        self.v2_v = np.zeros((nv2))
        for j, v2 in enumerate(data.vis2):
            vis2data.append(v2.vis2data)
            vis2err.append(v2.vis2err)
            self.v2_u[j] = v2.ucoord
            self.v2_v[j] = v2.vcoord
        self.vis2data = np.array(vis2data)
        self.vis2err = np.array(vis2err)

    def extract_multi_oifits(self, lfiles, verbose=True):
        '''Extract closure phase data from a list of oifits files.'''
        U, V, t3data, t3err = [], [], [], []
        for f in lfiles:
            self.extract_from_oifits(f)
            U.extend(self.u)
            V.extend(self.v)
            t3data.extend(self.t3data)
            t3err.extend(self.t3err)
            nbl = len(self.u)
        self.u = np.array(U)
        self.v = np.array(V)
        self.t3data = np.array(t3data)
        self.t3err = np.array(t3err)
        if verbose:
            cprint(r'PYMASK - %i oifits loaded : n=%ix%i=%i closure phases.' %
                   (len(lfiles), len(lfiles), nbl, len(t3data)), 'green')


class icpo():
    ''' Class used to manipulate multiple closure phase datasets.
    This one structures the data differently, so it can be used with a covariance
    matrix, or with projected closure phases'''

    def __init__(self, oifits=None, directory=None, tsize_targ=None, tsize_cal=None):
        # Default instantiation.

        # if the file is a complete (kpi + kpd) structure
        # additional data can be loaded.
        if oifits:
            try:
                self.extract_from_oifits(oifits)
            except Exception:
                print('Invalid file.')
        else:
            # Otherwise load directory from the IDL bs files
            self.extract_from_idl_directory(directory, tsize_targ=tsize_targ,
                                            tsize_cal=tsize_cal)

    def extract_from_oifits(self, filename):
        '''Extract closure phase data from an oifits file.'''

        data = oifits.open(filename)
        self.name = ''

        self.ndata = len(data.t3)

        for j in data.wavelength:
            wavel = data.wavelength[j].eff_wave
            self.wavel = wavel
            break
        self.nwavs = len(self.wavel)

        self.target = data.target[0].target

        t3data = []
        t3err = []
        self.u = np.zeros((self.ndata, 3))
        self.v = np.zeros((self.ndata, 3))

        for j, t3 in enumerate(data.t3):
            t3data.append(t3.t3phi)
            t3err.append(t3.t3phierr)
            self.u[j, :] = [t3.u1coord, t3.u2coord, -(t3.u1coord+t3.u2coord)]
            self.v[j, :] = [t3.v1coord, t3.v2coord, -(t3.v1coord+t3.v2coord)]

        self.t3data = np.array(t3data)
        self.t3err = np.array(t3err)

        # Also load the v2
        nv2 = len(data.vis2)
        vis2data = []
        vis2err = []
        self.v2_u = np.zeros((nv2))
        self.v2_v = np.zeros((nv2))
        for j, v2 in enumerate(data.vis2):
            vis2data.append(v2.vis2data)
            vis2err.append(v2.vis2err)
            self.v2_u[j] = v2.ucoord
            self.v2_v[j] = v2.vcoord
        self.vis2data = np.array(vis2data)
        self.vis2err = np.array(vis2err)

        print('  This probably doesnt work...')

    #########

    def extract_from_idl_directory(self, analysis_dir, tsize_targ=None, tsize_cal=None,
                                   idl_masking_dir='/Users/cheetham/code/masking/'):
        ''' Import the closure phase data from the IDL pipeline directly
        from the bispectrum (bs) files'''

        # Get all of the bispectrum files
        bs_files = glob.glob(analysis_dir+'bs*.idlvar')
        bs_files.sort()

        # Load the cubeinfo file
        cubeinfo_file = glob.glob(analysis_dir+'cubeinfo*.idlvar')
        cubeinfo = readsav(cubeinfo_file[0])

        tsize = cubeinfo['olog']['tsize'][0]
        uflip = cubeinfo['olog']['uflip'][0]
        pa = cubeinfo['olog']['pa'][0]
        del_pa = cubeinfo['olog']['del_pa'][0]

        # Check that tsize is the right size (i.e. that we found all the files)
        assert len(bs_files) == len(tsize)

        # Find which ones are the targets and calibrators
        # Default is that targets are all objects with tsize < 0
        if tsize_targ is None:
            targ_ix = tsize < 0
        else:
            targ_ix = np.zeros(len(tsize), dtype=bool)
            for ix in tsize_targ:
                targ_ix[tsize == ix] = True

        # Default is that cals are all objects with tsize > 0
        if tsize_cal is None:
            cal_ix = tsize > 0
        else:
            cal_ix = np.zeros(len(tsize), dtype=bool)
            for ix in tsize_cal:
                cal_ix[tsize == ix] = True

        # Set up all the arrays
        # CPs
        n_data_per_obs = []
        cal_cps = []
        targ_cps = []
        print('Found {0:4d} bs files from IDL'.format(len(bs_files)))

        # Loop through and load them
        for ix, f in enumerate(bs_files):

            data = readsav(f)
            cps = np.angle(data['bs_all'], deg=True)
            mean_cp = data['cp']
            err_cp = data['cp_sig']
            # Add a wavelength dimension to non-IFU data
            if mean_cp.ndim == 1:
                cps = cps[np.newaxis, :]
                mean_cp = mean_cp[np.newaxis, :]
                err_cp = err_cp[np.newaxis, :]

            # Check if targ or cal, then add the cps there
            if cal_ix[ix]:
                if len(cal_cps) == 0:
                    cal_cps = cps
                    mean_cal_cps = mean_cp[:, :, np.newaxis]
                    std_cal_cps = err_cp[:, :, np.newaxis]
                else:
                    cal_cps = np.append(cal_cps, cps, axis=2)
                    mean_cal_cps = np.append(
                        mean_cal_cps, mean_cp[:, :, np.newaxis], axis=2)
                    std_cal_cps = np.append(
                        std_cal_cps, err_cp[:, :, np.newaxis], axis=2)
            elif targ_ix[ix]:

                if len(targ_cps) == 0:
                    # We have lots of setup to do for the first file, and then we'll reuse it for the rest
                    # Load the matched filter file since it contains some useful info
                    mf_file = idl_masking_dir+'templates/'+data['mf_file']
                    mf_data = readsav(mf_file)

                    # Work out the wavelengths now
                    # u_ideal is the baseline divided by lambda so we can get them back from the first baseline like this
                    xy_coords = mf_data['xy_coords']
                    bl2h_ix = mf_data['bl2h_ix']
                    if 'u_ideal' in mf_data.keys():  # For the closing triangle approach
                        wavs = ((float(
                            xy_coords[0, bl2h_ix[0, 0]]) - xy_coords[0, bl2h_ix[0, 1]])/mf_data['u_ideal'][..., 0])
                    else:  # For the matched filter approach
                        wavs = ((float(
                            xy_coords[0, bl2h_ix[0, 0]]) - xy_coords[0, bl2h_ix[0, 1]])/mf_data['u'][..., 0])

                    wavs = np.atleast_1d(wavs)

                    # Get the uv coords from the matched filter file
                    # This should be the same at all wavelengths
                    if data['u'].ndim == 1:
                        # in metres
                        u_coords = data['u'][mf_data['bs2bl_ix']]*wavs[0]
                        # in metres
                        v_coords = data['v'][mf_data['bs2bl_ix']]*wavs[0]
                    else:
                        # in metres
                        u_coords = data['u'][0, mf_data['bs2bl_ix']]*wavs[0]
                        # in metres
                        v_coords = data['v'][0, mf_data['bs2bl_ix']]*wavs[0]

                # Rotate the uv coords
                pa_rad = (pa[ix] - 0.5*del_pa[ix])*np.pi/180.
                u_coords1 = uflip*u_coords * \
                    np.cos(pa_rad) + v_coords*np.sin(pa_rad)
                v_coords1 = -uflip*u_coords * \
                    np.sin(pa_rad) + v_coords*np.cos(pa_rad)

                if len(targ_cps) == 0:
                    # keep this a list to make it easier to separate them
                    targ_cps = [cps.transpose((1, 2, 0))]
                    mean_targ_cps = mean_cp[:, :, np.newaxis]
                    std_targ_cps = err_cp[:, :, np.newaxis]
                    targ_u_coords = u_coords1[:, np.newaxis, :]
                    targ_v_coords = v_coords1[:, np.newaxis, :]
                else:
                    #                    targ_cps = np.append(targ_cps,cps,axis=2)
                    targ_cps.append(cps.transpose((1, 2, 0)))
                    mean_targ_cps = np.append(
                        mean_targ_cps, mean_cp[:, :, np.newaxis], axis=2)
                    std_targ_cps = np.append(
                        std_targ_cps, err_cp[:, :, np.newaxis], axis=2)
                    targ_u_coords = np.append(
                        targ_u_coords, u_coords1[:, np.newaxis, :], axis=1)
                    targ_v_coords = np.append(
                        targ_v_coords, v_coords1[:, np.newaxis, :], axis=1)

                n_data_per_obs.append(cps.shape[2])

        # Convert to the right units
        mean_targ_cps *= 180./np.pi
        mean_cal_cps *= 180./np.pi
        std_targ_cps *= 180./np.pi
        std_cal_cps *= 180./np.pi

        cal_cov = np.zeros(
            (cal_cps.shape[0], cal_cps.shape[1], cal_cps.shape[1]))
        cal_cov_inv = 0*cal_cov
        for wav_ix, wav_clps in enumerate(cal_cps):
            # Get the covariance and its inverse
            wav_cov = np.cov(wav_clps)
            wav_cov_inv = np.linalg.inv(wav_cov)

            cal_cov[wav_ix] = wav_cov
            cal_cov_inv[wav_ix] = wav_cov_inv

        # Bug fix:
        # The uv coordinates above are all positive. The third baseline should be flipped
        targ_u_coords[:, :, 2] *= -1
        targ_v_coords[:, :, 2] *= -1

        # Save everything to the cpo object (if given)
        self.cal_cps = np.transpose(cal_cps, axes=(1, 2, 0))
#        self.targ_cps = np.transpose(targ_cps,axes=(1,2,0))
        self.targ_cps = targ_cps
        self.cal_cp_sig = np.transpose(std_cal_cps, axes=(1, 2, 0))
        self.targ_cp_sig = np.transpose(std_targ_cps, axes=(1, 2, 0))
        self.t3data = np.transpose(mean_targ_cps, axes=(1, 2, 0))
        self.t3err = np.transpose(std_targ_cps, axes=(1, 2, 0))
        self.u = targ_u_coords
        self.v = targ_v_coords
        self.wavel = wavs

        # ICPO specific info
        self.cal_cov = cal_cov
        self.cal_cov_inv = cal_cov_inv

        # Some useful info
        self.ncp = self.t3data.shape[0]
        self.nobs = self.t3data.shape[1]
        self.n_runs = self.nobs
        self.ndata = self.ncp * self.nobs
        self.nwavs = self.wavel.shape[0]
        self.name = ''

    def make_proj(self, cov_matrix, plot=True, n_remove=2, n_significant=None, tol=1e-2,
                  calibrate=True, silent=False):
        ''' Make a projection matrix and project the raw closure phases onto it,
        using the covariance matrix given.
        Options:
            cov_matrix: Should be the covariance matrix that you want to diagonalise
                    during the projection
            n_significant: The number of eigenvalues to take. Since n_clp > n_indep_clp,
                    some eigenvalues will be discarded. This allows a set
                    number to be taken, rather than relying on an automatic
                    detection using "tol" (see below)
            tol: the minimum value used to detect significant eigenvalues. This is
                    ignored if n_significant is set
        '''
        proj = []

        for wav_ix, wav in enumerate(self.wavel):
            if not silent:
                print(' Wavelength: '+str(wav_ix))
            #
            cov = cov_matrix[wav_ix, :, :]

            evals, evects = scipy.linalg.eigh(cov)

            # Put them in increasing order
            evects = evects[:, ::-1]
            evals = evals[::-1]

            if n_significant:
                good_evals = (np.arange(len(evals)) < n_significant)
                if not silent:
                    print('  Taking '+str(n_significant)+' eigenvalues')
            else:
                good_evals = evals > tol
                if not silent:
                    print('  Found '+str(np.sum(good_evals)) +
                          ' eigenvalues above tolerance')

            if plot:
                plt.clf()
                plt.plot(evals)
                plt.xlabel('Mode number')
                plt.ylabel('Eigenvalue')
                plt.yscale('log')
                xl = plt.xlim()
                plt.plot([xl[0], xl[1]], [tol, tol],
                         '--', label='Minimum eigenvalue')
                plt.xlim(xl)

            # Only (n-1)(n-2)/2 are independent
            # This is clearly shown in the evalues, with a clear break between 0-15 and >15
            evects = evects[:, good_evals]
            evals = evals[good_evals]

            # Throw away the ones that need calibrating
            evects = evects[:, n_remove:]
            evals = evals[n_remove:]
            if not silent:
                print('  Discarding first '+str(n_remove)+' modes')

#        for ix in range(evals.size):
#            evects[:,ix] /= np.sqrt(np.sum(evects[:,ix]**2)) # Make sure they have length 1 (i.e. normal)

            proj.append(evects)
        proj = np.array(proj)
        self.proj = proj
        # this assumes its the same number for all data...
        self.n_good = np.sum(good_evals) - n_remove

        print('  Projecting target data onto statistically indep. basis set')
        # Now project the data, starting with the raw data
        proj_t3data = np.zeros((self.n_good, self.nobs, self.nwavs))
        proj_t3err = np.zeros((self.n_good, self.nobs, self.nwavs))
        all_proj_cps = []
        for obs_ix in range(self.nobs):

            obs_cps = self.targ_cps[obs_ix]

            if calibrate:
                # Calibrate the target clps by subtracting the median calibrator cp
                systematic_cps = np.median(self.cal_cps, axis=1)
                obs_cps -= systematic_cps[:, np.newaxis, :]

            proj_cps = project_cps(obs_cps, proj)

            proj_t3data[:, obs_ix, :] = np.nanmean(proj_cps, axis=1)
            proj_t3err[:, obs_ix, :] = np.nanstd(
                proj_cps, axis=1) / np.sqrt(proj_cps.shape[1])  # SEM

            all_proj_cps.append(proj_cps)

        self.proj_t3data = proj_t3data
        self.proj_t3err = proj_t3err
        return all_proj_cps

# =========================================================================


class PolDiffObs():
    ''' Class used to manipulate polarimetric differential observables'''

    def __init__(self, filename):
        # Default instantiation.

        try:
            self.extract_from_idlvar(filename)
        except Exception:
            print('Invalid files.')

    def extract_from_idlvar(self, filename):
        '''Extract the differential visibility and closure phase data from
        the files output by the IDL masking pipeline.'''

        # Use scipy's idl reading function to get a dictionary of all the saved variables
        data = readsav(filename)

        # The differential visibility quantities
        self.u = data['u']
        self.v = data['v']
        self.vis = data['vis']
        self.vis_err = data['vis_err']

        # Closure phase quantities
        self.bs_u = data['bs_u']
        self.bs_v = data['bs_v']
        self.cp = data['cp']
        self.cp_err = data['cp_err']

        # Differential phase quantities
        self.ph = data['ph']
        self.ph_err = data['ph_err']

        # Generic quantities (that arent implemented yet)
#        self.wavel=data['wavel']
