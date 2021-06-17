# -*- coding: utf-8 -*-
"""
@author: Anthony Soulain (University of Sydney)

-------------------------------------------------------------------------
AMICAL: Aperture Masking Interferometry Calibration and Analysis Library
-------------------------------------------------------------------------

Set of functions to work with spectraly dispersed (IFU) NRM data.

-------------------------------------------------------------------------
"""

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from tqdm import tqdm

from .data_processing import select_clean_data
from .get_infos_obs import get_wavelength


def get_lambda(i_wl=None, filtname='YH', instrument='SPHERE-IFS'):
    """ Get spectral information for the given instrumental IFU setup.
    i_wl can be an integer or a list of 2 integers used to display the
    requested spectral channel."""
    wl = get_wavelength(instrument, filtname) * 1e6

    if np.isnan(wl.any()):
        return None

    print('\nInstrument: %s, spectral range: %s' % (instrument, filtname))
    print('-----------------------------')
    print('spectral coverage: %2.2f - %2.2f µm (step = %2.2f)' %
          (wl[0], wl[-1], np.diff(wl)[0]))

    one_wl = True
    if type(i_wl) is list:
        one_wl = False
        wl_range = wl[i_wl[0]:i_wl[1]]
        sp_range = np.arange(i_wl[0], i_wl[1], 1)
    elif i_wl is None:
        one_wl = False
        sp_range = np.arange(len(wl))
        wl_range = wl

    plt.figure(figsize=(4, 3))
    plt.title('--- SPECTRAL INFORMATION (IFU)---')
    plt.plot(wl, label='All spectral channels')
    if one_wl:
        plt.plot(np.arange(len(wl))[i_wl], wl[i_wl],
                 'ro', label='Selected (%2.2f µm)' % wl[i_wl])
    else:
        plt.plot(sp_range, wl_range, lw=5, alpha=.5,
                 label='Selected (%2.2f-%2.2f µm)' % (wl_range[0],
                                                      wl_range[-1]))
    plt.legend()
    plt.xlabel('Spectral channel')
    plt.ylabel('Wavelength [µm]')
    plt.tight_layout()

    if one_wl:
        output = np.round(wl[i_wl], 2)
    else:
        output = np.round(wl_range)
    return output


def clean_data(list_file, isz=256, r1=100, dr=10, edge=0,
               bad_map=None, add_bad=[], offx=0, offy=0,
               clip_fact=0.5, apod=True, sky=True, window=None,
               f_kernel=3, verbose=False, ihdu=0, display=False):
    """ Clean data using the standard procedure amical.select_clean_data()
    for each file in list_file. For IFU mode of SPHERE, the different frames
    are stored in different files and need to be reshaped into the appropriate
    4D datacube (i.e.: `cube_lambda.shape = [ndit, nlambda, isz,
    isz]`). Check amical.select_clean_data() for details about input parameters.
    """

    clean_param = {'isz': isz, 'r1': r1, 'dr': dr, 'edge': edge, 'clip': False,
                   'bad_map': bad_map, 'add_bad': add_bad, 'offx': offx, 'offy': offy,
                   'clip_fact': clip_fact, 'apod': apod, 'sky': sky, 'window': window,
                   'f_kernel': f_kernel, 'verbose': verbose, 'ihdu': ihdu,
                   'display': display
                   }

    hdu = fits.open(list_file[0])
    hdr = hdu[0].header

    nlambda = hdr['NAXIS3']
    nframe = len(list_file)

    cube_lambda = np.zeros([nframe, nlambda, isz, isz])

    for i in tqdm(range(len(list_file)), desc='Format/clean IFU (%s)' % (hdr['OBJECT']), ncols=100):
        cube_cleaned = select_clean_data(list_file[i], **clean_param)
        cube_lambda[i] = cube_cleaned

    return cube_lambda
