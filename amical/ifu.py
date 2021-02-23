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
from matplotlib import pyplot as plt

from .get_infos_obs import get_wavelength


def select_wl(i_wl, filtname='YH', instrument='SPHERE-IFS'):
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
