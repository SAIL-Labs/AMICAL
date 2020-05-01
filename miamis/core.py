# -*- coding: utf-8 -*-
"""
@author: Anthony Soulain (University of Sydney)

--------------------------------------------------------------------
MIAMIS: Multi-Instruments Aperture Masking Interferometry Software
--------------------------------------------------------------------

Core function to reduce NRM data. These functions are independant of 
the used method: mf method from Sydney code and xara method (not 
implemented yet).
-------------------------------------------------------------------- 
"""

import numpy as np
from matplotlib import pyplot as plt
from munch import munchify as dict2class
from termcolor import cprint
from uncertainties import ufloat

from miamis.dpfit import leastsqFit
from miamis.tools import (applyMaskApod, checkRadiusResize, crop_max,
                          skyCorrection)


def v2varfunc(X, parms):
    """
    This is a function to be fed to leastfit for fitting to normv2var
    to a 5-parameter windshake-seeing function.
    """
    b_lengths = X[0]
    b_angles = X[1]

    a = parms['a']
    b = parms['b']
    c = parms['c']
    d = parms['d']
    e = parms['e']

    Y = a + b*b_lengths + c*b_lengths**2\
        + d*b_lengths*abs(np.sin(b_angles))\
        + e*b_lengths**2*np.sin(b_angles)**2

    return Y


def calc_correctionAtm_vis2(data, v2=True, corr_const=1, nf=100, display=False,
                            verbose=False, normalizedUncer=False):
    """
    This function corrects V^2 for seeing and windshake. Use this on source and 
    cal before division. Returns a multiplicative correction to the V^2.

    Parameters:
    -----------
    `data` {class}:
        class like containting results from Extract_bispect function,

    `corr_const` {float}: 
        Correction constant (0.4 is good for V for NIRC experiment).
    Returns:
    --------
    `correction` {array}:
        Correction factor to apply.

    """

    vis = data.v2
    avar = data.avar
    u = data.u
    v = data.v

    err_avar = data.err_avar
    err_vis = data.v2_sig

    w = np.where((vis == 0) | (avar == 0))

    if len(w) == 0:
        print('Cannot continue - divide by zero imminent...')
        correction = None

    normvar = avar/vis**2
    w0 = np.where(u**2+v**2 >= 0)

    param = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0}

    if (len(err_avar) == 0):
        err_avar = normvar*2.0/np.sqrt(nf)
    if (len(err_vis) != 0):
        err_avar = abs(normvar)*np.sqrt(err_avar**2 /
                                        avar**2 + 2.0*err_vis**2/vis**2)

    X = [u[w0], v[w0]]
    fit = leastsqFit(v2varfunc, X, param, normvar[w0], err_avar[w0],
                     verbose=verbose, normalizedUncer=normalizedUncer)

    correction = np.ones(vis.shape)
    correction[w0] = np.exp(fit['model'] * corr_const)

    if display:
        plt.figure()
        plt.errorbar((X[0]**2+X[1]**2)**0.5, normvar[w0], yerr=err_avar[w0],
                     ls='None', ecolor='lightgray', marker='.', label='data')
        plt.plot((X[0]**2+X[1]**2)**0.5, fit['model'], 'rs')
        plt.show(block=False)

    return correction


def clean_data(data, isz=None, r1=None, dr=None, checkrad=False):
    """ Clean data (if not simulated data).

    Parameters:
    -----------

    `data` {np.array} -- datacube containing the NRM data\n
    `isz` {int} -- Size of the cropped image (default: {None})\n
    `r1` {int} -- Radius of the rings to compute background sky (default: {None})\n
    `dr` {int} -- Outer radius to compute sky (default: {None})\n
    `checkrad` {bool} -- If True, check the resizing and sky substraction parameters (default: {False})\n

    Returns:
    --------
    `cube` {np.array} -- Cleaned datacube.
    """
    if data.shape[1] % 2 == 1:
        data = np.array([im[:-1, :-1] for im in data])

    n_im = data.shape[0]
    npix = data.shape[1]

    if checkrad:
        img0 = applyMaskApod(data[0], r=int(npix//3))
        ref0_max, pos = crop_max(img0, isz, f=3)
        fig = checkRadiusResize(img0, isz, r1, dr, pos)
        fig.show()
        return None

    cube = []
    for i in range(n_im):
        img0 = applyMaskApod(data[i], r=int(npix//3))
        im_rec_max, pos = crop_max(img0, isz, f=3)
        img_biased, bg = skyCorrection(im_rec_max, r1=r1, dr=dr)
        try:
            img = applyMaskApod(img_biased, r=isz//5)
            cube.append(img)
        except ValueError:
            cprint(
                'Error: problem with centering process -> check isz/r1/dr parameters.', 'red')
            cprint(i, 'red')

    cube = np.array(cube)
    # If image size is odd, remove the last line and row (need even size image
    # for fft purposes.

    if cube.shape[1] % 2 == 1:
        cube = np.array([im[:-1, :-1] for im in cube])

    # cube = np.roll(np.roll(cube, npix//2, axis=1), npix//2, axis=2)
    return cube


def calibrate(res_t, res_c, apply_phscorr=False):
    """ Calibrate v2 and cp from a science target and its calibrator.

    Parameters
    ----------
    `res_t` : {dict}
        Dictionnary containing extracted NRM data of science target (see bispect.py),\n
    `res_c` : {dict}
        Dictionnary containing extracted NRM data of calibrator target,\n
    `apply_phscorr` : {bool}, optional
        If True, apply a phasor correction from seeing and wind shacking issues, by default False
        *alpha > beta*

    Returns
    -------
    `cal`: {class}
        Class of calibrated data, keys are: `v2`, `e_v2` (squared visibities and errors),
        `cp`, `e_cp` (closure phase and errors), `visamp`, `e_visamp` (visibility 
        ampliture and errors), `visphi`, `e_visphi` (visibility phase and errors), `u`, `v` 
        (u-v coordinates), `wl` (wavelength), `raw_t` and `raw_c` (dictionnary of extracted raw 
        NRM data, inputs of this function).
    """
    v2_corr_t = calc_correctionAtm_vis2(res_t)
    v2_corr_c = calc_correctionAtm_vis2(res_c)

    if apply_phscorr:
        v2_corr_t *= res_t.phs_v2corr
        v2_corr_c *= res_c.phs_v2corr

    v2_t = res_t.v2/v2_corr_t
    v2_c = res_c.v2/v2_corr_c

    v2_err_t = res_t.v2_sig/v2_corr_t
    v2_err_c = res_c.v2_sig/v2_corr_c

    cp = res_t.cp - res_c.cp
    e_cp = res_t.cp_sig + res_c.cp_sig

    V2_t = np.array([ufloat(v2_t[i], v2_err_t[i]) for i in range(len(v2_t))])
    V2_c = np.array([ufloat(v2_c[i], v2_err_c[i]) for i in range(len(v2_c))])
    V2 = V2_t/V2_c

    vis2 = np.array([x.nominal_value for x in V2])
    e_vis2 = np.array([x.std_dev for x in V2])

    visamp_t = np.abs(res_t.cvis_all)
    visamp_c = np.abs(res_c.cvis_all)
    visphi_t = np.angle(res_t.cvis_all)
    visphi_c = np.angle(res_c.cvis_all)

    fact_calib_visamp = np.mean(visamp_c, axis=0)
    fact_calib_visphi = np.mean(visphi_c, axis=0)

    visamp_calibrated = visamp_t/fact_calib_visamp
    visphi_calibrated = visphi_t - fact_calib_visphi

    visamp = np.mean(visamp_calibrated, axis=0)
    e_visamp = np.std(visamp_calibrated, axis=0)

    visphi = np.mean(visphi_calibrated, axis=0)
    e_visphi = np.std(visphi_calibrated, axis=0)

    u1 = res_t.u[res_t.bs2bl_ix[0, :]]
    v1 = res_t.v[res_t.bs2bl_ix[0, :]]
    u2 = res_t.u[res_t.bs2bl_ix[1, :]]
    v2 = res_t.v[res_t.bs2bl_ix[1, :]]

    cal = {'vis2': vis2, 'e_vis2': e_vis2, 'cp': cp, 'e_cp': e_cp,
           'visamp': visamp, 'e_visamp': e_visamp, 'visphi': visphi,
           'e_visphi': e_visphi, 'u': res_t.u, 'v': res_t.v, 'wl': res_t.wl,
           'u1': u1, 'v1': v1, 'u2': u2, 'v2': v2,
           'raw_t': res_t, 'raw_c': res_c}

    return dict2class(cal)
