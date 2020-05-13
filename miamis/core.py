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
from uncertainties import ufloat

from miamis.dpfit import leastsqFit


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


def computeMultiCal(res_c):
    """ Average calibration factors of visibilities and closure phases over 
    multiple calibrator files."""
    if type(res_c) is list:
        v2_c_all, cp_c_all, e_v2_c_all, e_cp_c_all = [], [], [], []
        list_res_c = np.array(res_c).copy()
        for res_c in list_res_c:
            v2_corr_c = calc_correctionAtm_vis2(res_c)
            v2_c = res_c.v2/v2_corr_c
            e_v2_c = res_c.v2_sig/v2_corr_c
            cp_c = res_c.cp
            e_cp_c = res_c.cp_sig

            v2_c_all.append(v2_c)
            cp_c_all.append(cp_c)
            e_v2_c_all.append(e_v2_c)
            e_cp_c_all.append(e_cp_c)

        v2_c_all = np.array(v2_c_all)
        e_v2_c_all = np.array(e_v2_c_all)
        cp_c_all = np.array(cp_c_all)
        e_cp_c_all = np.array(e_cp_c_all)

        # Average v2 calibrator over multiple files:
        v2_c_m = []
        for bl in range(v2_c_all.shape[1]):
            u_v2_c = []
            for ifile in range(v2_c_all.shape[0]):
                u_v2_c.append(ufloat(v2_c_all[ifile, bl],
                                     e_v2_c_all[ifile, bl]))
            u_v2_c = np.array(u_v2_c)
            v2_c_m.append(np.mean(u_v2_c))

        # Average cp calibrator over multiple files:
        cp_c_m = []
        for bs in range(cp_c_all.shape[1]):
            u_cp_c = []
            for ifile in range(cp_c_all.shape[0]):
                u_cp_c.append(ufloat(cp_c_all[ifile, bs],
                                     e_cp_c_all[ifile, bs]))
            u_cp_c = np.array(u_cp_c)
            cp_c_m.append(np.mean(u_cp_c))

        u_v2_c_m = np.array(v2_c_m)
        u_cp_c_m = np.array(cp_c_m)
    else:
        v2_corr_c = calc_correctionAtm_vis2(res_c)
        v2_c = res_c.v2/v2_corr_c
        v2_err_c = res_c.v2_sig/v2_corr_c
        cp_c = res_c.cp
        e_cp_c = res_c.cp_sig
        u_v2_c_m = np.array([ufloat(v2_c[i], v2_err_c[i])
                             for i in range(len(v2_c))])
        u_cp_c_m = np.array([ufloat(cp_c[i], e_cp_c[i])
                             for i in range(len(cp_c))])
    return u_v2_c_m, u_cp_c_m


def calibrate(res_t, res_c, apply_phscorr=False):
    """ Calibrate v2 and cp from a science target and its calibrator.

    Parameters
    ----------
    `res_t` : {dict}
        Dictionnary containing extracted NRM data of science target (see bispect.py),\n
    `res_c` : {list or dict}
        Dictionnary or a list of dictionnary containing extracted NRM data of calibrator target,\n
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

    u_v2_c_m, u_cp_c_m = computeMultiCal(res_c)

    v2_corr_t = calc_correctionAtm_vis2(res_t)

    if apply_phscorr:
        v2_corr_t *= res_t.phs_v2corr

    v2_t = res_t.v2/v2_corr_t
    v2_err_t = res_t.v2_sig/v2_corr_t

    cp_t = res_t.cp
    e_cp_t = res_t.cp_sig

    u_cp_t = np.array([ufloat(cp_t[i], e_cp_t[i]) for i in range(len(cp_t))])

    CP = u_cp_t - u_cp_c_m
    cp = np.array([x.nominal_value for x in CP])
    e_cp = np.array([x.std_dev for x in CP])
    # cp = res_t.cp - res_c.cp
    # e_cp = res_t.cp_sig + res_c.cp_sig

    V2_t = np.array([ufloat(v2_t[i], v2_err_t[i]) for i in range(len(v2_t))])
    V2 = V2_t/u_v2_c_m

    vis2 = np.array([x.nominal_value for x in V2])
    e_vis2 = np.array([x.std_dev for x in V2])

    visamp_t = np.abs(res_t.cvis_all)
    visphi_t = np.angle(res_t.cvis_all)

    if type(res_c) == list:
        visamp_c = np.abs(res_c[0].cvis_all)
        visphi_c = np.angle(res_c[0].cvis_all)

        fact_calib_visamp = np.mean(visamp_c, axis=0)
        fact_calib_visphi = np.mean(visphi_c, axis=0)
    else:
        visamp_c = np.abs(res_c.cvis_all)
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
