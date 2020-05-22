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
from astropy.io import fits
from astropy.stats import sigma_clip
from matplotlib import pyplot as plt
from munch import munchify as dict2class
from termcolor import cprint
from uncertainties import ufloat

from miamis.dpfit import leastsqFit
from miamis.tools import computeUfloatArr


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


def applySigClip(data, e_data, sig_thres=2, use_var=True, ymin=0, ymax=1.2, var='V2', display=False):
    """ Apply the sigma-clipping on the dataset and plot some diagnostic plots. """
    filtered_data = sigma_clip(data, sigma=sig_thres, axis=0)

    n_files = data.shape[0]
    n_pts = data.shape[1]

    u_data_clip = []
    for i in range(n_pts):
        cond = filtered_data[:, i].mask
        data_clip = data[:, i][~cond]
        e_data_clip = e_data[:, i][~cond]
        n_sel = len(data_clip)
        u_data = []
        for j in range(n_sel):
            u_data.append(ufloat(data_clip[j], e_data_clip[j]))

        if use_var:
            res = np.mean(u_data)
        else:
            res = ufloat(np.mean(data_clip), np.mean(e_data_clip))
        u_data_clip.append(res)

    data_med = np.median(data, axis=0)

    if var == 'V2':
        ylabel = r'Raw V$^2$'
        xlabel = '# baselines'
    else:
        ylabel = r'Raw closure Phases $\Phi$ [deg]'
        xlabel = '# bispectrum'
        ymin = -ymax

    if display:
        plt.figure()
        plt.title('CALIBRATOR')
        if n_files != 1:
            plt.plot(data[0], color='grey', alpha=.2, label='Data vs. files')
            plt.plot(data[1:, :].T, color='grey', alpha=.2)
        plt.plot(data_med, 'g--', lw=1, label='Median value')
        for i in range(n_files):
            data_ind = data[i, :]
            cond = filtered_data[i, :].mask
            x = np.arange(len(data_ind))
            bad = data_ind[cond]
            x_bad = x[cond]
            if i == 0:
                plt.plot(x_bad, bad, 'rx', ms=3,
                         label=r'Rejected points (>%i$\sigma$)' % sig_thres)
            else:
                plt.plot(x_bad, bad, 'rx', ms=3)
        plt.legend(loc='best', fontsize=9)
        plt.ylim(ymin, ymax)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
    return u_data_clip


def averageCalibFiles(list_nrm, use_var=True, sig_thres=2, display=False):
    """ Average NRM data extracted from multiple calibrator files. Additionaly,
    perform sigma-clipping to reject suspicious dataset.

    Parameters:
    -----------
    `list_nrm` : {list}
        List of classes containing extracted NRM data (see bispect.py) of multiple calibrator files,\n
    `use_var` : {bool}
        If True, the uncertainties are computed using the variance (uncertainties package). Else,
        the error are computed as the average of the uncertainties over the myultiple calibrator
        files,\n
    `sig_thres` : {float}
        Threshold of the sigma clipping (default: 2-sigma around the median is used),\n    
     """
    nfiles = len(list_nrm)
    l_pa = np.zeros(nfiles)
    cp_vs_file, e_cp_vs_file = [], []
    vis2_vs_file, e_vis2_vs_file = [], []
    u_vis2_all, u_cp_all = [], []

    # Fill array containing each vis2 and cp across files.

    for n in range(nfiles):
        nrm = list_nrm[n]
        hdu = fits.open(nrm.filename)
        hdr = hdu[0].header
        try:
            # todo: Check parallactic angle param of a real NIRISS header.
            l_pa[n] = hdr['PARANG']
        except KeyError:
            l_pa[n] = 0

        cp = nrm.cp
        e_cp = nrm.cp_sig
        vis2 = nrm.v2
        e_vis2 = nrm.v2_sig

        n_bs = len(cp)
        n_bl = len(vis2)

        cp_vs_file.append(cp)
        e_cp_vs_file.append(e_cp)
        vis2_vs_file.append(vis2)
        e_vis2_vs_file.append(e_vis2)

        u_vis2_all.append([ufloat(vis2[i], e_vis2[i]) for i in range(n_bl)])
        u_cp_all.append([ufloat(cp[i], e_cp[i]) for i in range(n_bs)])

    cp_vs_file = np.array(cp_vs_file)
    e_cp_vs_file = np.array(e_cp_vs_file)
    vis2_vs_file = np.array(vis2_vs_file)
    e_vis2_vs_file = np.array(e_vis2_vs_file)

    # Compute averages use variance (uncertainties package) or absolute errors.
    if use_var:
        u_vis2 = np.mean(u_vis2_all, axis=0)
        u_cp = np.mean(u_cp_all, axis=0)
    else:
        vis_m = np.mean(vis2_vs_file, axis=0)
        cp_m = np.mean(cp_vs_file, axis=0)
        abs_e_vis2 = np.mean(e_vis2_vs_file, axis=0)
        abs_e_cp = np.mean(e_cp_vs_file, axis=0)
        u_vis2 = computeUfloatArr(vis_m, abs_e_vis2)
        u_cp = computeUfloatArr(cp_m, abs_e_cp)

    # Apply sigma clipping on the averages
    u_vis2_clip = applySigClip(vis2_vs_file, e_vis2_vs_file,
                               sig_thres=sig_thres, use_var=use_var,
                               var='V2', display=display)
    u_cp_clip = applySigClip(cp_vs_file, e_cp_vs_file,
                             sig_thres=sig_thres, use_var=use_var,
                             ymax=10,
                             var='CP', display=display)

    res = {'f_v2_clip': np.array(u_vis2_clip),
           'f_v2': np.array(u_vis2),
           'f_cp_clip': np.array(u_cp_clip),
           'f_cp': np.array(u_cp),
           'bl': nrm.bl,
           'pa': l_pa}

    return dict2class(res)


def calibrate(res_t, res_c, use_var=False, clip=False, sig_thres=2, apply_phscorr=False, display=False):
    """ Calibrate v2 and cp from a science target and its calibrator.

    Parameters
    ----------
    `res_t` : {dict}
        Dictionnary containing extracted NRM data of science target (see bispect.py),\n
    `res_c` : {list or dict}
        Dictionnary or a list of dictionnary containing extracted NRM data of calibrator target,\n
    `use_var` : {bool}
        If True, the uncertainties are computed using the variance (uncertainties package). Else,
        the error are computed as the average of the uncertainties over the myultiple calibrator
        files,\n
    `clip` : {bool}
        If True, sigma clipping is performed over the calibrator files (if any) to reject bad
        observables due to seeing conditions, centering, etc.,\n
    `sig_thres` : {float}
        Threshold of the sigma clipping (default: 2-sigma around the median is used),\n    
    `apply_phscorr` : {bool}, optional
        If True, apply a phasor correction from seeing and wind shacking issues, by default False.\n
    `display`: {bool}
        If True, plot figures.


    Returns
    -------
    `cal`: {class}
        Class of calibrated data, keys are: `v2`, `e_v2` (squared visibities and errors),
        `cp`, `e_cp` (closure phase and errors), `visamp`, `e_visamp` (visibility
        ampliture and errors), `visphi`, `e_visphi` (visibility phase and errors), `u`, `v`
        (u-v coordinates), `wl` (wavelength), `raw_t` and `raw_c` (dictionnary of extracted raw
        NRM data, inputs of this function).
    """

    if type(res_c) is not list:
        res_c = [res_c]
        if clip:
            cprint('\nOnly one calibrator file is used: clip set to False.', 'green')
            clip = False

    calib_tab = averageCalibFiles(
        res_c, use_var=use_var, sig_thres=sig_thres, display=display)

    if clip:
        u_v2_c_m, u_cp_c_m = calib_tab.f_v2_clip, calib_tab.f_cp_clip
    else:
        u_v2_c_m, u_cp_c_m = calib_tab.f_v2, calib_tab.f_cp

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
