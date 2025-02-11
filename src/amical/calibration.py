"""
@author: Anthony Soulain (University of Sydney)

-------------------------------------------------------------------------
AMICAL: Aperture Masking Interferometry Calibration and Analysis Library
-------------------------------------------------------------------------

Set of functions to calibrate NRM data using a calibrator star data.

-------------------------------------------------------------------------
"""

import numpy as np

from amical.dpfit import leastsqFit
from amical.externals.munch import munchify as dict2class
from amical.tools import wtmn


def _v2varfunc(X, parms):
    """
    This is a function to be fed to leastfit for fitting to normv2var
    to a 5-parameter windshake-seeing function.
    """
    b_lengths = X[0]
    b_angles = X[1]

    a = parms["a"]
    b = parms["b"]
    c = parms["c"]
    d = parms["d"]
    e = parms["e"]

    Y = (
        a
        + b * b_lengths
        + c * b_lengths**2
        + d * b_lengths * abs(np.sin(b_angles))
        + e * b_lengths**2 * np.sin(b_angles) ** 2
    )

    return Y


def _apply_sig_clip(
    data, e_data, sig_thres=2, ymin=0, ymax=1.2, var="V2", display=False
):
    """Apply the sigma-clipping on the dataset and plot some diagnostic plots."""
    import matplotlib.pyplot as plt
    from astropy.stats import sigma_clip

    filtered_data = sigma_clip(data, sigma=sig_thres, axis=0)

    n_files = data.shape[0]
    n_pts = data.shape[1]  # baselines or bs number

    mn_data_clip, std_data_clip = [], []
    for i in range(n_pts):
        cond = filtered_data[:, i].mask
        data_clip = data[:, i][~cond]
        e_data_clip = e_data[:, i][~cond]
        cmn, std = wtmn(data_clip, weights=e_data_clip)
        mn_data_clip.append(cmn)
        std_data_clip.append(std)

    data_med = np.median(data, axis=0)

    if var == "V2":
        ylabel = r"Raw V$^2$"
        xlabel = "# baselines"
    else:
        ylabel = r"Raw closure Phases $\Phi$ [deg]"
        xlabel = "# bispectrum"
        ymin = -ymax

    if display:
        plt.figure()
        plt.title("CALIBRATOR")
        if n_files != 1:
            plt.plot(data[0], color="grey", alpha=0.2, label="Data vs. files")
            plt.plot(data[1:, :].T, color="grey", alpha=0.2)
        plt.plot(data_med, "g--", lw=1, label="Median value")
        for i in range(n_files):
            data_ind = data[i, :]
            cond = filtered_data[i, :].mask
            x = np.arange(len(data_ind))
            bad = data_ind[cond]
            x_bad = x[cond]
            if i == 0:
                plt.plot(
                    x_bad,
                    bad,
                    "rx",
                    ms=3,
                    label=rf"Rejected points (>{sig_thres}$\sigma$)",
                )
            else:
                plt.plot(x_bad, bad, "rx", ms=3)
        plt.legend(loc="best", fontsize=9)
        plt.ylim(ymin, ymax)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
    return np.array(mn_data_clip), np.array(std_data_clip)


def _calc_correction_atm_vis2(
    data, corr_const=1, nf=100, display=False, verbose=False, normalizedUncer=False
):
    """
    This function corrects V^2 for seeing and windshake. Use this on source and
    cal before division. Returns a multiplicative correction to the V^2.

    Parameters:
    -----------
    `data` {class}:
        class like containting results from extract_bs function,
    `corr_const` {float}:
        Correction constant (0.4 is good for V for NIRC experiment).
    Returns:
    --------
    `correction` {array}:
        Correction factor to apply.

    """
    import matplotlib.pyplot as plt

    vis = data.vis2
    avar = data.matrix.avar
    u = data.u
    v = data.v

    err_avar = data.matrix.err_avar
    err_vis = data.e_vis2

    w = np.where((vis == 0) | (avar == 0))

    if len(w) == 0:
        print("Cannot continue - divide by zero imminent...")
        correction = None

    normvar = avar / vis**2
    w0 = np.where(u**2 + v**2 >= 0)

    param = {"a": 0, "b": 0, "c": 0, "d": 0, "e": 0}

    if len(err_avar) == 0:
        err_avar = normvar * 2.0 / np.sqrt(nf)
    if len(err_vis) != 0:
        err_avar = abs(normvar) * np.sqrt(
            err_avar**2 / avar**2 + 2.0 * err_vis**2 / vis**2
        )

    X = [u[w0], v[w0]]
    fit = leastsqFit(
        _v2varfunc,
        X,
        param,
        normvar[w0],
        err_avar[w0],
        verbose=verbose,
        normalizedUncer=normalizedUncer,
    )

    correction = np.ones(vis.shape)
    correction[w0] = np.exp(fit["model"] * corr_const)

    if display:
        plt.figure()
        plt.errorbar(
            (X[0] ** 2 + X[1] ** 2) ** 0.5,
            normvar[w0],
            yerr=err_avar[w0],
            ls="None",
            ecolor="lightgray",
            marker=".",
            label="data",
        )
        plt.plot((X[0] ** 2 + X[1] ** 2) ** 0.5, fit["model"], "rs")
        plt.show(block=False)

    return correction


def average_calib_files(list_nrm, sig_thres=2, display=False):
    """Average NRM data extracted from multiple calibrator files. Additionaly,
    perform sigma-clipping to reject suspicious dataset.

    Parameters:
    -----------
    `list_nrm` : {list}
        List of classes containing extracted NRM data (see bispect.py) of multiple calibrator files,\n
    `sig_thres` : {float}
        Threshold of the sigma clipping (default: 2-sigma around the median is used),\n
    """
    from astropy.io import fits

    nfiles = len(list_nrm)
    l_pa = np.zeros(nfiles)
    cp_vs_file, e_cp_vs_file = [], []
    vis2_vs_file, e_vis2_vs_file = [], []

    # Fill array containing each vis2 and cp across files.
    for n in range(nfiles):
        nrm = list_nrm[n]
        with fits.open(nrm.infos.filename) as hdu:
            hdr = hdu[0].header
        try:
            # todo: Check parallactic angle param of a real NIRISS header.
            l_pa[n] = hdr["PARANG"]
        except KeyError:
            l_pa[n] = 0

        cp = nrm.cp
        e_cp = nrm.e_cp
        vis2 = nrm.vis2
        e_vis2 = nrm.e_vis2

        cp_vs_file.append(cp)
        e_cp_vs_file.append(e_cp)
        vis2_vs_file.append(vis2)
        e_vis2_vs_file.append(e_vis2)

    bl = list_nrm[0].bl

    cp_vs_file = np.array(cp_vs_file)
    e_cp_vs_file = np.array(e_cp_vs_file)
    vis2_vs_file = np.array(vis2_vs_file)
    e_vis2_vs_file = np.array(e_vis2_vs_file)

    zero_uncer = e_vis2_vs_file == 0
    e_vis2_vs_file[zero_uncer] = np.max(e_vis2_vs_file)

    cmn_vis2, std_vis2 = wtmn(vis2_vs_file, e_vis2_vs_file)
    cmn_cp, std_cp = wtmn(cp_vs_file, e_cp_vs_file)

    # Apply sigma clipping on the averages
    cmn_vis2_clip, std_vis2_clip = _apply_sig_clip(
        vis2_vs_file, e_vis2_vs_file, sig_thres=sig_thres, var="V2", display=display
    )
    cmn_cp_clip, std_cp_clip = _apply_sig_clip(
        cp_vs_file,
        e_cp_vs_file,
        sig_thres=sig_thres,
        ymax=10,
        var="CP",
        display=display,
    )

    res = {
        "f_v2_clip": np.array(cmn_vis2_clip),
        "f_v2": np.array(cmn_vis2),
        "std_vis2_clip": np.array(std_vis2_clip),
        "std_vis2": np.array(std_vis2),
        "f_cp_clip": np.array(cmn_cp_clip),
        "f_cp": np.array(cmn_cp),
        "std_cp_clip": np.array(std_cp_clip),
        "std_cp": np.array(std_cp),
        "bl": bl,
        "pa": l_pa,
    }

    return dict2class(res)


def calibrate(
    res_t,
    res_c,
    clip=False,
    sig_thres=2,
    apply_phscorr=False,
    apply_atmcorr=False,
    normalize_err_indep=False,
    display=False,
):
    """Calibrate v2 and cp from a science target and its calibrator.

    Parameters
    ----------
    `res_t` : {dict}
        Dictionnary containing extracted NRM data of science target (see bispect.py),\n
    `res_c` : {list or dict}
        Dictionnary or a list of dictionnary containing extracted NRM data of calibrator target,\n
    `clip` : {bool}
        If True, sigma clipping is performed over the calibrator files (if any) to reject bad
        observables due to seeing conditions, centering, etc.,\n
    `sig_thres` : {float}
        Threshold of the sigma clipping (default: 2-sigma around the median is used),\n
    `apply_phscorr` : {bool}, optional
        If True, apply a phasor correction due to piston between holes, by default False.\n
    `apply_atmcorr` : {bool}, optional
        If True, apply a atmospheric correction on V2 from seeing and wind shacking issues, by default False.\n
    `normalize_err_indep` : {bool}, optional
        If True, the CP uncertaintities are normalized by np.sqrt(n_holes/3.) to not over use
        the non-independant closure phases.\n
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

    if not isinstance(res_c, list):
        res_c = [res_c]

    calib_tab = average_calib_files(res_c, sig_thres=sig_thres, display=display)

    if clip:
        cmn_v2_c, cmn_cp_c, std_v2_c, std_cp_c = (
            calib_tab.f_v2_clip,
            calib_tab.f_cp_clip,
            calib_tab.std_vis2_clip,
            calib_tab.std_cp_clip,
        )
    else:
        cmn_v2_c, cmn_cp_c, std_v2_c, std_cp_c = (
            calib_tab.f_v2,
            calib_tab.f_cp,
            calib_tab.std_vis2,
            calib_tab.std_cp,
        )

    v2_corr_t = 1
    if apply_atmcorr:
        v2_corr_t = _calc_correction_atm_vis2(res_t)

    if apply_phscorr:
        v2_corr_t *= res_t.matrix.phs_v2corr

    # Raw V2 target (corrected from atm correction and phasors.)
    v2_t = res_t.vis2 / v2_corr_t
    e_v2_t = res_t.e_vis2 / v2_corr_t
    #  e_v2_c = res_c[0].e_vis2/v2_corr_t

    # Raw CP target
    cp_t = res_t.cp
    e_cp_t = res_t.e_cp

    # Calibration by the weighted averages and taking into accound the std of the calibrators.
    # ---------------------------------------------
    vis2_calib = v2_t / cmn_v2_c
    cp_calib = cp_t - cmn_cp_c

    #
    n_holes = res_t.mask.n_holes
    if normalize_err_indep:
        err_scale = np.sqrt(n_holes / 3.0)
    else:
        err_scale = 1

    # Quadratic added error due to calibrator dispersion (the average is weightened (see wtmn from amical.tools)).
    weightened_error = True
    if weightened_error:
        e_vis2_calib = np.sqrt(
            e_v2_t**2 / cmn_v2_c**2 + std_v2_c**2 * v2_t**2 / cmn_v2_c**4
        )
    else:
        e_vis2_calib = np.sqrt(e_v2_t**2 + std_v2_c**2)

    e_cp_calib = np.sqrt(e_cp_t**2 + std_cp_c**2) * err_scale

    u1 = res_t.u[res_t.mask.bs2bl_ix[0, :]]
    v1 = res_t.v[res_t.mask.bs2bl_ix[0, :]]
    u2 = res_t.u[res_t.mask.bs2bl_ix[1, :]]
    v2 = res_t.v[res_t.mask.bs2bl_ix[1, :]]

    cal = {
        "vis2": vis2_calib,
        "e_vis2": e_vis2_calib,
        "cp": cp_calib,
        "e_cp": e_cp_calib,
        "u": res_t.u,
        "v": res_t.v,
        "wl": res_t.wl,
        "u1": u1,
        "v1": v1,
        "u2": u2,
        "v2": v2,
        "raw_t": res_t,
        "raw_c": res_c,
    }
    return dict2class(cal)
