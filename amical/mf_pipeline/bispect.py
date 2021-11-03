"""
@author: Anthony Soulain (University of Sydney)

-------------------------------------------------------------------------
AMICAL: Aperture Masking Interferometry Calibration and Analysis Library
-------------------------------------------------------------------------

Matched filter sub-pipeline method.

Compute bispectrum for a given fits file (adapted from bispect.pro
and calc_bispect.pro).

--------------------------------------------------------------------
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from munch import munchify as dict2class
from PyPDF2 import PdfFileMerger
from PyPDF2 import PdfFileReader
from scipy.optimize import minimize
from termcolor import cprint
from tqdm import tqdm

from .idl_function import dblarr
from .idl_function import dist
from .idl_function import regress_noc
from amical.get_infos_obs import get_mask
from amical.mf_pipeline.ami_function import bs_multi_triangle
from amical.mf_pipeline.ami_function import compute_index_mask
from amical.mf_pipeline.ami_function import give_peak_info2d
from amical.mf_pipeline.ami_function import make_mf
from amical.mf_pipeline.ami_function import phase_chi2
from amical.mf_pipeline.ami_function import tri_pix
from amical.tools import compute_pa
from amical.tools import cov2cor


def _compute_complex_bs(
    ft_arr,
    index_mask,
    fringe_peak,
    mf,
    dark_ps=None,
    closing_tri_pix=None,
    bs_multi_tri=False,
):
    """Compute the complex visibilities and the bispectrum of an input ft_arr (_construct_ft_arr()).
    In addition, compute the phase phs of each frames, and some calibration array (relative to the
    dark frames).

    Parameters:
    -----------
    `ft_arr` {numpy.array}: Fourier transform of the input cube (from _construct_ft_arr()),\n
    `index_mask` {class/dict}: Object of indices computed with compute_index_mask(),\n
    `fringe_peak` {list}: List of fringe peak position and gain (see give_peak_info2d()),\n
    `mf` {class/dict}:  Object containing the matched filter informations (expected peak
    positions, make_mf()),\n
    `dark_ps` {array}: Calibration cube of the dark associated with the observations (default:
    None),\n
    `closing_tri_pix` {list}: List of all subset of closing triangle used to compute the bispectrum
    using the multiple triangle technique (default: None),\n
    `bs_multi_tri` {bool}: If True, the multiple triangle computation is applied (default=False).

    Returns:
    --------
    `complex_bs` {dict}: Dictionnary of observables ('vis_arr', 'bs_arr'), calibrations ('phs', 'calib_v2',
    'fluxes') and saved fft frames ('ps', 'dps').\n

    Observables arrays ('vis_arr', 'bs_arr') contain observables for each frames.\n

    >>> complex_bs['vis_arr'].shape()=[n_ps, n_baselines]

    complex_bs['vis_arr'] is a structured numpy array containing the complex visibilities ('complex'),
    the phase ('phase'), the amplitude ('amplitude') and the squared visibilities ('squared'). complex_bs['calib_v2']
    contains the 'dark' and the 'bias' arrays, and complex_bs['phs'] contains the phase values ('value').
    and the associated errors ('err').

    """
    n_baselines = index_mask.n_baselines
    n_bispect = index_mask.n_bispect
    bs2bl_ix = index_mask.bs2bl_ix

    n_ps = ft_arr.shape[0]
    npix = ft_arr.shape[1]
    aveps = np.zeros([npix, npix])
    avedps = np.zeros([npix, npix])

    # Extracted complex quantities
    vis_arr = np.zeros(
        (n_ps, n_baselines),
        dtype=[
            ("complex", complex),
            ("phase", float),
            ("amplitude", float),
            ("squared", float),
        ],
    )
    bs_arr = np.zeros([n_ps, n_bispect]).astype(complex)

    # Calibration
    calib_v2 = np.zeros(n_baselines, dtype=[("dark", float), ("bias", float)])
    phs = np.zeros((2, n_ps, n_baselines), dtype=[("value", float), ("err", float)])

    fluxes = np.zeros(n_ps)

    for i in tqdm(
        range(n_ps),
        ncols=100,
        desc="Extracting in the cube",
        leave=False,
        file=sys.stdout,
    ):
        ft_frame = ft_arr[i]
        ps = np.abs(ft_frame) ** 2

        if dark_ps is not None and (len(dark_ps.shape) == 3):
            dps = dark_ps[i]
        elif dark_ps is not None and (len(dark_ps.shape) == 2):
            dps = dark_ps
        else:
            dps = np.zeros([npix, npix])

        avedps += dps  # Cumulate ps (dark) to perform an average at the end
        aveps += ps  # Cumulate ps to perform an average at the end

        fluxes[i] = abs(ft_frame[0, 0]) - np.sqrt(dps[0, 0])

        # Extract complex visibilities of each fringe peak (each indices are
        # computed using make_mf function)
        cvis = np.zeros(n_baselines).astype(complex)
        for j in range(n_baselines):
            pix = fringe_peak[j][:, 0].astype(int), fringe_peak[j][:, 1].astype(int)
            gain = fringe_peak[j][:, 2]

            calib_v2["dark"][j] = np.sum(gain ** 2 * dps[pix])
            cvis[j] = np.sum(gain * ft_frame[pix])

            ftf1 = np.roll(ft_frame, 1, axis=0)
            ftf2 = np.roll(ft_frame, -1, axis=0)
            dummy = np.sum(
                ft_frame[pix] * np.conj(ftf1[pix]) + np.conj(ft_frame[pix]) * ftf2[pix]
            )

            phs["value"][0, i, j] = np.arctan2(dummy.imag, dummy.real)
            phs["err"][0, i, j] = 1 / abs(dummy)

            ftf1 = np.roll(ft_frame, 1, axis=1)
            ftf2 = np.roll(ft_frame, -1, axis=1)
            dummy = np.sum(
                ft_frame[pix] * np.conj(ftf1[pix]) + np.conj(ft_frame[pix]) * ftf2[pix]
            )

            phs["value"][1, i, j] = np.arctan2(dummy.imag, dummy.real)
            phs["err"][1, i, j] = 1 / abs(dummy)

        # Correct for overlapping baselines
        rvis = cvis.real
        ivis = cvis.imag

        rvis = np.dot(mf.rmat, rvis)
        ivis = np.dot(mf.imat, ivis)

        cvis_fixed = rvis + ivis * 1j
        vis_arr["complex"][i, :] = cvis_fixed
        vis_arr["phase"][i, :] = np.arctan2(cvis_fixed.imag, cvis_fixed.real)
        vis_arr["amplitude"][i, :] = np.abs(cvis_fixed)
        vis_arr["squared"][i] = np.abs(cvis_fixed) ** 2 - calib_v2["dark"]

        # Calculate Bispectrum
        if not bs_multi_tri:
            cvis_1 = cvis_fixed[bs2bl_ix[0, :]]
            cvis_2 = cvis_fixed[bs2bl_ix[1, :]]
            cvis_3 = cvis_fixed[bs2bl_ix[2, :]]
            bs_arr[i, :] = cvis_1 * cvis_2 * np.conj(cvis_3)
        else:
            bs_arr = bs_multi_triangle(
                i,
                bs_arr,
                ft_frame,
                bs2bl_ix,
                mf,
                closing_tri_pix,
            )

    ps = aveps / n_ps
    dps = avedps / n_ps

    complex_bs = {
        "vis_arr": vis_arr,
        "bs_arr": bs_arr,
        "phs": phs,
        "calib_v2": calib_v2,
        "fluxes": fluxes,
        "ps": ps,
        "dps": dps,
    }
    return complex_bs


def _construct_ft_arr(cube):
    """Open the data cube and perform a series of roll (both axis) to avoid grid artefact
    (negative fft values). Remove the last row/column in case of odd array.

    Parameters:
    -----------
    `cube` {array}: cleaned data cube from amical.select_data().

    Returns:
    --------
    `ft_arr` {array}: complex array of the Fourier transform of the cube,\n
    `n_ps` {int}: Number of frames,\n
    `n_pix` {int}: Dimensions of one frames,\n

    """
    if cube.shape[1] % 2 == 1:
        cube = np.array([im[:-1, :-1] for im in cube])

    n_pix = cube.shape[1]
    cube = np.roll(np.roll(cube, n_pix // 2, axis=1), n_pix // 2, axis=2)

    ft_arr = np.fft.fft2(cube)

    i_ps = ft_arr.shape
    n_ps = i_ps[0]

    return ft_arr, n_ps, n_pix


def _show_complex_ps(ft_arr, i_frame=0):
    """
    Show the complex fft image (real and imaginary) and power spectrum (abs(fft)) of the first frame
    to check the applied correction on the cube.
    """
    fig = plt.figure(figsize=(16, 6))
    ax1 = plt.subplot(1, 3, 1)
    plt.title("Real part")
    plt.imshow(ft_arr[i_frame].real, cmap="gist_stern", origin="lower")
    plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    plt.title("Imaginary part")
    plt.imshow(ft_arr[i_frame].imag, cmap="gist_stern", origin="lower")
    plt.subplot(1, 3, 3)
    plt.title("Power spectrum (centred)")
    plt.imshow(np.fft.fftshift(abs(ft_arr[i_frame])), cmap="gist_stern", origin="lower")
    plt.tight_layout()
    return fig


def _show_peak_position(
    ft_arr, n_baselines, mf, maskname, peakmethod, i_fram=0, aver=False
):
    """Show the expected position of the peak in the Fourier space using the
    mask coordinates and the chosen method."""
    dim1, dim2 = ft_arr.shape[1], ft_arr.shape[2]
    x, y = np.arange(dim1), np.arange(dim2)
    X, Y = np.meshgrid(x, y)
    lX, lY, lC = [], [], []
    for j in range(n_baselines):
        l_x = X.ravel()[mf.pvct[mf.ix[0, j] : mf.ix[1, j]]]
        l_y = Y.ravel()[mf.pvct[mf.ix[0, j] : mf.ix[1, j]]]
        g = mf.gvct[mf.ix[0, j] : mf.ix[1, j]]

        peak = [[l_y[k], l_x[k], g[k]] for k in range(len(l_x))]

        for x in peak:
            lX.append(x[1])
            lY.append(x[0])
            lC.append(x[2])

    ft_frame = ft_arr[i_fram]
    ps = ft_frame.real

    if aver:
        ps = np.zeros(ps.shape)
        for i_frame in ft_arr:
            ps += i_frame.real
        ps /= ft_arr.shape[0]

    fig, ax = plt.subplots(figsize=(9, 7))
    # plt.rc('xtick', labelsize=15)
    ax.set_title(
        f"Expected splodge position with mask {maskname} (method = {peakmethod})"
    )
    im = ax.imshow(ps, cmap="gist_stern", origin="lower")
    sc = ax.scatter(lX, lY, c=lC, s=20, cmap="viridis")
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size="3%", pad=0.5)
    fig.add_axes(cax)
    cb = fig.colorbar(im, cax=cax)

    cax2 = divider.new_horizontal(size="3%", pad=0.6, pack_start=True)
    fig.add_axes(cax2)
    cb2 = fig.colorbar(sc, cax=cax2)
    cb2.ax.yaxis.set_ticks_position("left")
    cb.set_label("Power Spectrum intensity")
    cb2.set_label("Relative weight [%]", fontsize=20)
    # x1, y1 = 23, 60
    # ax.set_xlim(x1, x1+8)
    # ax.set_ylim(y1, y1+8)
    plt.subplots_adjust(
        top=0.965, bottom=0.035, left=0.025, right=0.965, hspace=0.2, wspace=0.2
    )


def _show_norm_matrices(obs_norm, expert_plot=False):
    """Show covariances matrices of the V2, CP, and a combination
    bispectrum vs. V2."""

    v2_cov = obs_norm["v2_cov"]
    cp_cov = obs_norm["cp_cov"]
    bs_v2_cov = obs_norm["bs_v2_cov"]

    fig1 = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Covariance matrix $V^2$")
    plt.imshow(v2_cov, origin="upper")
    plt.xlabel("# BL")
    plt.ylabel("# BL")
    plt.subplot(1, 2, 2)
    plt.title("Covariance matrix CP")
    try:
        plt.imshow(cp_cov, origin="upper")
    except Exception:
        pass
    plt.xlabel("# CP")
    plt.ylabel("# CP")
    plt.tight_layout()

    if expert_plot:
        fig2 = plt.figure(figsize=(3, 6))
        plt.title("Cov matrix BS vs. $V^2$")
        plt.imshow(bs_v2_cov, origin="upper")
        plt.ylabel("# BL")
        plt.xlabel("# holes - 2")
        plt.tight_layout()
        return fig1, fig2
    return fig1


def _check_input_infos(hdr, targetname=None, filtname=None, instrum=None, verbose=True):
    """Extract informations from the header and fill the missing values with the
    input arguments. Return the infos class containing important informations of
    the input header (keys: target, seeing, instrument, ...)
    """
    target = hdr.get("OBJECT")
    filt = hdr.get("FILTER")
    instrument = hdr.get("INSTRUME", instrum)
    mod = hdr.get("HIERARCH ESO DET ID")

    if (mod == "IFS") & (instrument == "SPHERE"):
        instrument = instrument + "-" + mod

    # Check the target name
    if (target is None) or (target == "STD"):
        if targetname is not None:
            target = targetname
            if verbose:
                cprint(
                    "Warning: OBJECT is not in the header, targetname is used (%s)."
                    % targetname,
                    "green",
                )
        else:
            cprint("Warning: target name not found (header or as input).", "green")

    # Check the filter used
    if filt is None:
        if filtname is not None:
            filt = filtname
            if verbose:
                cprint(
                    "Warning: FILTER is not in the header, filtname is used (%s)."
                    % filtname,
                    "green",
                )

    # Check the instrument used
    if instrument is None:
        raise OSError("instrum not found (in the header or as input).")

    # Origin files
    orig = hdr.get("ORIGFILE", "SimulatedData")
    if orig == "SimulatedData":
        orig = hdr.get("ARCFILE", "SimulatedData")

    # Seeing informations
    seeing_start = float(hdr.get("HIERARCH ESO TEL AMBI FWHM START", 0))
    seeing_end = float(hdr.get("HIERARCH ESO TEL AMBI FWHM END", 0))
    seeing = np.mean([seeing_start, seeing_end])

    infos = {
        "filtname": filt,
        "target": target,
        "instrument": instrument,
        "orig": orig,
        "seeing": seeing,
    }
    return dict2class(infos)


def _format_closing_triangle(index_mask):
    """Use the index_mask from compute_index_mask() to compute the list of
    closing triangle in an appropriate format (e.g.: [[0,1,2], [0,1,3], ..., [4,5,6]]
    for a 7 mask holes)."""
    bs2bl_ix = index_mask.bs2bl_ix
    bl2h_ix = index_mask.bl2h_ix
    closing_tri = []
    for i_bs in range(len(bs2bl_ix.T)):
        tmp = []
        for x in bs2bl_ix.T[i_bs]:
            tmp.extend(bl2h_ix.T[x])
        closing_tri.append(list(set(tmp)))
    return closing_tri


def _set_good_nblocks(n_blocks, n_ps, verbose=False):
    """Check the given n_blocks to do the statistic on
    different block size. If the n_blocks is 0 or greater
    than the frame number, n_blocks is set to n_ps.
    """
    if (n_blocks == 0) or (n_blocks == 1):
        if verbose:
            cprint("! Warning: nblocks == 0 -> n_blocks set to n_ps", "green")
        n_blocks = n_ps
    elif n_blocks > n_ps:
        if verbose:
            cprint("------------------------------------", "green")
            cprint("! Warning: nblocks > n_ps -> n_blocks set to n_ps", "green")
        n_blocks = n_ps
    return n_blocks


def _compute_corr_noise(complex_bs, ft_arr, fringe_peak):
    """Compute the bias (science and dark) and the correlated noise of the
    Fourier transform (correlation between neighbouring terms in each
    frame after removing the signal)."""
    n_ps = ft_arr.shape[0]
    npix = ft_arr.shape[1]

    v2_arr = complex_bs["vis_arr"]["squared"]

    n_baselines = v2_arr.shape[1]

    aver_ps = complex_bs["ps"]
    aver_dps = complex_bs["dps"]

    signal_map = np.zeros([npix, npix])  # 2-D array where pixel=1 if signal
    # inside (i.e.: peak in the fft)
    for j in range(n_baselines):
        pix = fringe_peak[j][:, 0].astype(int), fringe_peak[j][:, 1].astype(int)
        signal_map[pix] = 1.0

    # Compute the center-symetric counterpart of the signal
    signal_map += np.roll(np.roll(np.rot90(np.rot90(signal_map)), 1, axis=0), 1, axis=1)

    # Compute the distance map centered on each corners (origin in the fft)
    corner_dist_map = dist(npix)

    # Indicate signal in the central peak
    signal_map[corner_dist_map < npix / 16.0] = 1.0

    bias, dark_bias = 0, 0
    for _ in range(3):
        signal_cond = signal_map == 0
        bias = np.median(aver_ps[signal_cond])
        dark_bias = np.median(aver_dps[signal_cond])
        signal_map[aver_ps >= 3 * bias] = 1.0

    signal_cond = signal_map != 0  # i.e. where there is signal.

    autocor_noise = np.zeros([npix, npix])
    for i in range(n_ps):
        ft_frame = ft_arr[i].copy()
        ft_frame[signal_cond] = 0.0
        autocor_noise += np.fft.fft2(np.abs(np.fft.ifft2(ft_frame)) ** 2).real
    autocor_noise /= n_ps

    return bias, dark_bias, autocor_noise


def _unbias_v2_arr(
    v2_arr, npix, fringe_peak, bias, dark_bias, autocor_noise, unbias=True
):
    """Unbias the squared visibilities array and add the dark bias substracted
    twice instead of one."""
    n_baselines = v2_arr.shape[1]

    bias_arr = np.zeros(n_baselines)
    for j in range(n_baselines):
        im_peak = dblarr(npix, npix)
        pix = fringe_peak[j][:, 0].astype(int), fringe_peak[j][:, 1].astype(int)
        im_peak[pix] = fringe_peak[j][:, 2]

        autocor_mf = np.fft.ifft2(np.abs(np.fft.ifft2(im_peak)) ** 2).real
        bias_arr[j] = (
            np.sum(autocor_mf * autocor_noise) * bias / autocor_noise[0, 0] * npix ** 2
        )

        if unbias:
            true_bias = bias_arr[j]
        else:
            true_bias = 0
        v2_arr[:, j] -= true_bias

    for j in range(n_baselines):
        v2_arr[:, j] += np.sum(fringe_peak[j][:, 2] ** 2) * dark_bias
    return v2_arr, bias_arr


def _compute_v2_quantities(v2_arr, bias_arr, n_blocks):
    """Compute the squared visibilities quantities: - average ('v2') over the
    cube, - covariance ('v2_cov'), - avar ('avar') and - 'err_avar'."""
    n_ps = v2_arr.shape[0]
    n_baselines = v2_arr.shape[1]

    v2 = np.zeros(n_baselines)
    v2_cov = np.zeros([n_baselines, n_baselines])
    v2_diff = np.zeros([n_blocks, n_baselines])

    # Compute vis. squared average
    v2 = np.mean(v2_arr, axis=0)

    # Compute vis. squared difference
    for j in range(n_baselines):
        for k in range(n_blocks):
            ind1 = k * n_ps // n_blocks
            ind2 = (k + 1) * n_ps // (n_blocks - 1)
            v2_diff[k, j] = np.mean(v2_arr[ind1:ind2, j]) - v2[j]
            # v2_diff[k, j] = np.mean(v2_arr[k, j]) - v2[j]

    # Compute vis. squared covariance
    for j in range(n_baselines):
        for k in range(n_baselines):
            num = np.sum(v2_diff[:, j] * v2_diff[:, k])
            v2_cov[j, k] = num / (n_blocks - 1) / n_blocks
            # Additonal "/ n_blocks" in the original code ???

    # AS. Comparison with numpy cov matrices
    # v2_cov_pyt = np.cov(v2_arr.T, bias=False)
    # v2_cov = v2_cov_pyt

    x = np.arange(n_baselines)
    avar = v2_cov[x, x] * n_ps - bias_arr ** 2 * (1 + (2.0 * v2) / bias_arr)
    err_avar = np.sqrt(
        2.0 / n_ps * (v2_cov[x, x]) ** 2 * n_ps ** 2
        + 4.0 * v2_cov[x, x] * bias_arr ** 2
    )

    v2_quantities = {
        "v2": v2,
        "v2_cov": v2_cov,
        "avar": avar,
        "err_avar": err_avar,
        "v2_arr": v2_arr,
    }
    return v2_quantities


def _compute_bs_quantities(
    bs_arr, v2, fluxes, index_mask, n_blocks, subtract_bs_bias=True
):
    """Compute the bispectrum quantities: - average ('bs') over the
    cube, - covariance ('bs_cov') and - variance ('bs_var')."""
    n_cov = index_mask.n_cov
    n_bispect = index_mask.n_bispect
    bs2bl_ix = index_mask.bs2bl_ix
    bscov2bs_ix = index_mask.bscov2bs_ix

    # Initiate quantities output
    bs_var = np.zeros([2, n_bispect])
    bs_cov = np.zeros([2, n_cov])

    # Compute bispectrum average
    bs = np.mean(bs_arr, axis=0)

    # Unbias bispectrum if required (suppose normalised mf filter)
    if subtract_bs_bias:
        a = 1  # Computed values for non-amplified detector
        b = 2
        bs_bias = a * (
            v2[bs2bl_ix[0, :]] + v2[bs2bl_ix[1, :]] + v2[bs2bl_ix[2, :]]
        ) - b * np.mean(fluxes)
        # bs_bias = (v2[bs2bl_ix[0, :]] + v2[bs2bl_ix[1, :]] + v2[bs2bl_ix[2, :]] + np.mean(fluxes))
        bs = bs - bs_bias

    bs_var = _compute_bs_var(bs_arr, bs, n_blocks)
    bs_cov = _compute_bs_cov(bs_arr, bs, bscov2bs_ix, n_cov)
    bs_quantities = {"bs": bs, "bs_var": bs_var, "bs_cov": bs_cov, "bs_arr": bs_arr}
    return bs_quantities


def _compute_bs_var(bs_arr, bs, n_blocks):
    """Compute the variance matrix of the bispectrum array."""
    n_ps = bs_arr.shape[0]
    n_bispect = bs_arr.shape[1]

    bs_var = np.zeros([2, n_bispect])
    # Compute bispectrum variance
    comp_diff = np.zeros(n_blocks).astype(complex)
    for j in range(n_bispect):
        # comp_diff is the complex difference from the mean, shifted so that the
        # real axis corresponds to amplitude and the imaginary axis phase.
        tmp = (bs_arr[:, j] - bs[j]) * np.conj(bs[j])
        for k in range(n_blocks):
            ind1 = k * n_ps // n_blocks
            ind2 = (k + 1) * n_ps // (n_blocks)
            comp_diff[k] = np.mean(tmp[ind1:ind2])

        num_real = np.sum(np.real(comp_diff) ** 2) / (n_blocks - 1) / n_blocks
        num_imag = np.sum(np.imag(comp_diff) ** 2) / (n_blocks - 1) / n_blocks

        bs_var[0, j] = num_real / (np.abs(bs[j]) ** 2)
        bs_var[1, j] = num_imag / (np.abs(bs[j]) ** 2)
    return bs_var


def _compute_bs_cov(bs_arr, bs, bscov2bs_ix, n_cov):
    """Compute the covariance matrix of the bispectrum array."""
    n_ps = bs_arr.shape[0]
    bs_cov = np.zeros([2, n_cov])
    for j in range(n_cov):
        temp1 = (bs_arr[:, bscov2bs_ix[0, j]] - bs[bscov2bs_ix[0, j]]) * np.conj(
            bs[bscov2bs_ix[0, j]]
        )
        temp2 = (bs_arr[:, bscov2bs_ix[1, j]] - bs[bscov2bs_ix[1, j]]) * np.conj(
            bs[bscov2bs_ix[1, j]]
        )
        denom = (
            abs(bs[bscov2bs_ix[0, j]]) * abs(bs[bscov2bs_ix[1, j]]) * (n_ps - 1) * n_ps
        )

        bs_cov[0, j] = np.sum(np.real(temp1) * np.real(temp2)) / denom
        bs_cov[1, j] = np.sum(np.imag(temp1) * np.imag(temp2)) / denom
    return bs_cov


def _compute_cp_cov(bs_arr, bs, index_mask):
    """Compute the covariance matrix of the closure phase."""
    n_ps = bs_arr.shape[0]
    n_bispect = index_mask.n_bispect

    cp_cov = dblarr(n_bispect, n_bispect)
    for i in tqdm(range(n_bispect), desc="CP covariance", ncols=100, leave=False):
        for j in range(n_bispect):
            temp1 = (bs_arr[:, i] - bs[i]) * np.conj(bs[i])
            temp2 = (bs_arr[:, j] - bs[j]) * np.conj(bs[j])
            denom = abs(bs[i]) ** 2 * abs(bs[j]) ** 2 * (n_ps - 1) * n_ps
            cp_cov[i, j] = np.sum(np.imag(temp1) * np.imag(temp2)) / denom
    return cp_cov


def _compute_bs_v2_cov(bs_arr, v2_arr, v2, bs, index_mask):
    """Compute covariance between power and bispectral amplitude."""
    n_ps = bs_arr.shape[0]
    n_baselines = index_mask.n_baselines
    n_holes = index_mask.n_holes
    bl2bs_ix = index_mask.bl2bs_ix

    # This complicated thing calculates the dot product between the bispectrum point and
    # its error term ie (x . del_x)/|x| and multiplies this by the power error term.
    # Note that this is not the same as using absolute value, and that this sum should be
    # zero where |bs| is zero within errors.
    bs_v2_cov = np.zeros([n_baselines, n_holes - 2])
    for j in range(n_baselines):
        for k in range(n_holes - 2):
            temp = bs_arr[:, bl2bs_ix[j, k]] - bs[bl2bs_ix[j, k]]
            bs_real_tmp = np.real(temp * np.conj(bs[bl2bs_ix[j, k]]))
            diff_v2 = v2_arr[:, j] - v2[j]
            norm = abs(bs[bl2bs_ix[j, k]]) / (n_ps - 1.0) / n_ps
            norm_bs_v2_cov = np.sum(bs_real_tmp * diff_v2) / norm
            bs_v2_cov[j, k] = norm_bs_v2_cov
    return bs_v2_cov


def _normalize_all_obs(
    bs_quantities,
    v2_quantities,
    cvis_arr,
    cp_cov,
    bs_v2_cov,
    fluxes,
    index_mask,
    infos,
    expert_plot=False,
    save=False,
):
    """Normalize all observables by the appropriate factor proportional to
    the averaged fluxes and the number of holes."""
    bs_arr = bs_quantities["bs_arr"]
    v2_arr = v2_quantities["v2_arr"]

    v2 = v2_quantities["v2"]
    v2_cov = v2_quantities["v2_cov"]
    avar = v2_quantities["avar"]
    err_avar = v2_quantities["err_avar"]

    bs = bs_quantities["bs"]
    bs_cov = bs_quantities["bs_cov"]
    bs_var = bs_quantities["bs_var"]

    n_holes = index_mask.n_holes

    bs_arr_norm = bs_arr / np.mean(fluxes ** 3) * n_holes ** 3
    v2_arr_norm = v2_arr / np.mean(fluxes ** 2) * n_holes ** 2
    cvis_arr_norm = cvis_arr / np.mean(fluxes) * n_holes

    v2_norm = (v2 / np.mean(fluxes ** 2)) * n_holes ** 2
    v2_cov_norm = (v2_cov / np.mean(fluxes ** 4)) * n_holes ** 4

    bs_norm = bs / np.mean(fluxes ** 3) * n_holes ** 3

    avar_norm = avar / np.mean(fluxes ** 4) * n_holes ** 4
    err_avar_norm = err_avar / np.mean(fluxes ** 4) * n_holes ** 4

    try:
        cp_cov_norm = cp_cov / np.mean(fluxes ** 6) * n_holes ** 6
    except TypeError:
        cp_cov_norm = None

    bs_cov_norm = bs_cov / np.mean(fluxes ** 6) * n_holes ** 6
    bs_v2_cov_norm = np.real(bs_v2_cov / np.mean(fluxes ** 5) * n_holes ** 5)
    bs_var_norm = bs_var / np.mean(fluxes ** 6) * n_holes ** 6

    if expert_plot:
        plt.figure(figsize=(12, 6))
        plt.title("DIAGNOSTIC PLOTS - V2 - %s" % infos.target)
        plt.plot(v2_arr_norm[0], color="grey", alpha=0.2, label="V$^2$ dispersion")
        plt.plot(v2_arr_norm.T, color="grey", alpha=0.2)
        plt.plot(v2_norm, color="crimson", label="Raw V$^2$")
        plt.grid(alpha=0.2)
        plt.legend()
        plt.xlabel("# baselines")
        plt.ylabel("Raw visibilities")
        plt.tight_layout()

    # We compute the correlation matrix (to be used lated)
    v2_cor = cov2cor(v2_cov)[0]

    norm_quantities = {
        "bs_arr": bs_arr_norm,
        "v2_arr": v2_arr_norm,
        "cvis_arr": cvis_arr_norm,
        "bs": bs_norm,
        "v2_cov": v2_cov_norm,
        "bs_cov": bs_cov_norm,
        "avar": avar_norm,
        "err_avar": err_avar_norm,
        "cp_cov": cp_cov_norm,
        "bs_var": bs_var_norm,
        "bs_v2_cov": bs_v2_cov_norm,
        "v2_cor": v2_cor,
    }
    return v2_norm, norm_quantities


def _compute_cp(obs_result, obs_norm, infos, expert_plot=False):
    """Compute the closure phases array (across the cube) and averaged cp using
    the normalized bispectrum (see _normalize_all_obs()). Note that for the CP, the
    extracted quantities are computed after the normalisation."""
    bs = obs_norm["bs"]
    bs_arr = obs_norm["bs_arr"]

    cp = np.rad2deg(np.arctan2(bs.imag, bs.real))
    cp_arr = np.rad2deg([np.arctan2(i_bs.imag, i_bs.real) for i_bs in bs_arr])

    obs_result["cp"] = cp
    obs_norm["cp_arr"] = cp_arr

    if expert_plot:
        plt.figure(figsize=(12, 6))
        plt.title("DIAGNOSTIC PLOTS - CP - %s" % infos.target)
        plt.plot(cp_arr[0], color="grey", alpha=0.2, label="CP dispersion")
        plt.plot(cp_arr.T, color="grey", alpha=0.2)
        plt.plot(cp, color="crimson", label="Raw CP")
        plt.grid(alpha=0.2)
        plt.legend()
        plt.xlabel("# BS")
        plt.ylabel("Raw closure phases [deg]")
        plt.tight_layout()
    return obs_result


def _compute_t3_coord(mf, index_mask):
    """Compute the closure phases coordinates u1, u2, v1, v2
    and the equivalent maximum baselines (used for the
    spatial frequencies)."""
    n_bispect = index_mask.n_bispect
    bs2bl_ix = index_mask.bs2bl_ix

    u1coord = mf.u[bs2bl_ix[0, :]]
    v1coord = mf.v[bs2bl_ix[0, :]]
    u2coord = mf.u[bs2bl_ix[1, :]]
    v2coord = mf.v[bs2bl_ix[1, :]]
    u3coord = -(u1coord + u2coord)
    v3coord = -(v1coord + v2coord)

    t3_coord = {"u1": u1coord, "u2": u2coord, "v1": v1coord, "v2": v2coord}

    bl_cp = np.zeros(n_bispect)
    for k in range(n_bispect):
        B1 = np.sqrt(u1coord[k] ** 2 + v1coord[k] ** 2)
        B2 = np.sqrt(u2coord[k] ** 2 + v2coord[k] ** 2)
        B3 = np.sqrt(u3coord[k] ** 2 + v3coord[k] ** 2)
        bl_cp[k] = np.max([B1, B2, B3])  # [m]
    return t3_coord, bl_cp


def _compute_uncertainties(obs_result, obs_norm, naive_err=False):
    """Compute the uncertainties using the covariance matrix for the v2
    and the variance matrix for the closure phase. Can also compute the
    so called naive error using the standard deviation of the cp and v2
    quantities along the cube (`naive_err`=True, default=False)."""
    bs = obs_norm["bs"]
    bs_var = obs_norm["bs_var"]
    v2_cov = obs_norm["v2_cov"]
    cp_arr = obs_norm["cp_arr"]
    v2_arr = obs_norm["v2_arr"]

    if not naive_err:
        e_cp = np.rad2deg(np.sqrt(bs_var[1] / abs(bs) ** 2))
        e_v2 = np.sqrt(np.diag(v2_cov))
    else:
        e_cp = np.std(cp_arr, axis=0)
        e_v2 = np.std(v2_arr, axis=0)

    obs_result["e_cp"] = e_cp
    obs_result["e_vis2"] = e_v2
    return obs_result


def _compute_phs_piston(
    complex_bs, index_mask, method="Nelder-Mead", tol=1e-4, verbose=False, display=False
):
    """Compute the phase piston to determine the additional phase error due to
    the wavefront differences between holes."""
    n_holes = index_mask.n_holes
    n_baselines = index_mask.n_baselines

    bl2h_ix = index_mask.bl2h_ix

    ph_arr = complex_bs["vis_arr"]["phase"]

    n_ps = ph_arr.shape[0]
    # In the MAPPIT-style, we define the relationship between hole phases
    # (or phase slopes) and baseline phases (or phase slopes) by the
    # use of a matrix, fitmat.
    fitmat = np.zeros([n_holes, n_baselines + 1])
    for j in range(n_baselines):
        fitmat[bl2h_ix[0, j], j] = 1.0
    for j in range(n_baselines):
        fitmat[bl2h_ix[1, j], j] = -1.0
    fitmat[0, n_baselines] = 1.0

    # Firstly, fit to the phases by doing a weighted least-squares fit
    # to baseline phasors.
    phasors = np.exp(ph_arr * 1j)
    phasors_sum = np.sum(phasors, axis=0)
    ph_mn = np.arctan2(phasors_sum.imag, phasors_sum.real)
    ph_err = np.ones(len(ph_mn))

    for j in range(n_baselines):
        ph_err[j] = np.std(
            ((ph_arr[:, j] - ph_mn[j] + 3 * np.pi) % (2 * np.pi)) - np.pi
        )

    ph_err = ph_err / np.sqrt(n_ps)

    p0 = np.zeros(n_holes)
    res = minimize(phase_chi2, p0, method=method, tol=tol, args=(fitmat, ph_mn, ph_err))

    find_piston = None
    if verbose:
        print("\nDetermining piston using %s minimisation..." % method)
    if res.success:
        if verbose:
            print(
                "Phase Chi^2: ",
                phase_chi2(res.x, fitmat, ph_mn, ph_err) / (n_baselines - n_holes + 1),
            )
        find_piston = np.dot(res.x, fitmat)
    else:
        if verbose:
            cprint("Error calculating hole pistons...", "red")
            pass

    if display:
        plt.figure()
        plt.errorbar(
            np.arange(len(ph_mn)),
            np.rad2deg(ph_mn),
            yerr=np.rad2deg(ph_err),
            ls="None",
            ecolor="lightgray",
            marker=".",
        )
        if res.success:
            plt.plot(
                np.rad2deg(find_piston[:-1]),
                ls="--",
                color="orange",
                lw=1,
                label=method + " minimisation",
            )
            plt.legend()
        plt.grid(alpha=0.1)
        plt.ylabel(r"Mean phase [$\degree$]")
        plt.xlabel("# baselines")
        plt.tight_layout()
        plt.show(block=False)
    return fitmat


def _calc_weight_reg(x, y, weights):
    """Apply a linear regression (IDL function) to fit the hole phase and error."""
    reg = regress_noc(x, y, weights)
    sig = cov2cor(reg.cov)[1]
    hole_ph = reg.coeff
    hole_ph_err = sig * np.sqrt(reg.MSE)
    return hole_ph, hole_ph_err


def _compute_phs_error(complex_bs, fitmat, index_mask, npix, imsize=3):
    """Compute the phase error"""
    n_holes = index_mask.n_holes
    n_baselines = index_mask.n_baselines
    bl2h_ix = index_mask.bl2h_ix

    phs_arr = complex_bs["phs"]["value"]
    phserr_arr = complex_bs["phs"]["err"]
    v2_arr = complex_bs["vis_arr"]["squared"]

    n_ps = phs_arr.shape[1]

    # Fit to the phase slopes using weighted linear regression.
    # Normalisation:  hole_phs was in radians per Fourier pixel.
    # Convert to phase slopes in pixels.
    phs_arr = phs_arr / 2.0 / np.pi * npix
    hole_phs = np.zeros([2, n_ps, n_holes])
    hole_err_phs = np.zeros([2, n_ps, n_holes])

    for j in range(n_baselines):
        fitmat[bl2h_ix[1, j], j] = 1

    fitmat = fitmat / 2.0
    fitmat = fitmat[:, 0:n_baselines]

    err, err_bias = np.zeros_like(v2_arr), np.zeros_like(v2_arr)
    for j in range(n_ps):
        y, weight = phs_arr[0, j, :], phserr_arr[0, j, :]
        hole_phs[0, j, :], hole_err_phs[0, j, :] = _calc_weight_reg(fitmat, y, weight)
        y, weight = phs_arr[1, j, :], phserr_arr[1, j, :]
        hole_phs[1, j, :], hole_err_phs[1, j, :] = _calc_weight_reg(fitmat, y, weight)

        tmp1 = hole_phs[0, j, bl2h_ix[0, :]] - hole_phs[0, j, bl2h_ix[1, :]]
        tmp2 = hole_phs[1, j, bl2h_ix[0, :]] - hole_phs[1, j, bl2h_ix[1, :]]
        err[j, :] = tmp1 ** 2 + tmp2 ** 2
        err_bias[j, :] = (
            hole_err_phs[0, j, bl2h_ix[0, :]] - hole_err_phs[0, j, bl2h_ix[1, :]]
        ) ** 2 + (
            hole_err_phs[1, j, bl2h_ix[0, :]] - hole_err_phs[1, j, bl2h_ix[1, :]]
        ) ** 2

    predictor = np.zeros_like(v2_arr)
    for j in range(n_baselines):
        predictor[:, j] = err[:, j] - np.mean(err_bias[:, j])

    # imsize is λ/hole_diameter in pixels. A factor of 3.0 was only
    # roughly correct based on simulations 2.5 seems to be better based
    # on real data (NB there is no window size adjustment here).
    phs_v2corr = np.zeros(n_baselines)
    for j in range(n_baselines):
        phs_v2corr[j] = np.mean(np.exp(-2.5 * predictor[:, j] / imsize ** 2))

    return phs_v2corr


def _add_infos_header(infos, hdr, mf, pa, filename, maskname, npix):
    """Save important informations and some parts of the original header."""
    infos["pixscale"] = mf.pixelSize
    infos["pa"] = pa
    infos["filename"] = filename
    infos["maskname"] = maskname
    infos["isz"] = npix

    # HACK: astropy _HeaderCommentaryCards are registered as mappings,
    # so munch tries to access their keys, leading to attribute error
    # to prevent this, we remove commentary cards as a temporary fix.
    # (As of June 23 2021, with astropy version 4.2.1)
    # See:
    # https://github.com/SydneyAstrophotonicInstrumentationLab/AMICAL/issues/31
    # https://github.com/astropy/astropy/issues/11866
    hdr_commentary_keys = fits.Card._commentary_keywords
    hdr = hdr.copy()
    for key in hdr_commentary_keys:
        hdr.remove(key, ignore_missing=True, remove_all=True)

    # Now that header is compatible with munch, we add it to infos
    infos["hdr"] = hdr

    # Save keys of the original header (as needed):
    add_keys = ["TELESCOP", "DATE-OBS", "MJD-OBS", "OBSERVER"]
    if infos.orig != "SimulatedData":
        for keys in add_keys:
            infos[keys.lower()] = hdr.get(keys)
    else:
        # For simulated data, add keys only if they exist
        # (missing keys will be filled later if needed)
        for keys in add_keys:
            try:
                infos[keys.lower()] = hdr[keys]
            except KeyError:
                pass
    return infos


def produce_result_pdf(figdir, filename):
    # Call the PdfFileMerger
    mergedObject = PdfFileMerger()

    for fileNumber in range(7):
        ifile = figdir + filename + "_" + str(fileNumber + 1) + ".pdf"
        mergedObject.append(PdfFileReader(ifile, "rb"))
        os.remove(ifile)

    # Write all the files into a file which is named as shown below
    mergedObject.write(figdir + filename + "_DIAGNOSTIC_PLOTS.pdf")
    return 0


def extract_bs(
    cube,
    filename,
    maskname,
    filtname=None,
    targetname=None,
    instrum=None,
    bs_multi_tri=False,
    peakmethod="gauss",
    hole_diam=0.8,
    cutoff=1e-4,
    fw_splodge=0.7,
    naive_err=False,
    n_wl=3,
    n_blocks=0,
    theta_detector=0,
    scaling_uv=1,
    i_wl=None,
    unbias_v2=True,
    compute_cp_cov=True,
    expert_plot=False,
    save=False,
    verbose=False,
    display=True,
):
    """Compute the bispectrum (bs, v2, cp, etc.) from a data cube.

    Parameters:
    -----------

    `cube` {array}:
        Cleaned and checked data cube ready to extract NRM data,\n
    `filename` {array}:
        Name of the file containing the datacube (to keep track on it),\n
    `maskname` {str}:
        Name of the mask,\n
    `filtname` {str}:
        By default, checks the header to extract the filter, if not in header
        uses filtname instead (e.g.: F430M, F480M),\n
    `targetname` {str}:
        By default, checks the header to extract the target, if not in header
        uses target_name instead,\n
    `bs_multi_tri` {bool}:
        Use the multiple triangle technique to compute the bispectrum
        (default: False),\n
    `peakmethod` {str}:
        3 methods are used to sample to u-v space: 'fft' uses fft between individual holes to compute
        the expected splodge position; 'square' compute the splodge in a square using the expected
        fraction of pixel to determine its weight; 'gauss' considers a gaussian splodge (with a gaussian
        weight) to get the same splodge side for each n(n-1)/2 baselines,\n
    `fw_splodge` {float}:
        Relative size of the splodge used to compute multiple triangle indices and the fwhm
        of the 'gauss' technique,\n
    `naive_err` {bool}:
        If True, the uncertainties are computed using the std of the overall
        cvis or bs array. Otherwise, the uncertainties are computed using
        covariance matrices,\n
    `n_wl` {int}:
        Number of elements to sample the spectral filters (default: 3),\n
    `n_blocks` {float}:
        Number of separated blocks use to split the data cube and get more
        accurate uncertainties (default: 0, n_blocks = n_ps),\n
    `theta_detector`: {float}
        Angle [deg] to rotate the mask compare to the detector (if the mask is not
        perfectly aligned with the detector, e.g.: VLT/VISIR) ,\n
    `i_wl`: {int}
        Only used for IFU data (e.g.: IFS/SPHERE), select the desired spectral channel
        to retrieve the appropriate wavelength and mask positions, \n
    `unbias_v2`: {bool}
        If True, the squared visibilities are unbiased using the Fourier base, \n
    `targetname` {str}:
        Name of the target to save in oifits file (if not in header of the
        cube),\n
    `verbose` {bool}:
        If True, print usefull informations during the process.\n
    `display` {bool}:
        If True, display all figures,\n

    Returns:
    --------
    `obs_result` {class object}:
        Return all interferometric observables (.vis2, .e_vis2, .cp, .e_cp, etc.), information relative
        to the used mask (.mask), the computed matrices and statistic (.matrix)
        and the important information (.infos). The .mask, .infos and .matrix are also class with
        various quantities (see .mask.__dict__.keys()).
    """
    if verbose:
        cprint("\n-- Starting extraction of observables --", "cyan")
    start_time = time.time()

    figdir = "amical_save_fig/"
    if save:
        if not os.path.exists(figdir):
            os.mkdir(figdir)

    with fits.open(filename) as hdu:
        hdr = hdu[0].header

    infos = _check_input_infos(
        hdr, targetname=targetname, filtname=filtname, instrum=instrum, verbose=False
    )

    if "INSTRUME" not in hdr.keys():
        hdr["INSTRUME"] = infos["instrument"]
    # 1. Open the data cube and perform a series of roll (both axis) to avoid
    # grid artefact (negative fft values).
    # ------------------------------------------------------------------------
    ft_arr, n_ps, npix = _construct_ft_arr(cube)

    # Number of aperture in the mask
    try:
        n_holes = len(get_mask(infos.instrument, maskname))
    except TypeError:
        return None

    # 2. Determine the number of different baselines (bl), bispectrums (bs) or
    # covariance matrices (cov) and associates each holes as couple for bl or
    # triplet for bs (or cp) using compute_index_mask function (see ami_function.py).
    # ------------------------------------------------------------------------
    index_mask = compute_index_mask(n_holes)

    n_baselines = index_mask.n_baselines

    closing_tri = _format_closing_triangle(index_mask)

    # 3. Compute the match filter mf
    # ------------------------------------------------------------------------
    mf = make_mf(
        maskname,
        infos.instrument,
        infos.filtname,
        npix,
        peakmethod=peakmethod,
        fw_splodge=fw_splodge,
        n_wl=n_wl,
        cutoff=cutoff,
        hole_diam=hole_diam,
        scaling=scaling_uv,
        theta_detector=theta_detector,
        i_wl=i_wl,
        display=display,
        save=save,
        figdir=figdir,
        filename=filename,
    )

    figname = figdir + Path(filename).stem
    ifig = 2
    if save:
        plt.savefig(figname + "_%i.pdf" % ifig)
    ifig += 1

    if mf is None:
        return None

    # We store the principal results in the new dictionnary to be save at the end
    obs_result = {"u": mf.u, "v": mf.v, "wl": mf.wl, "e_wl": mf.e_wl}

    # 4. Compute indices for the multiple triangle technique (tri_pix function)
    # -------------------------------------------------------------------------
    l_B = np.sqrt(mf.u ** 2 + mf.v ** 2)  # Length of different bl [m]
    minbl = np.min(l_B)

    if n_holes >= 15:
        sampledisk_r = minbl / 2 / mf.wl * mf.pixelSize * npix * 0.9
    else:
        sampledisk_r = minbl / 2 / mf.wl * mf.pixelSize * npix * fw_splodge

    if bs_multi_tri:
        closing_tri_pix = tri_pix(npix, sampledisk_r, display=display, verbose=verbose)
    else:
        closing_tri_pix = None

    # 5. Display the power spectrum of the first frame to check the computed
    # positions of the peaks.
    # ------------------------------------------------------------------------
    if display:
        _show_complex_ps(ft_arr)
        if save:
            plt.savefig(figname + "_%i.pdf" % ifig)
        ifig += 1

        _show_peak_position(ft_arr, n_baselines, mf, maskname, peakmethod)
        if save:
            plt.savefig(figname + "_%i.pdf" % ifig)
        ifig += 1

    if verbose:
        print("\nFilename: %s" % filename.split("/")[-1])
        print("# of frames = %i" % n_ps)

    n_blocks = _set_good_nblocks(n_blocks, n_ps)

    # 6. Extract the complex quantities from the fft_arr (complex vis, bispectrum,
    # phase, etc.)
    # ------------------------------------------------------------------------
    fringe_peak = give_peak_info2d(mf, n_baselines, npix, npix)

    if verbose:
        print("\nCalculating V^2 and BS...")

    complex_bs = _compute_complex_bs(
        ft_arr,
        index_mask,
        fringe_peak,
        mf,
        dark_ps=None,
        closing_tri_pix=closing_tri_pix,
        bs_multi_tri=bs_multi_tri,
    )

    cvis_arr = complex_bs["vis_arr"]["complex"]
    v2_arr = complex_bs["vis_arr"]["squared"]
    bs_arr = complex_bs["bs_arr"]
    fluxes = complex_bs["fluxes"]

    # 7. Compute correlated noise and bias at the peak position
    # ---------------------------------------------------------
    bias, dark_bias, autocor_noise = _compute_corr_noise(
        complex_bs, ft_arr, fringe_peak
    )

    v2_arr_unbiased, bias_arr = _unbias_v2_arr(
        v2_arr, npix, fringe_peak, bias, dark_bias, autocor_noise, unbias=unbias_v2
    )

    # 8. Turn Arrays into means and covariance matrices
    # -------------------------------------------------
    v2_quantities = _compute_v2_quantities(v2_arr_unbiased, bias_arr, n_blocks)

    bs_quantities = _compute_bs_quantities(
        bs_arr, v2_quantities["v2"], fluxes, index_mask, n_blocks
    )

    bs_v2_cov = _compute_bs_v2_cov(
        bs_arr, v2_arr_unbiased, v2_quantities["v2"], bs_quantities["bs"], index_mask
    )

    if compute_cp_cov:
        cp_cov = _compute_cp_cov(bs_arr, bs_quantities["bs"], index_mask)
    else:
        cp_cov = None

    # 9. Now normalize all extracted observables
    vis2_norm, obs_norm = _normalize_all_obs(
        bs_quantities,
        v2_quantities,
        cvis_arr,
        cp_cov,
        bs_v2_cov,
        fluxes,
        index_mask,
        infos,
        expert_plot=True,
    )
    if save:
        plt.savefig(figname + "_%i.pdf" % ifig)
    ifig += 1

    obs_result["vis2"] = vis2_norm

    # 10. Now we compute the cp quantities and store them with the other observables
    obs_result = _compute_cp(obs_result, obs_norm, infos, expert_plot=True)

    if save:
        plt.savefig(figname + "_%i.pdf" % ifig)
    ifig += 1

    if display:
        _show_norm_matrices(obs_norm, expert_plot=expert_plot)
        if save:
            plt.savefig(figname + "_%i.pdf" % ifig)
        ifig += 1

    t3_coord, bl_cp = _compute_t3_coord(mf, index_mask)
    bl_v2 = np.sqrt(mf.u ** 2 + mf.v ** 2)
    obs_result["bl"] = bl_v2
    obs_result["bl_cp"] = bl_cp

    # 11. Now we compute the uncertainties using the covariance matrix (for v2)
    # and the variance matrix for the cp.
    obs_result = _compute_uncertainties(obs_result, obs_norm, naive_err=naive_err)

    # 12. Compute scaling error due to phase error (piston) between holes.
    fitmat = _compute_phs_piston(complex_bs, index_mask, display=expert_plot)
    phs_v2corr = _compute_phs_error(complex_bs, fitmat, index_mask, npix)
    obs_norm["phs_v2corr"] = phs_v2corr

    # 13. Compute the absolute oriention (North-up, East-left)
    # ------------------------------------------------------------------------
    pa = compute_pa(hdr, n_ps, display=display, verbose=verbose)

    # Compile informations in the storage infos class
    infos = _add_infos_header(infos, hdr, mf, pa, filename, maskname, npix)

    mask = {
        "bl2h_ix": index_mask.bl2h_ix,
        "bs2bl_ix": index_mask.bs2bl_ix,
        "closing_tri": closing_tri,
        "xycoord": mf.xy_coords,
        "n_holes": index_mask.n_holes,
        "n_baselines": index_mask.n_baselines,
        "t3_coord": t3_coord,
    }

    # Finally we store the computed matrices (cov, var, arr, etc,), the informations
    # and the mask parameters to the final output.
    obs_result["mask"] = mask
    obs_result["infos"] = infos
    obs_result["matrix"] = obs_norm

    t = time.time() - start_time
    m = t // 60

    if save:
        produce_result_pdf(figdir, Path(filename).stem)

    if verbose:
        cprint("\nDone (exec time: %d min %2.1f s)." % (m, t - m * 60), color="magenta")
    return dict2class(obs_result)
