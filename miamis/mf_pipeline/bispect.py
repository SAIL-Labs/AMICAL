# -*- coding: utf-8 -*-
"""
@author: Anthony Soulain (University of Sydney)

--------------------------------------------------------------------
MIAMIS: Multi-Instruments Aperture Masking Interferometry Software
--------------------------------------------------------------------

Matched filter sub-pipeline method.

Compute bispectrum for a given fits file (adapted from bispect.pro 
and calc_bispect.pro).

-------------------------------------------------------------------- 
"""

import time
import warnings

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from munch import munchify as dict2class
from scipy.optimize import minimize
from termcolor import cprint

from miamis.core import clean_data
from miamis.getInfosObs import GetMaskPos
from miamis.tools import cov2cor

from .ami_function import (GivePeakInfo2d, bs_multiTriangle, index_mask,
                           make_mf, phase_chi2, tri_pix)
from .idl_function import dblarr, dist, regress_noc

warnings.filterwarnings("ignore")


def extract_bs_mf(filename, maskname, filtname=None, targetname=None, isz=256, r1=100, dr=20, clean=True,
                  checkrad=False, bs_MultiTri=False, n_blocks=0, peakmethod=True, hole_diam=0.8,
                  cutoff=1e-4, fw_splodge=0.7, naive_err=False, n_wl=3, verbose=False, display=True,):
    """Compute bispectrum (bs, v2, cp, etc.) from a data cube.

    Parameters:
    -----------

    `filename` {str}:
        filename of the cube,\n
    `maskname` {str}:
        Name of the mask,\n
    `filtname` {str}:
        By default, checks the header to extract the filter, if not in header
        uses filtname instead (e.g.: F430M, F480M),\n
    `target_name` {str}:
        By default, checks the header to extract the target, if not in header
        uses target_name instead,\n
    `BS_MultiTri` {bool}:
        Use the multiple triangle technique to compute the bispectrum
        (default: False),\n
    `n_blocks` {float}:
        Number of separated blocks use to split the data cube and get more
        accurate uncertainties (default: 0, n_blocks = n_ps),\n
    `peakmethod` {bool}:
        If True, perform FFTs to compute the peak position in the Fourier
        space (default). Otherwise, set the u-v peak sampled into 4 pixels,\n
    `fw_splodge` {float}:
        Relative size of the splodge used to compute mutliple triangle indices,\n
    `naive_err` {bool}:
        If True, the uncertainties are computed using the std of the overall
        cvis or bs array. Otherwise, the uncertainties are computed using
        covariance matrices,\n
    `n_wl` {int}:
        Number of elements to sample the spectral filters (default: 3),\n
    `targetname` {str}:
        Name of the target to save in oifits file (if not in header of the
        cube),\n
    `verbose` {bool}:
        If True, print usefull informations during the process.\n
    `display` {bool}:
        If True, display all figures,\n
    `zoom` {bool}:
        If True, display one zoomed splodge of the ps.

    Returns:
    --------
    `res` {class object}:
        Return all interferometric observables and informations as a class
        object res (res.v2, v2_sig, cp, cp_sig, u, v, etc.)
    """

    start_time = time.time()

    # 1. Open the data cube and perform a series of roll (both axis) to avoid
    # grid artefact (negative fft values).
    # ------------------------------------------------------------------------

    hdu = fits.open(filename)
    data = hdu[0].data
    hdr = hdu[0].header

    try:
        target = targetname  # hdr['OBJECT']
    except KeyError:
        target = targetname  # if cube if a simulated target use target_name
        cprint("Warning: OBJECT is not in the header, targetname is used.", "green")
    try:
        instrument = hdr["INSTRUME"]
    except KeyError:
        cprint("Error: no instrument name is included in the header.", "red")
        return None

    if filtname is None:
        try:
            filtname = hdr["FILTER"]
        except KeyError:
            cprint("Warning: FILTER is not in the header, filtname is used.", "green")
            return None

    # Clean data: centering, apodise, sky-substraction, rolling.
    if clean:
        cube = clean_data(data, isz=isz, r1=r1, dr=dr, checkrad=checkrad)
    else:
        cube = data.copy()

    if cube is None:
        return None

    npix = cube.shape[1]
    if cube.shape[1] % 2 == 1:
        cube = np.array([im[:-1, :-1] for im in cube])

    npix = cube.shape[1]
    cube = np.roll(np.roll(cube, npix // 2, axis=1), npix // 2, axis=2)

    ft_arr = np.fft.fft2(cube)
    save_ft = ft_arr[0].copy()

    i_ps = ft_arr.shape
    n_ps = i_ps[0]
    dim1 = i_ps[1]
    dim2 = i_ps[2]

    # Number of aperture in the mask
    n_holes = len(GetMaskPos(instrument, maskname))

    # 2. Determine the number of different baselines (bl), bispectrums (bs) or
    # covariance matrices (cov) and associates each holes as couple for bl or
    # triplet for bs (or cp) using index_mask function (see AMI_function.py).
    # ------------------------------------------------------------------------
    ind_mask_res = index_mask(n_holes)

    n_baselines = ind_mask_res[0]
    n_bispect = ind_mask_res[1]
    n_cov = ind_mask_res[2]
    bl2h_ix = ind_mask_res[4]
    bs2bl_ix = ind_mask_res[5]
    bl2bs_ix = ind_mask_res[6]
    bscov2bs_ix = ind_mask_res[7]

    closing_tri = []
    for i_bs in range(len(bs2bl_ix.T)):
        tmp = []
        for x in bs2bl_ix.T[i_bs]:
            tmp.extend(bl2h_ix.T[x])
        closing_tri.append(list(set(tmp)))

    # 3. Compute the match filter mf (make_mf function) which give the
    # indices of the peak positions (mf.pvct) and the associated gains (mf.gvct)
    # in the image. Contains also the u-v coordinates, wavelengths informations,
    # holes mask positions (mf.xy_coords), centred mf (mf.cpvct, mf.gpvct), etc.
    # ------------------------------------------------------------------------

    mf = make_mf(maskname, instrument, filtname, npix, peakmethod=peakmethod,
                 n_wl=n_wl, cutoff=cutoff, hole_diam=hole_diam, display=display,)

    # 4. Compute indices for the multiple triangle technique (tri_pix function)
    # -------------------------------------------------------------------------
    l_B = np.sqrt(mf.u ** 2 + mf.v ** 2)  # Length of different bl [m]
    minbl = np.min(l_B)

    if n_holes >= 15:
        sampledisk_r = minbl / 2 / mf.wl * mf.pixelSize * dim1 * 0.9
    else:
        sampledisk_r = minbl / 2 / mf.wl * mf.pixelSize * dim1 * fw_splodge

    if bs_MultiTri:
        closing_tri_pix = tri_pix(
            dim1, sampledisk_r, display=display, verbose=verbose)

    # 5. Display the power spectrum of the first frame to check the computed
    # positions of the peaks.
    # ------------------------------------------------------------------------

    if display:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Real part")
        plt.imshow(ft_arr[0].real, cmap="gist_stern", origin="lower")
        plt.subplot(1, 2, 2)
        plt.title("Imaginary part")
        plt.imshow(ft_arr[0].imag, cmap="gist_stern", origin="lower")
        plt.tight_layout()

        plt.figure(figsize=(6, 6))
        plt.title("Power spectrum")
        plt.imshow(np.fft.fftshift(
            abs(ft_arr[0])), cmap="gist_stern", origin="lower")
        plt.tight_layout()

        j = 0
        x, y = np.arange(dim1), np.arange(dim2)
        X, Y = np.meshgrid(x, y)

        # Compute determined peak position in the PS
        lX, lY, lC = [], [], []

        # l_peak = GivePeakInfo2d(mf, n_baselines, dim1, dim2)

        for j in range(n_baselines):
            l_x = X.ravel()[mf.pvct[mf.ix[0, j]: mf.ix[1, j]]]
            l_y = Y.ravel()[mf.pvct[mf.ix[0, j]: mf.ix[1, j]]]
            g = mf.gvct[mf.ix[0, j]: mf.ix[1, j]]

            peak = [[l_y[k], l_x[k], g[k]] for k in range(len(l_x))]

            for x in peak:
                lX.append(x[1])
                lY.append(x[0])
                lC.append(x[2])

        i_peak = 2
        i_fram = 0
        ft_frame = ft_arr[i_fram]
        ps = ft_frame.real

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_title("Peak #%i" % (i_peak))
        im = ax.imshow(ps, cmap="gist_stern", origin="lower")
        sc = ax.scatter(lX, lY, c=lC, s=3, cmap="viridis")
        cb = fig.colorbar(im)
        cb2 = fig.colorbar(sc)
        cb.set_label("Power Spectrum intensity")
        cb2.set_label("Relative weight [%]")
        plt.subplots_adjust(
            top=0.965, bottom=0.035, left=0.019, right=0.981, hspace=0.2, wspace=0.2
        )
        if checkrad:
            plt.show(block=True)

    # 6. Initialize arrays
    # ------------------------------------------------------------------------
    aveps = np.zeros([dim1, dim2])
    avedps = np.zeros([dim1, dim2])

    cvis = np.zeros(n_baselines).astype(complex)
    rvis = np.zeros(n_baselines)
    ivis = np.zeros(n_baselines)

    if verbose:
        print("\nFilename: %s" % filename.split("/")[-1])
        print("# of frames = %i" % n_ps)

    if (n_blocks == 0) or (n_blocks == 1):
        if verbose:
            cprint("! Warning: nblocks == 0 -> n_blocks set to n_ps", "green")
        n_blocks = n_ps
    elif n_blocks > n_ps:
        if verbose:
            cprint("------------------------------------", "green")
            cprint("! Warning: nblocks > n_ps -> n_blocks set to n_ps", "green")
        n_blocks = n_ps

    dark_v2 = np.zeros(n_baselines)
    v2_arr = np.zeros([n_ps, n_baselines])
    ph_arr = np.zeros([n_ps, n_baselines])
    phs_arr = np.zeros([2, n_ps, n_baselines])
    phserr_arr = np.zeros([2, n_ps, n_baselines])
    biasmn = np.zeros(n_baselines)
    bs_arr = np.zeros([n_ps, n_bispect]).astype(complex)
    cvis_arr = np.zeros([n_ps, n_baselines]).astype(complex)
    fluxes = np.zeros(n_ps)

    # 7. Fill up arrays
    # ------------------------------------------------------------------------
    dark_ps = np.zeros([dim1, dim2])

    List_peak = GivePeakInfo2d(mf, n_baselines, dim1, dim2)

    if verbose:
        print("\nCalculating V^2 and BS...")

    # Start to go through the cube
    for i in range(n_ps):
        ft_frame = ft_arr[i]
        ps = np.abs(ft_frame) ** 2

        if dark_ps is not None and (len(dark_ps.shape) == 3):
            dps = dark_ps[i]
        elif dark_ps is not None and (len(dark_ps.shape) == 2):
            dps = dark_ps
        else:
            dps = np.zeros([dim1, dim2])

        avedps += dps  # Cumulate ps (dark) to perform an average at the end
        aveps += ps  # Cumulate ps to perform an average at the end

        # Extract complex visibilities of each fringe peak (each indices are
        # computed using make_mf function above)
        for j in range(n_baselines):
            pix = List_peak[j][:, 0].astype(
                int), List_peak[j][:, 1].astype(int)
            gain = List_peak[j][:, 2]

            dark_v2[j] = np.sum(gain ** 2 * dps[pix])
            cvis[j] = np.sum(gain * ft_frame[pix])

            ftf1 = np.roll(ft_frame, 1, axis=0)
            ftf2 = np.roll(ft_frame, -1, axis=0)
            dummy = np.sum(
                ft_frame[pix] * np.conj(ftf1[pix]) +
                np.conj(ft_frame[pix]) * ftf2[pix]
            )

            phs_arr[0, i, j] = np.arctan2(dummy.imag, dummy.real)
            phserr_arr[0, i, j] = 1 / abs(dummy)  # ;NB only a relative error

            ftf1 = np.roll(ft_frame, 1, axis=1)
            ftf2 = np.roll(ft_frame, -1, axis=1)
            dummy = np.sum(
                ft_frame[pix] * np.conj(ftf1[pix]) +
                np.conj(ft_frame[pix]) * ftf2[pix]
            )

            phs_arr[1, i, j] = np.arctan2(dummy.imag, dummy.real)
            phserr_arr[1, i, j] = 1 / abs(dummy)  # ;NB only a relative error

        # Correct for overlapping baselines
        rvis = cvis.real
        ivis = cvis.imag

        rvis = np.dot(mf.rmat, rvis)
        ivis = np.dot(mf.imat, ivis)

        cvis2 = rvis + ivis * 1j
        cvis_arr[i, :] = cvis2

        ph_arr[i, :] = np.arctan2(cvis2.imag, cvis2.real)

        # Calculate square visibilities (correct for bias later)
        v2_arr[i] = np.abs(cvis2) ** 2 - dark_v2

        # Calculate Bispectrum
        if not bs_MultiTri:
            cvis_1 = cvis[bs2bl_ix[0, :]]
            cvis_2 = cvis[bs2bl_ix[1, :]]
            cvis_3 = cvis[bs2bl_ix[2, :]]
            bs_arr[i, :] = cvis_1 * cvis_2 * np.conj(cvis_3)
        else:
            bs_arr = bs_multiTriangle(i, bs_arr, ft_frame, bs2bl_ix, mf,
                                      closing_tri_pix,)
            # !! This is the April2013 code update which implements
            # the pixel-triangle-loops to explicitly populate the bispectrum
            # Firstly calculate up the complex spectrum multiplied by the
            # match filter (mfilter_spec)
            pass

        fluxes[i] = abs(ft_arr[i, 0, 0]) - np.sqrt(dps[0, 0])

    ps = aveps / n_ps
    dps = avedps / n_ps

    # 8. Compute correlated noise and bias at the peak position
    # ---------------------------------------------------------

    signal_map = np.zeros([dim1, dim2])  # 2-D array where pixel=1 if signal
    # inside (i.e.: peak in the fft)
    for j in range(n_baselines):
        pix = List_peak[j][:, 0].astype(int), List_peak[j][:, 1].astype(int)
        signal_map[pix] = 1.0

    # Compute the center-symetric counterpart of the signal
    signal_map += np.roll(np.roll(np.rot90(np.rot90(signal_map)),
                                  1, axis=0), 1, axis=1)

    # Compute the distance map centered on each corners (origin in the fft)
    corner_dist_map = dist(dim1)

    # Indicate signal if the central peak
    signal_map[corner_dist_map < dim1 / 16.0] = 1.0

    for i in range(3):
        signal_cond = signal_map == 0
        bias = np.median(ps[signal_cond])
        dark_bias = np.median(dps[signal_cond])
        signal_map[ps >= 3 * bias] = 1.0

    signal_cond = signal_map != 0  # i.e. where there is signal.

    # Now we know where the signal is, we can find the 'f' bias that
    # includes the correlation between neighbouring terms in each
    # ft_frame...

    autocor_noise = np.zeros([dim1, dim2])

    for i in range(n_ps):
        ft_frame = ft_arr[i].copy()
        ft_frame[signal_cond] = 0.0
        autocor_noise += np.fft.fft2(np.abs(np.fft.ifft2(ft_frame)) ** 2).real
    autocor_noise /= n_ps

    for j in range(n_baselines):
        fringe_peak = dblarr(dim1, dim2)
        pix = List_peak[j][:, 0].astype(int), List_peak[j][:, 1].astype(int)
        fringe_peak[pix] = List_peak[j][:, 2]

        autocor_mf = np.fft.ifft2(np.abs(np.fft.ifft2(fringe_peak)) ** 2).real
        biasmn[j] = (
            np.sum(autocor_mf * autocor_noise)
            * bias
            / autocor_noise[0, 0]
            * dim1
            * dim2
        )
        v2_arr[:, j] -= biasmn[j]

    # Problem: we've subtracted the dark noise as well here (already
    # subtracted as dark_v2 above), so we must add it back. It might look like
    # we've subtracted then added a similar thing (which is true) but it's not
    # the same, as the structure of dark_ps goes into the different values of
    # dark_v2, but dark_bias here is constant for all baselines.

    for j in range(n_baselines):
        v2_arr[:, j] += np.sum(List_peak[j][:, 2] ** 2) * dark_bias

    # 9. Turn Arrays into means and covariance matrices
    # ------------------------------------------------------------------------

    v2 = dblarr(n_baselines)
    v2_cov = dblarr(n_baselines, n_baselines)
    bs = dblarr(n_bispect).astype(complex)
    bs_var = dblarr(2, n_bispect)
    bs_cov = dblarr(2, n_cov)
    bs_v2_cov = dblarr(n_baselines, n_holes - 2)
    cp_var = dblarr(n_bispect)

    if verbose:
        print("Calculating mean V^2 and variance...")

    v2 = np.mean(v2_arr, axis=0)  # Computed unnormalized squared visibility

    v2diff = dblarr(n_blocks, n_baselines)

    for j in range(n_baselines):
        for k in range(n_blocks):
            ind1 = k * n_ps // n_blocks
            ind2 = (k + 1) * n_ps // (n_blocks)
            v2diff[k, j] = np.mean(v2_arr[ind1:ind2, j]) - v2[j]

    for j in range(n_baselines):
        for k in range(n_baselines):
            num = np.sum(v2diff[:, j] * v2diff[:, k])
            v2_cov[j, k] = num / (n_blocks - 1) / n_blocks

    # Now, for the case where we have coherent integration over splodges,
    # it is easy to calculate the bias in the variance. (note that the
    # variance of the bias is simply the square of it's mean)

    x = np.arange(n_baselines)
    avar = v2_cov[x, x] * n_ps - biasmn ** 2 * (1 + (2.0 * v2) / biasmn)
    err_avar = np.sqrt(
        2.0 / n_ps * (v2_cov[x, x]) ** 2 * n_ps ** 2 +
        4.0 * v2_cov[x, x] * biasmn ** 2
    )  # Assumes no error in biasmn

    if verbose:
        print("Calculating mean bispectrum and variance...")

    bs = np.mean(bs_arr, axis=0)
    # Bispectral bias subtraction. This assumes that the matched filter has
    # been correctly normalised...

    subtract_bs_bias = False
    if subtract_bs_bias:
        bs_bias = (
            v2[bs2bl_ix[0, :]]
            + v2[bs2bl_ix[1, :]]
            + v2[bs2bl_ix[2, :]]
            + np.mean(fluxes)
        )
        bs = bs - bs_bias

    temp = np.zeros(n_blocks).astype(complex)
    for j in range(n_bispect):
        # temp is the complex difference from the mean, shifted so that the
        # real axis corresponds to amplitude and the imaginary axis phase.

        temp2 = (bs_arr[:, j] - bs[j]) * np.conj(bs[j])
        for k in range(n_blocks):
            ind1 = k * n_ps // n_blocks
            ind2 = (k + 1) * n_ps // (n_blocks)
            temp[k] = np.mean(temp2[ind1:ind2])

        num_real = np.sum(np.real(temp2) ** 2) / n_blocks / (n_blocks - 1)
        num_imag = np.sum(np.imag(temp2) ** 2) / n_blocks / (n_blocks - 1)

        bs_var[0, j] = num_real / (np.abs(bs[j]) ** 2)
        bs_var[1, j] = num_imag / (np.abs(bs[j]) ** 2)

    if verbose:
        print("Calculating covariance between power and bispectral amplitude...")
    # This complicated thing below calculates the dot product between the
    # bispectrum point and its error term ie (x . del_x)/|x| and
    # multiplies this by the power error term. Note that this is not the
    # same as using absolute value, and that this sum should be zero where
    # |bs| is zero within errors.

    for j in range(n_baselines):
        for k in range(n_holes - 2):
            temp = bs_arr[:, bl2bs_ix[j, k]] - bs[bl2bs_ix[j, k]]
            bs_real_tmp = np.real(temp * np.conj(bs[bl2bs_ix[j, k]]))
            diff_v2 = v2_arr[:, j] - v2[j]
            norm = abs(bs[bl2bs_ix[j, k]]) / (n_ps - 1.0) / n_ps
            norm_bs_v2_cov = np.sum(bs_real_tmp * diff_v2) / norm
            bs_v2_cov[j, k] = norm_bs_v2_cov

    if verbose:
        print("Calculating the bispectral covariances...")

    for j in range(n_cov):
        temp1 = (bs_arr[:, bscov2bs_ix[0, j]] -
                 bs[bscov2bs_ix[0, j]]) * np.conj(bs[bscov2bs_ix[0, j]])
        temp2 = (bs_arr[:, bscov2bs_ix[1, j]] -
                 bs[bscov2bs_ix[1, j]]) * np.conj(bs[bscov2bs_ix[1, j]])
        denom = (abs(bs[bscov2bs_ix[0, j]]) *
                 abs(bs[bscov2bs_ix[1, j]]) * (n_ps - 1) * n_ps)

        bs_cov[0, j] = np.sum(np.real(temp1) * np.real(temp2)) / denom
        bs_cov[1, j] = np.sum(np.imag(temp1) * np.imag(temp2)) / denom

    cp_cov = dblarr(n_bispect, n_bispect)

    for i in range(n_bispect):
        for j in range(n_bispect):
            temp1 = (bs_arr[:, i] - bs[i]) * np.conj(bs[i])
            temp2 = (bs_arr[:, j] - bs[j]) * np.conj(bs[j])
            denom = abs(bs[i]) ** 2 * abs(bs[j]) ** 2 * (n_ps - 1) * n_ps
            cp_cov[i, j] = np.sum(np.imag(temp1) * np.imag(temp2)) / denom

    if display:
        plt.figure(figsize=(3, 6))
        plt.title("Cov matrix BS vs. $V^2$")
        plt.imshow(bs_v2_cov, origin="upper")
        plt.ylabel("# BL")
        plt.xlabel("# holes - 2")
        plt.tight_layout()

    # 10. Now normalise all return variables
    # ------------------------------------------------------------------------

    smallfloat = 1e-16

    bs_all = bs_arr / np.mean(fluxes ** 3) * n_holes ** 3
    v2_all = v2_arr / np.mean(fluxes ** 2) * n_holes ** 2
    cvis_all = cvis_arr / np.mean(fluxes) * n_holes

    v2 = v2 / np.mean(fluxes ** 2) * n_holes ** 2
    v2_cov = v2_cov / np.mean(fluxes ** 4) * n_holes ** 4
    v2_cov[v2_cov < smallfloat] = smallfloat

    bs = bs / np.mean(fluxes ** 3) * n_holes ** 3

    avar = avar / np.mean(fluxes ** 4) * n_holes ** 4
    err_avar = err_avar / np.mean(fluxes ** 4) * n_holes ** 4

    avar[avar < smallfloat] = smallfloat
    err_avar[err_avar < smallfloat] = smallfloat

    cp_cov = cp_cov / np.mean(fluxes ** 6) * n_holes ** 6

    bs_cov = bs_cov / np.mean(fluxes ** 6) * n_holes ** 6
    bs_v2_cov = np.real(bs_v2_cov / np.mean(fluxes ** 5) * n_holes ** 5)
    bs_var = bs_var / np.mean(fluxes ** 6) * n_holes ** 6

    if display:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Covariance matrix $V^2$")
        plt.imshow(v2_cov, origin="upper")
        plt.xlabel("# BL")
        plt.ylabel("# BL")
        plt.subplot(1, 2, 2)
        plt.title("Covariance matrix CP")
        plt.imshow(cp_cov, origin="upper")
        plt.xlabel("# CP")
        plt.ylabel("# CP")
        plt.tight_layout()

    cp = np.rad2deg(np.arctan2(bs.imag, bs.real))

    u1coord = mf.u[bs2bl_ix[0, :]]  # * res_t.wl
    v1coord = mf.v[bs2bl_ix[0, :]]  # * res_t.wl
    u2coord = mf.u[bs2bl_ix[1, :]]  # * res_t.wl
    v2coord = mf.v[bs2bl_ix[1, :]]  # * res_t.wl
    u3coord = -(u1coord + u2coord)
    v3coord = -(v1coord + v2coord)

    t3_coord = {
        "u1": u1coord,
        "u2": u2coord,
        "v1": v1coord,
        "v2": v2coord,
    }

    bl_cp = []
    for k in range(n_bispect):
        B1 = np.sqrt(u1coord[k] ** 2 + v1coord[k] ** 2)
        B2 = np.sqrt(u2coord[k] ** 2 + v2coord[k] ** 2)
        B3 = np.sqrt(u3coord[k] ** 2 + v3coord[k] ** 2)
        bl_cp.append(np.max([B1, B2, B3]))  # rad-1
    bl_cp = np.array(bl_cp)

    cp_all = np.rad2deg([np.arctan2(bs.imag, bs.real) for bs in bs_all])

    if not naive_err:
        e_cp = np.rad2deg(np.sqrt(bs_var[1] / abs(bs) ** 2))
        e_v2 = np.sqrt(np.diag(v2_cov))
    else:
        e_cp = np.std(cp_all, axis=0)
        e_v2 = np.std(v2_all, axis=0)

    v2_cor, dummy = cov2cor(v2_cov)

    # 11. Finally, convert baseline variables to hole variable.
    # ------------------------------------------------------------------------

    # 1) In the MAPPIT-style, we define the relationship between hole phases
    # (or phase slopes) and baseline phases (or phase slopes) by the
    # use of a matrix, fitmat.

    fitmat = dblarr(n_holes, n_baselines + 1)

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
            ((ph_arr[:, j] - ph_mn[j] + 3 * np.pi) % (2 * np.pi)) - np.pi)

    ph_err = ph_err / np.sqrt(n_ps)

    p0 = np.zeros(n_holes)
    method = ["Nelder-Mead", "Powell", "BFGS", "CG"]

    ii = 0
    tol = 1e-4
    res = minimize(phase_chi2, p0, method=method[ii], tol=tol,
                   args=(fitmat, ph_mn, ph_err))

    if verbose:
        print("\nDetermining piston using %s minimisation..." % method[ii])
    if res.success:
        if verbose:
            print("Phase Chi^2: ", phase_chi2(
                res.x) / (n_baselines - n_holes + 1))
        find_piston = np.dot(res.x, fitmat)
    else:
        if verbose:
            cprint("Error calculating hole pistons...", "red")
            pass

    if display:
        plt.figure()
        plt.errorbar(np.arange(len(ph_mn)), np.rad2deg(ph_mn), yerr=np.rad2deg(ph_err),
                     ls="None", ecolor="lightgray", marker=".",)
        if res.success:
            plt.plot(np.rad2deg(find_piston[:-1]), ls="--", color="orange", lw=1,
                     label=method[ii] + " minimisation",)
        plt.grid(alpha=0.1)
        plt.ylabel(r"Mean phase [$\degree$]")
        plt.xlabel("# baselines")
        plt.legend()
        plt.tight_layout()

    # 2) fit to the phase slopes using weighted linear regression.
    # Normalisation:  hole_phs was in radians per Fourier pixel.
    # Convert to phase slopes in pixels.

    phs_arr = phs_arr / 2.0 / np.pi * dim1
    hole_phs = np.zeros([2, n_ps, n_holes])
    hole_err_phs = np.zeros([2, n_ps, n_holes])

    for j in range(n_baselines):
        fitmat[bl2h_ix[1, j], j] = 1

    fitmat = fitmat / 2.0
    fitmat = fitmat[:, 0:n_baselines]
    err2 = np.zeros(v2_arr.shape)  # ie a matrix the same size
    err2_bias = err2.copy()

    x = fitmat
    for j in range(n_ps):
        y = phs_arr[0, j, :]
        weights = phserr_arr[0, j, :]

        reg = regress_noc(x, y, weights)

        hole_phs[0, j, :] = reg.coeff

        dummy, sig = cov2cor(reg.cov)

        hole_err_phs[0, j, :] = sig * np.sqrt(reg.MSE)

        y = phs_arr[1, j, :]
        weights = phserr_arr[1, j, :]

        reg = regress_noc(x, y, weights)

        hole_phs[1, j, :] = reg.coeff

        dummy, sig = cov2cor(reg.cov)

        hole_err_phs[1, j, :] = sig * np.sqrt(reg.MSE)

        tmp1 = hole_phs[0, j, bl2h_ix[0, :]] - hole_phs[0, j, bl2h_ix[1, :]]
        tmp2 = hole_phs[1, j, bl2h_ix[0, :]] - hole_phs[1, j, bl2h_ix[1, :]]
        err2[j, :] = tmp1 ** 2 + tmp2 ** 2

        err2_bias[j, :] = ((hole_err_phs[0, j, bl2h_ix[0, :]] -
                            hole_err_phs[0, j, bl2h_ix[1, :]]) ** 2
                           + (hole_err_phs[1, j, bl2h_ix[0, :]] -
                              hole_err_phs[1, j, bl2h_ix[1, :]]) ** 2
                           )

    phs_v2corr = np.zeros(n_baselines)
    predictor = np.zeros(v2_arr.shape)

    for j in range(n_baselines):
        predictor[:, j] = err2[:, j] - np.mean(err2_bias[:, j])

    # imsize is \lambda/hole_diameter in pixels. A factor of 3.0 was only
    # roughly correct based on simulations.
    # 2.5 seems to be better based on real data.
    # NB there is no window size adjustment here.

    imsize = 3
    for j in range(n_baselines):
        phs_v2corr[j] = np.mean(np.exp(-2.5 * predictor[:, j] / imsize ** 2))

    hdr = {"INSTRUME": hdr["INSTRUME"],
           "NRMNAME": maskname, "PIXELSCL": mf.pixelSize}

    res = {
        "v2": v2,
        "v2_sig": e_v2,
        "v2_cov": v2_cov,
        "v2_cor": v2_cor,
        "bs": bs,
        "bs_cov": bs_cov,
        "bs_var": bs_var,
        "bs_v2_cov": bs_v2_cov,
        "bs_all": bs_all,
        "cp": cp,
        "cp_sig": e_cp,
        "cp_cov": cp_cov,
        "cp_all": cp_all,
        "v2_all": v2_all,
        "cvis_all": cvis_all,
        "avar": avar,
        "err_avar": err_avar,
        "cp_var": cp_var,
        "phs_v2corr": phs_v2corr,
        "ps": ps,
        "u": mf.u,
        "v": mf.v,
        "wl": mf.wl,
        "e_wl": mf.e_wl,
        "bl": np.sqrt(mf.u ** 2 + mf.v ** 2),
        "bl_cp": bl_cp,
        "t3_coord": t3_coord,
        "save_ft": save_ft,
        "target": target,
        "filename": filename,
        "maskname": maskname,
        "filtname": filtname,
        "isz": dim1,
        "closing_tri": closing_tri,
        "bl2h_ix": bl2h_ix,
        "bs2bl_ix": bs2bl_ix,
        "n_baselines": n_baselines,
        "xycoord": mf.xy_coords,
        "hdr": hdr,
    }

    t = time.time() - start_time
    m = t // 60

    if verbose:
        cprint("\nDone (exec time: %d min %2.1f s)." %
               (m, t - m * 60), color="magenta")

    return dict2class(res)
