"""Bispectrum extraction routines for aperture-masking interferometry.

The module samples matched Fourier peaks, estimates calibrated visibility and
bispectrum statistics, and assembles AMICAL interferometric observables."""

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from rich import print as rprint
from rich.progress import track

from amical.externals.munch import munchify as dict2class
from amical.get_infos_obs import get_mask
from amical.mf_pipeline.ami_function import (
    bs_multi_triangle,
    compute_index_mask,
    give_peak_info2d,
    make_mf,
    phase_chi2,
    tri_pix,
)
from amical.tools import compute_pa, cov2cor

from .idl_function import dblarr, dist, regress_noc


def _compute_complex_bs(
    ft_arr,
    index_mask,
    fringe_peak,
    mf,
    dark_ps=None,
    closing_tri_pix=None,
    bs_multi_tri=False,
    verbose=True,
):
    """Extract frame-wise complex visibilities and bispectra from Fourier data.

    Parameters
    ----------
    ft_arr : numpy.ndarray of complex, shape (n_frames, npix, npix)
        Fourier transforms of the input image cube.
    index_mask : object
        Aperture-mask index mappings from ``compute_index_mask``.
    fringe_peak : numpy.ndarray of shape (n_baselines,)
        Per-baseline object arrays of Fourier ``(y, x, gain)`` peak samples.
    mf : object
        Matched filter with real and imaginary overlap-correction matrices.
    dark_ps : numpy.ndarray, optional
        Dark power-spectrum image of shape ``(npix, npix)`` or cube of shape
        ``(n_frames, npix, npix)``.
    closing_tri_pix : numpy.ndarray, optional
        Pixel closing-triangle combinations for multiple-triangle extraction.
    bs_multi_tri : bool, default=False
        Whether to use multiple-triangle bispectrum extraction.
    verbose : bool, default=True
        Whether to display extraction progress.

    Returns
    -------
    dict
        Frame-wise visibility and bispectrum arrays, phase-slope estimates, dark
        calibration, frame fluxes, and mean science and dark power spectra."""
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

    for i in track(
        range(n_ps),
        description="Extracting in the cube",
        disable=not verbose,
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

            calib_v2["dark"][j] = np.sum(gain**2 * dps[pix])
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
    """Center an image cube and compute its two-dimensional Fourier transforms.

    Parameters
    ----------
    cube : numpy.ndarray, shape (n_frames, ny, nx)
        Cleaned input image cube. Odd spatial dimensions are cropped by one pixel.

    Returns
    -------
    ft_arr : numpy.ndarray of complex, shape (n_frames, npix, npix)
        Centered two-dimensional Fourier transforms.
    n_ps : int
        Number of frames.
    n_pix : int
        Even spatial size in pixels."""
    if cube.shape[1] % 2 == 1:
        cube = np.array([im[:-1, :-1] for im in cube])

    n_pix = cube.shape[1]
    cube = np.roll(np.roll(cube, n_pix // 2, axis=1), n_pix // 2, axis=2)

    ft_arr = np.fft.fft2(cube)

    i_ps = ft_arr.shape
    n_ps = i_ps[0]

    return ft_arr, n_ps, n_pix


def _show_complex_ps(ft_arr, i_frame=0):
    """Plot real, imaginary, and centered power-spectrum views of one Fourier frame.

    Parameters
    ----------
    ft_arr : numpy.ndarray of complex, shape (n_frames, ny, nx)
        Fourier-transform cube.
    i_frame : int, default=0
        Frame index to display.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the three diagnostic panels."""
    import matplotlib.pyplot as plt

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
    """Plot matched-filter Fourier peak positions over a power spectrum.

    Parameters
    ----------
    ft_arr : numpy.ndarray of complex, shape (n_frames, ny, nx)
        Fourier-transform cube.
    n_baselines : int
        Number of aperture-mask baselines.
    mf : object
        Matched-filter object returned by ``make_mf``.
    maskname : str
        Aperture-mask identifier used in the plot title.
    peakmethod : str
        Matched-filter sampling method used in the plot title.
    i_fram : int, default=0
        Fourier-frame index to display.
    aver : bool, default=False
        Whether to display the mean real Fourier image across frames.

    Returns
    -------
    None
        Displays a diagnostic figure."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

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
    """Plot covariance matrices of normalized AMI observables.

    Parameters
    ----------
    obs_norm : dict
        Normalized observable quantities including ``v2_cov``, ``cp_cov``, and
        ``bs_v2_cov``.
    expert_plot : bool, default=False
        Whether to include the bispectrum-amplitude versus squared-visibility
        covariance plot.

    Returns
    -------
    matplotlib.figure.Figure or tuple of matplotlib.figure.Figure
        Visibility and closure-phase covariance figure, plus the cross-covariance
        figure when ``expert_plot`` is true."""
    import matplotlib.pyplot as plt

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
    """Extract observation metadata from a FITS header.

    Parameters
    ----------
    hdr : mapping
        FITS header providing observation metadata.
    targetname : str, optional
        Target-name fallback when ``OBJECT`` is missing or is ``"STD"``.
    filtname : str, optional
        Filter-name fallback when ``FILTER`` is missing.
    instrum : str, optional
        Instrument-name fallback when ``INSTRUME`` is missing.
    verbose : bool, default=True
        Whether to print metadata fallback warnings.

    Returns
    -------
    object
        Metadata with target, filter name, instrument, source filename, and mean
        seeing FWHM in the header's native angular unit.

    Raises
    ------
    OSError
        If neither the header nor ``instrum`` supplies an instrument."""

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
                rprint(
                    f"[green]Warning: OBJECT is not in the header, targetname is used ({targetname}).",
                    file=sys.stderr,
                )
        else:
            rprint(
                "[green]Warning: target name not found (header or as input).",
                file=sys.stderr,
            )

    # Check the filter used
    if filt is None:
        if filtname is not None:
            filt = filtname
            if verbose:
                rprint(
                    "[green]Warning: FILTER is not in the header, "
                    f"filtname is used ({filtname}).",
                    file=sys.stderr,
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
    """Convert bispectrum baseline indices to aperture-triangle lists.

    Parameters
    ----------
    index_mask : object
        Mask indices with ``bs2bl_ix`` and ``bl2h_ix`` arrays.

    Returns
    -------
    list of list of int
        Aperture indices in each closing triangle."""
    bs2bl_ix = index_mask.bs2bl_ix
    bl2h_ix = index_mask.bl2h_ix
    # Holes (i, j, k) in closure order: baselines ij and jk close with ki.
    # (A set would return them in hash order, which permutes triangles with
    # hole indices >= 8.)
    closing_tri = []
    for b1, b2 in bs2bl_ix[:2].T:
        closing_tri.append([bl2h_ix[0, b1], bl2h_ix[1, b1], bl2h_ix[1, b2]])
    return closing_tri


def _set_good_nblocks(n_blocks, n_ps, verbose=False):
    """Validate the number of statistical blocks used for uncertainty estimates.

    Parameters
    ----------
    n_blocks : int
        Requested number of frame blocks.
    n_ps : int
        Number of frames in the cube.
    verbose : bool, default=False
        Whether to print corrections to the requested block count.

    Returns
    -------
    int
        Usable block count, set to ``n_ps`` when ``n_blocks`` is 0, 1, or exceeds
        the number of frames."""
    if (n_blocks == 0) or (n_blocks == 1):
        if verbose:
            rprint(
                "[green]! Warning: nblocks == 0 -> n_blocks set to n_ps",
                file=sys.stderr,
            )
        n_blocks = n_ps
    elif n_blocks > n_ps:
        if verbose:
            rprint(
                "[green]------------------------------------\n"
                "! Warning: nblocks > n_ps -> n_blocks set to n_ps",
                file=sys.stderr,
            )
        n_blocks = n_ps
    return n_blocks


def _compute_corr_noise(complex_bs, ft_arr, fringe_peak):
    """Estimate Fourier bias and correlated noise outside sampled signal peaks.

    Parameters
    ----------
    complex_bs : dict
        Complex-observable data containing mean science and dark power spectra.
    ft_arr : numpy.ndarray of complex, shape (n_frames, npix, npix)
        Fourier-transform cube.
    fringe_peak : numpy.ndarray of shape (n_baselines,)
        Per-baseline Fourier ``(y, x, gain)`` peak samples.

    Returns
    -------
    bias : float
        Median science power-spectrum bias outside identified signal.
    dark_bias : float
        Median dark power-spectrum bias outside identified signal.
    autocor_noise : numpy.ndarray of shape (npix, npix)
        Mean Fourier autocorrelation of the masked noise."""
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
    """Subtract correlated-noise bias from frame-wise squared visibilities.

    Parameters
    ----------
    v2_arr : numpy.ndarray, shape (n_frames, n_baselines)
        Squared visibility samples, modified in place.
    npix : int
        Fourier-image side length in pixels.
    fringe_peak : numpy.ndarray of shape (n_baselines,)
        Per-baseline Fourier ``(y, x, gain)`` peak samples.
    bias, dark_bias : float
        Science and dark median power-spectrum biases.
    autocor_noise : numpy.ndarray of shape (npix, npix)
        Fourier autocorrelation of masked noise.
    unbias : bool, default=True
        Whether to subtract the estimated correlated science bias.

    Returns
    -------
    v2_arr : numpy.ndarray, shape (n_frames, n_baselines)
        Bias-corrected squared visibilities.
    bias_arr : numpy.ndarray, shape (n_baselines,)
        Estimated correlated science bias for each baseline."""
    n_baselines = v2_arr.shape[1]

    bias_arr = np.zeros(n_baselines)
    for j in range(n_baselines):
        im_peak = dblarr(npix, npix)
        pix = fringe_peak[j][:, 0].astype(int), fringe_peak[j][:, 1].astype(int)
        im_peak[pix] = fringe_peak[j][:, 2]

        autocor_mf = np.fft.ifft2(np.abs(np.fft.ifft2(im_peak)) ** 2).real
        bias_arr[j] = (
            np.sum(autocor_mf * autocor_noise) * bias / autocor_noise[0, 0] * npix**2
        )

        if unbias:
            true_bias = bias_arr[j]
        else:
            true_bias = 0
        v2_arr[:, j] -= true_bias

    for j in range(n_baselines):
        v2_arr[:, j] += np.sum(fringe_peak[j][:, 2] ** 2) * dark_bias
    return v2_arr, bias_arr


def _block_means(arr, n_blocks):
    """Average consecutive frame blocks of an array along its first axis.

    Block ``k`` spans frames ``k * n_ps // n_blocks`` to
    ``(k + 1) * n_ps // n_blocks``, as in the original IDL code.

    Parameters
    ----------
    arr : numpy.ndarray, shape (n_frames, ...)
        Frame-wise samples.
    n_blocks : int
        Number of blocks, at most ``n_frames`` so that no block is empty.

    Returns
    -------
    numpy.ndarray, shape (n_blocks, ...)
        Mean of each block."""
    n_ps = arr.shape[0]
    bounds = np.arange(n_blocks + 1) * n_ps // n_blocks
    counts = np.diff(bounds).reshape((n_blocks,) + (1,) * (arr.ndim - 1))
    # reduceat sums arr[bounds[k]:bounds[k + 1]] for each block start.
    return np.add.reduceat(arr, bounds[:-1], axis=0) / counts


def _compute_v2_quantities(v2_arr, bias_arr, n_blocks):
    """Compute mean squared visibilities and their block-estimated covariance.

    Parameters
    ----------
    v2_arr : numpy.ndarray, shape (n_frames, n_baselines)
        Bias-corrected squared visibility samples.
    bias_arr : numpy.ndarray, shape (n_baselines,)
        Per-baseline correlated-noise bias estimates.
    n_blocks : int
        Number of frame blocks used to estimate covariance.

    Returns
    -------
    dict
        Mean squared visibilities, their covariance, amplitude-variance estimates,
        associated uncertainties, and the input visibility array."""
    n_ps = v2_arr.shape[0]
    n_baselines = v2_arr.shape[1]

    # Compute vis. squared average
    v2 = np.mean(v2_arr, axis=0)

    # Compute vis. squared difference, shape (n_blocks, n_baselines)
    v2_diff = _block_means(v2_arr, n_blocks) - v2

    # Compute vis. squared covariance
    v2_cov = v2_diff.T @ v2_diff / (n_blocks - 1) / n_blocks
    # Additonal "/ n_blocks" in the original code ???

    # AS. Comparison with numpy cov matrices
    # v2_cov_pyt = np.cov(v2_arr.T, bias=False)
    # v2_cov = v2_cov_pyt

    x = np.arange(n_baselines)
    avar = v2_cov[x, x] * n_ps - bias_arr**2 * (1 + (2.0 * v2) / bias_arr)
    err_avar = np.sqrt(
        2.0 / n_ps * (v2_cov[x, x]) ** 2 * n_ps**2 + 4.0 * v2_cov[x, x] * bias_arr**2
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
    """Compute mean bispectra and bispectrum covariance statistics.

    Parameters
    ----------
    bs_arr : numpy.ndarray of complex, shape (n_frames, n_bispectra)
        Frame-wise bispectrum samples.
    v2 : numpy.ndarray, shape (n_baselines,)
        Mean squared visibilities.
    fluxes : numpy.ndarray, shape (n_frames,)
        Frame flux estimates.
    index_mask : object
        Mask indices including bispectrum and covariance mappings.
    n_blocks : int
        Number of frame blocks used for variance estimation.
    subtract_bs_bias : bool, default=True
        Whether to subtract the detector bispectrum-bias model.

    Returns
    -------
    dict
        Mean bispectrum, real and imaginary variance and covariance estimates, and
        the input frame-wise bispectrum array."""
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
    """Estimate real and imaginary bispectrum variances from frame blocks.

    Parameters
    ----------
    bs_arr : numpy.ndarray of complex, shape (n_frames, n_bispectra)
        Frame-wise bispectrum samples.
    bs : numpy.ndarray of complex, shape (n_bispectra,)
        Mean bispectrum values.
    n_blocks : int
        Number of frame blocks.

    Returns
    -------
    numpy.ndarray of float, shape (2, n_bispectra)
        Normalized variances along the bispectrum amplitude and phase directions."""
    # comp_diff is the complex difference from the mean, shifted so that the
    # real axis corresponds to amplitude and the imaginary axis phase.
    # Shape (n_blocks, n_bispect).
    comp_diff = _block_means((bs_arr - bs) * np.conj(bs), n_blocks)

    num_real = np.sum(np.real(comp_diff) ** 2, axis=0) / (n_blocks - 1) / n_blocks
    num_imag = np.sum(np.imag(comp_diff) ** 2, axis=0) / (n_blocks - 1) / n_blocks

    bs_var = np.array([num_real, num_imag]) / np.abs(bs) ** 2
    return bs_var


def _compute_bs_cov(bs_arr, bs, bscov2bs_ix, n_cov):
    """Estimate covariance between bispectra sharing a baseline.

    Parameters
    ----------
    bs_arr : numpy.ndarray of complex, shape (n_frames, n_bispectra)
        Frame-wise bispectrum samples.
    bs : numpy.ndarray of complex, shape (n_bispectra,)
        Mean bispectrum values.
    bscov2bs_ix : numpy.ndarray of int, shape (2, n_cov)
        Paired bispectrum indices for covariance calculation.
    n_cov : int
        Number of bispectrum covariance pairs.

    Returns
    -------
    numpy.ndarray of float, shape (2, n_cov)
        Real and imaginary normalized covariance for each bispectrum pair."""
    n_ps = bs_arr.shape[0]
    ix1, ix2 = bscov2bs_ix[0, :n_cov], bscov2bs_ix[1, :n_cov]

    # Deviations rotated so that real = amplitude and imaginary = phase
    # direction, shape (n_frames, n_bispect); then gathered for each pair.
    temp = (bs_arr - bs) * np.conj(bs)
    temp1, temp2 = temp[:, ix1], temp[:, ix2]
    denom = np.abs(bs[ix1]) * np.abs(bs[ix2]) * (n_ps - 1) * n_ps

    bs_cov = np.array(
        [
            np.sum(np.real(temp1) * np.real(temp2), axis=0),
            np.sum(np.imag(temp1) * np.imag(temp2), axis=0),
        ]
    )
    return bs_cov / denom


def _compute_cp_cov(bs_arr, bs, index_mask, disable=False):
    """Compute the closure-phase covariance from complex bispectrum samples.

    Parameters
    ----------
    bs_arr : numpy.ndarray of complex, shape (n_frames, n_bispectra)
        Frame-wise bispectrum samples.
    bs : numpy.ndarray of complex, shape (n_bispectra,)
        Mean bispectrum values.
    index_mask : object
        Mask indices providing the number of bispectra.
    disable : bool, default=False
        Unused; kept for backwards compatibility (the vectorised computation
        no longer reports progress).

    Returns
    -------
    numpy.ndarray of float, shape (n_bispectra, n_bispectra)
        Closure-phase covariance in radians squared."""
    n_ps = bs_arr.shape[0]

    # Phase-direction deviations of each bispectrum, shape (n_frames, n_bispect):
    # cp_cov[i, j] = sum_frames imag_i * imag_j / denom[i, j].
    imag_dev = np.imag((bs_arr - bs) * np.conj(bs))
    abs2 = np.abs(bs) ** 2
    denom = np.outer(abs2, abs2) * (n_ps - 1) * n_ps
    cp_cov = imag_dev.T @ imag_dev / denom
    return cp_cov


def _compute_bs_v2_cov(bs_arr, v2_arr, v2, bs, index_mask):
    """Compute covariance of bispectrum amplitude and squared visibility.

    Parameters
    ----------
    bs_arr : numpy.ndarray of complex, shape (n_frames, n_bispectra)
        Frame-wise bispectrum samples.
    v2_arr : numpy.ndarray, shape (n_frames, n_baselines)
        Frame-wise squared visibility samples.
    v2 : numpy.ndarray, shape (n_baselines,)
        Mean squared visibilities.
    bs : numpy.ndarray of complex, shape (n_bispectra,)
        Mean bispectrum values.
    index_mask : object
        Mask indices providing baseline-to-bispectrum mappings.

    Returns
    -------
    numpy.ndarray of float, shape (n_baselines, n_holes - 2)
        Normalized covariance of bispectrum amplitude and squared visibility."""
    n_ps = bs_arr.shape[0]
    bl2bs_ix = index_mask.bl2bs_ix

    # This complicated thing calculates the dot product between the bispectrum point and
    # its error term ie (x . del_x)/|x| and multiplies this by the power error term.
    # Note that this is not the same as using absolute value, and that this sum should be
    # zero where |bs| is zero within errors.
    # bl2bs_ix[j, k] is the k-th bispectrum containing baseline j; gathering
    # with it gives arrays of shape (n_frames, n_baselines, n_holes - 2).
    bs_ix = bs[bl2bs_ix]
    bs_real_tmp = np.real((bs_arr[:, bl2bs_ix] - bs_ix) * np.conj(bs_ix))
    diff_v2 = (v2_arr - v2)[:, :, None]
    norm = np.abs(bs_ix) / (n_ps - 1.0) / n_ps
    bs_v2_cov = np.sum(bs_real_tmp * diff_v2, axis=0) / norm
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
    """Normalize AMI observables by mean frame flux and aperture count.

    Parameters
    ----------
    bs_quantities, v2_quantities : dict
        Bispectrum and squared-visibility statistics.
    cvis_arr : numpy.ndarray of complex, shape (n_frames, n_baselines)
        Frame-wise complex visibilities.
    cp_cov : numpy.ndarray or None
        Closure-phase covariance matrix.
    bs_v2_cov : numpy.ndarray
        Bispectrum-amplitude to squared-visibility covariance.
    fluxes : numpy.ndarray, shape (n_frames,)
        Frame flux estimates.
    index_mask : object
        Mask indices providing the aperture count.
    infos : object
        Observation metadata used for optional diagnostic labels.
    expert_plot : bool, default=False
        Whether to display normalized-observable diagnostic plots.
    save : bool, default=False
        Reserved flag; it does not affect the computation.

    Returns
    -------
    v2_norm : numpy.ndarray, shape (n_baselines,)
        Flux- and aperture-normalized mean squared visibilities.
    norm_quantities : dict
        Normalized frame-wise observables, covariances, variances, and visibility
        correlation matrix."""
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

    bs_arr_norm = bs_arr / np.mean(fluxes**3) * n_holes**3
    v2_arr_norm = v2_arr / np.mean(fluxes**2) * n_holes**2
    cvis_arr_norm = cvis_arr / np.mean(fluxes) * n_holes

    v2_norm = (v2 / np.mean(fluxes**2)) * n_holes**2
    v2_cov_norm = (v2_cov / np.mean(fluxes**4)) * n_holes**4

    bs_norm = bs / np.mean(fluxes**3) * n_holes**3

    avar_norm = avar / np.mean(fluxes**4) * n_holes**4
    err_avar_norm = err_avar / np.mean(fluxes**4) * n_holes**4

    try:
        cp_cov_norm = cp_cov / np.mean(fluxes**6) * n_holes**6
    except TypeError:
        cp_cov_norm = None

    bs_cov_norm = bs_cov / np.mean(fluxes**6) * n_holes**6
    bs_v2_cov_norm = np.real(bs_v2_cov / np.mean(fluxes**5) * n_holes**5)
    bs_var_norm = bs_var / np.mean(fluxes**6) * n_holes**6

    if expert_plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        plt.title(f"DIAGNOSTIC PLOTS - V2 - {infos.target}")
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
    """Compute closure phases in degrees from normalized bispectra.

    Parameters
    ----------
    obs_result : dict
        Result dictionary updated with mean closure phases.
    obs_norm : dict
        Normalized observable data containing mean and frame-wise bispectra.
    infos : object
        Observation metadata used for optional diagnostic labels.
    expert_plot : bool, default=False
        Whether to display closure-phase diagnostics.

    Returns
    -------
    dict
        ``obs_result`` with closure phases in degrees added as ``cp``; ``obs_norm``
        is updated in place with frame-wise ``cp_arr`` values in degrees."""
    bs = obs_norm["bs"]
    bs_arr = obs_norm["bs_arr"]

    cp = np.rad2deg(np.arctan2(bs.imag, bs.real))
    cp_arr = np.rad2deg([np.arctan2(i_bs.imag, i_bs.real) for i_bs in bs_arr])

    obs_result["cp"] = cp
    obs_norm["cp_arr"] = cp_arr

    if expert_plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        plt.title(f"DIAGNOSTIC PLOTS - CP - {infos.target}")
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
    """Compute baseline coordinates for each closure triangle.

    Parameters
    ----------
    mf : object
        Matched-filter data containing baseline coordinates in metres.
    index_mask : object
        Mask indices containing ``n_bispect`` and ``bs2bl_ix``.

    Returns
    -------
    t3_coord : dict
        First and second baseline coordinates ``u1``, ``u2``, ``v1``, and ``v2``
        in metres for every closure triangle.
    bl_cp : numpy.ndarray of shape (n_bispectra,)
        Longest baseline in each closure triangle, in metres."""
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
    """Compute closure-phase and squared-visibility uncertainties.

    Parameters
    ----------
    obs_result : dict
        Result dictionary updated in place.
    obs_norm : dict
        Normalized observables containing bispectrum variance, visibility
        covariance, and frame-wise closure phases and visibilities.
    naive_err : bool, default=False
        Whether to use frame-wise standard deviations instead of covariance-based
        errors.

    Returns
    -------
    dict
        ``obs_result`` with ``e_cp`` in degrees and ``e_vis2`` added."""
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
    """Fit aperture piston phases from mean baseline phases.

    Parameters
    ----------
    complex_bs : dict
        Complex-observable data containing frame-wise visibility phases.
    index_mask : object
        Mask indices providing aperture-to-baseline mappings.
    method : str, default="Nelder-Mead"
        SciPy minimization method.
    tol : float, default=1e-4
        Minimizer tolerance in radians.
    verbose : bool, default=False
        Whether to print fit status and reduced chi-square.
    display : bool, default=False
        Whether to display fitted baseline phases.

    Returns
    -------
    numpy.ndarray of shape (n_holes, n_baselines + 1)
        Matrix mapping fitted aperture piston parameters to baseline phases and a
        reference constraint."""
    from scipy.optimize import minimize

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
        print(f"\nDetermining piston using {method} minimisation...")
    if res.success:
        if verbose:
            print(
                "Phase Chi^2: ",
                phase_chi2(res.x, fitmat, ph_mn, ph_err) / (n_baselines - n_holes + 1),
            )
        find_piston = np.dot(res.x, fitmat)
    else:
        if verbose:
            rprint("[red]Error calculating hole pistons...", file=sys.stderr)

    if display:
        import matplotlib.pyplot as plt

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
    """Fit aperture phase slopes with weighted linear regression.

    Parameters
    ----------
    x : numpy.ndarray of shape (n_holes, n_baselines)
        Aperture-to-baseline design matrix.
    y : numpy.ndarray of shape (n_baselines,)
        Measured baseline phase slopes.
    weights : numpy.ndarray of shape (n_baselines,)
        Regression weights derived from phase errors.

    Returns
    -------
    hole_ph : numpy.ndarray of shape (n_holes,)
        Fitted aperture phase slopes.
    hole_ph_err : numpy.ndarray of shape (n_holes,)
        Uncertainties on fitted aperture phase slopes."""
    reg = regress_noc(x, y, weights)
    sig = cov2cor(reg.cov)[1]
    hole_ph = reg.coeff
    hole_ph_err = sig * np.sqrt(reg.MSE)
    return hole_ph, hole_ph_err


def _compute_phs_error(complex_bs, fitmat, index_mask, npix, imsize=3):
    """Estimate visibility loss caused by frame-wise aperture phase slopes.

    Parameters
    ----------
    complex_bs : dict
        Complex-observable data with phase-slope measurements and squared
        visibilities.
    fitmat : numpy.ndarray
        Aperture-to-baseline phase design matrix.
    index_mask : object
        Mask indices providing aperture-to-baseline mappings.
    npix : int
        Fourier-image side length in pixels.
    imsize : float, default=3
        Approximate diffraction scale, lambda divided by hole diameter, in pixels.

    Returns
    -------
    numpy.ndarray of shape (n_baselines,)
        Mean multiplicative squared-visibility correction due to phase slopes."""
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
        err[j, :] = tmp1**2 + tmp2**2
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
        phs_v2corr[j] = np.mean(np.exp(-2.5 * predictor[:, j] / imsize**2))

    return phs_v2corr


def _add_infos_header(infos, hdr, mf, pa, filename, maskname, npix):
    """Add extraction metadata and selected FITS-header values to observation info.

    Parameters
    ----------
    infos : mapping
        Observation information mapping updated in place.
    hdr : mapping
        Original FITS header.
    mf : object
        Matched filter containing detector pixel scale in radians per pixel.
    pa : float or array-like
        Computed position angle in degrees.
    filename : str or path-like
        Processed data-cube filename.
    maskname : str
        Aperture-mask identifier.
    npix : int
        Detector image size in pixels.

    Returns
    -------
    mapping
        Updated observation information."""
    infos["pixscale"] = mf.pixelSize
    infos["pa"] = pa
    infos["filename"] = filename
    infos["maskname"] = maskname
    infos["isz"] = npix
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
    from pypdf import PdfReader, PdfWriter

    mergedObject = PdfWriter()

    for fileNumber in range(7):
        ifile = os.path.join(figdir, f"{filename}_{fileNumber + 1}.pdf")
        mergedObject.append(PdfReader(ifile, "rb"))
        os.remove(ifile)

    # Write all the files into a file which is named as shown below
    mergedObject.write(os.path.join(figdir, filename + "_DIAGNOSTIC_PLOTS.pdf"))
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
    save_to=None,
    verbose=False,
    display=True,
):
    """Extract calibrated aperture-masking interferometric observables from a data cube.

    Parameters
    ----------
    cube : numpy.ndarray, shape (n_frames, ny, nx)
        Cleaned image cube ready for non-redundant-mask extraction.
    filename : str or path-like
        FITS file from which observation headers are read.
    maskname : str
        Aperture-mask identifier.
    filtname : str, optional
        Filter-name fallback when the FITS header lacks ``FILTER``.
    targetname : str, optional
        Target-name fallback when the FITS header lacks ``OBJECT``.
    instrum : str, optional
        Instrument-name fallback when the FITS header lacks ``INSTRUME``.
    bs_multi_tri : bool, default=False
        Whether to compute bispectra from multiple pixel closure triangles.
    peakmethod : {"fft", "square", "unique", "gauss"}, default="gauss"
        Method used to sample Fourier splodges.
    hole_diam : float, default=0.8
        Aperture-hole diameter in metres.
    cutoff : float, default=1e-4
        Minimum matched-filter weight retained for a peak pixel.
    fw_splodge : float, default=0.7
        Relative Fourier-splodge size and Gaussian-method FWHM scale.
    naive_err : bool, default=False
        Whether to use frame-wise standard deviations instead of covariance-based
        uncertainties.
    n_wl : int, default=3
        Number of wavelengths used to sample filter bandwidth.
    n_blocks : int, default=0
        Number of frame blocks used for uncertainty estimation; 0 uses all frames.
    theta_detector : float, default=0
        Mask rotation relative to the detector in degrees.
    scaling_uv : float, default=1
        Multiplicative scale applied to aperture-mask coordinates.
    i_wl : int or sequence of int, optional
        SPHERE-IFS spectral-channel selector.
    unbias_v2 : bool, default=True
        Whether to subtract the correlated-noise bias from squared visibilities.
    compute_cp_cov : bool, default=True
        Whether to compute the closure-phase covariance matrix.
    expert_plot : bool, default=False
        Whether to show expert-level diagnostic plots.
    save_to : str or path-like, optional
        Directory in which diagnostic figures are saved.
    verbose : bool, default=False
        Whether to print processing progress.
    display : bool, default=True
        Whether to display diagnostic figures.

    Returns
    -------
    object or None
        AMI observables including squared visibilities, closure phases in degrees,
        uncertainties, baseline coordinates in metres, wavelength in metres, and
        mask, metadata, and covariance information. Returns ``None`` if the mask
        cannot be loaded or the matched filter cannot be constructed.

    Raises
    ------
    ValueError
        If SPHERE-IFS data are processed without ``i_wl``."""
    from astropy.io import fits

    if verbose:
        rprint("[cyan]\n-- Starting extraction of observables --")
    start_time = time.time()

    if save_to is not None and not display:
        warnings.warn(
            "Diagnostic figures are only saved when display=True: "
            "`save_to` is ignored.",
            stacklevel=2,
        )
        save_to = None

    if save_to is not None:
        if not os.path.exists(save_to):
            os.mkdir(save_to)

    with fits.open(filename) as hdu:
        hdr = hdu[0].header
        try:
            sci_hdr = hdu["SCI"].header
        except KeyError:
            sci_hdr = None

    infos = _check_input_infos(
        hdr, targetname=targetname, filtname=filtname, instrum=instrum, verbose=False
    )

    if infos.instrument == "SPHERE-IFS":
        if i_wl is None:
            raise ValueError(
                "Your file seems to be obtained with an IFU instrument: spectral "
                "channel index `i_wl` must be specified."
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
        save_to=save_to,
        filename=filename,
        index_mask=index_mask,
    )

    ifig = 2
    if save_to is not None:
        import matplotlib.pyplot as plt

        figname = os.path.join(save_to, Path(filename).stem)
        plt.savefig(f"{figname}_{ifig}.pdf")

    ifig += 1

    if mf is None:
        return None

    # We store the principal results in the new dictionnary to be save at the end
    obs_result = {"u": mf.u, "v": mf.v, "wl": mf.wl, "e_wl": mf.e_wl}

    # 4. Compute indices for the multiple triangle technique (tri_pix function)
    # -------------------------------------------------------------------------
    l_B = np.sqrt(mf.u**2 + mf.v**2)  # Length of different bl [m]
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
        if save_to is not None:
            plt.savefig(f"{figname}_{ifig}.pdf")
        ifig += 1

        _show_peak_position(ft_arr, n_baselines, mf, maskname, peakmethod)
        if save_to is not None:
            plt.savefig(f"{figname}_{ifig}.pdf")
        ifig += 1

    if verbose:
        print(f"\nFilename: {filename}")
        print(f"# of frames = {n_ps}")

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
        verbose=verbose,
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
        cp_cov = _compute_cp_cov(
            bs_arr, bs_quantities["bs"], index_mask, disable=np.invert(verbose)
        )
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
        expert_plot=display,
    )
    if save_to is not None:
        plt.savefig(f"{figname}_{ifig}.pdf")
    ifig += 1

    obs_result["vis2"] = vis2_norm

    # 10. Now we compute the cp quantities and store them with the other observables
    obs_result = _compute_cp(obs_result, obs_norm, infos, expert_plot=display)

    if save_to is not None:
        plt.savefig(f"{figname}_{ifig}.pdf")
    ifig += 1

    if display:
        _show_norm_matrices(obs_norm, expert_plot=expert_plot)
        if save_to is not None:
            plt.savefig(f"{figname}_{ifig}.pdf")
        ifig += 1

    t3_coord, bl_cp = _compute_t3_coord(mf, index_mask)
    bl_v2 = np.sqrt(mf.u**2 + mf.v**2)
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
    pa = compute_pa(hdr, n_ps, display=display, verbose=verbose, sci_hdr=sci_hdr)

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

    if save_to is not None:
        produce_result_pdf(save_to, Path(filename).stem)

    if verbose:
        rprint(f"[magenta]\nDone (exec time: {m} min {t - m * 60:2.1f} s).")
    return dict2class(obs_result)
