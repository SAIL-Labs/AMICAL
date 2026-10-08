"""Matched-filter utilities for aperture-masking interferometry.

The module builds Fourier-plane matched filters, aperture-mask index mappings,
and closure-triangle pixel combinations used by the AMICAL bispectrum pipeline."""

import os
import sys
from pathlib import Path

import numpy as np
from rich import print as rprint

from amical.dpfit import leastsqFit
from amical.externals.munch import munchify as dict2class
from amical.get_infos_obs import get_mask, get_pixel_size, get_wavelength
from amical.mf_pipeline.idl_function import array_coords, dist
from amical.tools import gauss_2d_asym, linear, norm_max, plot_circle


def _plot_mask_coord(xy_coords, maskname, instrument):
    import matplotlib.pyplot as plt

    if instrument == "NIRISS":
        marker = "H"
        D = 6.5
    else:
        D = 8.0
        marker = "o"

    fig = plt.figure(figsize=(6, 5.5))
    plt.title(f"{instrument} - mask {maskname}", fontsize=14)

    for i in range(xy_coords.shape[0]):
        plt.scatter(
            xy_coords[i][0],
            xy_coords[i][1],
            s=1e2,
            c="None",
            edgecolors="navy",
            marker=marker,
        )
        plt.text(xy_coords[i][0] + 0.1, xy_coords[i][1] + 0.1, i)

    plt.xlabel("Aperture x-coordinate [m]", fontsize=12)
    plt.ylabel("Aperture y-coordinate [m]", fontsize=12)
    plt.axis([-D / 2.0, D / 2.0, -D / 2.0, D / 2.0])
    plt.tight_layout()
    return fig


def _compute_uv_coord(
    xy_coords, index_mask, filt, pixelSize, npix, round_uv_to_pixel=False
):
    """Compute detector-sampled spatial-frequency coordinates.

    Parameters
    ----------
    xy_coords : numpy.ndarray of shape (n_holes, 2)
        Aperture coordinates in metres.
    index_mask : object
        Mask-index object with ``n_baselines`` and ``bl2h_ix`` attributes.
    filt : array-like of float
        Filter central wavelength and width in metres; only the central wavelength
        is used.
    pixelSize : float
        Detector pixel scale in radians per pixel.
    npix : int
        Detector image size in pixels.
    round_uv_to_pixel : bool, default=False
        Whether to quantize aperture coordinates to the detector Fourier-pixel grid.

    Returns
    -------
    u, v : numpy.ndarray of shape (n_baselines,)
        Baseline spatial frequencies along the aperture x and y axes, respectively,
        in inverse metres."""
    n_baselines = index_mask.n_baselines
    bl2h_ix = index_mask.bl2h_ix

    u_real = np.zeros(n_baselines)
    v_real = np.zeros(n_baselines)
    for i in range(n_baselines):
        if not round_uv_to_pixel:
            u_real[i] = (
                xy_coords[bl2h_ix[0, i], 0] - xy_coords[bl2h_ix[1, i], 0]
            ) / filt[0]
            v_real[i] = (
                xy_coords[bl2h_ix[0, i], 1] - xy_coords[bl2h_ix[1, i], 1]
            ) / filt[0]
        else:
            onepix = 1.0 / (npix * pixelSize)
            onepix_xy = onepix * filt[0]
            new_xy = (xy_coords / onepix_xy).astype(int) * onepix_xy

            u_real[i] = (new_xy[bl2h_ix[0, i], 0] - new_xy[bl2h_ix[1, i], 0]) / filt[0]
            v_real[i] = (new_xy[bl2h_ix[0, i], 1] - new_xy[bl2h_ix[1, i], 1]) / filt[0]
    return u_real, v_real


def _peak_fft_method(
    i, npix, xy_coords, wl, index_mask, pixelsize, innerpix, innerpix_center
):
    mf = np.zeros([npix, npix])
    n_holes = index_mask.n_holes
    bl2h_ix = index_mask.bl2h_ix
    npix = mf.shape[0]

    sum_xy = np.sum(xy_coords, axis=0) / n_holes
    shift_fact = np.ones([n_holes, 2])
    shift_fact[:, 0] = sum_xy[0]
    shift_fact[:, 1] = sum_xy[1]
    xy_coords2 = xy_coords.copy()
    xy_coords2 -= shift_fact

    for j in range(len(wl)):
        xyh = xy_coords2[bl2h_ix[1, i], :] / wl[j] * pixelsize * npix + npix // 2
        delta = xyh - np.floor(xyh)
        ap1 = np.zeros([npix, npix])
        x1 = int(xyh[1])
        y1 = int(xyh[0])
        ap1[x1, y1] = (1.0 - delta[0]) * (1.0 - delta[1])
        ap1[x1, y1 + 1] = delta[0] * (1.0 - delta[1])
        ap1[x1 + 1, y1] = (1.0 - delta[0]) * delta[1]
        ap1[x1 + 1, y1 + 1] = delta[0] * delta[1]

        xyh = xy_coords2[bl2h_ix[0, i], :] / wl[j] * pixelsize * npix + npix // 2
        delta = xyh - np.floor(xyh)
        ap2 = np.zeros([npix, npix])
        x2 = int(xyh[1])
        y2 = int(xyh[0])
        ap2[x2, y2] = (1.0 - delta[0]) * (1.0 - delta[1])
        ap2[x2, y2 + 1] = delta[0] * (1.0 - delta[1])
        ap2[x2 + 1, y2] = (1.0 - delta[0]) * delta[1]
        ap2[x2 + 1, y2 + 1] = delta[0] * delta[1]

        n_elts = npix**2

        tmf = np.fft.fft2(ap1) / n_elts * np.conj(np.fft.fft2(ap2) / n_elts)
        tmf = np.fft.fft2(tmf)
        mf = mf + np.real(tmf)

    mf_flat = norm_max(mf.ravel())
    mf_centered = norm_max(np.fft.fftshift(mf).ravel())

    mf_centered[innerpix_center] = 0.0
    mf_flat[innerpix] = 0.0

    dic_mf = {"flat": mf_flat, "centered": mf_centered}
    return dic_mf


def _peak_square_method(i, npix, u, v, pixelsize, innerpix, innerpix_center):
    mf = np.zeros([npix, npix])
    uv = np.array([v[i], u[i]]) * pixelsize * npix
    uv = (uv + npix) % npix
    uv_int = np.array(np.floor(uv), dtype=int)
    uv_frac = uv - uv_int
    mf[uv_int[0], uv_int[1]] = (1 - uv_frac[0]) * (1 - uv_frac[1])
    mf[uv_int[0], (uv_int[1] + 1) % npix] = (1 - uv_frac[0]) * uv_frac[1]
    mf[(uv_int[0] + 1) % npix, uv_int[1]] = uv_frac[0] * (1 - uv_frac[1])
    mf[(uv_int[0] + 1) % npix, (uv_int[1] + 1) % npix] = uv_frac[0] * uv_frac[1]
    mf = np.roll(mf, [0, 0])
    mf_flat = norm_max(mf.ravel())
    mf_centered = norm_max(np.fft.fftshift(mf).ravel())
    mf_flat[innerpix] = 0.0
    mf_centered[innerpix_center] = 0.0
    dic_mf = {"flat": mf_flat, "centered": mf_centered}
    return dic_mf


def _peak_one_method(i, npix, u, v, pixelsize, innerpix, innerpix_center):
    mf = np.zeros([npix, npix])
    uv = np.array([v[i], u[i]]) * pixelsize * npix
    uv = (uv + npix) % npix
    uv_int = np.array(np.round(uv), dtype=int)
    mf[uv_int[0], uv_int[1]] = 1
    mf = np.roll(mf, [0, 0])
    mf_flat = norm_max(mf.ravel())
    mf_centered = norm_max(np.fft.fftshift(mf).ravel())
    mf_flat[innerpix] = 0.0
    mf_centered[innerpix_center] = 0.0
    dic_mf = {"flat": mf_flat, "centered": mf_centered}
    return dic_mf


def _peak_gauss_method(
    i,
    npix,
    u,
    v,
    filt,
    index_mask,
    pixelsize,
    innerpix,
    innerpix_center,
    fw_splodge=0.7,
    hole_diam=0.8,
):
    mf = np.zeros([npix, npix])
    n_holes = index_mask.n_holes

    l_B = np.sqrt(u**2 + v**2)
    minbl = np.min(l_B) * filt[0]
    if n_holes >= 15:
        sampledisk_r = minbl / 2 / filt[0] * pixelsize * npix * 0.9
    else:
        sampledisk_r = minbl / 2.0 / filt[0] * pixelsize * npix * fw_splodge

    xspot = float(np.round(v[i] * pixelsize * npix + npix / 2.0))
    yspot = float(np.round(u[i] * pixelsize * npix + npix / 2.0))
    mf = plot_circle(mf, xspot, yspot, sampledisk_r, display=False)
    mf = np.roll(mf, npix // 2, axis=0)
    mf = np.roll(mf, npix // 2, axis=1)

    X = [np.arange(npix), np.arange(npix), 1]
    splodge_fwhm = hole_diam / filt[0] * pixelsize * npix / 1.9
    param = {
        "A": 1,
        "x0": -npix // 2 + yspot,
        "y0": -npix // 2 + xspot,
        "fwhm_x": splodge_fwhm,
        "fwhm_y": splodge_fwhm,
        "theta": 0,
    }
    gauss = gauss_2d_asym(X, param)
    gauss = np.roll(gauss, npix // 2, axis=0)
    gauss = np.roll(gauss, npix // 2, axis=1)
    mfg = gauss / np.sum(gauss)
    mf_gain_flat = mfg.ravel()
    mf_gain_centered = norm_max(np.fft.fftshift(mfg).ravel())
    mf_flat = norm_max(mf.ravel())
    mf_centered = norm_max(np.fft.fftshift(mf).ravel())

    mf_flat[innerpix] = 0.0
    mf_gain_flat[innerpix] = 0.0
    mf_centered[innerpix_center] = 0.0
    mf_gain_centered[innerpix_center] = 0.0

    mf = {
        "flat": mf_flat,
        "centered": mf_centered,
        "gain_f": mf_gain_flat,
        "gain_c": mf_gain_centered,
    }
    return dict2class(mf)


def _normalize_gain(
    mf_flat, mf_centered, pixelvector, pixelvector_c, normalize_pixelgain=True
):
    if normalize_pixelgain:
        pixelgain = mf_flat[pixelvector] / np.sum(mf_flat[pixelvector])
        pixelgain_c = mf_centered[pixelvector_c] / np.sum(mf_centered[pixelvector_c])
    else:
        pixelgain = (
            mf_flat[pixelvector]
            * np.max(mf_flat[pixelvector])
            / np.sum(mf_flat[pixelvector] ** 2)
        )
        pixelgain_c = (
            mf_centered[pixelvector_c]
            * np.max(mf_centered[pixelvector_c])
            / np.sum(mf_centered[pixelvector_c] ** 2)
        )
    return pixelgain, pixelgain_c


def _compute_center_splodge(
    npix,
    pixelsize,
    filt,
    hole_diam=0.8,
):
    tmp = dist(npix)
    innerpix = np.array(
        np.array(np.where(tmp < (hole_diam / filt[0] * pixelsize * npix) * 0.9)) * 0.6,
        dtype=int,
    )

    x, y = np.meshgrid(npix, npix)
    dist_c = np.sqrt((x - npix // 2) ** 2 + (y - npix // 2) ** 2)
    inner_pos = np.array(
        np.where(dist_c < (hole_diam / filt[0] * pixelsize * npix) * 0.9)
    )
    innerpix_center = np.array(inner_pos * 0.6, dtype=int)
    return innerpix, innerpix_center


def _make_overlap_mat(mf, n_baselines, display=False):
    overmat = np.zeros(
        [n_baselines, n_baselines], dtype=[("real", float), ("imag", float)]
    )
    # Now find the overlap matrices
    for i in range(n_baselines):
        pix_on = np.where(mf["norm"][:, :, i] != 0.0)
        for j in range(n_baselines):
            t1 = np.sum(mf["norm"][:, :, i][pix_on] * mf["norm"][:, :, j][pix_on])
            t2 = np.sum(mf["norm"][:, :, i][pix_on] * mf["conj"][:, :, j][pix_on])
            overmat["real"][i, j] = t1 + t2
            overmat["imag"][i, j] = t1 - t2

    mf_rmat = np.linalg.inv(overmat["real"])
    mf_imat = np.linalg.inv(overmat["imag"])
    mf_rmat[np.where(mf_rmat < 1e-6)] = 0.0
    mf_imat[np.where(mf_imat < 1e-6)] = 0.0
    mf_rmat[mf_rmat >= 2] = 2
    mf_imat[mf_imat >= 2] = 2
    mf_imat[mf_imat <= -2] = -2

    if display:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 6))
        plt.title("Overlap matrix", fontsize=14)
        plt.imshow(mf_imat, cmap="gray", origin="upper")
        plt.ylabel("# baselines", fontsize=12)
        plt.xlabel("# baselines", fontsize=12)
        plt.tight_layout()

    return mf_rmat, mf_imat


def make_mf(
    maskname,
    instrument,
    filtname,
    npix,
    i_wl=None,
    peakmethod="fft",
    n_wl=3,
    theta_detector=0,
    cutoff=1e-4,
    hole_diam=0.8,
    fw_splodge=0.7,
    scaling=1,
    diag_plot=False,
    verbose=False,
    display=True,
    save_to=None,
    filename=None,
    index_mask=None,
):
    """Construct the Fourier-plane matched filter for an aperture mask.

    Parameters
    ----------
    maskname : str
        Aperture-mask identifier.
    instrument : str
        Instrument identifier used to obtain mask coordinates and pixel scale.
    filtname : str
        Filter identifier.
    npix : int
        Square detector image size in pixels.
    i_wl : int or sequence of int, optional
        Spectral-channel selector for SPHERE-IFS data.
    peakmethod : {"fft", "square", "unique", "gauss"}, default="fft"
        Method used to sample each Fourier splodge.
    n_wl : int, default=3
        Number of wavelengths used to sample the filter bandwidth.
    theta_detector : float, default=0
        Mask rotation relative to the detector in degrees.
    cutoff : float, default=1e-4
        Minimum simulated matched-filter weight retained for a peak pixel.
    hole_diam : float, default=0.8
        Aperture-hole diameter in metres.
    fw_splodge : float, default=0.7
        Relative Fourier-splodge size; also sets the Gaussian-method FWHM.
    scaling : float, default=1
        Multiplicative scale applied to aperture-mask coordinates.
    diag_plot : bool, default=False
        Whether to display the matched-filter overlap matrix.
    verbose : bool, default=False
        Whether to print mask-index information.
    display : bool, default=True
        Whether to display mask and Fourier-plane figures.
    save_to : str or path-like, optional
        Directory in which the mask-coordinate figure is saved.
    filename : str or path-like, optional
        Source filename used to name the saved figure.
    index_mask : object, optional
        Precomputed ``compute_index_mask(n_holes)`` for this mask, to avoid
        rebuilding it. Computed here when not given.

    Returns
    -------
    object or None
        Matched-filter data containing Fourier-pixel vectors and gains, baseline
        coordinates in metres, wavelength information in metres, pixel scale in
        radians per pixel, and overlap-correction matrices. Returns ``None`` when
        the instrument pixel scale or peak method is unavailable.

    Raises
    ------
    ValueError
        If an SPHERE-IFS spectral channel is not specified."""

    # Get detector, filter and mask informations
    # ------------------------------------------
    pixelsize = get_pixel_size(instrument)  # Pixel size of the detector [rad]
    if np.isnan(pixelsize):
        rprint(f"[red]Error: Pixel size unknown for {instrument}.", file=sys.stderr)
        return None
    # Wavelength of the filter (filt[0]: central, filt[1]: width)
    filt = get_wavelength(instrument, filtname)

    if instrument == "SPHERE-IFS":
        if i_wl is None:
            raise ValueError(
                "Your file seems to be obtained with IFU instrument: spectral "
                f"channel index `i_wl` must be specified (nlambda = {len(filt)}) and "
                "should be strictly identical to the one used for the cleaning step."
            )
        if isinstance(i_wl, int | np.integer):
            filt = [filt[i_wl], 0.001 * filt[i_wl]]
        else:
            filt = [np.mean(filt[i_wl[0] : i_wl[1]]), filt[i_wl[1]] - filt[i_wl[0]]]

    xy_coords = get_mask(instrument, maskname)  # mask coordinates

    x_mask = xy_coords[:, 0] * scaling
    y_mask = xy_coords[:, 1] * scaling

    x_mask_rot = x_mask * np.cos(np.deg2rad(theta_detector)) + y_mask * np.sin(
        np.deg2rad(theta_detector)
    )
    y_mask_rot = -x_mask * np.sin(np.deg2rad(theta_detector)) + y_mask * np.cos(
        np.deg2rad(theta_detector)
    )

    xy_coords_rot = []
    for i in range(len(x_mask)):
        xy_coords_rot.append([x_mask_rot[i], y_mask_rot[i]])
    xy_coords = np.array(xy_coords_rot)

    if display:
        import matplotlib.pyplot as plt

        _plot_mask_coord(xy_coords, maskname, instrument)
        if save_to is not None:
            figname = os.path.join(save_to, Path(filename).stem)
            plt.savefig(f"{figname}_{1}.pdf")

    n_holes = xy_coords.shape[0]

    if index_mask is None:
        index_mask = compute_index_mask(n_holes)
    elif index_mask.n_holes != n_holes:
        raise ValueError(
            f"index_mask is for {index_mask.n_holes} holes but mask "
            f"{maskname} has {n_holes}."
        )
    n_baselines = index_mask.n_baselines
    n_bispect = index_mask.n_bispect

    ncp_i = int((n_holes - 1) * (n_holes - 2) / 2)
    if verbose:
        rprint(
            "[cyan]---------------------------\n"
            f"{instrument.upper()} ({filtname}): {n_holes} holes masks\n"
            "---------------------------\n"
            f"nbl = {n_baselines}, nbs = {n_bispect}, "
            f"ncp_i = {ncp_i}, ncov = {index_mask.n_cov}"
        )

    # Consider the filter to be made up of n_wl wavelengths
    wl = np.arange(n_wl) / n_wl * filt[1]
    wl = wl - np.mean(wl) + filt[0]

    Sum, Sum_c = 0, 0

    mf_ix = np.zeros([2, n_baselines], dtype=int)  # matched filter
    mf_ix_c = np.zeros([2, n_baselines], dtype=int)  # matched filter

    if verbose:
        print("\n- Calculating sampling of", n_holes, "holes array...")

    innerpix, innerpix_center = _compute_center_splodge(
        npix, pixelsize, filt, hole_diam=hole_diam
    )

    u, v = _compute_uv_coord(
        xy_coords, index_mask, filt, pixelsize, npix, round_uv_to_pixel=False
    )

    mf_pvct = mf_gvct = mfc_pvct = mfc_gvct = None

    for i in range(n_baselines):
        args = {
            "i": i,
            "npix": npix,
            "pixelsize": pixelsize,
            "innerpix": innerpix,
            "innerpix_center": innerpix_center,
        }

        if peakmethod == "fft":
            ind_peak = _peak_fft_method(
                xy_coords=xy_coords, wl=wl, index_mask=index_mask, **args
            )
        elif peakmethod == "square":
            ind_peak = _peak_square_method(u=u, v=v, **args)
        elif peakmethod == "unique":
            ind_peak = _peak_one_method(u=u, v=v, **args)
        elif peakmethod == "gauss":
            ind_peak = _peak_gauss_method(
                u=u,
                v=v,
                filt=filt,
                index_mask=index_mask,
                fw_splodge=fw_splodge,
                **args,
                hole_diam=hole_diam,
            )
        else:
            rprint(
                "[red]Error: choose the extraction method 'gauss', 'fft' or 'square'.",
                file=sys.stderr,
            )
            return None

        # Compute the cutoff limit before saving the gain map
        pixelvector = np.where(ind_peak["flat"] >= cutoff)[0]
        pixelvector_c = np.where(ind_peak["centered"] >= cutoff)[0]

        # Now normalise the pixel gain, so that using the matched filter
        # on an ideal splodge is equivalent to just looking at the peak...
        if peakmethod == "gauss":
            pixelgain, pixelgain_c = _normalize_gain(
                ind_peak["gain_f"], ind_peak["gain_c"], pixelvector, pixelvector_c
            )
        else:
            pixelgain, pixelgain_c = _normalize_gain(
                ind_peak["flat"], ind_peak["centered"], pixelvector, pixelvector_c
            )

        mf_ix[0, i] = Sum
        Sum = Sum + len(pixelvector)
        mf_ix[1, i] = Sum

        mf_ix_c[0, i] = Sum_c
        Sum_c = Sum_c + len(pixelvector_c)
        mf_ix_c[1, i] = Sum_c

        if i == 0:
            mf_pvct = list(pixelvector)
            mf_gvct = list(pixelgain)
            mfc_pvct = list(pixelvector_c)
            mfc_gvct = list(pixelgain_c)
        else:
            mf_pvct.extend(list(pixelvector))
            mf_gvct.extend(list(pixelgain))
            mfc_pvct.extend(list(pixelvector_c))
            mfc_gvct.extend(list(pixelgain_c))

    mf = np.zeros(
        [npix, npix, n_baselines],
        dtype=[("norm", float), ("conj", float), ("norm_c", float), ("conj_c", float)],
    )

    for i in range(n_baselines):
        mf_tmp = np.zeros([npix, npix])
        mf_tmp_c = np.zeros([npix, npix])

        ind = mf_pvct[mf_ix[0, i] : mf_ix[1, i]]
        ind_c = mfc_pvct[mf_ix_c[0, i] : mf_ix_c[1, i]]

        mf_tmp.ravel()[ind] = mf_gvct[mf_ix[0, i] : mf_ix[1, i]]
        mf_tmp_c.ravel()[ind_c] = mfc_gvct[mf_ix_c[0, i] : mf_ix_c[1, i]]

        mf_tmp = mf_tmp.reshape([npix, npix])
        mf_tmp_c = mf_tmp_c.reshape([npix, npix])

        mf["norm"][:, :, i] = np.roll(mf_tmp, 0, axis=1)
        mf["norm_c"][:, :, i] = np.roll(mf_tmp_c, 0, axis=1)

        mf_temp_rot = np.roll(np.roll(np.rot90(np.rot90(mf_tmp)), 1, axis=0), 1, axis=1)
        mf_temp_rot_c = np.roll(
            np.roll(np.rot90(np.rot90(mf_tmp_c)), 1, axis=0), 1, axis=1
        )

        mf["conj"][:, :, i] = mf_temp_rot
        mf["conj_c"][:, :, i] = mf_temp_rot_c

        norm = np.sqrt(np.sum(mf["norm"][:, :, i] ** 2))

        mf["norm"][:, :, i] = mf["norm"][:, :, i] / norm
        mf["conj"][:, :, i] = mf["conj"][:, :, i] / norm

        mf["norm_c"][:, :, i] = mf["norm_c"][:, :, i] / norm
        mf["conj_c"][:, :, i] = mf["conj_c"][:, :, i] / norm

    rmat, imat = _make_overlap_mat(mf, n_baselines, display=diag_plot)

    mf_tot = np.sum(mf["norm"], axis=2) + np.sum(mf["conj"], axis=2)
    mf_tot_m = np.sum(mf["norm"], axis=2) - np.sum(mf["conj"], axis=2)

    im_uv = np.roll(np.fft.fftshift(mf_tot), 1, axis=1)

    if display:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(9, 7))
        plt.title(f"(u-v) plan - mask {maskname}", fontsize=14)
        plt.imshow(im_uv, origin="lower")
        plt.plot(npix // 2 + 1, npix // 2, "r+")
        plt.ylabel("Y [pix]")  # , fontsize=12)
        plt.xlabel("X [pix]")  # , fontsize=12)
        plt.tight_layout()

    out = {
        "cube": mf["norm"],
        "imat": imat,
        "rmat": rmat,
        "uv": im_uv,
        "tot": mf_tot,
        "tot_m": mf_tot_m,
        "pvct": mf_pvct,
        "gvct": mf_gvct,
        "cpvct": mfc_pvct,
        "cgvct": mfc_gvct,
        "ix": mf_ix,
        "u": u * filt[0],
        "v": v * filt[0],
        "wl": filt[0],
        "e_wl": filt[1],
        "pixelSize": pixelsize,
        "xy_coords": xy_coords,
    }

    return dict2class(out)


def compute_index_mask(n_holes, verbose=False):
    """Build aperture-mask baseline, bispectrum, and covariance indices.

    Parameters
    ----------
    n_holes : int
        Number of apertures in the mask.
    verbose : bool, default=False
        Whether to print the generated index arrays.

    Returns
    -------
    object
        Index mappings with ``n_baselines``, ``n_bispect``, and ``n_cov`` counts;
        ``h2bl_ix`` of shape ``(n_holes, n_holes)``; ``bl2h_ix`` of shape
        ``(2, n_baselines)``; ``bs2bl_ix`` of shape ``(3, n_bispect)``;
        ``bl2bs_ix`` of shape ``(n_baselines, n_holes - 2)``; and ``bscov2bs_ix``
        of shape ``(2, n_cov)``."""

    n_baselines = int(n_holes * (n_holes - 1) / 2)
    n_bispect = int(n_holes * (n_holes - 1) * (n_holes - 2) / 6)
    n_cov = int(n_holes * (n_holes - 1) * (n_holes - 2) * (n_holes - 3) / 4)

    # Given a pair of holes i,j h2bl_ix(i,j) gives the number of the baseline
    h2bl_ix = np.zeros([n_holes, n_holes], dtype=int)
    count = 0
    for i in range(n_holes - 1):
        for j in np.arange(i + 1, n_holes):
            h2bl_ix[i, j] = int(count)
            count = count + 1

    if verbose:
        print(h2bl_ix.T)  # transpose to display as IDL

    # Given a baseline, bl2h_ix gives the 2 holes that go to make it up
    bl2h_ix = np.zeros([2, n_baselines], dtype=int)

    count = 0
    for i in range(n_holes - 1):
        for j in np.arange(i + 1, n_holes):
            bl2h_ix[0, count] = int(i)
            bl2h_ix[1, count] = int(j)
            count = count + 1

    if verbose:
        print(bl2h_ix.T)  # transpose to display as IDL

    # Given a point in the bispectrum, bs2bl_ix gives the 3 baselines which
    # make the triangle. bl2bs_ix gives the index of all points in the
    # bispectrum containing a given baseline.

    bs2bl_ix = np.zeros([3, n_bispect], dtype=int)
    temp = np.zeros([n_baselines], dtype=int)  # N_baselines * a count variable

    if verbose:
        print("Indexing bispectrum...")

    bl2bs_ix = np.zeros([n_baselines, n_holes - 2], dtype=int)
    count = 0

    for i in range(n_holes - 2):
        for j in np.arange(i + 1, n_holes - 1):
            for k in np.arange(j + 1, n_holes):
                bs2bl_ix[0, count] = int(h2bl_ix[i, j])
                bs2bl_ix[1, count] = int(h2bl_ix[j, k])
                bs2bl_ix[2, count] = int(h2bl_ix[i, k])
                bl2bs_ix[bs2bl_ix[0, count], temp[bs2bl_ix[0, count]]] = count
                bl2bs_ix[bs2bl_ix[1, count], temp[bs2bl_ix[1, count]]] = count
                bl2bs_ix[bs2bl_ix[2, count], temp[bs2bl_ix[2, count]]] = count
                temp[bs2bl_ix[0, count]] = temp[bs2bl_ix[0, count]] + 1
                temp[bs2bl_ix[1, count]] = temp[bs2bl_ix[1, count]] + 1
                temp[bs2bl_ix[2, count]] = temp[bs2bl_ix[2, count]] + 1
                count += 1

    if verbose:
        print(bl2bs_ix.T)  # transpose to display as IDL
        print("Indexing the bispectral covariance...")

    # bscov2bs_ix lists every pair of bispectra (i < j) that share a baseline,
    # in row-major (i, j) order. Two distinct triangles share at most one
    # baseline, so the pairs are exactly all pairs drawn from the same row of
    # bl2bs_ix (the n_holes - 2 bispectra containing that baseline), each
    # appearing once. Collect them per baseline, then sort by (i, j).
    a, b = np.triu_indices(n_holes - 2, 1)
    first = bl2bs_ix[:, a].ravel()
    second = bl2bs_ix[:, b].ravel()
    pairs = np.array([np.minimum(first, second), np.maximum(first, second)])
    pairs = pairs[:, np.lexsort((pairs[1], pairs[0]))]

    bscov2bs_ix = np.zeros([2, n_cov], dtype=int)
    bscov2bs_ix[:, : pairs.shape[1]] = pairs

    if verbose:
        print(bscov2bs_ix.T)

    indices_mask = dict2class(
        {
            "n_baselines": n_baselines,
            "n_bispect": n_bispect,
            "n_cov": n_cov,
            "h2bl_ix": h2bl_ix,
            "bl2h_ix": bl2h_ix,
            "bs2bl_ix": bs2bl_ix,
            "bl2bs_ix": bl2bs_ix,
            "bscov2bs_ix": bscov2bs_ix,
            "n_holes": n_holes,
        }
    )
    return indices_mask


def give_peak_info2d(mf, n_baselines, dim1, dim2):
    """Convert flattened matched-filter peak indices to image coordinates.

    Parameters
    ----------
    mf : object
        Matched-filter object returned by ``make_mf``.
    n_baselines : int
        Number of mask baselines.
    dim1, dim2 : int
        Image dimensions in pixels.

    Returns
    -------
    numpy.ndarray of shape (n_baselines,)
        Object array whose element for each baseline has rows ``(y, x, gain)`` for
        its sampled Fourier-peak pixels."""

    x, y = np.arange(dim1), np.arange(dim2)
    X, Y = np.meshgrid(x, y)

    List_peak = []
    for j in range(n_baselines):
        l_x = X.ravel()[mf.pvct[mf.ix[0, j] : mf.ix[1, j]]]  # .astype(int)
        l_y = Y.ravel()[mf.pvct[mf.ix[0, j] : mf.ix[1, j]]]  # .astype(int)
        g = mf.gvct[mf.ix[0, j] : mf.ix[1, j]]

        peak = [[int(l_y[k]), int(l_x[k]), g[k]] for k in range(len(l_x))]

        List_peak.append(np.array(peak))
    return np.array(List_peak, dtype=object)


def clos_unique(closing_tri_pix):
    """Remove duplicate closure triangles independent of pixel ordering.

    Parameters
    ----------
    closing_tri_pix : numpy.ndarray of shape (3, n_triangles)
        Flattened Fourier-pixel indices for candidate triangle vertices.

    Returns
    -------
    numpy.ndarray
        Columns of ``closing_tri_pix`` corresponding to unique sorted triplets."""
    L, L_i = [], []
    for i in range(closing_tri_pix.shape[1]):
        p1 = str(closing_tri_pix[0, i])
        p2 = str(closing_tri_pix[1, i])
        p3 = str(closing_tri_pix[2, i])

        p = np.sort(closing_tri_pix[:, i])

        p1 = str(p[0])
        p2 = str(p[1])
        p3 = str(p[2])

        val = p1 + p2 + p3

        if val not in L:
            L.append(val)
            L_i.append(i)
        else:
            pass

    return closing_tri_pix[:, L_i]


def tri_pix(array_size, sampledisk_r, verbose=True, display=True):
    """Enumerate pixel triangles that close within a Fourier splodge.

    Parameters
    ----------
    array_size : int
        Size of the square Fourier image in pixels.
    sampledisk_r : float
        Radius of the sampled splodge in pixels.
    verbose : bool, default=True
        Whether to print the number of candidate triangles.
    display : bool, default=True
        Whether to display sampled unique triangles.

    Returns
    -------
    numpy.ndarray of shape (3, n_triangles)
        Flattened Fourier-pixel indices for all candidate closing triangles."""

    if array_size % 2 == 1:
        rprint(
            f"[red]\n! Warnings: image dimension must be even ({array_size})\n"
            "Possible triangle inside the splodge should be incorrect.\n",
            file=sys.stderr,
        )

    d = np.zeros([array_size, array_size])
    d = plot_circle(d, array_size // 2, array_size // 2, sampledisk_r, display=False)
    pvct_flat = np.where(d.ravel() > 0)
    npx = len(pvct_flat[0])
    for px1 in range(npx):
        thispix1 = np.array(array_coords(pvct_flat[0][px1], array_size))

        roll1 = np.roll(d, int(array_size // 2 - thispix1[0]), axis=0)
        roll2 = np.roll(roll1, int(array_size // 2 - thispix1[1]), axis=1)

        xcor_12 = d + roll2

        valid_b12_vct = np.where(xcor_12.T.ravel() > 1)

        thisntriv = len(valid_b12_vct[0])

        thistrivct = np.zeros([3, thisntriv])

        for px2 in range(thisntriv):
            thispix2 = array_coords(valid_b12_vct[0][px2], array_size)
            thispix3 = array_size * 1.5 - (thispix1 + thispix2)
            bl3_px = thispix3[1] * array_size + (thispix3[0])
            thistrivct[:, px2] = [pvct_flat[0][px1], valid_b12_vct[0][px2], bl3_px]

        if px1 == 0:
            l_pix1 = list(thistrivct[0, :])
            l_pix2 = list(thistrivct[1, :])
            l_pix3 = list(thistrivct[2, :])
        else:
            l_pix1.extend(list(thistrivct[0, :]))
            l_pix2.extend(list(thistrivct[1, :]))
            l_pix3.extend(list(thistrivct[2, :]))

    closing_tri_pix = np.array([l_pix1, l_pix2, l_pix3]).astype(int)

    n_trip = closing_tri_pix.shape[1]

    if verbose:
        print(f"Closing triangle in r = {sampledisk_r:2.1f}: {n_trip}")

    cl_unique = clos_unique(closing_tri_pix)

    c0 = array_size // 2
    fw = 2 * sampledisk_r

    if display:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(5, 5))
        plt.title(
            "Splodge + unique triangle ("
            f"tri = {cl_unique.shape[1]}/{n_trip}, "
            f"d = {array_size}, r = {sampledisk_r:2.1f} pix"
            ")",
            fontsize=10,
        )
        plt.imshow(d)
        for i in range(cl_unique.shape[1]):
            trip1 = cl_unique[:, i]
            X, Y = array_coords(trip1, array_size)
            X = list(X)
            Y = list(Y)
            X.append(X[0])
            Y.append(Y[0])
            plt.plot(X, Y, "-", lw=1)
            plt.axis([c0 - fw, c0 + fw, c0 - fw, c0 + fw])

        plt.tight_layout()
        plt.show(block=False)
    return closing_tri_pix


def bs_multi_triangle(i, bs_arr, ft_frame, bs2bl_ix, mf, closing_tri_pix):
    """Accumulate a bispectrum sample using multiple closing triangles.

    Parameters
    ----------
    i : int
        Frame index in ``bs_arr``.
    bs_arr : numpy.ndarray of complex, shape (n_frames, n_bispectra)
        Bispectrum accumulator.
    ft_frame : numpy.ndarray of complex, shape (ny, nx)
        Fourier transform of the current image frame.
    bs2bl_ix : numpy.ndarray of int, shape (3, n_bispectra)
        Baseline indices forming each bispectrum.
    mf : object
        Matched-filter object returned by ``make_mf``.
    closing_tri_pix : numpy.ndarray
        Flattened-pixel combinations that close within a splodge.

    Returns
    -------
    numpy.ndarray of complex, shape (n_frames, n_bispectra)
        ``bs_arr`` with row ``i`` populated from the multiple-triangle products."""
    dim1 = ft_frame.shape[0]
    dim2 = ft_frame.shape[1]

    closing_tri_pix = closing_tri_pix.T
    n_bispect = bs_arr.shape[1]

    mfilter_spec = np.zeros([dim1, dim2])

    mfc_pvct = array_coords(mf.cpvct, dim1)
    coord_peak = array_coords(mf.cpvct, dim1)

    for j in range(len(mfc_pvct[0])):
        mfilter_spec[coord_peak[1][j], coord_peak[0][j]] = mf.cgvct[j]

    mfilter_spec_op = np.roll(
        np.roll(np.rot90(np.rot90(mfilter_spec)), 1, axis=0), 1, axis=1
    )

    mfilter_spec += mfilter_spec_op

    mfilter_spec = mfilter_spec * np.fft.fftshift(ft_frame)

    base_origin = closing_tri_pix[0, 0]

    n_closing_tri = closing_tri_pix.shape[0]

    All_multi_tri = []

    for this_bs in range(n_bispect):
        this_tri = bs2bl_ix[:, this_bs]

        mfc_vect = np.array(mf.cpvct)

        tri_splodge_origin = mfc_vect[mf.ix[0, this_tri]]  # -80

        splodge_shift = tri_splodge_origin - base_origin

        splodge_shift = splodge_shift.reshape([1, len(splodge_shift)])

        sh = np.ones([1, n_closing_tri])

        this_trisampling = closing_tri_pix + np.dot(splodge_shift.T, sh).T

        this_trisampling = this_trisampling.T

        a = (splodge_shift + ((dim1 / 2) * (dim1 + 1))).astype(int)[0]

        spl_offset = array_coords(a, dim1).T - dim1 // 2
        spl_offset = spl_offset.T

        this_trisampling[2, :] = (
            this_trisampling[2, :]
            - 2 * (spl_offset[0, 2] + 1 * spl_offset[1, 2] * dim1)
            + 0
        )
        this_trisampling[0, :] -= 0
        this_trisampling[1, :] -= 0
        this_trisampling = this_trisampling.astype(int)

        mfilter_spec2 = mfilter_spec.ravel()

        bs_arr[i, this_bs] = np.sum(
            mfilter_spec2[this_trisampling[0, :]]
            * mfilter_spec2[this_trisampling[1, :]]
            * (mfilter_spec2[this_trisampling[2, :]])
        )

        All_multi_tri.append(this_trisampling)

    return bs_arr


def find_bad_holes(bs, bmax=6, verbose=False, display=False):
    """Identify apertures with anomalously low calibrated visibility.

    Parameters
    ----------
    bs : object
        Extracted AMI observables with baseline coordinates, wavelength, squared
        visibilities, and mask indices.
    bmax : float, default=6
        Maximum baseline length in metres shown in the diagnostic fit.
    verbose : bool, default=False
        Whether to print linear-fit information.
    display : bool, default=False
        Whether to display the visibility-fit diagnostic.

    Returns
    -------
    numpy.ndarray of int
        Indices of apertures whose associated visibility ratio is at most 0.3."""
    u = bs.u / bs.wl
    v = bs.v / bs.wl
    X = np.sqrt(u**2 + v**2)
    Y = np.log(bs.vis2)

    n_holes = bs.mask.n_holes
    bl2h_ix = bs.mask.bl2h_ix

    param = {"a": 1, "b": 0}

    fit = leastsqFit(linear, X, param, Y, verbose=verbose)

    xm = np.linspace(0, bmax, 100) / bs.wl
    ym = linear(xm, fit["best"])

    pfit = list(fit["best"].values())

    if display:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(X / 1e6, Y, ".", label="data")
        plt.plot(xm / 1e6, ym, "--", label=f"fit (a={pfit[0]:2.1e}, b={pfit[1]:2.2f})")
        plt.grid(alpha=0.1)
        plt.legend()
        plt.xlim(0, xm.max() / 1e6)
        plt.ylabel(r"$\log(V^2)$ (calibrator)")
        plt.xlabel(r"Sp. Freq. [M$\lambda$]")
        plt.tight_layout()

    bad_holes = []
    for j in range(n_holes):
        w = np.where((bl2h_ix[0, :] == j) | (bl2h_ix[1, :] == j))
        if (np.mean(bs.vis2[w] / np.exp(fit["model"][w]))) <= 0.3:
            bad_holes.append(j)

    bad_holes = np.unique(bad_holes)

    return bad_holes


def find_bad_BL_BS(bad_holes, bs):
    """Find observables affected by a set of rejected apertures.

    Parameters
    ----------
    bad_holes : array-like of int
        Aperture indices identified as bad.
    bs : object
        Extracted AMI observables containing mask baseline and bispectrum indices.

    Returns
    -------
    bad_baselines : numpy.ndarray of int
        Baseline indices containing a bad aperture.
    bad_bispect : numpy.ndarray of int
        Bispectrum indices containing a bad baseline.
    good_baselines : array-like of int
        Baseline indices not rejected.
    good_bispectrum : array-like of int
        Bispectrum indices not rejected."""

    bl2h_ix = bs.mask.bl2h_ix
    bs2bl_ix = bs.mask.bs2bl_ix

    n_baselines = bl2h_ix.shape[1]
    n_bispect = bs2bl_ix.shape[1]

    bad_baselines, bad_bispect = [], []

    good_baselines = np.arange(n_baselines)
    good_bispectrum = np.arange(n_bispect)

    if len(bad_holes) == 0:
        good_baselines = np.arange(n_baselines)
        good_bispectrum = np.arange(n_bispect)
        res = bad_baselines, bad_bispect, good_baselines, good_bispectrum
    else:
        for i in range(len(bad_holes)):
            new_bad = list(
                np.where(
                    (bl2h_ix[0, :] == bad_holes[i]) | (bl2h_ix[1, :] == bad_holes[i])
                )[0]
            )
            bad_baselines.extend(new_bad)

        bad_baselines = np.unique(bad_baselines)

        if len(bad_baselines) != 0:
            for i in range(len(bad_baselines)):
                new_bad = np.where(
                    (bs2bl_ix[0, :] == bad_baselines[i])
                    | (bs2bl_ix[1, :] == bad_baselines[i])
                    | (bs2bl_ix[2, :] == bad_baselines[i])
                )[0]
                bad_bispect.extend(new_bad)

        bad_bispect = np.unique(bad_bispect)

        good_baselines = [bl for bl in good_baselines if bl not in bad_baselines]
        good_bispectrum = [
            bs_el for bs_el in good_bispectrum if bs_el not in bad_bispect
        ]

        res = bad_baselines, bad_bispect, good_baselines, good_bispectrum

    return res


def phase_chi2(p, fitmat, ph_mn, ph_err):
    """Evaluate the wrapped phase chi-square for a piston model.

    Parameters
    ----------
    p : numpy.ndarray of shape (n_holes,)
        Fitted aperture-piston parameters in radians.
    fitmat : numpy.ndarray of shape (n_holes, n_baselines + 1)
        Matrix mapping aperture pistons to baseline phases plus a reference term.
    ph_mn : numpy.ndarray of shape (n_baselines,)
        Mean baseline phases in radians.
    ph_err : numpy.ndarray of shape (n_baselines,)
        Uncertainties on mean baseline phases in radians.

    Returns
    -------
    float
        Sum of squared wrapped phase residuals weighted by phase variance."""
    piston = np.dot(p, fitmat)

    tmp = list(ph_mn)
    tmp.append(0)
    tmp = np.array(tmp)
    e_tmp = list(ph_err)
    e_tmp.append(0.01)
    e_tmp = np.array(e_tmp) ** 2
    arg = np.array(tmp - piston) * 1j
    phase_chi2 = np.sum(np.abs(1 - np.exp(arg)) ** 2 / e_tmp)
    return phase_chi2
