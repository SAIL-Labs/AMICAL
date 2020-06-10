# -*- coding: utf-8 -*-
"""
@author: Anthony Soulain (University of Sydney)

--------------------------------------------------------------------
MIAMIS: Multi-Instruments Aperture Masking Interferometry Software
--------------------------------------------------------------------

Matched filter sub-pipeline method.

All AMI related function, the most important are:
- make_mf: compute splodge positions for a given mask,
- bs_multiTriangle: compute bispectrum using multiple triangle method,
- tri_pix: compute unique closing triangle for a given splodge.

--------------------------------------------------------------------
"""

import numpy as np
from matplotlib import pyplot as plt
from munch import munchify as dict2class
from termcolor import cprint

from miamis.dpfit import leastsqFit
from miamis.getInfosObs import GetMaskPos, GetPixelSize, GetWavelength
from miamis.mf_pipeline.idl_function import array_coords, dist
from miamis.tools import linear, plot_circle


def index_mask(n_holes, verbose=False):
    """
    This function generates index arrays for an N-hole mask.

    Parameters:
    -----------

    `n_holes`: int
       number of holes in the array.

    Returns:
    --------

    `n_baselines`: int
        The number of different baselines (n_holes*(n_holes-1)/2),\n
    `n_bispect`: int
        The number of bispectrum elements (n_holes*(n_holes-1)*(n_holes-2)/6),\n
    `n_cov`: int
        The number of bispectrum covariance
        (n_holes*(n_holes-1)*(n_holes-2)*(n_holes-3)/4),\n
    `h2bl_ix`: numpy.array
        Holes to baselines index,\n
    `bl2h_ix`: numpy.array
                Baselines to holes index,\n
    `bs2bl_ix`: numpy.array
        Bispectrum to baselines index,\n
    `bl2bs_ix`	: numpy.array
        Baselines to bispectrum index,\n
    `bscov2bs_ix`: numpy.array,
        Bispectrum covariance to bispectrum index.

    """

    n_baselines = int(n_holes*(n_holes-1)/2)
    n_bispect = int(n_holes*(n_holes-1)*(n_holes-2)/6)
    n_cov = int(n_holes*(n_holes-1)*(n_holes-2)*(n_holes-3)/4)

    # Given a pair of holes i,j h2bl_ix(i,j) gives the number of the baseline
    h2bl_ix = np.zeros([n_holes, n_holes], dtype=int)
    count = 0
    for i in range(n_holes-1):
        for j in np.arange(i+1, n_holes):
            h2bl_ix[i, j] = int(count)
            count = count+1

    if verbose:
        print(h2bl_ix.T)  # transpose to display as IDL

    # Given a baseline, bl2h_ix gives the 2 holes that go to make it up
    bl2h_ix = np.zeros([2, n_baselines], dtype=int)

    count = 0
    for i in range(n_holes-1):
        for j in np.arange(i+1, n_holes):
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
        print('Indexing bispectrum...')

    bl2bs_ix = np.zeros([n_baselines, n_holes-2], dtype=int)
    count = 0

    for i in range(n_holes-2):
        for j in np.arange(i+1, n_holes-1):
            for k in np.arange(j+1, n_holes):
                bs2bl_ix[0, count] = int(h2bl_ix[i, j])
                bs2bl_ix[1, count] = int(h2bl_ix[j, k])
                bs2bl_ix[2, count] = int(h2bl_ix[i, k])
                bl2bs_ix[bs2bl_ix[0, count], temp[bs2bl_ix[0, count]]] = count
                bl2bs_ix[bs2bl_ix[1, count], temp[bs2bl_ix[1, count]]] = count
                bl2bs_ix[bs2bl_ix[2, count], temp[bs2bl_ix[2, count]]] = count
                temp[bs2bl_ix[0, count]] = temp[bs2bl_ix[0, count]]+1
                temp[bs2bl_ix[1, count]] = temp[bs2bl_ix[1, count]]+1
                temp[bs2bl_ix[2, count]] = temp[bs2bl_ix[2, count]]+1
                count += 1

    if verbose:
        print(bl2bs_ix.T)  # transpose to display as IDL
        print('Indexing the bispectral covariance...')

    bscov2bs_ix = np.zeros([2, n_cov], dtype=int)

    count = 0

    for i in range(n_bispect-1):
        for j in np.arange(i+1, n_bispect):
            if ((bs2bl_ix[0, i] == bs2bl_ix[0, j]) or (bs2bl_ix[1, i] == bs2bl_ix[0, j]) or
                (bs2bl_ix[2, i] == bs2bl_ix[0, j]) or (bs2bl_ix[0, i] == bs2bl_ix[1, j]) or
                (bs2bl_ix[1, i] == bs2bl_ix[1, j]) or (bs2bl_ix[2, i] == bs2bl_ix[1, j]) or
                (bs2bl_ix[0, i] == bs2bl_ix[2, j]) or (bs2bl_ix[1, i] == bs2bl_ix[2, j]) or
                    (bs2bl_ix[2, i] == bs2bl_ix[2, j])):
                bscov2bs_ix[0, count] = i
                bscov2bs_ix[1, count] = j
                count += 1

    if verbose:
        print(bscov2bs_ix.T)

    return n_baselines, n_bispect, n_cov, h2bl_ix, bl2h_ix, bs2bl_ix, bl2bs_ix, bscov2bs_ix


def make_mf(maskname, instrument, filtname, npix,
            peakmethod=True, n_wl=3, cutoff=1e-4, D=6.5,
            hole_diam=0.8, verbose=False,
            display=True):
    """
    Parameters:
    -----------
    `maskname`: str
        Name of the mask (number of holes),\n
    `npix`: int
        Size of the image,\n
    `instrument`: str
        Instrument used (default = jwst),\n
    `mas_pixel`: float
        Pixel size of the detector [mas] (default = 65.6 mas for NIRISS),\n
    `peakmethod`: boolean
        If True, perform FFTs to compute the peak position in the Fourier space (default).
        Otherwise, set the u-v peak sampled into 4 pixels,\n
    `n_wl`: int
        number of wavelengths to use to simulate bandwidth,\n
    `cutoff`: float
        cut between noise and signal pixels in simulated transforms,\n
    `D`: float
        Diameter of the primary mirror,\n
    `hole_diam`: float
        Diameter of a single aperture (0.8 for JWST).
    """

    # Get detector, filter and mask informations
    # ------------------------------------------
    pixelSize = GetPixelSize(instrument)  # Pixel size of the detector [rad]
    # Wavelength of the filter (filt[0]: central, filt[1]: width)
    filt = GetWavelength(instrument, filtname)
    xy_coords = GetMaskPos(instrument, maskname)  # mask coordinates

    if display:
        if instrument == 'NIRISS':
            marker = 'H'
        else:
            marker = 'o'

        plt.figure(figsize=(6, 5.5))
        plt.title('%s - mask %s' % (instrument, maskname), fontsize=14)

        xy_coords_tel = list(xy_coords.copy())
        xy_coords_tel.append([0, 2.64])
        xy_coords_tel.append([1.14315, -1.98])
        xy_coords_tel.append([-1.14315, -1.98])
        xy_coords_tel.append([2.28631, 0])
        xy_coords_tel.append([-2.28631, -1.32])
                
        xy_coords_tel = np.array(xy_coords_tel)
        print(xy_coords_tel.shape)
        for i in range(xy_coords.shape[0]):
            plt.scatter(xy_coords[i][0], xy_coords[i][1],
                        s=1e2, c='', edgecolors='navy', marker=marker)
            plt.text(xy_coords[i][0]+0.1, xy_coords[i][1]+0.1, i)

        # if instrument == 'NIRISS':
        #     for i in range(len(xy_coords_tel)):
        #         plt.scatter(xy_coords_tel[i][0], xy_coords_tel[i][1],
        #                     s=5.3e3, c='', edgecolors='k', marker=marker)

        plt.xlabel('Aperture x-coordinate [m]', fontsize=12)
        plt.ylabel('Aperture y-coordinate [m]', fontsize=12)
        plt.axis([-D/2., D/2., -D/2., D/2.])
        plt.tight_layout()
        plt.show(block=False)

    # Normalize total( mf[pixelgain] ) = 1
    # (Unnecessary for calibrated amplitudes, necessary for uncalibrated amplitudes.)
    # ---------------------------------------
    normalize_pixelgain = True

    # ----------------------------------------------------------
    # Automatic from here
    # ----------------------------------------------------------

    n_holes = xy_coords.shape[0]

    n_baselines, n_bispect, n_cov, h2bl_ix, bl2h_ix, bs2bl_ix, bl2bs_ix, bscov2bs_ix = index_mask(
        n_holes)

    ncp_i = int((n_holes - 1)*(n_holes - 2)/2)
    if verbose:
        cprint('---------------------------', 'cyan')
        cprint('%s (%s): %i holes masks' %
               (instrument.upper(), filtname, n_holes), 'cyan')
        cprint('---------------------------', 'cyan')
        cprint('nbl = %i, nbs = %i, ncp_i = %i, ncov = %i' %
               (n_baselines, n_bispect, ncp_i, n_cov), 'cyan')

    # Consider the filter to be made up of n_wl wavelengths
    wl = np.arange(n_wl)/n_wl*filt[1]
    wl = wl - np.mean(wl) + filt[0]

    u = np.zeros(n_baselines)
    v = np.zeros(n_baselines)

    Sum, Sum_c = 0, 0

    mf_ix = np.zeros([2, n_baselines], dtype=int)  # matched filter
    mf_ix_c = np.zeros([2, n_baselines], dtype=int)  # matched filter

    if verbose:
        print('\n- Calculating sampling of', n_holes, 'holes array...')

    # Why 0.9 and 0.6 factor here ???
    tmp = dist(npix)  # .ravel()
    innerpix = np.array(np.array(
        np.where(tmp < (hole_diam/filt[0]*pixelSize*npix)*0.9))*0.6, dtype=int)

    ap1all = []
    ap2all = []
    mfall = []

    round_uv_to_pixel = False

    for i in range(n_baselines):

        if not round_uv_to_pixel:
            u[i] = (xy_coords[bl2h_ix[0, i], 0] -
                    xy_coords[bl2h_ix[1, i], 0])/filt[0]
            v[i] = (xy_coords[bl2h_ix[0, i], 1] -
                    xy_coords[bl2h_ix[1, i], 1])/filt[0]
        else:
            onepix = 1./(npix*pixelSize)
            onepix_xy = onepix*filt[0]
            new_xy = (xy_coords/onepix_xy).astype(int)*onepix_xy

            u[i] = (new_xy[bl2h_ix[0, i], 0] -
                    new_xy[bl2h_ix[1, i], 0])/filt[0]
            v[i] = (new_xy[bl2h_ix[0, i], 1] -
                    new_xy[bl2h_ix[1, i], 1])/filt[0]

        mf = np.zeros([npix, npix])

        if peakmethod:
            sum_xy = np.sum(xy_coords, axis=0)/n_holes
            shift_fact = np.ones([n_holes, 2])
            shift_fact[:, 0] = sum_xy[0]
            shift_fact[:, 1] = sum_xy[1]
            xy_coords2 = xy_coords.copy()
            xy_coords2 -= shift_fact

            for j in range(n_wl):
                xyh = xy_coords2[bl2h_ix[0, i], :]/wl[j]*pixelSize * \
                    npix + npix//2  # round 5 precision IDL
                delta = xyh-np.floor(xyh)
                ap1 = np.zeros([npix, npix])
                x1 = int(xyh[1])
                y1 = int(xyh[0])
                ap1[x1, y1] = (1.-delta[0]) * (1.-delta[1])
                ap1[x1, y1+1] = delta[0]*(1.-delta[1])
                ap1[x1+1, y1] = (1.-delta[0])*delta[1]
                ap1[x1+1, y1+1] = delta[0]*delta[1]

                ap1all.append(np.roll(ap1, 0, axis=0))

                xyh = xy_coords2[bl2h_ix[1, i], :]/wl[j]*pixelSize * \
                    npix + npix//2
                delta = xyh-np.floor(xyh)
                ap2 = np.zeros([npix, npix])
                x2 = int(xyh[1])
                y2 = int(xyh[0])
                ap2[x2, y2] = (1.-delta[0])*(1.-delta[1])
                ap2[x2, y2+1] = delta[0]*(1.-delta[1])
                ap2[x2+1, y2] = (1.-delta[0])*delta[1]
                ap2[x2+1, y2+1] = delta[0]*delta[1]

                ap2all.append(np.roll(ap2, [0, 0]))

                n_elts = npix**2

                tmf = (np.fft.fft2(ap1)/n_elts *
                       np.conj(np.fft.fft2(ap2)/n_elts))
                tmf = np.fft.fft2(tmf)
                mf = mf+np.real(tmf)
                mfall.append(mf)

        else:
            uv = np.array([v[i], u[i]])*pixelSize*npix
            uv = (uv + npix) % npix
            uv_int = np.array(np.floor(uv), dtype=int)
            uv_frac = uv - uv_int
            mf[uv_int[0], uv_int[1]] = (1-uv_frac[0])*(1-uv_frac[1])
            mf[uv_int[0], (uv_int[1]+1) % npix] = (1-uv_frac[0])*uv_frac[1]
            mf[(uv_int[0]+1) % npix, uv_int[1]] = uv_frac[0]*(1-uv_frac[1])
            mf[(uv_int[0]+1) % npix, (uv_int[1]+1) %
                npix] = uv_frac[0]*uv_frac[1]
            mf = np.roll(mf, [0, 0])

        mf_flat = mf.ravel()

        mf_centered = np.fft.fftshift(mf).ravel()

        mf_flat = mf_flat/np.max(mf_flat)  # normalize for cutoff purposes...
        mf_centered = mf_centered/np.max(mf_centered)

        x, y = np.meshgrid(npix, npix)
        dist_c = np.sqrt((x-npix//2)**2 + (y-npix//2)**2)
        innerpix_center = np.array(np.array(
            np.where(dist_c < (hole_diam/filt[0]*pixelSize*npix)*0.9))*0.6, dtype=int)

        mf_centered[innerpix_center] = 0.0
        mf_flat[innerpix] = 0.0

        pixelvector = np.where(mf_flat >= cutoff)[0]
        pixelvector_c = np.where(mf_centered >= cutoff)[0]

        # Now normalise the pixel gain, so that using the matched filter
        # on an ideal splodge is equivalent to just looking at the peak...
        if normalize_pixelgain:
            pixelgain = mf_flat[pixelvector] / np.sum(mf_flat[pixelvector])
            pixelgain_c = mf_centered[pixelvector_c] / \
                np.sum(mf_centered[pixelvector_c])
        else:
            pixelgain = mf_flat[pixelvector] * \
                np.max(mf_flat[pixelvector])/np.sum(mf_flat[pixelvector]**2)

        mf_ix[0, i] = Sum
        Sum = Sum + len(pixelvector)
        mf_ix[1, i] = Sum

        mf_ix_c[0, i] = Sum_c
        Sum_c = Sum_c + len(pixelvector_c)
        mf_ix_c[1, i] = Sum_c

        if (i == 0):
            mf_pvct = list(pixelvector)
            mf_gvct = list(pixelgain)
            mfc_pvct = list(pixelvector_c)
            mfc_gvct = list(pixelgain_c)
        else:
            mf_pvct.extend(list(pixelvector))
            mf_gvct.extend(list(pixelgain))
            mfc_pvct.extend(list(pixelvector_c))
            mfc_gvct.extend(list(pixelgain_c))

    mf = np.zeros([npix, npix, n_baselines])
    mf_c = np.zeros([npix, npix, n_baselines])

    mf_conj = np.zeros([npix, npix, n_baselines])
    mf_conj_c = np.zeros([npix, npix, n_baselines])

    mf_rmat = np.zeros([n_baselines, n_baselines])
    mf_imat = np.zeros([n_baselines, n_baselines])

    if verbose:
        print('- Compute matched filters...')
    # ;Now fill-in the huge matched-filter cube (to be released later)
    for i in range(n_baselines):
        mf_temp = np.zeros([npix, npix])
        mf_temp_c = np.zeros([npix, npix])

        ind = mf_pvct[mf_ix[0, i]:mf_ix[1, i]]
        ind_c = mfc_pvct[mf_ix_c[0, i]:mf_ix_c[1, i]]

        mf_temp.ravel()[ind] = mf_gvct[mf_ix[0, i]:mf_ix[1, i]]
        mf_temp_c.ravel()[ind_c] = mfc_gvct[mf_ix_c[0, i]:mf_ix_c[1, i]]

        mf_temp2 = mf_temp.reshape([npix, npix])
        mf_temp2_c = mf_temp_c.reshape([npix, npix])

        mf[:, :, i] = np.roll(mf_temp2, 0, axis=1)
        mf_c[:, :, i] = np.roll(mf_temp2_c, 0, axis=1)

        mf_temp2_rot = np.roll(
            np.roll(np.rot90(np.rot90(mf_temp2)), 1, axis=0), 1, axis=1)
        mf_temp2_rot_c = np.roll(
            np.roll(np.rot90(np.rot90(mf_temp2_c)), 1, axis=0), 1, axis=1)

        mf_conj[:, :, i] = mf_temp2_rot
        mf_conj_c[:, :, i] = mf_temp2_rot_c

        norm = np.sqrt(np.sum(mf[:, :, i]**2))

        mf[:, :, i] = mf[:, :, i]/norm
        mf_conj[:, :, i] = mf_conj[:, :, i]/norm

        mf_c[:, :, i] = mf_c[:, :, i]/norm
        mf_conj_c[:, :, i] = mf_conj_c[:, :, i]/norm

    # Now find the overlap matrices
    for i in range(n_baselines):
        pix_on = np.where(mf[:, :, i] != 0.0)
        for j in range(n_baselines):
            t1 = np.sum(mf[:, :, i][pix_on]*mf[:, :, j][pix_on])
            t2 = np.sum(mf[:, :, i][pix_on]*mf_conj[:, :, j][pix_on])
            mf_rmat[i, j] = t1 + t2
            mf_imat[i, j] = t1 - t2

    if display:
        plt.figure(figsize=(6, 6))
        plt.title('Overlap matrix', fontsize=14)
        plt.imshow(mf_imat, cmap='gray')
        plt.ylabel('# baselines', fontsize=12)
        plt.xlabel('# baselines', fontsize=12)
        plt.tight_layout()

    # This next big is for diagnostics...
    mf_tot = np.sum(mf, axis=2) + np.sum(mf_conj, axis=2)
    mf_tot_m = np.sum(mf, axis=2) - np.sum(mf_conj, axis=2)

    w = np.where(mf_tot == 0)
    mask = np.zeros([npix, npix])
    mask[w] = 1.0

    if verbose:
        print('- Inverting Matrices.')
    mf_rmat = np.linalg.inv(mf_rmat)
    mf_imat = np.linalg.inv(mf_imat)
    mf_rmat[np.where(mf_rmat < 1e-6)] = 0.0
    mf_imat[np.where(mf_imat < 1e-6)] = 0.0
    mf_rmat[mf_rmat >= 2] = 2
    mf_imat[mf_imat >= 2] = 2
    mf_imat[mf_imat <= -2] = -2

    im_uv = np.roll(np.fft.fftshift(mf_tot), 1, axis=1)

    if display:
        plt.figure(figsize=(6, 6))
        plt.title('(u-v) plan - mask %s' %
                  (maskname), fontsize=14)
        plt.imshow(im_uv, origin='lower')
        plt.plot(npix//2+1, npix//2, 'r+')
        plt.ylabel('Y [pix]', fontsize=12)
        plt.xlabel('X [pix]', fontsize=12)
        plt.tight_layout()

    out = {'cube': mf,
           'imat': mf_imat,
           'rmat': mf_rmat,
           'uv': im_uv,
           'tot': mf_tot,
           'tot_m': mf_tot_m,
           'pvct': mf_pvct,
           'gvct': mf_gvct,
           'cpvct': mfc_pvct,
           'cgvct': mfc_gvct,
           'ix': mf_ix,
           'u': u*filt[0],
           'v': v*filt[0],
           'wl': filt[0],
           'e_wl': filt[1],
           'pixelSize': pixelSize,
           'xy_coords': xy_coords
           }

    return dict2class(out)


def GivePeakInfo2d(mf, n_baselines, dim1, dim2):
    """ 
    Transform mf.pvct indices from flatten 1-D array to 2-D coordinates and the
    associated gains.

    Parameters:
    -----------

    `mf` {object class}: 
        Match filter class (see make_mf function),\n
    `n_baselines` {int}:
        Number of baselines,\n
    `dim1`, `dim2` {int}:
        Size of the 2-D image.\n

    Returns:
    --------

    `l_peak` {list}:
        List of the n_baselines peak positions (2-D) and gains.
    """

    x, y = np.arange(dim1), np.arange(dim2)
    X, Y = np.meshgrid(x, y)

    List_peak = []
    for j in range(n_baselines):
        l_x = X.ravel()[mf.pvct[mf.ix[0, j]:mf.ix[1, j]]]  # .astype(int)
        l_y = Y.ravel()[mf.pvct[mf.ix[0, j]:mf.ix[1, j]]]  # .astype(int)
        g = mf.gvct[mf.ix[0, j]:mf.ix[1, j]]

        peak = [[int(l_y[k]), int(l_x[k]), g[k]] for k in range(len(l_x))]

        List_peak.append(np.array(peak))

    return np.array(List_peak)


def clos_unique(closing_tri_pix):
    """Compute the list of unique triplets in multiple triangle list"""
    l, l_i = [], []
    for i in range(closing_tri_pix.shape[1]):

        p1 = str(closing_tri_pix[0, i])
        p2 = str(closing_tri_pix[1, i])
        p3 = str(closing_tri_pix[2, i])

        p = np.sort(closing_tri_pix[:, i])

        p1 = str(p[0])
        p2 = str(p[1])
        p3 = str(p[2])

        val = p1+p2+p3

        if val not in l:
            l.append(val)
            l_i.append(i)
        else:
            pass

    return closing_tri_pix[:, l_i]


def tri_pix(array_size, sampledisk_r, itrip=1, verbose=True, display=True):
    """Compute all combination of triangle for a given splodge size"""

    if array_size % 2 == 1:
        cprint('\n! Warnings: image dimension must be even (%i)' % array_size,
               'red')
        cprint('Possible triangle inside the splodge should be incorrect.\n',
               'red')

    d = np.zeros([array_size, array_size])
    d = plot_circle(d, array_size//2, array_size//2, sampledisk_r,
                    display=False)
    pvct_flat = np.where(d.ravel() > 0)
    npx = len(pvct_flat[0])
    for px1 in range(npx):
        thispix1 = np.array(array_coords(pvct_flat[0][px1], array_size))

        roll1 = np.roll(d, int(array_size//2 - thispix1[0]), axis=0)
        roll2 = np.roll(roll1, int(array_size//2 - thispix1[1]), axis=1)

        xcor_12 = d + roll2

        valid_b12_vct = np.where(xcor_12.T.ravel() > 1)

        thisntriv = len(valid_b12_vct[0])

        thistrivct = np.zeros([3, thisntriv])

        for px2 in range(thisntriv):
            thispix2 = array_coords(valid_b12_vct[0][px2], array_size)
            thispix3 = array_size*1.5 - (thispix1 + thispix2)
            bl3_px = thispix3[1] * array_size + (thispix3[0])
            thistrivct[:, px2] = [pvct_flat[0][px1], valid_b12_vct[0][px2],
                                  bl3_px]

        if (px1 == 0):
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
        print('Closing triangle in r = %2.1f: %i' % (sampledisk_r, n_trip))

    cl_unique = clos_unique(closing_tri_pix)

    c0 = array_size//2
    fw = 2*sampledisk_r

    if display:
        plt.figure(figsize=(5, 5))
        plt.title('Splodge + unique triangle (tri = %i/%i, d = %i, r = %2.1f pix)' % (cl_unique.shape[1],
                                                                                      n_trip, array_size,
                                                                                      sampledisk_r), fontsize=10)
        plt.imshow(d)
        for i in range(cl_unique.shape[1]):

            trip1 = cl_unique[:, i]
            X, Y = array_coords(trip1, array_size)
            X = list(X)
            Y = list(Y)
            X.append(X[0])
            Y.append(Y[0])
            plt.plot(X, Y, '-', lw=1)
            plt.axis([c0 - fw, c0 + fw, c0 - fw, c0 + fw])

        plt.tight_layout()
        plt.show(block=False)
    return closing_tri_pix


def bs_multiTriangle(i, bs_arr, ft_frame, bs2bl_ix, mf, closing_tri_pix):
    """
    Compute the bispectrum using the multiple triangle technique

    Parameters:
    -----------

    `i` {int}: 
        Indice number of the bispectrum array (bs_arr),\n
    `bs_arr` {array}:
        Empty bispectrum array (bs_arr.shape[0] = n_bs),\n
    `ft_frame` {array}:
        fft of the frame where is extracted the bs value,\n
    `bs2bl_ix` {list}:
        Bispectrum to baselines indix,\n
    `mf` {class}:
        See make_mf function,\n
    `closing_tri_pix` {array}:
        Array of possible combination of indices in a splodge.\n

    Returns:
    --------

    `bs_arr` {array}:
        Filled bispectrum array.


    """
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
        np.roll(np.rot90(np.rot90(mfilter_spec)), 1, axis=0), 1, axis=1)

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

        a = (splodge_shift+((dim1/2)*(dim1+1))).astype(int)[0]

        spl_offset = array_coords(a, dim1).T - dim1//2
        spl_offset = spl_offset.T

        this_trisampling[2, :] = this_trisampling[2, :] - \
            2*(spl_offset[0, 2]+1*spl_offset[1, 2]*dim1) + 0
        this_trisampling[0, :] -= 0
        this_trisampling[1, :] -= 0
        this_trisampling = this_trisampling.astype(int)

        mfilter_spec2 = mfilter_spec.ravel()

        bs_arr[i, this_bs] = np.sum(mfilter_spec2[this_trisampling[0, :]] *
                                    mfilter_spec2[this_trisampling[1, :]] * (mfilter_spec2[this_trisampling[2, :]]))

        All_multi_tri.append(this_trisampling)

    return bs_arr


def find_bad_holes(res_c, n_holes, bl2h_ix, bmax=6, verbose=False, display=False):
    """ Find bad apertures using a linear fit of the v2 vs. spatial frequencies.

    Parameters
    ----------
    `res_c` : {class}
        Class containing NRM data of the calibrator (bispect.py),\n
    `n_holes` : {int}
        Number of apertures,\n
    `bl2h_ix` : {array}
        Baselines to apertures (holes) indices,\n
    `bmax` : {int}, optional
        Maximum baseline used to plot the fit, by default 6,\n
    `verbose` : {bool}, optional
        If True, print useful informations , by default False,\n
    `display` : {bool}, optional
        If True, display figures, by default False.

    Returns
    -------
    `bad_holes`: {array}
        List of determined bad holes,
    """
    u = res_c.u/res_c.wl
    v = res_c.v/res_c.wl
    X = np.sqrt(u**2+v**2)
    Y = np.log(res_c.v2)

    param = {'a': 1,
             'b': 0}

    fit = leastsqFit(linear, X, param, Y, verbose=verbose)

    xm = np.linspace(0, bmax, 100)/res_c.wl
    ym = linear(xm, fit['best'])

    pfit = list(fit['best'].values())

    if display:
        plt.figure()
        plt.plot(X/1e6, Y, '.', label='data')
        plt.plot(xm/1e6, ym, '--', label='fit (a=%2.1e, b=%2.2f)' %
                 (pfit[0], pfit[1]))
        plt.grid(alpha=.1)
        plt.legend()
        plt.xlim(0, xm.max()/1e6)
        plt.ylabel('$\log(V^2)$ (calibrator)')
        plt.xlabel('Sp. Freq. [M$\lambda$]')
        plt.tight_layout()

    bad_holes = []
    for j in range(n_holes):
        pass
        w = np.where((bl2h_ix[0, :] == j) | (bl2h_ix[1, :] == j))
        if (np.mean(res_c.v2[w]/np.exp(fit['model'][w]))) <= 0.3:
            bad_holes.append(j)

    bad_holes = np.unique(bad_holes)

    return bad_holes


def find_bad_BL_BS(bad_holes, bl2h_ix, bs2bl_ix):
    """
    Give indices of bad BS and V2 using a given bad holes list.

    Parameters:
    -----------
    `bad_holes` {array}:
        Bad holes list from find_bad_holes (using calibrator data),\n
    `bl2h_ix`, `bs2bl_ix` {array}:
        Corresponding indices of baselines and bispectrums from a given mask 
        positions (see index_mask function).\n

    Returns:
    --------
    `bad_baselines` {array}:
        Bad baselines indices,\n
    `bad_bispect` {array}:
        Bad bispectrum indices,\n
    `good_baselines` {array}:
        Good baselines indices,\n
    `good_bispectrum` {array}:
        Good bispectrum indices.

    """
    n_baselines = bl2h_ix.shape[1]
    n_bispect = bs2bl_ix.shape[1]

    bad_baselines, bad_bispect = [], []

    good_baselines = np.arange(n_baselines)
    good_bispectrum = np.arange(n_bispect)

    if (len(bad_holes) == 0):
        good_baselines = np.arange(n_baselines)
        good_bispectrum = np.arange(n_bispect)
        res = bad_baselines, bad_bispect, good_baselines, good_bispectrum
    else:
        for i in range(len(bad_holes)):
            new_bad = list(np.where((bl2h_ix[0, :] == bad_holes[i]) |
                                    (bl2h_ix[1, :] == bad_holes[i])
                                    )[0])
            bad_baselines.extend(new_bad)

        bad_baselines = np.unique(bad_baselines)

        if len(bad_baselines) != 0:
            for i in range(len(bad_baselines)):
                new_bad = np.where((bs2bl_ix[0, :] == bad_baselines[i]) |
                                   (bs2bl_ix[1, :] == bad_baselines[i]) |
                                   (bs2bl_ix[2, :] == bad_baselines[i]))
                if len(bad_bispect) == 0:
                    bad_bispect.extend(new_bad)

        bad_bispect = np.unique(bad_bispect)

        good_baselines = [
            bl for bl in good_baselines if bl not in bad_baselines]
        good_bispectrum = [
            bs_el for bs_el in good_bispectrum if bs_el not in bad_bispect]

        res = bad_baselines, bad_bispect, good_baselines, good_bispectrum

    return res


def phase_chi2(p, fitmat, ph_mn, ph_err):
    """Compute chi2 of the phase used to fit piston"""
    piston = np.dot(p, fitmat)

    tmp = list(ph_mn)
    tmp.append(0)
    tmp = np.array(tmp)
    e_tmp = list(ph_err)
    e_tmp.append(0.01)
    e_tmp = np.array(e_tmp)**2
    arg = (np.array(tmp - piston) * 1j)
    phase_chi2 = np.sum(np.abs(1 - np.exp(arg))**2/e_tmp)
    return phase_chi2
