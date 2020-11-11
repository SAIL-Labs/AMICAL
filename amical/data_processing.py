"""
@author: Anthony Soulain (University of Sydney)

-------------------------------------------------------------------------
AMICAL: Aperture Masking Interferometry Calibration and Analysis Library
-------------------------------------------------------------------------

Function related to data cleaning (ghost, background correction,
centering, etc.) and data selection (sigma-clipping, centered flux,).

-------------------------------------------------------------------- 
"""

import numpy as np
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm
from termcolor import cprint
from tqdm import tqdm

from amical.tools import apply_windowing, crop_max


def _apply_patch_ghost(cube, xc, yc, radius=20, dx=0, dy=-200, method='bg'):
    """Apply a patch on an eventual artifacts/ghosts on the spectral filter (i.e.
    K1 filter of SPHERE presents an artifact/ghost at (392, 360)).

    Arguments:
    ----------
    `cube` {array} -- Data cube,\n
    `xc` {int} -- x-axis position of the artifact,\n
    `yc` {int} -- y-axis position of the artifact.

    Keyword Arguments:
    ----------
    `radius` {int} -- Radius to apply the patch in a circle (default: {10}),\n
    `dy` {int} -- Offset pixel number to compute background values (default: {0}),\n
    `dx` {int} -- Same along y-axis (default: {0}),\n
    `method` {str} -- If 'bg', the replacement values are the background computed at
    xc+dx, yx+dy, else zero is apply (default: {'bg'}).
    """
    cube_corrected = []
    for i in range(len(cube)):
        imA = cube[i].copy()
        isz = imA.shape[0]
        xc_off, yc_off = xc+dx, yc+dy
        xx, yy = np.arange(isz), np.arange(isz)
        xx_c = (xx-xc)
        yy_c = (yc-yy)
        xx_off = (xx-xc_off)
        yy_off = (yc_off-yy)
        distance = np.sqrt(xx_c**2 + yy_c[:, np.newaxis]**2)
        distance_off = np.sqrt(xx_off**2 + yy_off[:, np.newaxis]**2)
        cond_patch = (distance <= radius)
        cond_bg = (distance_off <= radius)
        if method == 'bg':
            imA[cond_patch] = imA[cond_bg]
        elif method == 'zero':
            imA[cond_patch] = 0
        cube_corrected.append(imA)
    cube_corrected = np.array(cube_corrected)
    return cube_corrected


def select_data(cube, clip_fact=0.5, clip=False, verbose=True, display=True):
    """ Check the cleaned data cube using the position of the maximum in the
    fft image (supposed to be zero). If not in zero position, the fram is 
    rejected. It can apply a sigma-clipping to select only the frames with the
    highest total fluxes.

    Parameters:
    -----------
    `cube` {array} -- Data cube,\n
    `clip_fact` {float} -- Relative sigma if rejecting frames by 
    sigma-clipping (default=False),\n
    `clip` {bool} -- If True, sigma-clipping is used,\n
    `verbose` {bool} -- If True, print informations in the terminal,\n
    `display` {bool} -- If True, plot figures.


    """
    fft_fram = abs(np.fft.fft2(cube))
    # flag_fram, cube_flagged, cube_cleaned_checked = [], [], []

    fluxes, flag_fram, good_fram = [], [], []
    for i in range(len(fft_fram)):
        fluxes.append(fft_fram[i][0, 0])
        pos_max = np.argmax(fft_fram[i])
        if pos_max != 0:
            flag_fram.append(i)
        else:
            good_fram.append(cube[i])

    fluxes = np.array(fluxes)
    flag_fram = np.array(flag_fram)

    best_fr = np.argmax(fluxes)
    worst_fr = np.argmin(fluxes)

    std_flux = np.std(fluxes)
    med_flux = np.median(fluxes)

    if verbose:
        if (med_flux/std_flux) <= 5.:
            cprint('\nStd of the fluxes along the cube < 5 (%2.1f):\n -> sigma clipping is suggested (clip=True).' % (
                (med_flux/std_flux)), 'cyan')

    limit_flux = med_flux - clip_fact*std_flux

    if clip:
        cond_clip = (fluxes > limit_flux)
        cube_cleaned_checked = cube[cond_clip]
        ind_clip = np.where(fluxes <= limit_flux)[0]
    else:
        ind_clip = []
        cube_cleaned_checked = np.array(good_fram)

    ind_clip2 = np.where(fluxes <= limit_flux)[0]
    if ((worst_fr in ind_clip2) and clip) or (worst_fr in flag_fram):
        ext = '(rejected)'
    else:
        ext = ''

    diffmm = 100*abs(np.max(fluxes) - np.min(fluxes))/med_flux
    if display:
        plt.figure()
        plt.plot(fluxes, label=r'|$\Delta F$|/$\sigma_F$=%2.0f (%2.2f %%)' %
                 (med_flux/std_flux, diffmm))
        if len(flag_fram) > 0:
            plt.scatter(flag_fram, fluxes[flag_fram],
                        s=52, facecolors='none', edgecolors='r', label='Rejected frames (maximum fluxes)')
        if clip:
            if len(ind_clip) > 0:
                plt.plot(ind_clip, fluxes[ind_clip], 'rx',
                         label='Rejected frames (clipping)')
            else:
                print('0')
        plt.hlines(limit_flux, 0, len(fluxes), lw=1,
                   ls='--', label='Clipping limit', zorder=10)
        plt.legend(loc='best', fontsize=9)
        plt.ylabel('Flux [counts]')
        plt.xlabel('# frames')
        plt.grid(alpha=.2)
        plt.tight_layout()

        plt.figure(figsize=(7, 7))
        plt.subplot(2, 2, 1)
        plt.title('Best fram (%i)' % best_fr)
        plt.imshow(cube[best_fr], norm=PowerNorm(.5), cmap='afmhot', vmin=0)
        plt.subplot(2, 2, 2)
        plt.imshow(np.fft.fftshift(fft_fram[best_fr]), cmap='gist_stern')
        plt.subplot(2, 2, 3)
        plt.title('Worst fram (%i) %s' % (worst_fr, ext))
        plt.imshow(cube[worst_fr], norm=PowerNorm(.5), cmap='afmhot', vmin=0)
        plt.subplot(2, 2, 4)
        plt.imshow(np.fft.fftshift(fft_fram[worst_fr]), cmap='gist_stern')
        plt.tight_layout()
        plt.show(block=False)
    if verbose:
        n_good = len(cube_cleaned_checked)
        n_bad = len(cube) - n_good
        if clip:
            cprint('\n---- σ-clip + centered fluxes selection ---', 'cyan')
        else:
            cprint('\n---- centered fluxes selection ---', 'cyan')
        print('%i/%i (%2.1f%%) are flagged as bad frames' %
              (n_bad, len(cube), 100*float(n_bad)/len(cube)))
    return cube_cleaned_checked


def sky_correction(imA, r1=100, dr=20, verbose=False):
    """
    Perform background sky correction to be as close to zero as possible.
    """
    isz = imA.shape[0]
    xc, yc = isz//2, isz//2
    xx, yy = np.arange(isz), np.arange(isz)
    xx2 = (xx-xc)
    yy2 = (yc-yy)
    r2 = r1 + dr

    distance = np.sqrt(xx2**2 + yy2[:, np.newaxis]**2)
    cond_bg = (r1 <= distance) & (distance <= r2)

    try:
        minA = imA.min()
        imB = imA + 1.01*abs(minA)
        backgroundB = np.mean(imB[cond_bg])
        imC = imB - backgroundB
        backgroundC = np.mean(imC[cond_bg])
    except IndexError:
        imC = imA.copy()
        backgroundC = 0
        if verbose:
            cprint('Warning: Background not computed', 'green')
            cprint(
                '-> check the inner and outer radius rings (checkrad option).', 'green')
    return imC, backgroundC


def fix_bad_pixels(image, bad_map, add_bad=[], x_stddev=1):
    """ Replace bad pixels with values interpolated from their neighbors (interpolation
    is made with a gaussian kernel convolution)."""
    if len(add_bad) != 0:
        for j in range(len(add_bad)):
            bad_map[add_bad[j][0], add_bad[j][1]] = 1

    img_nan = image.copy()
    img_nan[bad_map == 1] = np.nan
    kernel = Gaussian2DKernel(x_stddev=x_stddev)
    fixed_image = interpolate_replace_nans(img_nan, kernel)
    return fixed_image


def check_data_params(filename, isz, r1, dr, bad_map=None, add_bad=[],
                      edge=0, remove_bad=True, nframe=0, ihdu=0, f_kernel=3,
                      offx=0, offy=0, apod=False, window=None):
    """ Check the input parameters for the cleaning.

    Parameters:
    -----------

    `filename` {str}: filename containing the datacube,\n
    `isz` {int}: Size of the cropped image (default: 256)\n
    `r1` {int}: Radius of the rings to compute background sky (default: 100)\n
    `dr` {int}: Outer radius to compute sky (default: 10)\n
    `bad_map` {array}: Bad pixel map with 0 and 1 where 1 set for a bad pixel (default: None),\n
    `add_bad` {list}: List of 2d coordinates of bad pixels/cosmic rays (default: []),\n
    `edge` {int}: Number of pixel to be removed on the edge of the image (SPHERE),\n
    `remove_bad` {bool}: If True, the bad pixels are removed using a gaussian interpolation,\n
    `nframe` {int}: Frame number to be shown (default: 0),\n
    `ihdu` {int}: Hdu number of the fits file. Normally 1 for NIRISS and 0 for SPHERE (default: 0).
    """
    data = fits.open(filename)[ihdu].data
    img0 = data[nframe]
    if edge != 0:
        img0[:, 0:edge] = 0
        img0[:, -edge:-1] = 0
        img0[0:edge, :] = 0
        img0[-edge:-1, :] = 0
    if (bad_map is not None) & (remove_bad):
        img1 = fix_bad_pixels(img0, bad_map, add_bad=add_bad)
    else:
        img1 = img0.copy()
    cropped_infos = crop_max(img1, isz, offx=offx, offy=offy, f=f_kernel)
    pos = cropped_infos[1]

    noBadPixel = False
    bad_pix_x, bad_pix_y = [], []
    if (bad_map is not None) or (len(add_bad) != 0):
        for j in range(len(add_bad)):
            bad_map[add_bad[j][0], add_bad[j][1]] = 1
        bad_pix = np.where(bad_map == 1)
        bad_pix_x = bad_pix[0]
        bad_pix_y = bad_pix[1]
    else:
        noBadPixel = True

    r2 = r1 + dr
    theta = np.linspace(0, 2*np.pi, 100)
    x0 = pos[0]
    y0 = pos[1]

    x1 = r1 * np.cos(theta) + x0
    y1 = r1 * np.sin(theta) + y0
    x2 = r2 * np.cos(theta) + x0
    y2 = r2 * np.sin(theta) + y0
    if window is not None:
        r3 = window
        x3 = r3 * np.cos(theta) + x0
        y3 = r3 * np.sin(theta) + y0

    xs1, ys1 = x0 + isz//2, y0 + isz//2
    xs2, ys2 = x0 - isz//2, y0 + isz//2
    xs3, ys3 = x0 - isz//2, y0 - isz//2
    xs4, ys4 = x0 + isz//2, y0 - isz//2

    max_val = img1[y0, x0]
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(img1, norm=PowerNorm(.5), cmap='afmhot', vmin=0, vmax=max_val)
    plt.plot(x1, y1, label='Inner radius for sky subtraction')
    plt.plot(x2, y2, label='Outer radius for sky subtraction')
    if apod:
        if window is not None:
            plt.plot(x3, y3, label='Super-gaussian windowing')
    plt.plot(x0, y0, '+', color='g', ms=10, label='Centering position')
    plt.plot([xs1, xs2, xs3, xs4, xs1], [ys1, ys2, ys3, ys4, ys1], 'w--',
             label='Resized image')
    if not noBadPixel:
        if remove_bad:
            label = 'Fixed hot/bad pixels'
        else:
            label = 'Hot/bad pixels'
        plt.scatter(bad_pix_y, bad_pix_x, color='', marker='s',
                    edgecolors='r', s=20, label=label)

    plt.legend(fontsize=7, loc=1)
    plt.tight_layout()
    return fig


def clean_data(data, isz=None, r1=None, dr=None, edge=0,
               r2=None, bad_map=None, add_bad=[], apod=True,
               offx=0, offy=0, sky=True, window=None,
               f_kernel=3, verbose=False):
    """ Clean data.

    Parameters:
    -----------

    `data` {np.array} -- datacube containing the NRM data\n
    `isz` {int} -- Size of the cropped image (default: {None})\n
    `r1` {int} -- Radius of the rings to compute background sky (default: {None})\n
    `dr` {int} -- Outer radius to compute sky (default: {None})\n
    `edge` {int} -- Patch the edges of the image (VLT/SPHERE artifact, default: {200}),\n
    `checkrad` {bool} -- If True, check the resizing and sky substraction parameters (default: {False})\n

    Returns:
    --------
    `cube` {np.array} -- Cleaned datacube.
    """
    # print(data.shape[1])
    # if data.shape[1] % 2 == 1:
    #     data = np.array([im[:-1, :-1] for im in data])

    n_im = data.shape[0]
    cube_cleaned = np.zeros([n_im, isz, isz])
    for i in tqdm(range(n_im), ncols=100, desc='Cleaning', leave=False):
        img0 = data[i]
        if edge != 0:
            img0[:, 0:edge] = 0
            img0[:, -edge:-1] = 0
            img0[0:edge, :] = 0
            img0[-edge:-1, :] = 0
        if bad_map is not None:
            img1 = fix_bad_pixels(img0, bad_map, add_bad=add_bad)
        else:
            img1 = img0.copy()
        im_rec_max = crop_max(img1, isz, offx=offx, offy=offy, f=f_kernel)[0]
        if sky:
            img_biased = sky_correction(im_rec_max, r1=r1, dr=dr,
                                        verbose=verbose)[0]
        else:
            img_biased = im_rec_max.copy()
        img_biased[img_biased < 0] = 0  # Remove negative pixels

        if img_biased.shape[0] != img_biased.shape[1]:
            cprint(
                '\nCropped image do not have same X, Y dimensions -> check isz', 'red')
            return None

        if apod:
            if r2 is None:
                r2 = isz//3
            img = apply_windowing(img_biased, window=window)

        else:
            img = img_biased.copy()
        cube_cleaned[i] = img
    return cube_cleaned


def select_clean_data(filename, isz=256, r1=100, r2=None, dr=10, edge=0,
                      clip=True, bad_map=None, add_bad=[], offx=0, offy=0,
                      clip_fact=0.5, apod=True, sky=True, window=None,
                      f_kernel=3, verbose=False, ihdu=0, display=False):
    """ Clean and select good datacube (sigma-clipping using fluxes variations).

    Parameters:
    -----------

    `filename` {str}: filename containing the datacube,\n
    `isz` {int}: Size of the cropped image (default: 256)\n
    `r1` {int}: Radius of the rings to compute background sky (default: 100)\n
    `dr` {int}: Outer radius to compute sky (default: 10)\n
    `edge` {int}: Patch the edges of the image (VLT/SPHERE artifact, default: {100}),\n
    `clip` {bool}: If True, sigma-clipping is used to reject frames with low integrated flux,\n
    `clip_fact` {float}: Relative sigma if rejecting frames by sigma-clipping 
    (default=0.5),\n

    Returns:
    --------
    `cube_final` {np.array}: Cleaned and selected datacube.
    """
    hdu = fits.open(filename)
    cube = hdu[ihdu].data
    hdr = hdu[0].header

    if hdr['INSTRUME'] == 'SPHERE':
        seeing_start = float(hdr['HIERARCH ESO TEL AMBI FWHM START'])
        seeing = float(hdr['HIERARCH ESO TEL IA FWHM'])
        seeing_end = float(hdr['HIERARCH ESO TEL AMBI FWHM END'])

        if verbose:
            print('\n----- Seeing conditions -----')
            print("%2.2f (start), %2.2f (end), %2.2f (Corrected AirMass)" %
                  (seeing_start, seeing_end, seeing))

    raw_size = cube.shape[1]
    if isz > raw_size:
        raise ValueError(
            'Reshape factor is larger than the data size (choose a smaller isz).')

    cube_cleaned = clean_data(cube, isz=isz, r1=r1, edge=edge,
                              r2=r2, bad_map=bad_map, add_bad=add_bad,
                              dr=dr, sky=sky, apod=apod, window=window,
                              f_kernel=f_kernel, offx=offx, offy=offy,
                              verbose=verbose)

    if cube_cleaned is None:
        return None

    cube_final = select_data(cube_cleaned, clip=clip, clip_fact=clip_fact,
                             verbose=verbose, display=display)
    return cube_final
