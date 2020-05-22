# -*- coding: utf-8 -*-
"""
@author: Anthony Soulain (University of Sydney)

--------------------------------------------------------------------
MIAMIS: Multi-Instruments Aperture Masking Interferometry Software
--------------------------------------------------------------------

General tools.
-------------------------------------------------------------------- 
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from matplotlib.colors import PowerNorm
from munch import munchify as dict2class
from scipy.signal import medfilt2d
from termcolor import cprint

warnings.filterwarnings("ignore", module='astropy.io.votable.tree')
warnings.filterwarnings("ignore", module='astropy.io.votable.xmlutil')


def linear(x, param):
    """Linear model used in dpfit"""
    a = param['a']
    b = param['b']
    y = a*x + b
    return y


def mas2rad(mas):
    """ Convert angle in milli-arcsec to radians """
    rad = mas * (10**(-3)) / (3600 * 180 / np.pi)
    return rad


def rad2mas(rad):
    """ Convert input angle in radians to milli-arcsec """
    mas = rad * (3600. * 180 / np.pi) * 10.**3
    return mas


def crop_max(img, dim, filtmed=True, f=3):
    """
    Summary
    -------------
    Resize an image on the brightest pixel.

    Parameters
    ----------
    `img` : {numpy.array}
        input image,\n
    `dim` : {int}
        resized dimension,\n
    `filtmed` : {boolean}, (optionnal)
        True if perform a median filter on the image (to blur bad pixels),\n
    `f` : {float}, (optionnal),
        If filtmed == True, kernel size of the median filter.


    Returns
    -------
    `cutout`: {numpy.array}
        Resized image.
    """
    if filtmed:
        im_med = medfilt2d(img, f)
    else:
        im_med = img.copy()

    pos_max = np.where(im_med == im_med.max())
    X = pos_max[1][0]
    Y = pos_max[0][0]

    position = (X, Y)

    cutout = Cutout2D(img, position, dim)
    return cutout.data, position


def norm_max(tab):
    """
    Short Summary
    -------------
    Normalize an array or a list by the maximum.

    Parameters
    ----------
    `tab` : {numpy.array}, {list}
        input array or list.

    Returns
    -------
    `tab_norm` : {numpy.array}, {list}
        Normalized array.
    """
    tab_norm = tab/np.max(tab)
    return tab_norm


def crop_center(img, dim):
    """
    Short Summary
    -------------
    Resize an image on the center.

    Parameters
    ----------
    `img` : {numpy.array}
        input image,\n
    `dim` : {int}
        resized dimension.

    Returns
    -------
    `cutout`: {numpy.array}
        Resized image.
    """
    b = img.shape[0]
    position = (b//2, b//2)
    cutout = Cutout2D(img, position, dim)
    return cutout.data


def crop_position(img, X, Y, dim):
    """
    Short Summary
    -------------
    Resize an image on a defined position.

    Parameters
    ----------
    `img` : {numpy.array}
        input image,\n
    `X`, `Y` : {int}
        Position to resize (new center of the image),\n
    `dim` : {int}
        resized dimension.

    Returns
    -------
    `cutout`: {numpy.array}
        Resized image.
    """
    position = (X, Y)
    cutout = Cutout2D(img, position, dim)
    return cutout.data


def gauss_2d_asym(X, param):
    """
    Short Summary
    -------------
    Creates 2D oriented gaussian with an asymmetrical grid.

    Parameters
    ----------
    `X` : {list}. 
        Input values :
         - `X[0]` : x coordinates [pixels]
         - `X[1]` : y coordinates [pixels]
         - `X[2]` : pixels scale [mas]\n

    `param` : {dict}.
        Input parameters, with the keys:
            - `A` : amplitude.
            - `x0` : x offset from the center [mas].
            - `y0` : y offset from the center [mas].
            - `fwhm_x` : width in x direction [mas].
            - `fwhm_y` : width in y direction [mas].
            - `theta` : orientation [deg].()
    Returns
    -------
    `im`: {numpy.array}
        image of a 2D gaussian function.
    """

    x_1d = X[0]
    y_1d = X[1]
    pixel_scale = X[2]

    dim = len(x_1d)

    x, y = np.meshgrid(x_1d, y_1d)

    fwhmx = param['fwhm_x']/pixel_scale
    fwhmy = param['fwhm_y']/pixel_scale

    sigma_x = (fwhmx / np.sqrt(8 * np.log(2)))
    sigma_y = (fwhmy / np.sqrt(8 * np.log(2)))

    amplitude = param['A']
    x0 = dim//2 + param['x0']/pixel_scale
    y0 = dim//2 + param['y0']/pixel_scale
    theta = np.deg2rad(param['theta'])
    size_x = len(x)
    size_y = len(y)
    im = np.zeros([size_y, size_x])
    x0 = float(x0)
    y0 = float(y0)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    im = amplitude*np.exp(- (a*((x-x0)**2) + 2*b*(x-x0)
                             * (y-y0) + c*((y-y0)**2)))
    return im


def conv_fft(image, psf):
    """
    Compute 2D convolution with the PSF, passing through Fourier space.
    """
    fft_im = np.fft.fft2(image)
    fft_psf = np.fft.fft2(psf)
    fft_conv = fft_im*fft_psf
    conv = abs(np.fft.fftshift(np.fft.ifft2(fft_conv)))
    return conv


def plot_circle(d, x, y, hole_radius, sz=1, display=True):
    """ Return an image with a disk = sz at x, y position and zero elsewhere"""
    chipsz = np.shape(d)[0]

    im = np.zeros([chipsz, chipsz])
    info = [len(im.shape), im.shape[0], im.shape[1], 3, len(im.ravel())]

    if (type(x) == float) or (type(x) == int):
        n_circ = 1
        xx = [x]
        yy = [y]
    else:
        n_circ = len(x)
        xx = x
        yy = y

    r = hole_radius
    for c in range(n_circ):
        ind1 = int(max([0.0, xx[c] - r - 1]))
        ind2 = int(min([info[1]-1, xx[c] + r + 1]))
        for i in np.arange(ind1, ind2, 1):
            ind3 = int(max([0.0, yy[c] - r - 1]))
            ind4 = int(min([info[2] - 1, yy[c] + r + 1]))
            for j in np.arange(ind3, ind4, 1):
                r_d = np.sqrt(
                    (float(i)-float(xx[c]))**2+(float(j)-float(yy[c]))**2)
                if (r_d <= r):
                    im[i, j] = sz

    if display:
        plt.figure()
        plt.imshow(im)

    return im


def cov2cor(cov):
    """
    Convert the input covariance matrix to a correlation matrix 
    corr[i,j] = cov[i,j]/sqrt(cov[i,i]*cov[j,j]).

    Parameters
    ----------
    `cov`: {square array}
        An NxN covariance matrix.

    Outputs
    -------
    `cor`: {square array}
        The NxN correlation matrix.
    """
    cor = np.zeros(cov.shape)

    sigma = np.sqrt(np.diag(cov))
    for ix in range(cov.shape[0]):
        cxx = cov[ix, ix]
        if cxx <= 0.0:
            str_err = "diagonal cov[%d,%d]=%e is not positive" % (ix, ix, cxx)
            raise ValueError(str_err)
        for iy in range(cov.shape[1]):
            cyy = cov[iy, iy]
            if cyy <= 0.0:
                str_err = "diagonal cov[%d,%d]=%e is not positive" % (iy, iy,
                                                                      cyy)
                raise ValueError(str_err)
            cor[ix, iy] = cov[ix, iy] / np.sqrt(cxx * cyy)

    return cor, sigma


def skyCorrection(imA, r1=100, dr=20, verbose=False):
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
        cprint('Warning: Background not computed', 'green')
        cprint('-> check the inner and outer radius rings (checkrad option).', 'green')

    return imC, backgroundC


def applyMaskApod(img, r=80, sig=10):
    isz = len(img)

    X = [np.arange(isz), np.arange(isz), 1]

    sig = 10
    param = {'A': 1,
             'x0': 0,
             'y0': 0,
             'fwhm_x': sig,
             'fwhm_y': sig,
             'theta': 0
             }

    gauss = gauss_2d_asym(X, param)

    xx, yy = np.arange(isz), np.arange(isz)
    xx2 = (xx-isz//2)
    yy2 = (isz//2-yy)

    distance = np.sqrt(xx2**2 + yy2[:, np.newaxis]**2)

    mask = np.zeros([isz, isz])

    mask[distance < r] = 1

    conv_mask = conv_fft(mask, gauss)

    mask_apod = conv_mask/np.max(conv_mask)

    img_apod = img * mask_apod
    return img_apod


def checkRadiusResize(img, isz, r1, dr, pos):
    # isz = len(img)
    r2 = r1 + dr
    theta = np.linspace(0, 2*np.pi, 100)
    x0 = pos[0]
    y0 = pos[1]

    x1 = r1 * np.cos(theta) + x0
    y1 = r1 * np.sin(theta) + y0
    x2 = r2 * np.cos(theta) + x0
    y2 = r2 * np.sin(theta) + y0

    xs1, ys1 = x0 + isz//2, y0 + isz//2
    xs2, ys2 = x0 - isz//2, y0 + isz//2
    xs3, ys3 = x0 - isz//2, y0 - isz//2
    xs4, ys4 = x0 + isz//2, y0 - isz//2

    max_val = img[y0, x0]
    fig = plt.figure()
    plt.imshow(img, norm=PowerNorm(.5), cmap='afmhot', vmin=0, vmax=max_val)
    plt.plot(x1, y1)
    plt.plot(x2, y2)
    plt.plot(x0, y0, '+', color='g', ms=10)
    plt.plot([xs1, xs2, xs3, xs4, xs1], [ys1, ys2, ys3, ys4, ys1], 'w--')
    return fig


def sanitize_array(dic):
    """ Recursively convert values in a nested dictionnary from np.bool_ to builtin bool type
    This is required for json serialization.
    """
    d2 = dic.copy()
    for k, v in dic.items():
        if isinstance(v, np.ndarray):
            d2[k] = sanitize_array(v)
        if isinstance(v, list):
            d2[k] = np.array(v)
    return d2


def checkSeeingCond(list_nrm):
    """ Extract the seeing conditions, parang, averaged vis2
    and cp of a list of nrm classes extracted with extract_bs_mf 
    function (bispect.py).

    Output
    ------
    If output is **res**, access to parallactic angle by `res.pa`, or
    `res.seeing` for the seeing across multiple nrm data (files).

    """
    l_seeing, l_vis2, l_cp, l_pa, l_mjd = [], [], [], [], []
    for nrm in list_nrm:
        hdr = fits.open(nrm.filename)[0].header
        pa = hdr['PARANG']
        seeing = nrm.hdr['SEEING']
        mjd = hdr['MJD-OBS']
        l_vis2.append(np.mean(nrm.v2))
        l_cp.append(np.mean(nrm.cp))
        l_seeing.append(seeing)
        l_pa.append(pa)
        l_mjd.append(mjd)

    res = {'pa': l_pa,
           'seeing': l_seeing,
           'vis2': l_vis2,
           'cp': l_cp,
           'mjd': l_mjd,
           'target': hdr['OBJECT']}

    return dict2class(sanitize_array(res))


def plotSeeingCond(cond_t, cond_c, lim_seeing=None):
    """ Plot seeing condition between calibrator and target files. """
    m_mjd = abs(np.min(np.diff(cond_c.mjd)))/2.
    xmin = np.min([cond_c.mjd.min(), cond_t.mjd.min()])-m_mjd
    xmax = np.max([cond_c.mjd.max(), cond_t.mjd.max()])+m_mjd

    fig = plt.figure()
    ax1 = plt.gca()
    ax1.set_xlabel('mjd [days]')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Seeing ["]', color="#c62d42")
    ax2.tick_params(axis='y', labelcolor="#c62d42")
    ax1.set_ylabel('Uncalibrated mean V$^2$')

    ax1.plot(cond_t.mjd, cond_t.vis2, '.',
             color="#20b2aa", label=cond_t.target)
    ax2.plot(cond_t.mjd, cond_t.seeing, '+', color="#c62d42")
    ax1.plot(cond_c.mjd, cond_c.vis2, '.',
             color="#00468c", label="%s (cal)" % cond_c.target)
    ax2.plot(cond_c.mjd, cond_c.seeing, '+', color="#c62d42", label='Seeing')
    if lim_seeing is not None:
        ax2.hlines(lim_seeing, xmin, xmax, color='g', label='Seeing threshold')
    ax1.set_ylim(0, 1.2)
    ax2.set_ylim(0.6, 1.8)
    ax1.set_xlim(xmin, xmax)
    ax1.grid(alpha=.1, color='grey')
    ax1.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.show(block=False)
    return fig
