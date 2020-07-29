# -*- coding: utf-8 -*-
"""
@author: Anthony Soulain (University of Sydney)

-------------------------------------------------------------------------
AMICAL: Aperture Masking Interferometry Calibration and Analysis Library
-------------------------------------------------------------------------

General tools.

--------------------------------------------------------------------
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.time import Time
from munch import munchify as dict2class
from scipy.signal import medfilt2d
from termcolor import cprint
from uncertainties import ufloat

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
    cutout = Cutout2D(img, (X, Y), dim)
    return cutout.data, (X, Y)


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


def computeUfloatArr(data, e_data):
    """ Compute the array containing ufloat format used by uncertainties package. """
    u_data = np.array([ufloat(data[i], e_data[i]) for i in range(len(data))])
    return u_data


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


def wtmn(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    mn = np.average(values, weights=weights, axis=0)

    ndim = values.ndim

    # Fast and numerically precise:
    variance = np.average((values-mn)**2, weights=weights, axis=0)
    std = np.sqrt(variance)

    if ndim == 2:
        e_max = []
        for i in range(values.shape[1]):
            e_max.append(np.max(weights[:, i]))
        std_unbias = []
        for i in range(values.shape[1]):
            std_unbias.append(np.max([e_max[i], std[i]]))
        std_unbias = np.array(std_unbias)
    else:
        e_max = weights.copy()
        std_unbias = np.max([np.max(e_max), std])
    return (mn, std_unbias)


def jd2lst(lng, jd):
    '''Convert Julian date to LST '''
    c = [280.46061837, 360.98564736629, 0.000387933, 38710000.0]
    jd2000 = 2451545.0
    t0 = jd - jd2000
    t = t0/36525.

    # Compute GST in seconds.
    theta = c[0] + (c[1] * t0) + t**2*(c[2] - t / c[3])

    # Compute LST in hours.
    lst = (theta + lng)/15.0
    neg = np.where(lst < 0.0)
    n = neg[0].size
    if n > 0:
        lst[neg] = 24.0 + (lst[neg] % 24)
    lst = lst % 24
    return lst


def compute_pa(hdr, n_ps, verbose=False, display=False):

    list_fct_pa = {'SPHERE': sphere_parang,
                   }

    instrument = hdr['INSTRUME']
    if instrument not in list(list_fct_pa.keys()):
        try:
            nframe = hdr['NAXIS3']
        except KeyError:
            nframe = n_ps
        if verbose:
            cprint('Warning: %s not in known pa computation -> set to zero.\n' %
                   instrument, 'green')
        pa_exist = False
        l_pa = np.zeros(nframe)
    else:
        l_pa = list_fct_pa[instrument](hdr)
        pa_exist = True

    pa = np.mean(l_pa)
    std_pa = np.std(l_pa)

    if display and pa_exist:
        plt.figure(figsize=(4, 3))
        plt.plot(
            l_pa, '.-', label='pa=%2.1f, $\sigma_{pa}$=%2.1f deg' % (pa, std_pa))
        plt.legend(fontsize=7)
        plt.grid(alpha=.2)
        plt.xlabel("# frames")
        plt.ylabel("Position angle [deg]")
        plt.tight_layout()

    return pa


def sphere_parang(hdr):
    """
    Reads the header and creates an array giving the paralactic angle for each frame,
    taking into account the inital derotator position.
    The columns of the output array contains:
    frame_number, frame_time, paralactic_angle
    """

    r2d = 180/np.pi
    d2r = np.pi/180

    detector = hdr['HIERARCH ESO DET ID']
    if detector.strip() == 'IFS':
        offset = 135.87-100.46  # from the SPHERE manual v4
    elif detector.strip() == 'IRDIS':
        # correspond to the difference between the PUPIL tracking ant the FIELD tracking for IRDIS taken here: http://wiki.oamp.fr/sphere/AstrometricCalibration (PUPOFFSET)
        offset = 135.87
    else:
        offset = 0
        print('WARNING: Unknown instrument in create_parang_list_sphere: '+str(detector))

    try:
        # Get the correct RA and Dec from the header
        actual_ra = hdr['HIERARCH ESO INS4 DROT2 RA']
        actual_dec = hdr['HIERARCH ESO INS4 DROT2 DEC']

        # These values were in weird units: HHMMSS.ssss
        actual_ra_hr = np.floor(actual_ra/10000.)
        actual_ra_min = np.floor(actual_ra/100. - actual_ra_hr*100.)
        actual_ra_sec = (actual_ra - actual_ra_min*100. - actual_ra_hr*10000.)

        ra_deg = (actual_ra_hr + actual_ra_min/60. +
                  actual_ra_sec/60./60.) * 360./24.

        # the sign makes this complicated, so remove it now and add it back at the end
        sgn = np.sign(actual_dec)
        actual_dec *= sgn

        actual_dec_deg = np.floor(actual_dec/10000.)
        actual_dec_min = np.floor(actual_dec/100. - actual_dec_deg*100.)
        actual_dec_sec = (actual_dec - actual_dec_min *
                          100. - actual_dec_deg*10000.)

        dec_deg = (actual_dec_deg + actual_dec_min /
                   60. + actual_dec_sec/60./60.)*sgn
        geolat_rad = float(hdr['ESO TEL GEOLAT'])*d2r
    except Exception:
        print('WARNING: No RA/Dec Keywords found in header')
        ra_deg = 0
        dec_deg = 0
        geolat_rad = 0

    if 'NAXIS3' in hdr:
        n_frames = hdr['NAXIS3']
    else:
        n_frames = 1

    # We want the exposure time per frame, derived from the total time from when the shutter
    # opens for the first frame until it closes at the end.
    # This is what ACC thought should be used
    # total_exptime = hdr['ESO DET SEQ1 EXPTIME']
    # This is what the SPHERE DC uses
    total_exptime = (Time(hdr['HIERARCH ESO DET FRAM UTC']) -
                     Time(hdr['HIERARCH ESO DET SEQ UTC'])).sec
    # print total_exptime-total_exptime2
    delta_dit = total_exptime / n_frames
    dit = hdr['ESO DET SEQ1 REALDIT']

    # Set up the array to hold the parangs
    parang_array = np.zeros((n_frames))

    # Output for debugging
    hour_angles = []

    if ('ESO DET SEQ UTC' in hdr.keys()) and ('ESO TEL GEOLON' in hdr.keys()):
        # The SPHERE DC method
        jd_start = Time(hdr['ESO DET SEQ UTC']).jd
        lst_start = jd2lst(hdr['ESO TEL GEOLON'], jd_start)*3600
        # Use the old method
        lst_start = float(hdr['LST'])
    else:
        lst_start = 0.
        print('WARNING: No LST keyword found in header')

    # delta dit and dit are in seconds so we need to multiply them by this factor to add them to an LST
    time_to_lst = (24.*3600.)/(86164.1)

    if 'ESO INS4 COMB ROT' in hdr.keys() and hdr['ESO INS4 COMB ROT'] == 'PUPIL':

        for i in range(n_frames):

            ha_deg = ((lst_start+i*delta_dit*time_to_lst +
                       time_to_lst*dit/2.)*15./3600)-ra_deg
            hour_angles.append(ha_deg)

            # VLT TCS formula
            f1 = float(np.cos(geolat_rad) * np.sin(d2r*ha_deg))
            f2 = float(np.sin(geolat_rad) * np.cos(d2r*dec_deg) -
                       np.cos(geolat_rad) * np.sin(d2r*dec_deg) * np.cos(d2r*ha_deg))
            pa = -r2d*np.arctan2(-f1, f2)

            pa = pa+offset

            # Also correct for the derotator issues that were fixed on 12 July 2016 (MJD = 57581)
            if hdr['MJD-OBS'] < 57581:
                alt = hdr['ESO TEL ALT']
                drot_begin = hdr['ESO INS4 DROT2 BEGIN']
                # Formula from Anne-Lise Maire
                correction = np.arctan(
                    np.tan((alt-2*drot_begin)*np.pi/180))*180/np.pi
                pa += correction

            pa = ((pa + 360) % 360)
            parang_array[i] = pa

    else:
        if 'ARCFILE' in hdr.keys():
            print(hdr['ARCFILE']+' does seem to be taken in pupil tracking.')
        else:
            print('Data does not seem to be taken in pupil tracking.')

        for i in range(n_frames):
            parang_array[i] = 0

    # And a sanity check at the end
    try:
        # The parang start and parang end refer to the start and end of the sequence, not in the middle of the first and last frame.
        # So we need to correct for that
        expected_delta_parang = (hdr['HIERARCH ESO TEL PARANG END'] -
                                 hdr['HIERARCH ESO TEL PARANG START']) * (n_frames-1)/n_frames
        delta_parang = (parang_array[-1]-parang_array[0])
        if np.abs(expected_delta_parang - delta_parang) > 1.:
            print(
                "WARNING! Calculated parallactic angle change is >1degree more than expected!")

    except Exception:
        pass

    return parang_array


def checkSeeingCond(list_nrm):
    """ Extract the seeing conditions, parang, averaged vis2
    and cp of a list of nrm classes extracted with extract_bs
    function (bispect.py).

    Output
    ------
    If output is **res**, access to parallactic angle by `res.pa`, or
    `res.seeing` for the seeing across multiple nrm data (files).

    """
    l_seeing, l_vis2, l_cp, l_pa, l_mjd = [], [], [], [], []
    
    hdr = fits.open(list_nrm[0].filename)[0].header
    for nrm in list_nrm:
        hdr = fits.open(nrm.filename)[0].header
        pa = np.mean(sphere_parang(hdr))
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


def plotSeeingCond(cond, lim_seeing=None):
    """ Plot seeing condition between calibrator and target files. """

    l_xmin, l_xmax = [], []
    for x in cond:
        m_mjd = abs(np.min(np.diff(x.mjd)))/2.
        xmin = np.min([x.mjd.min(), x.mjd.min()])-m_mjd
        xmax = np.max([x.mjd.max(), x.mjd.max()])+m_mjd
        l_xmin.append(xmin)
        l_xmax.append(xmax)

    xmin = np.min(l_xmin)
    xmax = np.min(l_xmax)

    # m_mjd=abs(np.min(np.diff(cond_t.mjd)))/2.
    # xmin=np.min([cond_c.mjd.min(), cond_t.mjd.min()])-m_mjd
    # xmax=np.max([cond_c.mjd.max(), cond_t.mjd.max()])+m_mjd

    fig = plt.figure()
    ax1 = plt.gca()
    ax1.set_xlabel('mjd [days]')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Seeing ["]', color="#c62d42")
    ax2.tick_params(axis='y', labelcolor="#c62d42")
    ax1.set_ylabel('Uncalibrated mean V$^2$')

    for x in cond:
        ax1.plot(x.mjd, x.vis2, '.', label=x.target)  # color="#20b2aa"
        ax2.plot(x.mjd, x.seeing, '+', color="#c62d42")
    # ax1.plot(cond_c.mjd, cond_c.vis2, '.',
    #          label="%s (cal)" % cond_c.target)
    # ax2.plot(cond_c.mjd, cond_c.seeing, '+',
    #          color="#c62d42", label='Seeing')
    if lim_seeing is not None:
        ax2.hlines(lim_seeing, xmin, xmax, color='g',
                   label='Seeing threshold')
    ax1.set_ylim(0, 1.2)
    ax2.set_ylim(0.6, 1.8)
    ax1.set_xlim(xmin, xmax)
    ax1.grid(alpha=.1, color='grey')
    ax1.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.show(block=False)
    return fig
