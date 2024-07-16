"""
@author: Anthony Soulain (University of Sydney)

-------------------------------------------------------------------------
AMICAL: Aperture Masking Interferometry Calibration and Analysis Library
-------------------------------------------------------------------------

General tools.

--------------------------------------------------------------------
"""

import math as m
import sys
import warnings

import numpy as np
from rich import print as rprint

from amical.externals.munch import munchify as dict2class


def linear(x, param):
    """Linear model used in dpfit"""
    a = param["a"]
    b = param["b"]
    y = a * x + b
    return y


def mas2rad(mas):
    """Convert angle in milli-arcsec to radians"""
    rad = mas * (10 ** (-3)) / (3600 * 180 / np.pi)
    return rad


def rad2mas(rad):
    """Convert input angle in radians to milli-arcsec"""
    mas = rad * (3600.0 * 180 / np.pi) * 10.0**3
    return mas


def find_max(img, filtmed=True, f=3):
    """
    Summary
    -------------
    Find brightest pixel of an image

    Parameters
    ----------
    `img` : {numpy.array}
        input image,\n
    `filtmed` : {boolean}, (optionnal)
        True if perform a median filter on the image (to blur bad pixels),\n
    `f` : {float}, (optionnal),
        If filtmed == True, kernel size of the median filter.


    Returns
    -------
    `Max coordinates`: {tuple}
        X and Y positions of max pixel
    """
    from scipy.signal import medfilt2d

    if filtmed:
        try:
            im_med = medfilt2d(img, f)
        except ValueError:
            img = img.astype(float)
            im_med = medfilt2d(img, f)
    else:
        im_med = img.copy()

    pos_max = np.where(im_med == im_med.max())

    X = pos_max[1][0]
    Y = pos_max[0][0]

    return X, Y


def crop_max(img, dim, offx=0, offy=0, filtmed=True, f=3):
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
    from astropy.nddata import Cutout2D

    xmax, ymax = find_max(img, filtmed=filtmed, f=f)

    X = xmax + offx
    Y = ymax + offy
    isz_max = 2 * np.min([X, img.shape[1] - X - 1, Y, img.shape[0] - Y - 1]) + 1
    if isz_max < dim:
        size_msg = (
            f"The specified cropped image size, {dim}, is greater than the distance to"
            " the PSF center in at least one dimension. The max size for this image is"
            f" {isz_max}"
        )
        raise ValueError(size_msg)
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
    tab_norm = tab / np.max(tab)
    return tab_norm


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

    fwhmx = param["fwhm_x"] / pixel_scale
    fwhmy = param["fwhm_y"] / pixel_scale

    sigma_x = fwhmx / np.sqrt(8 * np.log(2))
    sigma_y = fwhmy / np.sqrt(8 * np.log(2))

    amplitude = param["A"]
    x0 = dim // 2 + param["x0"] / pixel_scale
    y0 = dim // 2 + param["y0"] / pixel_scale
    theta = np.deg2rad(param["theta"])
    size_x = len(x)
    size_y = len(y)
    im = np.zeros([size_y, size_x])
    x0 = float(x0)
    y0 = float(y0)
    a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (
        2 * sigma_y**2
    )
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (np.cos(theta) ** 2) / (
        2 * sigma_y**2
    )
    im = amplitude * np.exp(
        -(a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0) + c * ((y - y0) ** 2))
    )
    return im


def plot_circle(d, x, y, hole_radius, sz=1, display=True):
    """Return an image with a disk = sz at x, y position and zero elsewhere"""
    chipsz = np.shape(d)[0]

    im = np.zeros([chipsz, chipsz])
    info = [len(im.shape), im.shape[0], im.shape[1], 3, len(im.ravel())]

    if isinstance(x, (float, int)):
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
        ind2 = int(min([info[1] - 1, xx[c] + r + 1]))
        for i in np.arange(ind1, ind2, 1):
            ind3 = int(max([0.0, yy[c] - r - 1]))
            ind4 = int(min([info[2] - 1, yy[c] + r + 1]))
            for j in np.arange(ind3, ind4, 1):
                r_d = np.sqrt(
                    (float(i) - float(xx[c])) ** 2 + (float(j) - float(yy[c])) ** 2
                )
                if r_d <= r:
                    im[i, j] = sz

    if display:
        import matplotlib.pyplot as plt

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
        if cxx < 0.0:
            str_err = "diagonal cov[%d,%d]=%e is not positive" % (ix, ix, cxx)
            raise ValueError(str_err)
        for iy in range(cov.shape[1]):
            cyy = cov[iy, iy]
            if cyy < 0.0:
                str_err = "diagonal cov[%d,%d]=%e is not positive" % (iy, iy, cyy)
                raise ValueError(str_err)
            cor[ix, iy] = cov[ix, iy] / np.sqrt(cxx * cyy)

    return cor, sigma


def super_gaussian(
    x: np.ndarray, sigma: float, m: float = 3.0, amp: float = 1.0, x0: float = 0.0
) -> np.ndarray:
    """
    Function for creating a super-Gaussian window.

    Parameters
    ----------
    x : np.ndarray
        2D array with the distances of each pixel to the image center.
    sigma : float
        Full width at half maximum (FWHM) of the super-Gaussian
        window. It is therefore not the standard deviation, as the
        parameter name would suggest.
    m : float
        Exponent used for the super-Gaussian function (default: 3.0).
    amp : float
        Amplitude of the Gaussian function (default: 1.0).
    x0 : float
        Offset applied to the distances (default: 0.0)

    Returns
    -------
    np.ndarray
        2D array with the super-Gaussian window function.
    """

    return amp * (
        (
            np.exp(
                -(2 ** (2 * m - 1)) * np.log(2) * (((x - x0) ** 2) / (sigma**2)) ** m
            )
        )
        ** 2
    )


def apply_windowing(
    img: np.ndarray, window: float = 80.0, m: float = 3.0
) -> np.ndarray:
    """
    Function for applying a super-Gaussian window to an image.

    Parameters
    ----------
    img : np.ndarray
        2D array with the input image.
    window : float
        Half width at half maximum (HWHM) of the window function
        (default: 80.0).
    m : float
        Exponent used for the super-Gaussian function (default: 3.0).

    Returns
    -------
    np.ndarray
        2D array with the windowed input image.
    """
    isz = len(img)
    xx, yy = np.arange(isz), np.arange(isz)
    xx2 = xx - isz // 2
    yy2 = isz // 2 - yy
    # Distance map
    distance = np.sqrt(xx2**2 + yy2[:, np.newaxis] ** 2)

    # Super-gaussian windowing
    # Mutiply the window value with 2 to change from HWHM to FWHM
    super_gauss = super_gaussian(distance, sigma=window * 2, m=m)

    # Apply the windowing
    return img * super_gauss


def sanitize_array(dic):  # pragma: no cover
    """Recursively convert values in a nested dictionnary from np.bool_ to builtin bool type
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
    variance = np.average((values - mn) ** 2, weights=weights, axis=0)
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
    """Convert Julian date to LST"""
    c = [280.46061837, 360.98564736629, 0.000387933, 38710000.0]
    jd2000 = 2451545.0
    t0 = jd - jd2000
    t = t0 / 36525.0

    # Compute GST in seconds.
    theta = c[0] + (c[1] * t0) + t**2 * (c[2] - t / c[3])

    # Compute LST in hours.
    lst = (theta + lng) / 15.0
    cond_neg = lst < 0.0
    n = lst[cond_neg].size
    if n > 0:
        lst[cond_neg] = 24.0 + (lst[cond_neg] % 24)
    lst = lst % 24
    return lst


def compute_pa(hdr, n_ps, verbose=False, display=False, *, sci_hdr=None):
    list_fct_pa = {
        "SPHERE": (sphere_parang, {"hdr": hdr, "n_dit_ifs": n_ps}),
        "NIRISS": (niriss_parang, {"hdr": sci_hdr}),
    }

    instrument = hdr["INSTRUME"]
    if instrument not in list(list_fct_pa.keys()):
        try:
            nframe = hdr["NAXIS3"]
        except KeyError:
            nframe = n_ps
        if verbose:
            rprint(
                f"[green]Warning: {instrument} not in known pa computation"
                " -> set to zero.\n",
                file=sys.stderr,
            )
        pa_exist = False
        l_pa = np.zeros(nframe)
    else:
        fct_pa, kwargs_pa = list_fct_pa[instrument]
        l_pa = fct_pa(**kwargs_pa)
        pa_exist = True

    pa = np.mean(l_pa)
    std_pa = np.std(l_pa)

    if display and pa_exist:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(4, 3))
        plt.plot(l_pa, ".-", label=rf"pa={pa:2.1f}, $\sigma_{{pa}}$={std_pa:2.1f} deg")
        plt.legend(fontsize=7)
        plt.grid(alpha=0.2)
        plt.xlabel("# frames")
        plt.ylabel("Position angle [deg]")
        plt.tight_layout()

    return pa


def niriss_parang(hdr):
    if hdr is None:
        warnings.warn(
            "No SCI header for NIRISS. No PA correction will be applied.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 0.0
    v3i_yang = hdr["V3I_YANG"]  # Angle from V3 axis to ideal y axis (deg)
    roll_ref_pa = hdr["ROLL_REF"]  # Offset between V3 and N in local aperture coord

    return roll_ref_pa - v3i_yang


def sphere_parang(hdr, n_dit_ifs=None):
    """
    Reads the header and creates an array giving the paralactic angle for each frame,
    taking into account the inital derotator position.
    The columns of the output array contains:
    frame_number, frame_time, paralactic_angle
    """
    from astropy.time import Time

    r2d = 180 / np.pi
    d2r = np.pi / 180

    detector = hdr["HIERARCH ESO DET ID"]
    if detector.strip() == "IFS":
        offset = 135.87 - 100.46  # from the SPHERE manual v4
    elif detector.strip() == "IRDIS":
        # correspond to the difference between the PUPIL tracking ant the FIELD tracking for IRDIS taken here: http://wiki.oamp.fr/sphere/AstrometricCalibration (PUPOFFSET)
        offset = 135.87
    else:
        offset = 0
        print(
            "WARNING: Unknown instrument in create_parang_list_sphere: " + str(detector)
        )

    try:
        # Get the correct RA and Dec from the header
        actual_ra = hdr["HIERARCH ESO INS4 DROT2 RA"]
        actual_dec = hdr["HIERARCH ESO INS4 DROT2 DEC"]

        # These values were in weird units: HHMMSS.ssss
        actual_ra_hr = np.floor(actual_ra / 10000.0)
        actual_ra_min = np.floor(actual_ra / 100.0 - actual_ra_hr * 100.0)
        actual_ra_sec = actual_ra - actual_ra_min * 100.0 - actual_ra_hr * 10000.0

        ra_deg = (
            (actual_ra_hr + actual_ra_min / 60.0 + actual_ra_sec / 60.0 / 60.0)
            * 360.0
            / 24.0
        )

        # the sign makes this complicated, so remove it now and add it back at the end
        sgn = np.sign(actual_dec)
        actual_dec *= sgn

        actual_dec_deg = np.floor(actual_dec / 10000.0)
        actual_dec_min = np.floor(actual_dec / 100.0 - actual_dec_deg * 100.0)
        actual_dec_sec = actual_dec - actual_dec_min * 100.0 - actual_dec_deg * 10000.0

        dec_deg = (
            actual_dec_deg + actual_dec_min / 60.0 + actual_dec_sec / 60.0 / 60.0
        ) * sgn
        geolat_rad = float(hdr["ESO TEL GEOLAT"]) * d2r
    except Exception:
        print("WARNING: No RA/Dec Keywords found in header")
        ra_deg = 0
        dec_deg = 0
        geolat_rad = 0

    if "NAXIS3" in hdr:
        if detector.strip() == "IFS":
            n_frames = n_dit_ifs
        else:
            n_frames = hdr["NAXIS3"]
    else:
        n_frames = 1

    # We want the exposure time per frame, derived from the total time from when the shutter
    # opens for the first frame until it closes at the end.
    # This is what ACC thought should be used
    # total_exptime = hdr['ESO DET SEQ1 EXPTIME']
    # This is what the SPHERE DC uses
    total_exptime = (
        Time(hdr["HIERARCH ESO DET FRAM UTC"]) - Time(hdr["HIERARCH ESO DET SEQ UTC"])
    ).sec
    # print total_exptime-total_exptime2
    delta_dit = total_exptime / n_frames
    dit = hdr["ESO DET SEQ1 REALDIT"]

    # Set up the array to hold the parangs
    parang_array = np.zeros(n_frames)

    # Output for debugging
    hour_angles = []

    if ("ESO DET SEQ UTC" in hdr.keys()) and ("ESO TEL GEOLON" in hdr.keys()):
        # The SPHERE DC method
        jd_start = Time(hdr["ESO DET SEQ UTC"]).jd
        lst_start = jd2lst(hdr["ESO TEL GEOLON"], jd_start) * 3600
        # Use the old method
        lst_start = float(hdr["LST"])
    else:
        lst_start = 0.0
        print("WARNING: No LST keyword found in header")

    # delta dit and dit are in seconds so we need to multiply them by this factor to add them to an LST
    time_to_lst = (24.0 * 3600.0) / (86164.1)

    if "ESO INS4 COMB ROT" in hdr.keys() and hdr["ESO INS4 COMB ROT"] == "PUPIL":
        for i in range(n_frames):
            ha_deg = (
                (lst_start + i * delta_dit * time_to_lst + time_to_lst * dit / 2.0)
                * 15.0
                / 3600
            ) - ra_deg
            hour_angles.append(ha_deg)

            # VLT TCS formula
            f1 = float(np.cos(geolat_rad) * np.sin(d2r * ha_deg))
            f2 = float(
                np.sin(geolat_rad) * np.cos(d2r * dec_deg)
                - np.cos(geolat_rad) * np.sin(d2r * dec_deg) * np.cos(d2r * ha_deg)
            )
            pa = -r2d * np.arctan2(-f1, f2)

            pa = pa + offset

            # Also correct for the derotator issues that were fixed on 12 July 2016 (MJD = 57581)
            if hdr["MJD-OBS"] < 57581:
                alt = hdr["ESO TEL ALT"]
                drot_begin = hdr["ESO INS4 DROT2 BEGIN"]
                # Formula from Anne-Lise Maire
                correction = (
                    np.arctan(np.tan((alt - 2 * drot_begin) * np.pi / 180))
                    * 180
                    / np.pi
                )
                pa += correction

            pa = (pa + 360) % 360
            parang_array[i] = pa

    else:
        if "ARCFILE" in hdr.keys():
            print(hdr["ARCFILE"] + " does seem to be taken in pupil tracking.")
        else:
            print("Data does not seem to be taken in pupil tracking.")

        for i in range(n_frames):
            parang_array[i] = 0

    # And a sanity check at the end
    try:
        # The parang start and parang end refer to the start and end of the sequence, not in the middle of the first and last frame.
        # So we need to correct for that
        expected_delta_parang = (
            (hdr["HIERARCH ESO TEL PARANG END"] - hdr["HIERARCH ESO TEL PARANG START"])
            * (n_frames - 1)
            / n_frames
        )
        delta_parang = parang_array[-1] - parang_array[0]
        if np.abs(expected_delta_parang - delta_parang) > 1.0:
            print(
                "WARNING! Calculated parallactic angle change is >1degree more than expected!"
            )

    except Exception:
        pass

    return parang_array


def check_seeing_cond(list_nrm):  # pragma: no cover
    """Extract the seeing conditions, parang, averaged vis2
    and cp of a list of nrm classes extracted with extract_bs
    function (bispect.py).

    Output
    ------
    If output is **res**, access to parallactic angle by `res.infos.pa`, or
    `res.infos.seeing` for the seeing across multiple nrm data (files).

    """
    from astropy.io import fits

    l_seeing, l_vis2, l_cp, l_pa, l_mjd = [], [], [], [], []

    with fits.open(list_nrm[0].infos.filename) as fd:
        hdr = fd[0].header
    for nrm in list_nrm:
        with fits.open(nrm.infos.filename) as fd:
            hdr = fd[0].header
        pa = np.mean(sphere_parang(hdr))
        seeing = nrm.infos.seeing
        mjd = hdr["MJD-OBS"]
        l_vis2.append(np.mean(nrm.vis2))
        l_cp.append(np.mean(nrm.cp))
        l_seeing.append(seeing)
        l_pa.append(pa)
        l_mjd.append(mjd)

    res = {
        "pa": l_pa,
        "seeing": l_seeing,
        "vis2": l_vis2,
        "cp": l_cp,
        "mjd": l_mjd,
        "target": hdr["OBJECT"],
    }

    return dict2class(sanitize_array(res))


def plot_seeing_cond(cond, lim_seeing=None):  # pragma: no cover
    """Plot seeing condition between calibrator and target files."""
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax1 = plt.gca()
    ax1.set_xlabel("mjd [days]")
    ax2 = ax1.twinx()
    ax2.set_ylabel('Seeing ["]', color="#c62d42")
    ax2.tick_params(axis="y", labelcolor="#c62d42")
    ax1.set_ylabel("Uncalibrated mean V$^2$")

    for x in cond:
        ax1.plot(x.mjd, x.vis2, ".", label=x.target)
        ax2.plot(x.mjd, x.seeing, "+", color="#c62d42")
    if lim_seeing is not None:
        ax2.axhline(lim_seeing, color="g", label="Seeing threshold")
    ax1.set_ylim(0, 1.2)
    ax2.set_ylim(0.6, 1.8)
    ax1.grid(alpha=0.1, color="grey")
    ax1.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.show(block=False)
    return fig


def roundSciDigit(number):
    """Rounds a float number with a significant digit number."""
    ff = str(number).split(".")[0]
    d = str(number).split(".")[1]
    d, ff = m.modf(number)
    if ff == 0:
        res = str(d).split(".")[1]
        for i in range(len(res)):
            if float(res[i]) != 0.0:
                sig_digit = i + 1
                break
    else:
        sig_digit = 1

    return float(np.round(number, sig_digit)), sig_digit


def save_bs_hdf5(bs, filename):
    """Save results from `amical.extract_bs()` into hdf5 file."""
    import h5py

    if ".h5" not in filename:
        filename += ".h5"

    # Step 1 - Create hdf5 and hierarchy tree/grp
    hf = h5py.File(filename, "w")

    grp_obs = hf.create_group("obs")
    grp_matrix = hf.create_group("matrix")
    grp_info = hf.create_group("infos")
    grp_mask = hf.create_group("mask")
    grp_hdr = hf.create_group("hdr")

    # Step 2 - Save observable (not in tree matrix, infos and mask).
    for key in bs:
        if key not in ["matrix", "mask", "infos"]:
            grp_obs.create_dataset(key, data=bs[key])

    # Step 3 - Save matrix (contains all individual observable
    # frame by frame and statistics matrix (covariance, variance, etc.)).
    matrix = bs.matrix
    for mat in matrix:
        if matrix[mat] is not None:
            grp_matrix.create_dataset(mat, data=matrix[mat])
        else:
            grp_matrix.create_dataset(mat, data=[0])

    # Step 4 - Save mask (contains all mask informations) as
    # well as u1, v1, u2, v2 coordinates (for the CP).
    mask = bs.mask
    for key in mask:
        if (mask[key] is not None) and (key != "t3_coord"):
            grp_mask.create_dataset(key, data=mask[key])

    t3_coord = mask["t3_coord"]
    for key in t3_coord:
        grp_mask.create_dataset(key, data=t3_coord[key])

    # Step 5 - Save informations (target, date, etc.).
    infos = bs.infos
    for key in infos:
        if key != "hdr":
            info = infos[key]
            if info is None:
                info = "Unknown"
            grp_info.attrs[key] = info

    # Step 6 - Save original header keywords.
    hdr = bs.infos.hdr
    for key in hdr:
        grp_hdr.attrs[key] = str(hdr[key])

    # Last step - close hdf5
    hf.close()
    return None


def load_bs_hdf5(filename):
    """Load hdf5 file and format as class like object (same
    format as `amical.extract_bs()`
    """
    import h5py

    dict_bs = {"matrix": {}, "infos": {"hdr": {}}, "mask": {}}
    with h5py.File(filename, "r") as hf2:
        obs = hf2["obs"]
        for o in obs:
            dict_bs[o] = np.array(obs.get(o))

        matrix = hf2["matrix"]
        for key in matrix:
            dict_bs["matrix"][key] = np.array(matrix.get(key))

        if len(dict_bs["matrix"]["cp_cov"]) == 1:
            dict_bs["matrix"]["cp_cov"] = None

        mask = hf2["mask"]
        for key in mask:
            if key not in ["u1", "u2", "v1", "v2"]:
                dict_bs["mask"][key] = np.array(mask.get(key))

        t3_coord = {
            "u1": np.array(mask.get("u1")),
            "u2": np.array(mask.get("u2")),
            "v1": np.array(mask.get("v1")),
            "v2": np.array(mask.get("v2")),
        }

        dict_bs["mask"]["t3_coord"] = t3_coord

        infos = hf2["infos"]
        for key in hf2["infos"].attrs.keys():
            dict_bs["infos"][key] = infos.attrs[key]

        hdr = hf2["hdr"]
        for key in hdr.attrs.keys():
            dict_bs["infos"]["hdr"][key] = hdr.attrs[key]

        bs_save = dict2class(dict_bs)
    return bs_save
