"""Deprecated utilities for spectrally dispersed IFU NRM data."""

import warnings

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from rich.progress import track

from .data_processing import select_clean_data
from .get_infos_obs import get_wavelength

warnings.warn(
    "The amical.ifu module is deprecated "
    "and will be removed in a future version. "
    "Please do not rely on it.",
    category=UserWarning,
    stacklevel=2,
)


def get_lambda(i_wl=None, filtname="YH", instrument="SPHERE-IFS"):
    """Display and return wavelengths for the selected IFU channels.

    Parameters
    ----------
    i_wl : int or list of int or None, default=None
        One channel, a two-index channel range, or all channels.
    filtname : str, default="YH"
        IFU filter name.
    instrument : str, default="SPHERE-IFS"
        Instrument name.

    Returns
    -------
    float, numpy.ndarray, or None
        Selected wavelength values in micrometres, or None when unavailable.
    """
    wl = get_wavelength(instrument, filtname) * 1e6

    if np.isnan(wl.any()):
        return None

    print(f"\nInstrument: {instrument}, spectral range: {filtname}")
    print("-----------------------------")
    print(
        f"spectral coverage: {wl[0]:2.2f} - {wl[-1]:2.2f} µm (step = {np.diff(wl)[0]:2.2f})"
    )

    one_wl = True
    if isinstance(i_wl, list):
        one_wl = False
        wl_range = wl[i_wl[0] : i_wl[1]]
        sp_range = np.arange(i_wl[0], i_wl[1], 1)
    elif i_wl is None:
        one_wl = False
        sp_range = np.arange(len(wl))
        wl_range = wl

    plt.figure(figsize=(4, 3))
    plt.title("--- SPECTRAL INFORMATION (IFU)---")
    plt.plot(wl, label="All spectral channels")
    if one_wl:
        plt.plot(
            np.arange(len(wl))[i_wl],
            wl[i_wl],
            "ro",
            label=f"Selected ({wl[i_wl]:2.2f} µm)",
        )
    else:
        plt.plot(
            sp_range,
            wl_range,
            lw=5,
            alpha=0.5,
            label=f"Selected ({wl_range[0]:2.2f}-{wl_range[-1]:2.2f} µm)",
        )
    plt.legend()
    plt.xlabel("Spectral channel")
    plt.ylabel("Wavelength [µm]")
    plt.tight_layout()

    if one_wl:
        output = np.round(wl[i_wl], 2)
    else:
        output = np.round(wl_range)
    return output


def clean_data(
    list_file,
    isz=256,
    r1=100,
    dr=10,
    edge=0,
    bad_map=None,
    add_bad=None,
    offx=0,
    offy=0,
    clip_fact=0.5,
    apod=True,
    sky=True,
    window=None,
    f_kernel=3,
    verbose=False,
    ihdu=0,
    display=False,
):
    """Clean IFU files and reshape their frames into a four-dimensional cube.

    Parameters
    ----------
    list_file : list of str or path-like
        IFU FITS files.
    isz : int, default=256
        Crop size.
    r1, dr : int, default=100, 10
        Sky-ring parameters.
    edge : int, default=0
        Detector-edge width to remove.
    bad_map : numpy.ndarray or None, default=None
        Bad-pixel map.
    add_bad : list or None, default=None
        Additional bad-pixel coordinates.
    offx, offy : int, default=0
        Crop-center offsets.
    clip_fact : float, default=0.5
        Relative sigma threshold for frame selection.
    apod, sky : bool, default=True
        Whether to apodize and sky-subtract frames.
    window : float or None, default=None
        Apodization width.
    f_kernel : int or None, default=3
        Median-filter kernel size for centering.
    verbose : bool, default=False
        Whether to print progress.
    ihdu : int, default=0
        FITS HDU containing the data.
    display : bool, default=False
        Whether to display cleaning parameters.

    Returns
    -------
    numpy.ndarray
        Cube with shape (ndit, nlambda, isz, isz).
    """

    clean_param = {
        "isz": isz,
        "r1": r1,
        "dr": dr,
        "edge": edge,
        "clip": False,
        "bad_map": bad_map,
        "add_bad": add_bad,
        "offx": offx,
        "offy": offy,
        "clip_fact": clip_fact,
        "apod": apod,
        "sky": sky,
        "window": window,
        "f_kernel": f_kernel,
        "verbose": verbose,
        "ihdu": ihdu,
        "display": display,
    }

    # Add check to create default add_bad list (not use mutable data)
    if add_bad is None:
        add_bad = []

    with fits.open(list_file[0]) as fd:
        hdr = fd[0].header

    nlambda = hdr["NAXIS3"]
    nframe = len(list_file)

    cube_lambda = np.zeros([nframe, nlambda, isz, isz])

    for i in track(
        range(len(list_file)),
        desription="Format/clean IFU ({})".format(hdr["OBJECT"]),
    ):
        cube_cleaned = select_clean_data(list_file[i], **clean_param)
        cube_lambda[i] = cube_cleaned

    return cube_lambda
