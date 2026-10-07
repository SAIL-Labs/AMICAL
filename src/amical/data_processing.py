"""Clean and select aperture-masking data cubes."""

import sys
import warnings

import numpy as np
from rich import print as rprint
from rich.progress import track

from amical.tools import apply_windowing, crop_max, find_max


def _apply_patch_ghost(cube, xc, yc, radius=20, dx=0, dy=-200, method="bg"):
    """Replace a circular spectral-filter ghost in each cube frame.

    Parameters
    ----------
    cube : numpy.ndarray
        Data cube.
    xc, yc : int
        Artifact x and y coordinates.
    radius : int, default=20
        Radius of the replacement circle.
    dx, dy : int, default=0, -200
        Offset used to sample replacement background values.
    method : str, default="bg"
        Use sampled background values when "bg"; otherwise use zeros.

    Returns
    -------
    numpy.ndarray
        Ghost-corrected cube.
    """
    cube_corrected = []
    for i in range(len(cube)):
        imA = cube[i].copy()
        isz = imA.shape[0]
        xc_off, yc_off = xc + dx, yc + dy
        xx, yy = np.arange(isz), np.arange(isz)
        xx_c = xx - xc
        yy_c = yc - yy
        xx_off = xx - xc_off
        yy_off = yc_off - yy
        distance = np.sqrt(xx_c**2 + yy_c[:, np.newaxis] ** 2)
        distance_off = np.sqrt(xx_off**2 + yy_off[:, np.newaxis] ** 2)
        cond_patch = distance <= radius
        cond_bg = distance_off <= radius
        if method == "bg":
            imA[cond_patch] = np.mean(imA[cond_bg])
        elif method == "zero":
            imA[cond_patch] = 0
        cube_corrected.append(imA)
    cube_corrected = np.array(cube_corrected)
    return cube_corrected


def select_data(cube, clip_fact=0.5, clip=False, verbose=True, display=True):
    """Select frames centered in Fourier space and, optionally, by flux.

    Parameters
    ----------
    cube : numpy.ndarray
        Cleaned data cube.
    clip_fact : float, default=0.5
        Relative sigma threshold for flux selection.
    clip : bool, default=False
        Whether to apply flux sigma clipping.
    verbose : bool, default=True
        Whether to print selection information.
    display : bool, default=True
        Whether to display diagnostic plots.

    Returns
    -------
    numpy.ndarray
        Selected frames.
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
        if (med_flux / std_flux) <= 5.0:
            rprint(
                "[cyan]\n"
                f"Std of the fluxes along the cube < 5 ({med_flux / std_flux:2.1f}):\n"
                " -> sigma clipping is suggested (clip=True).",
            )

    limit_flux = med_flux - clip_fact * std_flux

    if clip:
        is_flagged = np.isin(np.arange(len(fluxes)), flag_fram)
        cond_clip = (fluxes > limit_flux) & ~is_flagged
        cube_cleaned_checked = cube[cond_clip]
        ind_clip = np.where(fluxes <= limit_flux)[0]
    else:
        ind_clip = []
        cube_cleaned_checked = np.array(good_fram)

    ind_clip2 = np.where(fluxes <= limit_flux)[0]
    if ((worst_fr in ind_clip2) and clip) or (worst_fr in flag_fram):
        ext = "(rejected)"
    else:
        ext = ""

    diffmm = 100 * abs(np.max(fluxes) - np.min(fluxes)) / med_flux
    if display:
        import matplotlib.pyplot as plt
        from matplotlib.colors import PowerNorm

        plt.figure(figsize=(10, 5))
        plt.plot(
            fluxes,
            label=rf"|$\Delta F$|/$\sigma_F$={med_flux / std_flux:2.0f} ({diffmm:2.2f} %)",
            lw=1,
        )
        if len(flag_fram) > 0:
            plt.scatter(
                flag_fram,
                fluxes[flag_fram],
                s=52,
                facecolors="none",
                edgecolors="r",
                label="Rejected frames (maximum fluxes)",
            )
        if clip:
            if len(ind_clip) > 0:
                plt.plot(
                    ind_clip,
                    fluxes[ind_clip],
                    "x",
                    color="crimson",
                    label="Rejected frames (clipping)",
                )
            else:
                print("0")
        # plt.hlines(limit_flux, 0, len(fluxes), )
        plt.axhline(
            limit_flux,
            lw=3,
            color="#00b08b",
            ls="--",
            label="Clipping threshold",
            zorder=10,
        )
        plt.legend(loc="best", fontsize=9)
        plt.ylabel("Flux [counts]")
        plt.xlabel("# frames")
        plt.grid(alpha=0.2)
        plt.tight_layout()

        plt.figure(figsize=(7, 7))
        plt.subplot(2, 2, 1)
        plt.title(f"Best frame ({best_fr})")
        plt.imshow(cube[best_fr], norm=PowerNorm(0.5, vmin=0), cmap="afmhot")
        plt.subplot(2, 2, 2)
        plt.imshow(np.fft.fftshift(fft_fram[best_fr]), cmap="gist_stern")
        plt.subplot(2, 2, 3)
        plt.title(f"Worst frame ({worst_fr}) {ext}")
        plt.imshow(cube[worst_fr], norm=PowerNorm(0.5, vmin=0), cmap="afmhot")
        plt.subplot(2, 2, 4)
        plt.imshow(np.fft.fftshift(fft_fram[worst_fr]), cmap="gist_stern")
        plt.tight_layout()
        plt.show(block=False)
    if verbose:
        n_good = len(cube_cleaned_checked)
        n_bad = len(cube) - n_good
        if clip:
            rprint("[cyan]\n---- σ-clip + centered fluxes selection ---")
        else:
            rprint("[cyan]\n---- centered fluxes selection ---")
        print(
            f"{n_bad}/{len(cube)} ({100 * n_bad / len(cube):2.1f}%) are flagged as bad frames"
        )
    return cube_cleaned_checked


def _get_ring_mask(r1, dr, isz, center=None):
    if center is None:
        xc, yc = isz // 2, isz // 2
    else:
        xc, yc = center
    xx, yy = np.arange(isz), np.arange(isz)
    xx2 = xx - xc
    yy2 = yc - yy
    distance = np.sqrt(xx2**2 + yy2[:, np.newaxis] ** 2)
    inner_cond = r1 <= distance
    if dr is not None:
        r2 = r1 + dr
        outer_cond = distance <= r2
    else:
        outer_cond = True
    cond_bg = inner_cond & outer_cond

    if dr is not None and np.all(outer_cond):
        warnings.warn(
            "The outer radius is out of the image, using everything beyond r1 as background",
            RuntimeWarning,
            stacklevel=2,
        )

    return cond_bg


def sky_correction(imA, r1=None, dr=None, verbose=False, *, center=None, mask=None):
    """Subtract a sky background estimated from a ring or mask.

    Parameters
    ----------
    imA : numpy.ndarray
        Image to correct.
    r1 : float or None, default=None
        Inner radius of the background ring.
    dr : float or None, default=None
        Background ring width.
    verbose : bool, default=False
        Whether to print correction information.
    center : tuple of float or None, keyword-only
        Ring center.
    mask : numpy.ndarray of bool or None, keyword-only
        Background mask with the same image shape.

    Returns
    -------
    image : numpy.ndarray
        Sky-corrected image.
    background : float
        Estimated background.
    """
    # FUTURE: Future AMICAL release should raise error
    if r1 is None and mask is None:
        warnings.warn(
            "The default value of r1 and dr is now None. Either mask or r1 must be set"
            " explicitely. In the future, this will result in an error."
            " Setting r1=100 and dr=20",
            PendingDeprecationWarning,
            stacklevel=2,
        )
        r1 = 100
        dr = 20

    if r1 is not None and mask is not None:
        raise TypeError("Only one of mask and r1 can be specified")
    elif r1 is None and dr is not None:
        raise TypeError("dr cannot be set when r1 is None")
    elif r1 is not None:
        isz = imA.shape[0]
        cond_bg = _get_ring_mask(r1, dr, isz, center=center)
    elif mask is not None:
        if mask.shape != imA.shape:
            raise ValueError("mask should have the same shape as image")
        elif not mask.any():
            warnings.warn(
                "Background not computed because mask has no True values",
                RuntimeWarning,
                stacklevel=2,
            )
        cond_bg = mask

    do_bg = cond_bg.any()

    if do_bg:
        try:
            minA = imA.min()
            imB = imA + 1.01 * abs(minA)
            backgroundB = np.mean(imB[cond_bg])
            imC = imB - backgroundB
            backgroundC = np.mean(imC[cond_bg])
        except IndexError:
            do_bg = False

    # Not using else because do_bg can change in except above
    if not do_bg:
        imC = imA.copy()
        backgroundC = 0
        warnings.warn(
            "Background not computed, likely because specified radius is out of bounds",
            RuntimeWarning,
            stacklevel=2,
        )

    return imC, backgroundC


def fix_bad_pixels(image, bad_map, add_bad=None, x_stddev=1):
    """Interpolate bad pixels from neighboring values with a Gaussian kernel.

    Parameters
    ----------
    image : numpy.ndarray
        Image to correct.
    bad_map : numpy.ndarray
        Bad-pixel map.
    add_bad : list or None, default=None
        Additional bad-pixel coordinates.
    x_stddev : float, default=1
        Gaussian-kernel standard deviation.

    Returns
    -------
    numpy.ndarray
        Image with interpolated bad pixels.
    """
    from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans

    if add_bad is None:
        add_bad = []

    if len(add_bad) != 0:
        bad_map = bad_map.copy()  # Don't modify input bad pixel map, use a copy
        for j in range(len(add_bad)):
            bad_map[add_bad[j][1], add_bad[j][0]] = 1

    img_nan = image.astype(float)
    img_nan[bad_map == 1] = np.nan
    kernel = Gaussian2DKernel(x_stddev=x_stddev)
    fixed_image = interpolate_replace_nans(img_nan, kernel)
    return fixed_image


def _get_3d_bad_pixels(bad_map, add_bad, data):
    """Format bad-pixel inputs for a three-dimensional data cube.

    Parameters
    ----------
    bad_map : numpy.ndarray or None
        Two- or three-dimensional bad-pixel map.
    add_bad : list or None
        Bad-pixel coordinates.
    data : numpy.ndarray
        Data corresponding to the bad-pixel map.

    Returns
    -------
    bad_map : numpy.ndarray
        Three-dimensional bad-pixel map matching the data cube.
    add_bad : list
        Coordinates compatible with the three-dimensional data set.
    """
    n_im = data.shape[0]

    # Add check to create default add_bad list (not use mutable data)
    if add_bad is None or len(add_bad) == 0:
        # Reshape add_bad to simplify indexing in loop
        add_bad = [[]] * n_im
    else:
        add_bad = np.array(add_bad)
        if add_bad.ndim == 2 and len(add_bad[0]) != 0:
            add_bad = np.repeat(add_bad[np.newaxis, :], n_im, axis=0)
        elif add_bad.ndim == 3:
            if add_bad.shape[0] != n_im:
                raise ValueError("3D add_bad should have one list per frame")

    if (bad_map is None) and (len(add_bad) != 0):
        # If we have extra bad pixels, define bad_map with same shape as image
        bad_map = np.zeros_like(data, dtype=bool)
    elif bad_map is not None:
        # Shape should match data
        if bad_map.ndim == 2 and bad_map.shape != data[0].shape:
            raise ValueError(
                f"2D bad_map should have the same shape as a frame ({data[0].shape}),"
                f" but has shape {bad_map.shape}"
            )
        elif bad_map.ndim == 3 and bad_map.shape != data.shape:
            raise ValueError(
                f"3D bad_map should have the same shape as data cube ({data.shape}),"
                f" but has shape {bad_map.shape}"
            )
        elif bad_map.ndim == 2:
            bad_map = np.repeat(bad_map[np.newaxis, :], n_im, axis=0)

    return bad_map, add_bad


def show_clean_params(
    filename,
    isz=None,
    r1=None,
    dr=None,
    bad_map=None,
    add_bad=None,
    edge=0,
    remove_bad=True,
    nframe=0,
    ihdu=0,
    f_kernel=3,
    offx=0,
    offy=0,
    apod=False,
    window=None,
    *,
    ifu=False,
    mask=None,
):
    """Display the parameters used to clean a FITS data cube.

    Parameters
    ----------
    filename : str or path-like
        Data-cube filename.
    isz, r1, dr : int or None, default=None
        Crop size and sky-ring parameters.
    bad_map : numpy.ndarray or None, default=None
        Bad-pixel map.
    add_bad : list or None, default=None
        Additional bad-pixel coordinates.
    edge : int, default=0
        Detector-edge width to remove.
    remove_bad : bool, default=True
        Whether to interpolate bad pixels.
    nframe, ihdu : int, default=0
        Frame and FITS HDU to display.
    f_kernel : int or None, default=3
        Median-filter kernel size for centering.
    offx, offy : int, default=0
        Crop-center offsets.
    apod : bool, default=False
        Whether to show apodization.
    window : float or None, default=None
        Apodization width.
    ifu : bool, keyword-only, default=False
        Whether the input is an IFU cube.
    mask : numpy.ndarray of bool or None, keyword-only, default=None
        Sky-background mask.

    Returns
    -------
    matplotlib.figure.Figure
        Cleaning-parameter figure.
    """
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from matplotlib.colors import PowerNorm

    with fits.open(filename) as fd:
        data = fd[ihdu].data

    if ifu:
        data = data[0, :, :, :]

    img0 = data[nframe]
    dims = img0.shape

    bad_map, add_bad = _get_3d_bad_pixels(bad_map, add_bad, data)
    bmap0 = bad_map[nframe]
    ab0 = add_bad[nframe]

    if edge != 0:
        img0[:, 0:edge] = 0
        img0[:, -edge:] = 0
        img0[0:edge, :] = 0
        img0[-edge:, :] = 0
    if (bad_map is not None) & (remove_bad):
        img1 = fix_bad_pixels(img0, bmap0, add_bad=ab0)
    else:
        img1 = img0.copy()

    if isz is None:
        pos = (img1.shape[0] // 2, img1.shape[1] // 2)

        print(
            "Warning: isz not found (None by default). "
            f"isz is set to the original image size ({dims[0]}).",
            file=sys.stderr,
        )

        isz = dims[0]

    else:
        # Get expected center for sky correction
        filtmed = f_kernel is not None
        _, pos = crop_max(img1, isz, offx=offx, offy=offy, filtmed=filtmed, f=f_kernel)

    noBadPixel = False
    bad_pix_x, bad_pix_y = [], []
    if np.any(bmap0):
        if len(ab0) != 0:
            for j in range(len(ab0)):
                bmap0[ab0[j][1], ab0[j][0]] = 1
        bad_pix = np.where(bmap0 == 1)
        bad_pix_x = bad_pix[0]
        bad_pix_y = bad_pix[1]
    else:
        noBadPixel = True

    theta = np.linspace(0, 2 * np.pi, 100)
    sky_method = None
    x0 = pos[0]
    y0 = pos[1]
    if r1 is not None:
        x1 = r1 * np.cos(theta) + x0
        y1 = r1 * np.sin(theta) + y0
        if dr is not None:
            r2 = r1 + dr
            x2 = r2 * np.cos(theta) + x0
            y2 = r2 * np.sin(theta) + y0
        sky_method = "ring"
    elif mask is not None:
        bg_coords = np.where(mask == 1)
        bg_x = bg_coords[0]
        bg_y = bg_coords[1]
        sky_method = "mask"
    if window is not None:
        r3 = window
        x3 = r3 * np.cos(theta) + x0
        y3 = r3 * np.sin(theta) + y0

    xs1, ys1 = x0 + isz // 2, y0 + isz // 2
    xs2, ys2 = x0 - isz // 2, y0 + isz // 2
    xs3, ys3 = x0 - isz // 2, y0 - isz // 2
    xs4, ys4 = x0 + isz // 2, y0 - isz // 2

    max_val = img1[y0, x0]
    fig = plt.figure(figsize=(5, 5))
    plt.title("--- CLEANING PARAMETERS ---")
    plt.imshow(img1, norm=PowerNorm(0.5, vmin=0, vmax=max_val), cmap="afmhot")
    if sky_method == "ring":
        if dr is not None:
            plt.plot(x1, y1, label="Inner radius for sky subtraction")
            plt.plot(x2, y2, label="Outer radius for sky subtraction")
        else:
            plt.plot(x1, y1, label="Boundary for sky subtraction")
    elif sky_method == "mask":
        plt.scatter(
            bg_y,
            bg_x,
            color="None",
            marker="s",
            edgecolors="C0",
            s=20,
            label="Pixels used for sky subtraction",
        )
    if apod:
        if window is not None:
            plt.plot(x3, y3, "--", label="Super-gaussian windowing")
    plt.plot(x0, y0, "+", color="c", ms=10, label="Centering position")
    plt.plot(
        [xs1, xs2, xs3, xs4, xs1],
        [ys1, ys2, ys3, ys4, ys1],
        "w--",
        label="Resized image",
    )
    plt.xlim((0, dims[0] - 1))
    plt.ylim((0, dims[1] - 1))
    if not noBadPixel:
        if remove_bad:
            label = "Fixed hot/bad pixels"
        else:
            label = "Hot/bad pixels"
        plt.scatter(
            bad_pix_y,
            bad_pix_x,
            color="None",
            marker="s",
            edgecolors="r",
            facecolors="None",
            s=20,
            label=label,
        )

    plt.xlabel("X [pix]")
    plt.ylabel("Y [pix]")
    plt.legend(fontsize=8, loc=1)
    plt.tight_layout()
    return fig


def _apply_edge_correction(img0, edge=0):
    """Set bright detector edges to zero.

    Parameters
    ----------
    img0 : numpy.ndarray
        Image to correct.
    edge : int, default=0
        Width of the edge to remove.

    Returns
    -------
    numpy.ndarray
        Edge-corrected image.
    """
    if edge != 0:
        img0[:, 0:edge] = 0
        img0[:, -edge:] = 0
        img0[0:edge, :] = 0
        img0[-edge:, :] = 0
    return img0


def _remove_dark(img1, darkfile=None, ihdu=0, verbose=False):
    if darkfile is not None:
        from astropy.io import fits

        with fits.open(darkfile) as hdu:
            dark = hdu[ihdu].data
        if verbose:
            print("Dark cube shape is:", dark.shape)
        master_dark = np.mean(dark, axis=0)
        img1 -= master_dark
    return img1


def clean_data(
    data,
    isz=None,
    r1=None,
    dr=None,
    edge=0,
    bad_map=None,
    add_bad=None,
    apod=True,
    offx=0,
    offy=0,
    sky=True,
    window=None,
    darkfile=None,
    f_kernel=3,
    verbose=False,
    *,
    mask=None,
):
    """Clean an aperture-masking data cube.

    Parameters
    ----------
    data : numpy.ndarray
        Input data cube.
    isz, r1, dr : int or None, default=None
        Crop size and sky-ring parameters.
    edge : int, default=0
        Detector-edge width to remove.
    bad_map : numpy.ndarray or None, default=None
        Bad-pixel map.
    add_bad : list or None, default=None
        Additional bad-pixel coordinates.
    apod : bool, default=True
        Whether to apodize frames.
    offx, offy : int, default=0
        Crop-center offsets.
    sky : bool, default=True
        Whether to apply sky subtraction.
    window : float or None, default=None
        Apodization width.
    darkfile : str or path-like or None, default=None
        Dark-cube filename.
    f_kernel : int or None, default=3
        Median-filter kernel size for centering.
    verbose : bool, default=False
        Whether to print cleaning information.
    mask : numpy.ndarray of bool or None, keyword-only, default=None
        Sky-background mask.

    Returns
    -------
    numpy.ndarray
        Cleaned data cube.
    """
    # Work on a float copy: integer cubes cannot hold NaN (bad pixels) and the
    # edge correction would otherwise modify the input cube in place.
    data = np.array(data, dtype=float)
    n_im = data.shape[0]
    cube_cleaned = []  # np.zeros([n_im, isz, isz])
    l_bad_frame = []

    bad_map, add_bad = _get_3d_bad_pixels(bad_map, add_bad, data)

    for i in track(range(n_im), description="Cleaning"):
        img0 = data[i]
        img0 = _apply_edge_correction(img0, edge=edge)
        if bad_map is not None:
            img1 = fix_bad_pixels(img0, bad_map[i], add_bad=add_bad[i])
        else:
            img1 = img0.copy()

        img1 = _remove_dark(img1, darkfile=darkfile, verbose=verbose)

        if isz is not None:
            # Get expected center for sky correction
            filtmed = f_kernel is not None
            center = find_max(img1, filtmed=filtmed, f=f_kernel)
        else:
            center = None

        if sky and (r1 is not None or mask is not None):
            img_biased = sky_correction(
                img1, r1=r1, dr=dr, verbose=verbose, center=center, mask=mask
            )[0]
        elif sky:
            warnings.warn(
                "sky is set to True, but r1 and mask are set to None. Skipping sky correction",
                RuntimeWarning,
                stacklevel=2,
            )
            img_biased = img1.copy()
        else:
            img_biased = img1.copy()
        img_biased[img_biased < 0] = 0  # Remove negative pixels

        if isz is not None:
            # Get expected center for sky correction
            filtmed = f_kernel is not None
            im_rec_max = crop_max(
                img_biased, isz, offx=offx, offy=offy, filtmed=filtmed, f=f_kernel
            )[0]
        else:
            im_rec_max = img_biased.copy()

        if (
            (im_rec_max.shape[0] != im_rec_max.shape[1])
            or (isz is not None and im_rec_max.shape[0] != isz)
            or (isz is None and im_rec_max.shape[0] != img0.shape[0])
        ):
            l_bad_frame.append(i)
        else:
            if apod and window is not None:
                img = apply_windowing(im_rec_max, window=window)
            elif apod:
                warnings.warn(
                    "apod is set to True, but window is None. Skipping apodisation",
                    RuntimeWarning,
                    stacklevel=2,
                )
                img = im_rec_max.copy()
            else:
                img = im_rec_max.copy()
            cube_cleaned.append(img)
    if verbose:
        print("Bad centering frame number:", l_bad_frame)
    cube_cleaned = np.array(cube_cleaned)
    return cube_cleaned


def select_clean_data(
    filename,
    isz=None,
    r1=None,
    dr=None,
    edge=0,
    clip=True,
    bad_map=None,
    add_bad=None,
    offx=0,
    offy=0,
    clip_fact=0.5,
    apod=True,
    sky=True,
    window=None,
    darkfile=None,
    f_kernel=3,
    verbose=False,
    ihdu=0,
    display=False,
    *,
    remove_bad=True,
    nframe=0,
    mask=None,
    i_wl=None,
):
    """Clean and select frames from a FITS data cube.

    Parameters
    ----------
    filename : str or path-like
        Data-cube filename.
    isz, r1, dr : int or None, default=None
        Crop size and sky-ring parameters.
    edge : int, default=0
        Detector-edge width to remove.
    clip : bool, default=True
        Whether to reject low-flux frames.
    bad_map : numpy.ndarray or None, default=None
        Bad-pixel map.
    add_bad : list or None, default=None
        Additional bad-pixel coordinates.
    offx, offy : int, default=0
        Crop-center offsets.
    clip_fact : float, default=0.5
        Relative sigma threshold for flux selection.
    apod, sky : bool, default=True
        Whether to apodize frames and subtract the sky.
    window : float or None, default=None
        Apodization width.
    darkfile : str or path-like or None, default=None
        Dark-cube filename.
    f_kernel : int or None, default=3
        Median-filter kernel size for centering.
    verbose : bool, default=False
        Whether to print cleaning information.
    ihdu : int, default=0
        FITS HDU containing the data.
    display : bool, default=False
        Whether to show cleaning parameters.
    remove_bad : bool, keyword-only, default=True
        Whether to interpolate bad pixels.
    nframe : int, keyword-only, default=0
        Frame to use for parameter display.
    mask : numpy.ndarray of bool or None, keyword-only, default=None
        Sky-background mask.
    i_wl : int or None, keyword-only, default=None
        IFU spectral-channel index.

    Returns
    -------
    numpy.ndarray or None
        Cleaned and selected data cube, or None when processing cannot
        continue.
    """
    from astropy.io import fits

    with fits.open(filename) as hdu:
        cube = hdu[ihdu].data
        hdr = hdu[0].header

    ins = hdr.get("INSTRUME", None)

    ifu = False
    if ins == "SPHERE":
        seeing_start = float(hdr["HIERARCH ESO TEL AMBI FWHM START"])
        seeing = float(hdr["HIERARCH ESO TEL IA FWHM"])
        seeing_end = float(hdr["HIERARCH ESO TEL AMBI FWHM END"])

        if verbose:
            print("\n----- Seeing conditions -----")
            print(
                f"{seeing_start:2.2f} (start), {seeing_end:2.2f} (end), {seeing:2.2f} (Corrected AirMass)"
            )

        n_axis = len(cube.shape)
        if n_axis == 4:
            ifu = True
            naxis4 = hdr["NAXIS4"]
            if i_wl is None:
                raise ValueError(
                    "Your file seems to be obtained with an IFU instrument: spectral "
                    f"channel index `i_wl` must be specified (nlambda = {naxis4})."
                )
            if i_wl >= naxis4:
                iwl_msg = f"The choosen spectral channel {i_wl} do not exist (i_wl <= {naxis4 - 1})"
                raise ValueError(iwl_msg)

    # Add check to create default add_bad list (not use mutable data)
    if add_bad is None:
        add_bad = []

    if r1 is None and mask is None and sky:
        warnings.warn(
            "The default value of r1 is now None. Either r1 or mask should be set explicitely. This will raise an error in the future.",
            PendingDeprecationWarning,
            stacklevel=2,
        )
        r1 = 100
        if dr is None:
            dr = 10
    elif r1 is not None and dr is None and mask is None and sky:
        warnings.warn(
            "The default value of dr is now None. dr must be set explicitely to be used.",
            PendingDeprecationWarning,
            stacklevel=2,
        )
        dr = 10

    if display:
        show_clean_params(
            filename,
            isz,
            r1,
            dr,
            bad_map=bad_map,
            add_bad=add_bad,
            edge=edge,
            remove_bad=remove_bad,
            nframe=nframe,
            ihdu=ihdu,
            f_kernel=f_kernel,
            offx=offx,
            offy=offy,
            apod=apod,
            window=window,
            ifu=ifu,
            mask=mask,
        )

    if ifu:
        cube = cube[i_wl]

    cube_cleaned = clean_data(
        cube,
        isz=isz,
        r1=r1,
        edge=edge,
        bad_map=bad_map,
        add_bad=add_bad,
        dr=dr,
        sky=sky,
        apod=apod,
        window=window,
        f_kernel=f_kernel,
        offx=offx,
        offy=offy,
        darkfile=darkfile,
        verbose=verbose,
        mask=mask,
    )

    if cube_cleaned is None:
        return None

    cube_final = select_data(
        cube_cleaned,
        clip=clip,
        clip_fact=clip_fact,
        verbose=verbose,
        display=display,
    )

    return cube_final
