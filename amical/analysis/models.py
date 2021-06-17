import numpy as np
from scipy import special

from amical.tools import mas2rad


def shiftFourier(Utable, Vtable, wl, C_in, x0, y0):
    """ Shift the image (apply a phasor in Fourier space."""
    u = Utable / wl
    v = Vtable / wl
    C_out = C_in * np.exp(-2j*np.pi*(u*x0+v*y0))
    return C_out


def visUniformDisk(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of an uniform disk

    Params:
    -------
    diam: {float}
        Diameter of the disk [rad],\n
    x0, y0: {float}
        Shift along x and y position [rad].
    """
    u = Utable / Lambda
    v = Vtable / Lambda

    diam = mas2rad(param["diam"])

    r = np.sqrt(u ** 2 + v ** 2)

    C_centered = 2 * special.j1(np.pi * r * diam) / (np.pi * r * diam)
    C = shiftFourier(Utable, Vtable, Lambda, C_centered, mas2rad(param["x0"]),
                     mas2rad(param["y0"]))
    return C


def visPointSource(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of a point source.

    Params:
    -------
    x0, y0: {float}
        Shift along x and y position [rad].
    """
    C_centered = np.ones(np.size(Utable))
    C = shiftFourier(Utable, Vtable, Lambda, C_centered,
                     param["x0"], param["y0"])
    return C


def visBinary(Utable, Vtable, Lambda, param):
    sep = mas2rad(param["sep"])
    dm = param["dm"]
    theta = np.deg2rad(90-param["theta"])

    if dm < 0:
        return np.array([np.nan]*len(Lambda))
    f1 = 1
    f2 = f1 / 2.5 ** dm
    ftot = f1 + f2

    rel_f1 = f1 / ftot
    rel_f2 = f2 / ftot

    p_s1 = {"x0": 0, "y0": 0}
    p_s2 = {"x0": sep * np.cos(theta), "y0": sep * np.sin(theta)}
    s1 = rel_f1 * visPointSource(Utable, Vtable, Lambda, p_s1)
    s2 = rel_f2 * visPointSource(Utable, Vtable, Lambda, p_s2)
    C_centered = s1 + s2
    return C_centered


def visBinary_res(Utable, Vtable, Lambda, param):
    sep = mas2rad(param["sep"])
    dm = param["dm"]
    theta = np.deg2rad(90-param["theta"])
    diam = param["diam"]

    if dm < 0:
        return np.array([np.nan]*len(Lambda))
    f1 = 1
    f2 = f1 / 2.5 ** dm
    ftot = f1 + f2

    rel_f1 = f1 / ftot
    rel_f2 = f2 / ftot

    if diam == 0:
        p_s1 = {"x0": 0, "y0": 0}
    else:
        p_s1 = {'diam': diam, "x0": 0, "y0": 0}
    p_s2 = {"x0": sep * np.cos(theta), "y0": sep * np.sin(theta)}

    if diam == 0:
        s1 = rel_f1 * visPointSource(Utable, Vtable, Lambda, p_s1)
    else:
        s1 = rel_f1 * visUniformDisk(Utable, Vtable, Lambda, p_s1)
    s2 = rel_f2 * visPointSource(Utable, Vtable, Lambda, p_s2)
    C_centered = s1 + s2
    return C_centered


def visGaussianDisk(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of a gaussian disk

    Params:
    -------
    fwhm: {float}
        fwhm of the disk [rad],\n
    x0, y0: {float}
        Shift along x and y position [rad].
    """
    u = Utable / Lambda
    v = Vtable / Lambda

    fwhm = param["fwhm"]
    x0 = param["x0"]
    y0 = param["y0"]

    r2 = ((np.pi ** 2) * (u ** 2 + v ** 2) * (fwhm ** 2)) / (4.0 * np.log(2.0))
    C_centered = np.exp(-r2)

    # Deplacement du plan image
    C = shiftFourier(Utable, Vtable, Lambda, C_centered, x0, y0)
    return C


def visDebrisDisk(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of an elliptical thick ring.

    Params:
    -------
    majorAxis: {float}
        Major axis of the disk [rad],\n
    minorAxis: {float}
        Minor axis of the disk [rad],\n
    angle: {float}
        Orientation of the disk [rad],\n
    thickness: {float}
        Thickness of the ring [rad],\n
    x0, y0: {float}
        Shift along x and y position [rad].
    """

    majorAxis = mas2rad(param["majorAxis"])*2
    inclination = np.deg2rad(param['incl'])
    posang = np.deg2rad(param["posang"])
    thickness = mas2rad(param["thickness"])
    cr_star = param["cr"]
    x0 = param["x0"]
    y0 = param["y0"]

    minorAxis = majorAxis * np.cos(inclination)

    u = Utable / Lambda
    v = Vtable / Lambda

    r = np.sqrt(
        ((u * np.sin(posang) + v * np.cos(posang)) * majorAxis) ** 2
        + ((u * np.cos(posang) - v * np.sin(posang)) * minorAxis) ** 2
    )

    C_centered = special.j0(np.pi * r)
    C_shifted = shiftFourier(Utable, Vtable, Lambda, C_centered, x0, y0)
    C = C_shifted * visGaussianDisk(Utable, Vtable, Lambda,
                                    {"fwhm": thickness, "x0": 0.0, "y0": 0.0})

    fstar = cr_star
    fdisk = 1
    total_flux = fstar + fdisk

    rel_star = fstar / total_flux
    rel_disk = fdisk / total_flux

    p_s1 = {'x0': x0, 'y0': y0}
    s1 = rel_star * visPointSource(Utable, Vtable, Lambda, p_s1)
    s2 = rel_disk * C
    return s1 + s2


def visClumpDebrisDisk(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of an elliptical thick ring.

    Params:
    -------
    majorAxis: {float}
        Major axis of the disk [rad],\n
    minorAxis: {float}
        Minor axis of the disk [rad],\n
    angle: {float}
        Orientation of the disk [rad],\n
    thickness: {float}
        Thickness of the ring [rad],\n
    x0, y0: {float}
        Shift along x and y position [rad].
    """

    majorAxis = mas2rad(param["majorAxis"])*2
    inclination = np.deg2rad(param['incl'])
    posang = np.deg2rad(param["posang"])
    thickness = mas2rad(param["thickness"])
    cr_star = param["cr"]
    x0 = param["x0"]
    y0 = param["y0"]

    minorAxis = majorAxis * np.cos(inclination)

    #majorAxis = majorAxis_c * np.cos(inclination) - minorAxis_c * np.sin(inclination)
    #minorAxis = -majorAxis_c * np.sin(inclination) + minorAxis_c * np.cos(inclination)

    u = Utable / Lambda
    v = Vtable / Lambda

    r = np.sqrt(
        ((u * np.sin(posang) + v * np.cos(posang)) * majorAxis) ** 2
        + ((u * np.cos(posang) - v * np.sin(posang)) * minorAxis) ** 2
    )

    d_clump = mas2rad(param['d_clump'])
    cr_clump = param['cr_clump']/100.

    x1 = 0
    y1 = majorAxis * np.cos(inclination)
    x_clump = ((x1 * np.cos(posang) - y1 * np.sin(posang))/2.)
    y_clump = ((x1 * np.sin(posang) + y1 * np.cos(posang))/2.)

    p_clump = {"fwhm": d_clump,
               "x0": x_clump,
               "y0": y_clump}

    C_clump = visGaussianDisk(Utable, Vtable, Lambda, p_clump)

    C_centered = special.j0(np.pi * r)
    C_shifted = shiftFourier(Utable, Vtable, Lambda, C_centered, x0, y0)
    C = C_shifted * visGaussianDisk(Utable, Vtable, Lambda,
                                    {"fwhm": thickness, "x0": 0.0, "y0": 0.0})

    fstar = cr_star
    fdisk = 1
    total_flux = fstar + fdisk

    f_clump = cr_clump * total_flux
    f_debrisdisk = (1 - cr_clump) * total_flux

    rel_star = fstar / total_flux
    rel_disk = fdisk / total_flux

    p_s1 = {'x0': x0, 'y0': y0}
    s1 = rel_star * visPointSource(Utable, Vtable, Lambda, p_s1)
    s2 = rel_disk * C
    deb_disk = s1 + s2
    return f_debrisdisk * deb_disk + f_clump * C_clump
