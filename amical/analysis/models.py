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
