"""
Created on Tue Mar  8 14:56:35 2016

Tools for analysing and dealing with polarimetry data.
Also able to read the outputs from the IDL pipeline

@author: cheetham
"""
import numpy as np

from .cp_tools import mas2rad


# =========================================================================
def binary_model(u, v, wavel, p):
    """Calculate the complex visibility observed by an array on a binary star
    ----------------------------------------------------------------
    p: 3-component vector (+2 optional), the binary "parameters":
    - p[0] = sep (mas)
    - p[1] = PA (deg) E of N.
    - p[2] = contrast ratio (primary/secondary)

    optional:
    - p[3] = angular size of primary (mas)
    - p[4] = angular size of secondary (mas)

    - u,v: baseline coordinates (meters)
    - wavel: wavelength (meters)
    ----------------------------------------------------------------"""

    p = np.array(p)
    # relative locations
    th = (p[1] + 90.0) * np.pi / 180.0
    ddec = mas2rad(p[0] * np.sin(th))
    dra = -mas2rad(p[0] * np.cos(th))

    # baselines into number of wavelength
    x = np.sqrt(u * u + v * v) / wavel

    # decompose into two "luminosity"
    l2 = 1.0 / (p[2] + 1)
    l1 = 1 - l2

    # phase-factor
    phi = np.zeros(u.size, dtype=complex)
    phi.real = np.cos(-2 * np.pi * (u * dra + v * ddec) / wavel)
    phi.imag = np.sin(-2 * np.pi * (u * dra + v * ddec) / wavel)

    # optional effect of resolved individual sources
    if p.size == 5:
        th1, th2 = mas2rad(p[3]), mas2rad(p[4])
        v1 = 2 * j1(np.pi * th1 * x) / (np.pi * th1 * x)
        v2 = 2 * j1(np.pi * th2 * x) / (np.pi * th2 * x)
    else:
        v1 = np.ones(u.size)
        v2 = np.ones(u.size)

    cvis = l1 * v1 + l2 * v2 * phi
    return cvis


# =========================================================================


def diff_vis_binary(u, v, wavel, p):
    """The expected differential visibility signal assuming a model of the form:
    separation, position angle, contrast.
        This is just a really simple model that puts all of the complicated physics
    into the contrast term.
        Contrast is really the polarization fraction with some factor due to the size
    of the object, but they are completely degenerate so I merged them."""

    # Get the expected signal from binary_model
    binary_cvis = binary_model(u, v, wavel, p)


# =========================================================================
