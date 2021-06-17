# -*- coding: utf-8 -*-
"""
@author: Anthony Soulain (University of Sydney)

-------------------------------------------------------------------------
AMICAL: Aperture Masking Interferometry Calibration and Analysis Library
-------------------------------------------------------------------------

Matched filter sub-pipeline method.

All required IDL function translated into python.

--------------------------------------------------------------------
"""

import numpy as np
from munch import munchify as dict2class
from termcolor import cprint


def regress_noc(x, y, weights):
    """ Python version of IDL regress_noc. """
    sx = x.shape
    sy = y.shape
    nterm = sx[0]           # # OF TERMS
    npts = sy[0]            # # OF OBSERVATIONS

    if (len(weights) != sy[0]) or (len(sx) != 2) or (sy[0] != sx[1]):
        cprint('Incompatible arrays to compute slope error.', 'red')

    xwy = np.dot(x, (weights * y))
    wx = np.zeros([npts, nterm])
    for i in range(npts):
        wx[i, :] = x[:, i] * weights[i]
    xwx = np.dot(x, wx)
    cov = np.linalg.inv(xwx)
    coeff = np.dot(cov, xwy)
    yfit = np.dot(x.T, coeff)
    if npts != nterm:
        MSE = np.sum(weights * (yfit - y)**2) / (npts - nterm)

    var_yfit = np.zeros(npts)

    for i in range(npts):
        var_yfit[i] = np.dot(np.dot(x[:, i].T, cov),
                             x[:, i])  # Neter et al pg 233

    dic = {'coeff': coeff,
           'cov': cov,
           'yfit': yfit,
           'MSE': MSE,
           'var_yfit': var_yfit
           }
    return dict2class(dic)


def dist(naxis):
    """Returns a rectangular array in which the value of each element is proportional to its frequency.
    >>> dist(3)
    array([[ 0.        ,  1.        ,  1.        ],
           [ 1.        ,  1.41421356,  1.41421356],
           [ 1.        ,  1.41421356,  1.41421356]])
    >>> dist(4)
    array([[ 0.        ,  1.        ,  2.        ,  1.        ],
           [ 1.        ,  1.41421356,  2.23606798,  1.41421356],
           [ 2.        ,  2.23606798,  2.82842712,  2.23606798],
           [ 1.        ,  1.41421356,  2.23606798,  1.41421356]])
    """
    xx, yy = np.arange(naxis), np.arange(naxis)
    xx2 = (xx-naxis//2)
    yy2 = (naxis//2-yy)

    distance = np.sqrt(xx2**2 + yy2[:, np.newaxis]**2)
    output = np.roll(distance, -1*(naxis//2), axis=(0, 1))
    return output


def array_coords(ind, dim):
    """Transform 1-D coordinates indices (ind) into 2-D coordinates"""
    x, y = np.arange(dim), np.arange(dim)
    X, Y = np.meshgrid(x, y)
    output = [X.ravel()[ind], Y.ravel()[ind]]
    return np.array(output)


def dblarr(dim1, dim2=None):
    """Python version of idl dblarr"""
    if dim2 is None:
        tab = np.zeros(dim1)
    else:
        tab = np.zeros([dim1, dim2])
    return tab
