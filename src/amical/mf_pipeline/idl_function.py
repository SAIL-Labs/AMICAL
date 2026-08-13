"""IDL-compatible numerical helpers for the matched-filter pipeline.

These functions reproduce small array and weighted-regression utilities used by
the original IDL AMICAL implementation."""

import sys

import numpy as np
from rich import print as rprint

from amical.externals.munch import munchify as dict2class


def regress_noc(x, y, weights):
    """Perform a weighted linear regression without a constant term.

    Parameters
    ----------
    x : numpy.ndarray of shape (n_terms, n_observations)
        Regression design matrix.
    y : numpy.ndarray of shape (n_observations,)
        Observed values.
    weights : numpy.ndarray of shape (n_observations,)
        Multiplicative weights for each observation.

    Returns
    -------
    object
        Regression results containing coefficients, coefficient covariance, fitted
        values, mean squared error, and fitted-value variances."""

    sx = x.shape
    sy = y.shape
    nterm = sx[0]  # # OF TERMS
    npts = sy[0]  # # OF OBSERVATIONS

    if (len(weights) != sy[0]) or (len(sx) != 2) or (sy[0] != sx[1]):
        rprint("[red]Incompatible arrays to compute slope error.", file=sys.stderr)

    xwy = np.dot(x, (weights * y))
    wx = np.zeros([npts, nterm])
    for i in range(npts):
        wx[i, :] = x[:, i] * weights[i]
    xwx = np.dot(x, wx)
    cov = np.linalg.inv(xwx)
    coeff = np.dot(cov, xwy)
    yfit = np.dot(x.T, coeff)
    if npts != nterm:
        MSE = np.sum(weights * (yfit - y) ** 2) / (npts - nterm)

    var_yfit = np.zeros(npts)

    for i in range(npts):
        var_yfit[i] = np.dot(np.dot(x[:, i].T, cov), x[:, i])  # Neter et al pg 233

    dic = {"coeff": coeff, "cov": cov, "yfit": yfit, "MSE": MSE, "var_yfit": var_yfit}
    return dict2class(dic)


def dist(naxis):
    """Compute a periodic radial-distance map.

    Parameters
    ----------
    naxis : int
        Side length of the square output array in pixels.

    Returns
    -------
    numpy.ndarray of shape (naxis, naxis)
        Euclidean distance from the Fourier origin in pixels, with periodic
        frequency ordering.

    Examples
    --------
    >>> dist(3)
    array([[0.        , 1.        , 1.        ],
           [1.        , 1.41421356, 1.41421356],
           [1.        , 1.41421356, 1.41421356]])"""
    xx, yy = np.arange(naxis), np.arange(naxis)
    xx2 = xx - naxis // 2
    yy2 = naxis // 2 - yy

    distance = np.sqrt(xx2**2 + yy2[:, np.newaxis] ** 2)
    output = np.roll(distance, -1 * (naxis // 2), axis=(0, 1))
    return output


def array_coords(ind, dim):
    """Convert flattened square-array indices to two-dimensional coordinates.

    Parameters
    ----------
    ind : int or array-like of int
        Flattened indices into a ``(dim, dim)`` array.
    dim : int
        Side length of the square array in pixels.

    Returns
    -------
    numpy.ndarray of shape (2, ...)
        x then y coordinates corresponding to ``ind``."""
    x, y = np.arange(dim), np.arange(dim)
    X, Y = np.meshgrid(x, y)
    output = [X.ravel()[ind], Y.ravel()[ind]]
    return np.array(output)


def dblarr(dim1, dim2=None):
    """Create a zero-filled floating-point array.

    Parameters
    ----------
    dim1 : int
        Length of a one-dimensional array or first array dimension.
    dim2 : int, optional
        Second array dimension. If omitted, return a one-dimensional array.

    Returns
    -------
    numpy.ndarray
        Zero-filled array of shape ``(dim1,)`` or ``(dim1, dim2)``."""
    if dim2 is None:
        tab = np.zeros(dim1)
    else:
        tab = np.zeros([dim1, dim2])
    return tab
