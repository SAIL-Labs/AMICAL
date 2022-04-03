"""
Created on Mon Aug 25 13:17:03 2014

@author: anthony
"""
import time
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp

from .cp_tools import cp_loglikelihood
from .cp_tools import cp_loglikelihood_proj
from .cp_tools import cp_model
from .cp_tools import mas2rad
from .cp_tools import project_cps
from .cp_tools import rad2mas


def phase_binary_flux(u, v, wavel, p, return_cvis=False):
    """Calculate the phases observed by an array on a binary star
    ----------------------------------------------------------------
    p: 3-component vector (+2 optional), the binary "parameters":
    - p[0] = sep (mas)
    - p[1] = PA (deg) E of N.
    - p[2] = flux (primary is assumed to be 1)

    optional:
    - p[2:] = contrast ratio for several wavelengths that we want
             to calculate the cps over

    - u,v: baseline coordinates (meters)
    - wavel: wavelength (meters)
    ----------------------------------------------------------------"""
    p = np.array(p)
    # relative locations
    th = (p[1] + 90.0) * np.pi / 180.0
    ddec = mas2rad(p[0] * np.sin(th))
    dra = -mas2rad(p[0] * np.cos(th))

    # decompose into two "luminosities"
    # but first, a little trick so this works whether
    # p is a single value or a list of contrasts
    spec = p[2:]
    if len(spec) == 1:
        spec = spec[0]
    l2 = spec
    l1 = 1 - l2

    # phase-factor
    output_shape = list(u.shape)
    output_shape[-1] = np.size(wavel)
    phi = np.zeros(output_shape, dtype=complex)
    phi.real = np.cos(-2 * np.pi * (u * dra + v * ddec) / wavel)
    phi.imag = np.sin(-2 * np.pi * (u * dra + v * ddec) / wavel)

    cvis = l1 + l2 * phi

    phase = np.angle(cvis, deg=True)
    if return_cvis:
        return cvis
    else:
        return np.mod(phase + 10980.0, 360.0) - 180.0


# =========================================================================
def cp_model_flux(params, u, v, wavels, model="constant"):
    """Function to model closure phases. Takes a parameter list, u,v triangles and range of wavelengths.
    Allows fitting of a model to contrast vs wavelength.
    Models for contrast ratio:
            constant (contrast is constant with wavelength, default)
            linear (params[2,3]=contrast ratios at end wavelengths),
            free (params[2:]=contrast ratios).
            ndof (the wavelength channels are evenly spaced cubic interpolations in params[2:])
            polynomial (of the form Sum[n] params[n+2]*(wavelength*1e6)**n )
    NOTE: This doesn't allow for nonzero size of each component!"""
    nwav = wavels.size
    if model == "constant":
        cons = np.repeat(params[2], nwav)
    elif model == "linear":
        cons = params[2] + (params[3] - params[2]) * (wavels - wavels[0]) / (
            wavels[-1] - wavels[0]
        )
    elif model == "ndof":
        ndof = params[2:].size
        wavs = np.linspace(np.min(wavels), np.max(wavels), ndof)
        f = interp.interp1d(wavs, params[2:], kind="cubic")
        cons = f(wavels)
    elif model == "free":
        # no model, crat vs wav is free to vary.
        cons = params[2:]
    elif model == "polynomial":
        coefficients = params[2:]
        ndof = len(coefficients)
        cons = np.repeat(0.0, nwav)
        xax = (wavels - np.min(wavels)) / (np.max(wavels) - np.min(wavels))
        for order in range(ndof):
            cons += coefficients[order] * xax**order
    else:
        raise NameError("Unknown model input to cp_model")

    # vectorize the arrays to speed up multi-wavelength calculations
    u = u[..., np.newaxis]  # (ncp x n_runs x 3 x 1) or (ncp x 3 x 1)
    v = v[..., np.newaxis]  # (ncp x n_runs x 3 x 1) or (ncp x 3 x 1)
    wavels = wavels[np.newaxis, np.newaxis, :]  # (1 x 1 x 1 x nwav) or (1x1xnwav)
    if u.ndim == 4:
        wavels = wavels[np.newaxis]
    phases = phase_binary_flux(u, v, wavels, params)
    cps = np.sum(phases, axis=-2)

    return cps


# =========================================================================
# =========================================================================


def cp_loglikelihood_proj_flux(
    params, u, v, wavel, proj_t3data, proj_t3err, proj, model="constant"
):
    """Calculate loglikelihood for projected closure phase data.
    Used both in the MultiNest and MCMC Hammer implementations.
    Here proj is the eigenvector array"""

    # hacky way to introduce priors
    #    if (params[2] > 50000) or (params[2] < 0.):
    #        return -np.inf
    if (params[0] > 350.0) or (params[0] < 0.0):
        return -np.inf
    if (params[1] > 360.0) or (params[1] < 0.0):
        return -np.inf

    cps = cp_model_flux(params, u, v, wavel, model=model)

    proj_mod_cps = project_cps(cps, proj)

    chi2 = np.sum(((proj_t3data - proj_mod_cps) / proj_t3err) ** 2)

    loglike = -chi2 / 2
    return loglike


# =========================================================================


def chi2_grid(everything):
    """Function for multiprocessing, does 2d chi2 grid for xy_grid"""
    cpo = everything["cpo"]
    chi2 = np.zeros((len(everything["ys"]), len(everything["cons"])))
    x = everything["x"]
    ys = everything["ys"]
    seps = np.sqrt(x**2 + ys**2)
    pas = np.angle(np.complex(0, 1) * ys + np.complex(1, 0) * x, True) % 360

    projected = everything["projected"]

    for ix in range(ys.size):
        for k, con in enumerate(everything["cons"]):
            params = [seps[ix], pas[ix], con]
            if projected:
                chi2[ix, k] = -2 * cp_loglikelihood_proj_flux(
                    params,
                    cpo.u,
                    cpo.v,
                    cpo.wavel,
                    cpo.proj_t3data,
                    cpo.proj_t3err,
                    cpo.proj,
                )
            else:
                chi2[ix, k] = -2 * cp_loglikelihood(
                    params, cpo.u, cpo.v, cpo.wavel, cpo.t3data, cpo.t3err
                )

    return chi2


# =========================================================================

# =========================================================================


def xy_grid(
    cpo,
    nxy=30,
    ncon=32,
    xymax="Default",
    cmin=10.0,
    cmax=500.0,
    threads=0,
    err_scale=1.0,
    extra_error=0.0,
    fix_crat=False,
    cmap="ds9cool",
    plot_as_mags=False,
    projected=False,
):

    """An attempt to copy Sylvestre's chi2 grid plots, using x and y instead
    of separation and position angle.

    Written by A Cheetham, with some parts stolen from other pysco/pymask routines."""

    # ------------------------
    # first, load your data!
    # ------------------------

    ndata = cpo.ndata

    u, v = cpo.u, cpo.v

    cpo.t3err = np.sqrt(cpo.t3err**2 + extra_error**2)
    cpo.t3err *= err_scale

    wavel = cpo.wavel

    w = np.array(np.sqrt(u**2 + v**2)) / np.median(wavel)

    if xymax == "Default":
        #        xymax = cpt.rad2mas(1./np.min(w/np.max(wavel)))
        xymax = rad2mas(1.0 / np.min(w))

    # ------------------------
    # initialise grid params
    # ------------------------

    xys = np.linspace(-xymax, xymax, nxy)
    #    cons = cmin  + (cmax-cmin)  * np.linspace(0,1,ncon)
    cons = np.linspace(cmin, cmax, ncon)

    if fix_crat != False:
        cons = np.array([fix_crat])
        ncon = 1

    # ------------------------
    # Calculate chi2 at each point
    # ------------------------

    tic = time.time()  # start the clock
    chi2 = np.zeros((nxy, nxy, ncon))
    if threads == 0:
        toc = time.time()
        for ix, x in enumerate(xys):
            everything = {
                "x": x,
                "cons": cons,
                "ys": xys,
                "cpo": cpo,
                "ix": ix,
                "projected": projected,
            }
            chi2[ix, :, :] = chi2_grid(everything)
            if (ix % 50) == 0:
                tc = time.time()
                print("Done " + str(ix) + ". Time taken: " + str(tc - toc) + "seconds")
                toc = tc
    else:
        all_vars = []
        for ix in range(nxy):
            everything = {
                "x": xys[ix],
                "cons": cons,
                "ys": xys,
                "cpo": cpo,
                "ix": ix,
                "projected": projected,
            }
            all_vars.append(everything)
        pool = Pool(processes=threads)
        chi2 = pool.map(chi2_grid, all_vars)
        pool.close()
    tf = time.time()
    if tf - tic > 60:
        print("Total time elapsed: " + str((tf - tic) / 60.0) + "mins")
    elif tf - tic <= 60:
        print("Total time elapsed: " + str(tf - tic) + " seconds")
    chi2 = np.array(chi2)
    best_ix = np.where(chi2 == np.amin(chi2))

    # hack: if the best chi2 is at more than one location, take the first.
    bestx = xys[best_ix[0][0]]
    besty = xys[best_ix[1][0]]
    sep = np.sqrt(bestx**2 + besty**2)
    pa = np.angle(np.complex(bestx, besty), True) % 360
    best_params = [sep, pa, cons[best_ix[2][0]]]
    best_params = np.array(np.array(best_params).ravel())
    print("Separation " + str(best_params[0]) + " mas")
    print("Position angle " + str(best_params[1]) + " deg")
    print("Contrast Ratio " + str(best_params[2]))
    # ---------------------------------------------------------------
    #                        sum over each variable so we can visualise it all
    # ---------------------------------------------------------------
    temp_chi2 = ndata * chi2 / np.amin(chi2)
    like = np.exp(-(temp_chi2 - ndata) / 2)
    x_y = np.sum(like, axis=2)

    # ---------------------------------------------------------------
    #                        contour plot!
    # ---------------------------------------------------------------
    names = ["Chi2", "Likelihood", "Best Contrast Ratio"]
    plots = [np.min(chi2, axis=2), x_y, cons[np.argmin(chi2, axis=2)]]
    for ix, n in enumerate(names):

        plt.figure(n)
        plt.clf()
        # Plot it with RA on the X axis
        plt.imshow(
            plots[ix],
            extent=[np.amin(xys), np.amax(xys), np.amin(xys), np.amax(xys)],
            aspect="auto",
            cmap=cmap,
        )
        plt.colorbar()
        plt.ylabel("Dec (mas)")
        plt.xlabel("RA (mas)")

        plt.plot([0], [0], "wo")
        plt.xlim(xys[-1], xys[0])
        plt.ylim(xys[0], xys[-1])

    # ---------------------------------------------------------------
    #               And the detection limits that come for free!
    # ---------------------------------------------------------------
    chi2_null = np.sum((cpo.t3data / cpo.t3err) ** 2)
    # Define the detec limits to be the contrast at which chi2_binary - chi2_null < 25
    detecs = (chi2 - chi2_null) < 25
    detec_lim = np.zeros((nxy, nxy))
    for x_ix in range(nxy):
        for y_ix in range(nxy):
            detectable_cons = cons[detecs[x_ix, y_ix, :]]
            if len(detectable_cons) == 0:
                detec_lim[x_ix, y_ix] = cons[-1]
            else:
                detec_lim[x_ix, y_ix] = np.min(detectable_cons)

    if plot_as_mags:
        detec_lim_plot = -2.5 * np.log10(detec_lim)
    else:
        detec_lim_plot = detec_lim

    plt.figure(1)
    plt.clf()
    #    plt.imshow(detec_lim,extent=(xys[0],xys[-1],xys[0],xys[-1]),cmap=cmap)
    # Plot it with RA on the X axis
    plt.imshow(detec_lim_plot, extent=(xys[0], xys[-1], xys[0], xys[-1]), cmap=cmap)
    plt.colorbar()
    plt.title("Detection limits")
    plt.xlabel("RA (mas)")
    plt.ylabel("Dec (mas)")
    plt.xlim(xys[-1], xys[0])
    plt.ylim(xys[0], xys[-1])

    # And we should also print whether the likelihood peak is a detection
    #  according to the limits we just calculated
    limit_at_pos = detec_lim[best_ix[0][0], best_ix[1][0]]
    print("Contrast limit at best fit position: " + str(limit_at_pos))
    if limit_at_pos > best_params[2]:
        print("Detection!")
    else:
        print("No significant detection found")

    data = {
        "chi2": chi2,
        "like": like,
        "xys": xys,
        "cons": cons,
        "best_params": best_params,
        "detec_lim": detec_lim,
    }
    return data


# =========================================================================
