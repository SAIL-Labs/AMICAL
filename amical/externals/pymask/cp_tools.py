import multiprocessing
import pickle
import time

import corner
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp
from scipy.optimize import leastsq
from tqdm import tqdm

multiprocessing.set_start_method("fork", force=True)

"""------------------------------------------------------------------------
cp_tools.py - a collection of functions useful for closure phase analysis
in Python. This includes mas2rad, rad2mas and phase_binary from pysco;
it depends on PyMultiNest, MultiNest and emcee
------------------------------------------------------------------------"""


def mas2rad(x):
    """Convenient little function to convert milliarcsec to radians"""
    return x * np.pi / (180 * 3600 * 1000)


# =========================================================================
# =========================================================================


def rad2mas(x):
    """Convenient little function to convert radians to milliarcseconds"""
    return x / np.pi * (180 * 3600 * 1000)


# =========================================================================
# =========================================================================


def phase_binary(u, v, wavel, p, return_cvis=False):
    """Calculate the phases observed by an array on a binary star
    ----------------------------------------------------------------
    p: 3-component vector (+2 optional), the binary "parameters":
    - p[0] = sep (mas)
    - p[1] = PA (deg) E of N.
    - p[2] = contrast ratio (primary/secondary)

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
    l2 = 1.0 / (spec + 1)
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
# =========================================================================


def cp_loglikelihood(params, u, v, wavel, t3data, t3err, model="constant"):
    """Calculate loglikelihood for closure phase data.
    Used both in the MultiNest and MCMC Hammer implementations."""

    # hacky way to introduce priors
    if (params[2] > 5000) or (params[2] < 0.0):
        return -np.inf
    # if (params[0] > 250.) or (params[0] < 0.):
    #    return -np.inf
    if (params[1] > 380.0) or (params[1] < -5.0):
        return -np.inf

    cps = cp_model(params, u, v, wavel, model=model)
    chi2 = np.sum(((t3data - cps) / t3err) ** 2)
    loglike = -chi2 / 2
    return loglike


# =========================================================================
# =========================================================================


def cp_loglikelihood_cov(params, u, v, wavel, t3data, cov_inv, model="constant"):
    """Calculate loglikelihood for closure phase data. Uses the inverse
    covariance matrix rather than the uncertainties
    Used both in the MultiNest and MCMC Hammer implementations."""

    # hacky way to introduce priors
    if (params[2] > 5000) or (params[2] < 0.0):
        return -np.inf
    if (params[0] > 250.0) or (params[0] < 0.0):
        return -np.inf
    if (params[1] > 380.0) or (params[1] < -5):
        return -np.inf

    cps = cp_model(params, u, v, wavel, model=model)
    resids = t3data - cps
    # Loop through wavelengths and calculate the chi2 for each one and add them
    chi2 = 0
    for wav in range(wavel.size):
        # We want obs.T * Cov**-1 * obs
        # But since obs is not a vector, we would need to loop over the second dimension.
        # Instead, here's a trick to do this in a faster way
        temp = resids[:, :, wav].transpose().dot(cov_inv[:, :, wav])
        chi2 += np.sum(temp.transpose() * resids[:, :, wav])
    loglike = -chi2 / 2
    return loglike


# =========================================================================
# =========================================================================


def cp_loglikelihood_proj(
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
    if (params[1] > 380.0) or (params[1] < -5.0):
        return -np.inf

    cps = cp_model(params, u, v, wavel, model=model)

    proj_mod_cps = project_cps(cps, proj)

    chi2 = np.sum(((proj_t3data - proj_mod_cps) / proj_t3err) ** 2)

    loglike = -chi2 / 2
    return loglike


# =========================================================================
# =========================================================================


def cp_loglikelihood_multiple(
    params, u, v, wavel, t3data, t3err, model="constant", ncomp=1
):
    """Calculate loglikelihood for closure phase data and multiple companions.
    Used both in the MultiNest and MCMC Hammer implementations."""
    cps = cp_model(params[0:3], u, v, wavel, model=model)
    for ix in range(1, ncomp):
        # 3 since there are 3 parameters per companion
        cps += cp_model(params[ix * 3 : (ix + 1) * 3], u, v, wavel, model=model)

    chi2 = np.sum(((t3data.ravel() - cps.ravel()) / t3err.ravel()) ** 2)
    loglike = -chi2 / 2
    return loglike


# =========================================================================
# =========================================================================


def cp_model(params, u, v, wavels, model="constant"):
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
    model_params = np.zeros(nwav + 2)
    model_params[0:2] = params[0:2]
    if model == "constant":
        cons = np.repeat(params[2], nwav)
    elif model == "linear":
        cons = params[2] + (params[3] - params[2]) * (wavels - wavels[0]) / (
            wavels[-1] - wavels[0]
        )
    elif model == "ndof":
        ndof = params[2:].size
        wavs = np.linspace(np.min(wavels), np.max(wavels), ndof)
        f = interp.interp1d(wavs, params[2:], kind="linear")
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
            cons += coefficients[order] * xax ** order
    else:
        raise NameError("Unknown model input to cp_model")
    model_params[2:] = cons
    # vectorize the arrays to speed up multi-wavelength calculations
    u = u[..., np.newaxis]  # (ncp x n_runs x 3 x 1) or (ncp x 3 x 1)
    v = v[..., np.newaxis]  # (ncp x n_runs x 3 x 1) or (ncp x 3 x 1)
    # (1 x 1 x 1 x nwav) or (1x1xnwav)
    wavels = wavels[np.newaxis, np.newaxis, :]
    if u.ndim == 4:
        wavels = wavels[np.newaxis]
    phases = phase_binary(u, v, wavels, model_params)
    cps = np.sum(phases, axis=-2)

    return cps


# =========================================================================
# =========================================================================


def project_cps(cps, proj):
    """Short wrapper program to do the projection of a set of closure phases
    onto another basis. proj is the projection matrix, usually the eigenvectors
    for projecting onto a statistically independent basis set."""
    proj_cps = np.zeros((proj.shape[2], cps.shape[1], cps.shape[2]))
    for wav in range(cps.shape[2]):
        proj_cps[:, :, wav] = np.dot(proj[wav].transpose(), cps[:, :, wav])

    return proj_cps


# =========================================================================
# =========================================================================


def hammer(
    cpo,
    ivar=[52.0, 192.0, 1.53],
    ndim="Default",
    nwalcps=50,
    plot=False,
    projected=False,
    niters=1000,
    threads=1,
    model="constant",
    sep_prior=None,
    pa_prior=None,
    crat_prior=None,
    err_scale=1.0,
    extra_error=0.0,
    use_cov=False,
    burn_in=0,
    verbose=False,
):
    import emcee

    """Default implementation of emcee, the MCMC Hammer, for closure phase
    fitting. Requires a closure phase object cpo, and is best called with
    ivar chosen to be near the peak - it can fail to converge otherwise.
    Also allows fitting of a contrast vs wavlength model. See cp_model for details!
    Prior ranges introduce a flat (tophat) prior between the two values specified
    burn_in = the number of iterations to discard due to burn-in"""
    if ndim == "Default":
        ndim = len(ivar)

    ivar = np.array(ivar)  # initial parameters for model-fit

    # Starting parameters for the walkers
    p0 = []
    scatter = np.zeros(ndim) + 0.01
    scatter[0] = 0.05
    scatter[1] = 0.05
    for walker_ix in range(nwalcps):
        p0.append(ivar + ivar * scatter * np.random.rand(ndim))
    #    p0 = [ivar + 0.1*ivar*np.random.rand(ndim) for i in range(nwalcps)] # initialise walcps in a ball
    #    p0 = [ivar + 0.75*ivar*np.random.rand(ndim) for i in range(nwalcps)] # initialise walcps in a ball
    # print('\n -- Running emcee --')

    t3err = np.sqrt(cpo.t3err ** 2 + extra_error ** 2)
    t3err *= err_scale

    t0 = time.time()
    if projected is False:
        sampler = emcee.EnsembleSampler(
            nwalcps,
            ndim,
            cp_loglikelihood,
            args=[cpo.u, cpo.v, cpo.wavel, cpo.t3data, t3err, model],
            threads=threads,
        )
    elif use_cov:
        sampler = emcee.EnsembleSampler(
            nwalcps,
            ndim,
            cp_loglikelihood_cov,
            args=[cpo.u, cpo.v, cpo.wavel, cpo.t3data, cpo.cov_inv, model],
            threads=threads,
        )
    else:
        proj_t3err = np.sqrt(cpo.proj_t3err ** 2 + extra_error ** 2)
        proj_t3err *= err_scale
        sampler = emcee.EnsembleSampler(
            nwalcps,
            ndim,
            cp_loglikelihood_proj,
            args=[
                cpo.u,
                cpo.v,
                cpo.wavel,
                cpo.proj_t3data,
                proj_t3err,
                cpo.proj,
                model,
            ],
            threads=threads,
        )

    sampler.run_mcmc(p0, niters, progress=True)
    tf = time.time()

    if verbose:
        print("Time elapsed =", tf - t0, "s")

    chain = sampler.flatchain

    # Remove the burn in
    chain = chain[burn_in:]

    seps = chain[:, 0]
    ths = chain[:, 1]
    cs = chain[:, 2:][:, 0]

    # Now introduce the prior, by ignoring values outside of the range
    if sep_prior is not None:
        wh = (seps >= sep_prior[0]) & (seps <= sep_prior[1])
        seps = seps[wh]
        ths = ths[wh]
        cs = cs[wh]
    if pa_prior is not None:
        wh = (ths >= pa_prior[0]) & (ths <= pa_prior[1])
        seps = seps[wh]
        ths = ths[wh]
        cs = cs[wh]
    if crat_prior is not None:
        wh = (cs >= crat_prior[0]) & (cs <= crat_prior[1])
        seps = seps[wh]
        ths = ths[wh]
        cs = cs[wh]
    # if crat_prior is not None:
    #     #     for ix in range(ndim-2):
    #     #         c = cs[:, ix]
    #     wh = (cs[:, 0] >= crat_prior[0]) & (cs[:, 0] <= crat_prior[1])
    #     seps = seps[wh]
    #     ths = ths[wh]
    #     cs = cs[wh, :]

    # check that there are still some points left!
    if seps.size > 0:
        ngood = len(seps)
        chain = np.zeros((ngood, ndim))
        chain[:, 0] = seps
        chain[:, 1] = ths
        chain[:, 2] = cs
    else:
        print("WARNING: Your priors eliminated all points!")

    meansep = np.mean(seps)
    dsep = np.std(seps)

    meanth = np.mean(ths)
    dth = np.std(ths)

    meanc = np.mean(cs, axis=0)
    dc = np.std(cs, axis=0)

    if verbose:
        print("Separation", meansep, "pm", dsep, "mas")
        print("Position angle", meanth, "pm", dth, "deg")
        print("Contrast", meanc[0], "pm", dc[0])

    if model == "linear":
        print("Contrast2", meanc[1], "pm", dc[1])
        extra_pars = ["Contrast "]
        extra_dims = ["Ratio"]
    elif model == "free":
        extra_pars = np.repeat("Contrast", ndim - 2)
        extra_dims = np.repeat("Ratio", ndim - 2)
    else:
        extra_pars = "None"
        extra_dims = "None"

        paramdims = ["(mas)", "(deg)", "Ratio"]
        for ix, par in enumerate(extra_pars):
            # paramnames.append(par)
            paramdims.append(extra_dims[ix])

        res_p, err_p, err_m = [], [], []
        for i in range(ndim):
            mcmc = np.percentile(chain[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            res_p.append(mcmc[1])
            err_m.append(q[0])
            err_p.append(q[1])

    if plot:
        # plt.figure(figsize=(5, 7))
        # plt.subplot(3, 1, 1)
        # plt.plot(sampler.chain[:, :, 0].T, color='grey', alpha=.5)
        # # plt.plot(len(chain_sep), sep, marker='*', color='#0085ca', zorder=1e3)
        # plt.ylabel('Separation [mas]')
        # plt.subplot(3, 1, 2)
        # plt.plot(sampler.chain[:, :, 1].T, color='grey', alpha=.5)
        # # plt.plot(len(chain_sep), pa, marker='*', color='#0085ca', zorder=1e3)
        # plt.ylabel('PA [deg]')
        # plt.subplot(3, 1, 3)
        # plt.plot(sampler.chain[:, :, 2].T, color='grey', alpha=.2)
        # # plt.plot(len(chain_sep), cr, marker='*', color='#0085ca', zorder=1e3)
        # plt.xlabel('Step')
        # plt.ylabel('CR')
        # plt.tight_layout()
        # plt.show(block=False)

        fig = corner.corner(
            chain,
            labels=["SEP [mas]", "PA [deg]", "CONTRAST"],
            quantiles=(0.16, 0.84),
            show_titles=True,
            title_kwargs={"fontsize": 10},
            color="#096899",
        )
        axes = np.array(fig.axes).reshape((ndim, ndim))
        # Loop over the diagonal
        for i in range(ndim):
            ax = axes[i, i]
            # ax.axvline(value1[i], color="g")
            ax.axvline(res_p[i], color="#ce0056e6", lw=1)

        # Loop over the histograms
        for yi in range(ndim):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(res_p[xi], color="#ce0056e6", lw=1)
                ax.axhline(res_p[yi], color="#ce0056e6", lw=1)
                ax.plot(res_p[xi], res_p[yi], "#ce0056e6", lw=1)
        plt.tight_layout()
        plt.show(block=False)

    data = {
        "sep": meansep,
        "delsep": dsep,
        "pa": meanth,
        "delpa": dth,
        "con": meanc,
        "delcon": dc,
        "chain": sampler.chain,
    }
    data2 = {
        "sep": res_p[0],
        "delsepm": err_m[0],
        "delsepp": err_p[0],
        "pa": res_p[1],
        "delpam": err_m[1],
        "delpap": err_p[1],
        "cr": res_p[2],
        "delcrm": err_m[2],
        "delcrp": err_p[2],
        "chain": sampler.chain,
    }

    return data, data2


def detec_sim_loopfit(everything):
    """Function for multiprocessing in detec_limits. Takes a
    single separation and full angle, contrast lists.
    For each sep,pa,contrast, it calculates 10,000 simulations of that binary
     (adding noise to each). A detection is defined as having chi2_bin - chi2_null <0
     It then counts the number of detections over all separations and"""
    detec_count = np.zeros((everything["nth"], everything["ncon"]))
    ndim = len(everything["error"].shape)

    # This can be done once since it doesn't change with the binary params
    # error should be ncp x nwav, rands should be ncp x nwav x n
    err = everything["error"]
    resids = err[..., np.newaxis] * everything["rands"]

    for j, th in enumerate(everything["ths"]):
        for k, con in enumerate(everything["cons"]):

            bin_cp = cp_model(
                [everything["sep"], th, con],
                everything["u"],
                everything["v"],
                everything["wavel"],
            )
            rnd_cp = bin_cp[..., np.newaxis] + resids

            # We want the difference in chi2 between the binary and null hypothesis.
            #  i.e. using rnd_cp for the single star and rnd_cp-bin_cp for the binary
            #  but this simplifies to the following equation
            chi2_diff = np.sum(
                (resids ** 2 - rnd_cp ** 2) / everything["error"][..., np.newaxis] ** 2,
                axis=tuple(range(ndim)),
            )

            #            chi2_sngl = np.sum(np.sum((((rnd_cp)/ everything['error'][:,:,np.newaxis])**2),axis=0),axis=0)
            #            chi2_binr = np.sum(np.sum((((rnd_cp-bin_cp[:,:,np.newaxis]) / everything['error'][:,:,np.newaxis])**2),axis=0),axis=0)
            #            chi2_diff=chi2_binr-chi2_sngl
            # this counts the number of detections
            detec_count[j, k] = (chi2_diff < (-0.0)).sum()

    # print('done one separation')
    # print(err.shape,bin_cp.shape,rnd_cp.shape,everything['rands'].shape)
    return detec_count


# =========================================================================
# =========================================================================


def detec_sim_loopfit_cov(everything):
    """Function for multiprocessing in detec_limits. Takes a
    single separation and full angle, contrast lists.
    For each sep,pa,contrast, it calculates 10,000 simulations of that binary
     (adding noise to each). A detection is defined as having chi2_bin - chi2_null <0
     It then counts the number of detections over all separations and"""
    detec_count = np.zeros((everything["nth"], everything["ncon"]))
    for j, th in enumerate(everything["ths"]):
        for k, con in enumerate(everything["cons"]):
            bin_cp = cp_model(
                [everything["sep"], th, con],
                everything["u"],
                everything["v"],
                everything["wavel"],
            )
            errs = everything["rands"]
            # binary cp model
            # ----------------------
            rnd_cp = bin_cp[:, :, np.newaxis] + errs
            chi2_sngl = rnd_cp.transpose().dot(everything["cov_inv"]).dot(rnd_cp)
            chi2_binr = errs.transpose().dot(everything["cov_inv"]).dot(errs)
            chi2_diff = chi2_binr - chi2_sngl
            # this counts the number of detections
            detec_count[j, k] = (chi2_diff < (-0.0)).sum()

    # print('done one separation')
    return detec_count


# =========================================================================
# =========================================================================


def detec_sim_loopfit_proj(everything):
    """Function for multiprocessing in detec_limits. Takes a
    single separation and full angle, contrast lists. Made for projected data"""
    detec_count = np.zeros((everything["nth"], everything["ncon"]))
    proj = everything["proj"]
    ndim = len(everything["error"].shape)

    # This can be done once since it doesn't change with the binary params
    # Note that error and rands are already in the projected basis
    proj_resids = everything["error"][..., np.newaxis] * everything["rands"]

    for j, th in enumerate(everything["ths"]):
        for k, con in enumerate(everything["cons"]):

            bin_cp = cp_model(
                [everything["sep"], th, con],
                everything["u"],
                everything["v"],
                everything["wavel"],
            )

            # Project the data:
            proj_bin_cp = project_cps(bin_cp, proj)
            proj_rnd_cp = proj_bin_cp[..., np.newaxis] + proj_resids

            # We want the difference in chi2 between the binary and null hypothesis.
            #  i.e. using rnd_cp for the single star and rnd_cp-bin_cp for the binary
            #  but this simplifies to the following equation
            chi2_diff = np.sum(
                (proj_resids ** 2 - proj_rnd_cp ** 2)
                / everything["error"][..., np.newaxis] ** 2,
                axis=tuple(range(ndim)),
            )

            #            chi2_sngl = np.sum(np.sum((((rnd_cp)/ everything['error'][:,:,np.newaxis])**2),axis=0),axis=0)
            #            chi2_binr = np.sum(np.sum((((rnd_cp-bin_cp[:,:,np.newaxis]) / everything['error'][:,:,np.newaxis])**2),axis=0),axis=0)
            #            chi2_diff=chi2_binr-chi2_sngl
            # this counts the number of detections
            detec_count[j, k] = (chi2_diff < (-0.0)).sum()

    return detec_count


# =========================================================================
# =========================================================================


def detec_limits(
    cpo,
    nsim=2000,
    nsep=32,
    nth=20,
    ncon=32,
    smin="Default",
    smax="Default",
    cmin=1.0001,
    cmax=500.0,
    extra_error=0,
    threads=0,
    save=False,
    projected=False,
    use_cov=False,
    icpo=False,
    err_scale=1.0,
    no_plot=False,
    linear_in_mags=False,
):
    """uses a Monte Carlo simulation to establish contrast-separation
    detection limits given an array of standard deviations per closure phase.

    Because different separation-contrast grid points are entirely
    separate, this task is embarrassingly parallel. If you want to
    speed up the calculation, use multiprocessing with a threads
    argument equal to the number of available cores.

    Make nseps a multiple of threads! This uses the cores most efficiently.

    Hyperthreading (2x processes per core) in my experience gets a ~20%
    improvement in speed.

    Written by F. Martinache and B. Pope.
    ACC added option for projected data and a few tweaks.

    Use_cov option allows the random clps to be generated using the sample covariance matrix.

    Note also that the calculation of the model closure phases could be done outside
    the big loop, which would be efficient on CPU but not RAM. However, ACC
    tried adding this and ran out of RAM (8GB) on GPI data (8880 clps), so removed it."""

    # Note that the accuracy of these sims are limited by the number of fake clps sets you take.
    # e.g. if you only have 10 sims you can't get accuracy better than 10%
    # (and even then, you will need several times more than this to get a robust 10% limit).
    print("Detection limit resolution:", 100.0 / (nsim * nth), "%")
    if 100.0 / (nsim * nth) > 0.01:
        print(
            "It is recommended that you increase nsim if you want robust 99.9% detection limits."
        )

    # ------------------------
    # first, load your data!
    # ------------------------
    if projected is True:
        proj = cpo.proj
        error = np.sqrt(cpo.proj_t3err ** 2 + extra_error ** 2) * err_scale
        n_clps = cpo.proj.shape[-1]
        n_runs = cpo.n_runs
    elif icpo is True:
        proj = []
        cpo.t3err = np.sqrt(cpo.t3err ** 2 + extra_error ** 2)
        error = cpo.t3err * err_scale
        n_clps = cpo.n_clps
        n_runs = cpo.n_runs
    else:
        proj = []
        cpo.t3err = np.sqrt(cpo.t3err ** 2 + extra_error ** 2)
        error = cpo.t3err * err_scale
        n_clps = cpo.ndata
        n_runs = 1

    if use_cov:
        cov_inv = cpo.cov_inv
    else:
        cov_inv = []

    # nwav = cpo.wavel.size
    # ndata = cpo.u.shape[0]
    # u=np.repeat(np.resize(cpo.u,[ndata,3,1]),nwav,2).ravel()
    # v=np.repeat(np.resize(cpo.v,[ndata,3,1]),nwav,2).ravel()
    # wavel=np.repeat(np.repeat(np.resize(cpo.wavel,[1,1,nwav]),ndata,0),3,1).ravel()
    wavel = cpo.wavel
    u = cpo.u
    v = cpo.v

    w = np.array(np.sqrt(u ** 2 + v ** 2)) / np.median(wavel)

    if smin == "Default":
        smin = rad2mas(1.0 / 4 / np.max(w))

    if smax == "Default":
        smax = rad2mas(1.0 / np.min(w))

    # ------------------------
    # initialise Monte Carlo
    # ------------------------

    seps = smin + (smax - smin) * np.linspace(0, 1, nsep)
    ths = 0.0 + 360.0 * np.linspace(0, 1, nth)
    cons = cmin + (cmax - cmin) * np.linspace(0, 1, ncon)
    if linear_in_mags:
        cons = 10 ** (
            np.linspace(2.5 * np.log10(cmin), 2.5 * np.log10(cmax), ncon) / 2.5
        )

    # Generate the random numbers first to save time
    rands_shape = list(error.shape)
    rands_shape.append(nsim)
    if projected:
        # The random numbers should be in the projected basis
        rands = np.random.normal(size=rands_shape)

    elif use_cov:
        # use full multivariate gaussian with sample covariance to get rands. Dont use with projected!
        rands = np.zeros((n_clps * n_runs, nsim))
        for run in range(n_runs):
            rands[n_clps * (run) : n_clps * (run + 1), :] = np.transpose(
                np.random.multivariate_normal(
                    np.zeros(n_clps), np.atleast_3d(cpo.sample_cov)[:, :, run], (nsim)
                )
            )
    else:
        rands = np.random.normal(size=rands_shape)
    print(rands.shape, error.shape)
    all_vars = []
    print("Setting up the big dictionary to store variables for loop")
    for ix in range(nsep):
        everything = {
            "sep": seps[ix],
            "cons": cons,
            "ths": ths,
            "ix": ix,
            "nsep": nsep,
            "ncon": ncon,
            "nth": nth,
            "u": cpo.u,
            "v": cpo.v,
            "nsim": nsim,
            "rands": rands,
            "error": error,
            "wavel": cpo.wavel,
            "proj": proj,
            "n_clps": n_clps,
            "n_runs": n_runs,
            "cov_inv": cov_inv,
        }
        all_vars.append(everything)

    print("Starting big loop over separations")
    # ------------------------
    # Run Monte Carlo
    # ------------------------
    tic = time.time()  # start the clock'
    if threads == 0:
        ndetec = np.zeros((nsep, nth, ncon))
        for ix, sep in enumerate(seps):
            if projected is True:
                ndetec[ix, :, :] = detec_sim_loopfit_proj(all_vars[ix])
            elif use_cov:
                ndetec[ix, :, :] = detec_sim_loopfit_cov(all_vars[ix])
            else:
                ndetec[ix, :, :] = detec_sim_loopfit(all_vars[ix])
            toc = time.time()
            if ix != 10:
                remaining = (toc - tic) * (nsep - ix) / float(ix + 1)
                if remaining > 60:
                    print("Estimated time remaining: %.2f mins" % (remaining / 60.0))
                else:
                    print("Estimated time remaining: %.2f seconds" % (remaining))
    else:
        pool = multiprocessing.Pool(processes=threads)
        ndetec = np.zeros((nsep, nth, ncon))
        if projected:
            ndetec = pool.map(detec_sim_loopfit_proj, all_vars)
        elif use_cov:
            ndetec = pool.map(detec_sim_loopfit_cov, all_vars)
        else:
            ndetec = pool.map(detec_sim_loopfit, all_vars)
        ndetec = np.array(ndetec)

        # and clean up
        pool.close()

    tf = time.time()
    if tf - tic > 60:
        print("Total time elapsed:", (tf - tic) / 60.0, "mins")
    elif tf - tic <= 60:
        print("Total time elapsed:", tf - tic, "seconds")
    # nc, ns = int(ncon), int(nsep)

    # for each contrast and sep, how many detections?
    nd = ndetec
    ndetec = ndetec.sum(axis=1)
    nbtot = nsim * nth
    ndetec /= float(nbtot)
    # and as usual, ACC got the dimensions wrong (still in IDL mode)
    ndetec = np.transpose(ndetec)

    # ---------------------------------------------------------------
    #                        contour plot!
    # ---------------------------------------------------------------
    levels = [0.5, 0.9, 0.99, 0.999]
    mycols = ("k", "k", "k", "k")
    if not no_plot:
        plt.figure(0)
        plt.clf()
        contours = plt.contour(
            ndetec, levels, colors=mycols, extent=[smin, smax, cmin, cmax]
        )
        plt.clabel(contours)
        plt.contourf(seps, cons, ndetec, levels, cmap=plt.cm.bone)
        plt.colorbar()
        plt.xlabel("Separation (mas)")
        plt.ylabel("Contrast Ratio")
        plt.title("Contrast Detection Limits")
        plt.draw()
        plt.show()

    data = {
        "levels": levels,
        "seps": seps,
        "angles": ths,
        "cons": cons,
        "name": cpo.name,
        "ndetec": nd,
        "nsim": nsim,
        "limits": ndetec,
    }

    if save:
        if type(save) is str:
            save_file = save
        else:
            save_file = "limit_lowc" + cpo.name + ".pick"
        print("Saving contrast limits as: " + save_file)

        with open(save_file, "w") as myf:
            pickle.dump(data, myf)

    return data


# =========================================================================
# =========================================================================


def binary_fit(cpo, p0):
    """Performs a best binary fit search for the dataset.
    -------------------------------------------------------------
    p0 is the initial guess for the parameters 3 parameter vector
    typical example would be : [100.0, 0.0, 5.0].
    returns the full solution of the least square fit:
    - soluce[0] : best-fit parameters
    - soluce[1] : covariance matrix
    -------------------------------------------------------------"""

    if np.all(cpo.t3err == 0.0):
        print("Closure phase object instance is not calibrated.\n")
        soluce = leastsq(cpo.bin_fit_residuals, p0, args=(cpo), full_output=1)
    else:

        def lmcpmodel(params, cpo=cpo):
            model = cp_model(params, cpo.u, cpo.v, cpo.wavel)
            return ((model - cpo.t3data) / cpo.t3err).ravel()

        soluce = leastsq(lmcpmodel, p0, args=cpo, full_output=1)
    cpo.covar = soluce[1]
    # to get consistent position angle measurements
    soluce[0][1] = np.mod(soluce[0][1], 360.0)
    return soluce[0:2]  # only return the best params and covariance matrix


# =========================================================================
# =========================================================================


def bin_fit_residuals(params, cpo):
    """Function for binary_fit without errorbars"""
    test = cp_model(params, cpo.u, cpo.v, cpo.wavel)
    err = cpo.t3data - test
    return err


# =========================================================================
# =========================================================================


def brute_force_chi2_grid(everything):
    """Function for multiprocessing, fills in part of the big 3d chi2 grid in a
    way that minimises repeated calculations (i.e. each model only calculated once)."""
    sim_cps = everything["sim_cps"]
    sep = everything["sep"]
    nth = everything["nth"]
    ncon = everything["ncon"]
    nsim = everything["nsim"]
    chi2 = np.zeros((nth, ncon, nsim))
    for j, th in enumerate(everything["ths"]):
        for k, con in enumerate(everything["cons"]):
            mod_cps = cp_model(
                [sep, th, con], everything["u"], everything["v"], everything["wavel"]
            )
            for i in range(nsim):
                chi2[j, k, i] = np.sum(
                    ((sim_cps[:, :, i] - mod_cps) / everything["error"]) ** 2
                )
    return chi2


def lmfit(everything):
    """Unpacks a cpo and calls the proper binary L-M fitting function"""
    this_cpo = everything["sim_cpo"]
    best_chi2_params = everything["best_chi2_params"]
    [new_params, cov] = binary_fit(this_cpo, best_chi2_params)
    return new_params


# =========================================================================
# =========================================================================


def chi2_grid(everything):
    """Function for multiprocessing, does 2d chi2 grid for coarse_grid"""
    cpo = everything["cpo"]
    data_cp = cpo.t3data
    chi2 = np.zeros((len(everything["ths"]), len(everything["cons"])))
    sep = everything["sep"]
    for j, th in enumerate(everything["ths"]):
        for k, con in enumerate(everything["cons"]):
            mod_cps = cp_model([sep, th, con], cpo.u, cpo.v, cpo.wavel)
            chi2[j, k] = np.sum(
                ((data_cp.ravel() - mod_cps.ravel()) / cpo.t3err.ravel()) ** 2
            )
    return chi2


# =========================================================================
# =========================================================================


def chi2_grid_cov(everything):
    """Function for multiprocessing, does 2d chi2 grid for coarse_grid, using
    the covariance matrix instead of uncertainties"""
    cpo = everything["cpo"]
    data_cp = cpo.t3data
    chi2 = np.zeros((len(everything["ths"]), len(everything["cons"])))
    sep = everything["sep"]
    for j, th in enumerate(everything["ths"]):
        for k, con in enumerate(everything["cons"]):
            mod_cps = cp_model([sep, th, con], cpo.u, cpo.v, cpo.wavel)
            resids = data_cp.ravel() - mod_cps.ravel()

            chi2[j, k] = resids.transpose().dot(cpo.cov_inv).dot(resids)
    return chi2


# =========================================================================
# =========================================================================


def chi2_grid_proj(everything):
    """Function for multiprocessing, does 2d chi2 grid for coarse_grid, with projected data"""
    cpo = everything["cpo"]
    chi2 = np.zeros((len(everything["ths"]), len(everything["cons"])))
    sep = everything["sep"]
    for j, th in enumerate(everything["ths"]):
        for k, con in enumerate(everything["cons"]):
            mod_cps = cp_model([sep, th, con], cpo.u, cpo.v, cpo.wavel)
            proj_mod_cps = project_cps(mod_cps, cpo.proj)
            chi2[j, k] = np.sum(
                ((cpo.proj_t3data - proj_mod_cps) / cpo.proj_t3err) ** 2
            )
    return chi2


def coarse_grid(
    cpo,
    nsep=32,
    nth=20,
    ncon=32,
    smin="Default",
    smax="Default",
    cmin=10.0,
    cmax=500.0,
    threads=0,
    projected=False,
    err_scale=1.0,
    extra_error=0.0,
    thmin=0.0,
    thmax=360.0,
    use_cov=False,
    plot=True,
    verbose=True,
):
    """Does a coarse grid search for the best fit parameters. This is
    helpful for finding a good initial point for hammer or multinest.

    Because different separation-contrast grid points are entirely
    separate, this task is embarrassingly parallel. If you want to
    speed up the calculation, use multiprocessing with a threads
    argument equal to the number of available cores.

    Make nseps a multiple of threads! This uses the cores most efficiently.

    Hyperthreading (2x processes per core) in my experience gets a ~20%
    improvement in speed.

    Written by A Cheetham, with some parts stolen from other pysco/pymask routines.

    use_cov : use inverse covariance matrix stored in cpo.cov_inv instead of uncertainties

    """

    # ------------------------
    # first, load your data!
    # ------------------------
    try:
        if projected is True:
            ndata = cpo.n_runs * cpo.n_good
        else:
            ndata = cpo.ndata
    except AttributeError:
        return None

    u, v = cpo.u, cpo.v

    cpo.t3err = np.sqrt(cpo.t3err ** 2 + extra_error ** 2)
    cpo.t3err *= err_scale

    wavel = cpo.wavel

    w = np.array(np.sqrt(u ** 2 + v ** 2))

    if smin == "Default":
        smin = rad2mas(1.0 / 4 / np.max(w / np.min(wavel)))

    if smax == "Default":
        smax = rad2mas(1.0 / np.min(w / np.max(wavel)))

    # ------------------------
    # initialise grid params
    # ------------------------

    seps = smin + (smax - smin) * np.linspace(0, 1, nsep)
    ths = np.linspace(thmin, thmax, num=nth)
    cons = cmin + (cmax - cmin) * np.linspace(0, 1, ncon)

    # ------------------------
    # Calculate chi2 at each point
    # ------------------------
    tic = time.time()  # start the clock
    chi2 = np.zeros((nsep, nth, ncon))

    if threads == 0:
        toc = time.time()
        for ix, sep in enumerate(seps):
            everything = {
                "sep": seps[ix],
                "cons": cons,
                "ths": ths,
                "cpo": cpo,
                "ix": ix,
            }
            if projected:
                chi2[ix, :, :] = chi2_grid_proj(everything)
            elif use_cov:
                chi2[ix, :, :] = chi2_grid_cov(everything)
            else:
                chi2[ix, :, :] = chi2_grid(everything)
            if (ix % 50) == 0:
                tc = time.time()
                if verbose:
                    print("Done", ix, ". Time taken:", (tc - toc), "seconds")
                toc = tc
    else:
        all_vars = []
        for ix in range(nsep):
            everything = {
                "sep": seps[ix],
                "cons": cons,
                "ths": ths,
                "cpo": cpo,
                "ix": ix,
            }
            all_vars.append(everything)
        pool = multiprocessing.Pool(processes=threads)
        if projected:
            chi2 = pool.map(chi2_grid_proj, all_vars)
        elif use_cov:
            chi2 = pool.map(chi2_grid_cov, all_vars)
        else:
            chi2 = pool.map(chi2_grid, all_vars)
        pool.close()
    tf = time.time()
    if tf - tic > 60:
        if verbose:
            print("Total time elapsed:", (tf - tic) / 60.0, "mins")
    elif tf - tic <= 60:
        if verbose:
            print("Total time elapsed:", tf - tic, "seconds")
    chi2 = np.array(chi2)
    best_ix = np.where(chi2 == np.amin(chi2))
    # If the best chi2 is at more than one location, take the first.
    best_params = [seps[best_ix[0][0]], ths[best_ix[1][0]], cons[best_ix[2][0]]]
    best_params = np.array(np.array(best_params).ravel())
    print("\nMaximum likelihood estimation (χ2=%2.1f):" % (np.amin(chi2) / (ndata)))
    print("-------------------------------")
    print("Separation = %2.2f mas" % best_params[0])
    print("PA = %2.2f deg" % best_params[1])
    print(
        "Contrast Ratio = %2.1f (%2.1f mag)\n"
        % (best_params[2], 2.5 * np.log10(best_params[2]))
    )
    # ---------------------------------------------------------------
    #   sum over each variable so we can visualise it all
    # ---------------------------------------------------------------
    temp_chi2 = ndata * chi2 / np.amin(chi2)
    like = np.exp(-(temp_chi2 - ndata) / 2)
    sep_pa = np.sum(like, axis=2)
    sep_crat = np.sum(like, axis=1)
    pa_crat = np.sum(like, axis=0)

    # ---------------------------------------------------------------
    #                        contour plot!
    # ---------------------------------------------------------------
    if plot:
        vmax = np.max([sep_pa.max(), sep_crat.max(), pa_crat.max()])
        plt.figure(figsize=(6, 5))
        plt.subplot(2, 2, 1)
        plt.imshow(
            sep_pa,
            extent=[np.amin(ths), np.amax(ths), np.amin(seps), np.amax(seps)],
            cmap="YlGnBu_r",
            vmax=vmax,
            aspect="auto",
        )
        plt.colorbar()
        plt.ylabel("Separation [mas]")
        plt.xlabel("PA [deg]")
        plt.subplot(2, 2, 2)
        plt.imshow(
            sep_crat,
            extent=[np.amin(cons), np.amax(cons), np.amin(seps), np.amax(seps)],
            cmap="YlGnBu_r",
            vmax=vmax,
            aspect="auto",
        )
        plt.colorbar()
        plt.gca().tick_params(axis="x", label1On=False)
        plt.gca().tick_params(axis="y", label1On=False)
        plt.subplot(2, 2, 4)
        plt.imshow(
            pa_crat,
            extent=[np.amin(cons), np.amax(cons), np.amin(ths), np.amax(ths)],
            cmap="YlGnBu_r",
            vmax=vmax,
            aspect="auto",
        )
        plt.colorbar()
        plt.ylabel("PA [deg]")
        plt.xlabel("Contrast Ratio")
        plt.subplots_adjust(
            top=0.974, bottom=0.092, left=0.097, right=0.99, hspace=0.1, wspace=0.05
        )
        plt.show(block=False)

    data = {
        "chi2": chi2,
        "like": like,
        "seps": seps,
        "ths": ths,
        "cons": cons,
        "best": best_params,
    }
    return data


def test_significance(cpo, params, projected=False):
    """Tests whether a certain set of binary parameters is a better fit than a
    single star model"""
    # first, calculate the null hypothesis chi squared
    if projected:
        t3data = cpo.proj_t3data
        t3err = cpo.proj_t3err
    else:
        t3data = cpo.t3data
        t3err = cpo.t3err
        u = cpo.u
        v = cpo.v
        wavel = cpo.wavel

    null_chi2 = np.sum((t3data.ravel() / t3err.ravel()) ** 2)
    mod_cps = cp_model(params, u, v, wavel)
    if projected:
        mod_cps = np.reshape(mod_cps, (cpo.n_clps, cpo.n_runs), order="F")
        mod_cps = (np.transpose(cpo.proj).dot(mod_cps))[: len(t3data), :]
    bin_chi2 = np.sum(((t3data.ravel() - mod_cps.ravel()) / t3err.ravel()) ** 2)
    print("Chi2_null:", null_chi2)
    print("Chi2_bin:", bin_chi2)
    print("Significance:", np.sqrt(null_chi2 - bin_chi2), "=sqrt(chi2_null-chi2_bin)")
    return [null_chi2, bin_chi2]


def multiple_companions_hammer(
    cpo,
    ivar=[[50.0, 0.0, 2.0], [50.0, 90.0, 2.0]],
    ndim="Default",
    nwalcps=50,
    plot=False,
    projected=False,
    niters=1000,
    threads=1,
    model="constant",
    sep_prior=None,
    pa_prior=None,
    crat_prior=None,
    err_scale=1.0,
    extra_error=0.0,
):
    """Implementation of emcee, the MCMC Hammer, for closure phase
    fitting to multiple companions. Requires a closure phase object cpo, and is best called with
    ivar chosen to be near the peak - it can fail to converge otherwise.
    See cp_model for details!
    Prior ranges introduce a flat (tophat) prior between the two values specified
    NOTE: This doesn't work with the wavelength model yet!"""

    import emcee

    ivar = np.array(ivar)  # initial parameters for model-fit
    if ndim == "Default":
        ndim = len(ivar)

    if ivar.ndim != 1:
        ncomp = ivar.shape[0]
        ndim = ivar.size
        nparam = ndim / ncomp
    else:
        ncomp = 1
        nparam = ndim

    cpo.t3err = np.sqrt(cpo.t3err ** 2 + extra_error ** 2)
    cpo.t3err *= err_scale

    print("Fitting for:", ncomp, "companions")

    p0 = [
        ivar.ravel() + 0.1 * ivar.ravel() * np.random.rand(ndim) for i in range(nwalcps)
    ]  # initialise walcps in a ball
    print("Running emcee now!")

    t0 = time.time()
    if projected is False:
        sampler = emcee.EnsembleSampler(
            nwalcps,
            ndim,
            cp_loglikelihood_multiple,
            args=[cpo.u, cpo.v, cpo.wavel, cpo.t3data, cpo.t3err, model, ncomp],
            threads=threads,
        )
    else:
        print("I havent coded this yet...")
        raise NameError("cp_loglikelihood_multiple_proj doesnt exist")

    sampler.run_mcmc(p0, niters)
    tf = time.time()

    print("Time elapsed =", tf - t0, "s")

    chain = sampler.flatchain

    seps = np.zeros((chain.shape[0], ncomp))
    ths = np.zeros((chain.shape[0], ncomp))
    cs = np.zeros((chain.shape[0], ncomp, nparam - 2))
    for ix in range(ncomp):
        seps[:, ix] = chain[:, ix * nparam]
        ths[:, ix] = chain[:, ix * nparam + 1]
        cs[:, ix, :] = chain[:, ix * nparam + 2 : (ix + 1) * nparam]

    # Now introduce the prior, by ignoring values outside of the range
    if sep_prior is not None:
        for c_ix in range(ncomp):
            wh = (seps[:, c_ix] >= sep_prior[0]) & (seps[:, c_ix] <= sep_prior[1])
            seps = seps[wh, :]
            ths = ths[wh, :]
            cs = cs[wh, :]
    if pa_prior is not None:
        for c_ix in range(ncomp):
            wh = (ths[:, c_ix] >= pa_prior[0]) & (ths[:, c_ix] <= pa_prior[1])
            seps = seps[wh, :]
            ths = ths[wh, :]
            cs = cs[wh, :, :]
    if crat_prior is not None:
        for c_ix in range(ncomp):
            for ix in range(nparam - 2):
                wh = (cs[:, c_ix, ix] >= crat_prior[0]) & (
                    cs[:, c_ix, ix] <= crat_prior[1]
                )
                seps = seps[wh]
                ths = ths[wh]
                cs = cs[wh, :, :]

    # Now put the "good" values back into chain for later
    chain = np.zeros((seps.shape[0], nparam * ncomp))
    for ix in range(ncomp):
        chain[:, ix * nparam] = seps[:, ix]
        chain[:, ix * nparam + 1] = ths[:, ix]
        chain[:, ix * nparam + 2 : (ix + 1) * nparam] = cs[:, ix, :]

    if plot is True:
        plt.clf()

    # Loop over number of companions to calculate best params and do plots
    meanseps = []
    dseps = []
    meanths = []
    dths = []
    meancs = []
    dcs = []
    for c_ix in range(ncomp):
        meansep = np.mean(seps[:, c_ix])
        dsep = np.std(seps[:, c_ix])

        meanth = np.mean(ths[:, c_ix])
        dth = np.std(ths[:, c_ix])

        meanc = np.mean(cs[:, c_ix, :], axis=0)
        dc = np.std(cs[:, c_ix], axis=0)

        print("Companion #" + str(c_ix))
        print("    Separation", meansep, "pm", dsep, "mas")
        print("    Position angle", meanth, "pm", dth, "deg")
        print("    Contrast", meanc[0], "pm", dc[0])

        meanseps.append(meansep)
        dseps.append(dsep)
        meanths.append(meanth)
        dths.append(dth)
        meancs.append(meanc)
        dcs.append(dc)

        if model == "linear":
            print("    Contrast2", meanc[1], "pm", dc[1])
            extra_pars = ["Contrast "]
            extra_dims = ["Ratio"]
        elif model == "free":
            extra_pars = np.repeat("Contrast", ndim - 2)
            extra_dims = np.repeat("Ratio", ndim - 2)
        else:
            extra_pars = "None"
            extra_dims = "None"

        if plot is True:
            paramnames = ["Separation", "Position Angle", "Contrast"]
            paramdims = ["(mas)", "(deg)", "Ratio"]
            for ix, par in enumerate(extra_pars):
                paramnames.append(par)
                paramdims.append(extra_dims[ix])

            for i in range(nparam):
                if i < 3:
                    plt.figure(i)
                    if c_ix == 0:
                        plt.clf()
                plt.hist(chain[:, c_ix * nparam + i], niters / 5, histtype="step")
                plt.title(paramnames[i])
                plt.ylabel("Counts")
                plt.xlabel(paramnames[i] + " " + paramdims[i])

            plt.show()

    data = {
        "sep": meanseps,
        "delsep": dseps,
        "pa": meanths,
        "delpa": dths,
        "con": meancs,
        "delcon": dcs,
        "chain": sampler.chain,
        "seps": seps,
    }
    # and clean up
    sampler.pool.terminate()
    return data


# =========================================================================
# =========================================================================


def hammer_spectrum(
    cpo,
    params,
    ivar=[1.53],
    nwalcps=50,
    plot=False,
    model="free",
    niters=1000,
    threads=1,
    crat_prior=None,
    err_scale=1.0,
    extra_error=0.0,
):
    """ACC's modified version of hammer that allows fitting to a spectrum only, by
    fixing the position angle and separation of the companion. Made for HD142527
    Requires a closure phase object cpo, and is best called with
    ivar chosen to be near the peak - it can fail to converge otherwise.
    Also allows fitting of a contrast vs wavlength model. See cp_model for details!
    Prior ranges introduce a flat (tophat) prior between the two values specified.

    Ndim is the number of d.o.f in the spectral channels you want to fit to.
    It only works if ndim=nwav at the moment. Whoops. Otherwise need a polynomial model for cp_model"""

    import emcee

    [sep, pa] = params

    ivar = np.array(ivar)  # initial parameters for spectrum fit
    ndim = len(ivar)
    if ndim != 37:
        print(
            "This is still a work in progress, and only works for 37 channel data (GPI default)"
        )

    # initialise walcps in a ball
    p0 = [ivar + 0.1 * ivar * np.random.rand(ndim) for i in range(nwalcps)]

    p0 = np.array(p0)
    params = np.array(params)
    cpo.t3err = np.sqrt(cpo.t3err ** 2 + extra_error ** 2)
    cpo.t3err *= err_scale

    print("Running emcee now!")

    t0 = time.time()
    sampler = emcee.EnsembleSampler(
        nwalcps,
        ndim,
        cp_loglikelihood_spectrum,
        args=[params, cpo.u, cpo.v, cpo.wavel, cpo.t3data, cpo.t3err, model],
        threads=threads,
    )
    sampler.run_mcmc(p0, niters)
    # x=cp_loglikelihood_spectrum(ivar,params,cpo.u,cpo.v,cpo.wavel,cpo.t3data,cpo.t3err,model)
    tf = time.time()

    print("Time elapsed =", tf - t0, "s")

    chain = sampler.flatchain
    cs = chain

    # Now introduce the prior, by ignoring values outside of the range
    if crat_prior is not None:
        for ix in range(ndim - 2):
            c = cs[:, ix]
            wh = (c >= crat_prior[0]) & (c <= crat_prior[1])
            cs = cs[wh, :]

    # check that there are still some points left!
    if cs.size > 0:
        chain = np.zeros((len(cs), ndim))
        chain = cs
    else:
        print("WARNING: Your priors eliminated all points!")

    meanc = np.mean(cs, axis=0)
    dc = np.std(cs, axis=0)
    print("Mean contrast", np.mean(meanc), "pm", np.mean(dc))
    for ix in range(ndim):
        print("Channel", ix, ":", meanc[ix], "pm", dc[ix])

    if plot is True:

        plt.clf()

        for i in range(ndim):
            plt.hist(chain[:, i], niters / 10, histtype="step")
            plt.title("Contrast")
            plt.ylabel("Counts")
            plt.xlabel("Contrast Ratio")

        plt.show()

    data = {
        "con": meanc,
        "delcon": dc,
        "chain": sampler.chain,
        "posterior": chain,
        "wavelength": cpo.wavel,
    }
    # and clean up
    sampler.pool.terminate()
    return data


# =========================================================================
# =========================================================================


def cp_loglikelihood_spectrum(
    spec_params, bin_params, u, v, wavel, t3data, t3err, model="free"
):
    """Calculate loglikelihood for closure phase data, with fixed sep,
    pa and con so you can measure a spectrum. Used in hammer_spectrum."""
    params = np.concatenate((bin_params, spec_params))
    cps = cp_model(params, u, v, wavel, model=model)
    chi2 = np.sum(((t3data.ravel() - cps.ravel()) / t3err.ravel()) ** 2)
    loglike = -chi2 / 2
    return loglike


# =========================================================================
# =========================================================================


def find_extra_error(
    params, cpo, err_scale=1.0, dof="Default", model="constant", projected=False
):
    """Finds the extra error needed for reduced chi squared to be 1. This is
    done by trying for many values of extra error and then interpolating.
    If dof is not set, then the degrees of freedom are taken as cpo.t3data.size."""

    model_cps = cp_model(params, cpo.u, cpo.v, cpo.wavel, model=model)
    if projected:
        model_cps = project_cps(model_cps, cpo.proj)
        data_cps = cpo.proj_t3data
        cp_err = cpo.proj_t3err
    else:
        data_cps = cpo.t3data
        cp_err = cpo.t3err

    cp_resids = data_cps - model_cps

    print("Max resid:" + str(np.max(np.abs(cp_resids))))

    n = 1000  # number of extra errors to try.
    chis_null = np.zeros(n)
    chis_bin = np.zeros(n)

    extra_errors = np.logspace(-2, 2, num=n)

    for ix, err in enumerate(extra_errors):

        # Calculate the closure phase uncertainties for this amount of extra_error
        t3err = np.sqrt(cp_err ** 2 + err ** 2)
        t3err *= err_scale

        chis_bin[ix] = np.sum(cp_resids ** 2 / t3err ** 2)
        chis_null[ix] = np.sum(data_cps ** 2 / t3err ** 2)

    # scale to red.chi2
    if dof == "Default":
        dof = data_cps.size

    chis_bin /= dof - 3.0  # -3 due to the fit
    chis_null /= dof

    print("Binary Chi2 with no extra error: " + str(np.max(chis_bin)))

    # now find where it goes to 1. with interpolation
    func_bin = interp.interp1d(chis_bin, extra_errors)
    func_null = interp.interp1d(chis_null, extra_errors)
    try:
        err_bin = func_bin(1.0)
    except Exception:
        if np.min(chis_bin) < 1.0:
            print("Binary Reduced Chi2 less than 1")
        else:
            print("Problem trying to calculate extra error.")
        err_bin = 0.0

    try:
        err_null = func_null(1.0)
    except Exception:
        if np.min(chis_null) < 1.0:
            print("Null Reduced Chi2 less than 1")
        else:
            print("Problem trying to calculate extra error.")
        err_null = 0.0

    print("Extra error needed:")
    print("Binary:", err_bin)
    print("Null  :", err_null)

    # Diagnostic plots
    cp_err_plot = np.sqrt(cp_err ** 2 + err_bin ** 2)

    plt.clf()
    plt.subplot(211)
    plt.errorbar(model_cps.ravel(), data_cps.ravel(), cp_err_plot.ravel(), fmt="o")
    plt.xlabel("Model CLP (deg)")
    plt.ylabel("Measured CLP (deg)")
    plt.subplot(212)
    plt.errorbar(model_cps.ravel(), cp_resids.ravel(), cp_err_plot.ravel(), fmt="o")
    plt.xlabel("Model CLP (deg)")
    plt.ylabel("Residual CLP (deg)")
    plt.tight_layout()

    return chis_bin, extra_errors
