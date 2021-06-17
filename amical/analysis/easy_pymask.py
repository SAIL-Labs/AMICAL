import numpy as np
from matplotlib import pyplot as plt

from amical.externals import pymask


def pymask_grid(
    input_data,
    ngrid=40,
    pa_prior=None,
    sep_prior=None,
    cr_prior=None,
    err_scale=1.0,
    extra_error_cp=0.0,
    ncore=1,
    verbose=False,
):
    """Compute chi2 map of a binary model over a regular grid.

    Parameters:
    -----------
    `input_data` {str}:
        Oifits file name,\n
    `ngrid` {int}:
        Number of points in the grid (for each parameters: pa, sep,
        cr, i.e.: ngrid**3),\n
    `pa_prior` {list}:
        Bounds of the position angle (default: [0, 360]),\n
    `sep_prior` {list}:
        Bounds of the separation (default: [0, 100]),\n
    `cr_prior` {list}:
        Bounds of the contrast ratio (default: [0, 150]),\n
    `err_scale` {float}:
        Scaling factor apply on the errorbars (multiplicative),\n
    `extra_error_cp` {float}:
        Additional error [deg] (additive),\n
    `ncore` {int}:
        Number of threads used for multiprocessing.

    Return:
    -------
    `like_grid` (np.array):
        Compute likelyhood map.
    """
    if pa_prior is None:
        pa_prior = [0, 360]
    if sep_prior is None:
        sep_prior = [0, 100]
    if cr_prior is None:
        cr_prior = [1, 150]

    cpo = pymask.cpo(input_data)
    like_grid = pymask.coarse_grid(
        cpo,
        nsep=ngrid,
        nth=ngrid,
        ncon=ngrid,
        thmin=pa_prior[0],
        thmax=pa_prior[1],
        smin=sep_prior[0],
        smax=sep_prior[1],
        cmin=cr_prior[0],
        cmax=cr_prior[1],
        threads=ncore,
        err_scale=err_scale,
        extra_error=extra_error_cp,
        verbose=verbose,
    )
    return like_grid


def pymask_mcmc(
    input_data,
    initial_guess,
    niters=1000,
    pa_prior=None,
    sep_prior=None,
    cr_prior=None,
    err_scale=1,
    extra_error_cp=0,
    ncore=1,
    burn_in=500,
    walkers=100,
    display=True,
    verbose=True,
):
    if pa_prior is None:
        pa_prior = [0, 360]

    cpo = pymask.cpo(input_data)
    hammer_data = pymask.hammer(
        cpo,
        ivar=initial_guess,
        niters=niters,
        model="constant",
        nwalcps=walkers,
        sep_prior=sep_prior,
        pa_prior=pa_prior,
        crat_prior=cr_prior,
        err_scale=err_scale,
        extra_error=extra_error_cp,
        plot=display,
        burn_in=burn_in,
        threads=ncore,
    )

    res_corner = hammer_data[1]
    chain = hammer_data[0]["chain"]

    dm = 2.5 * np.log10(res_corner["cr"])
    dmm = 2.5 * np.log10(res_corner["cr"] - res_corner["delcrm"])
    dmp = 2.5 * np.log10(res_corner["cr"] + res_corner["delcrp"])
    e_dmm = abs(dm - dmm)
    e_dmp = abs(dm - dmp)

    if verbose:
        print("MCMC estimation")
        print("---------------")
        print(
            "Separation = %2.1f +%2.1f/-%2.1f mas"
            % (res_corner["sep"], res_corner["delsepp"], res_corner["delsepm"])
        )
        print(
            "PA = %2.1f +%2.1f/-%2.1f deg"
            % (res_corner["pa"], res_corner["delpap"], res_corner["delpam"])
        )
        print(
            "Contrast Ratio = %2.1f +%2.1f/-%2.1f"
            % (res_corner["cr"], res_corner["delcrp"], res_corner["delcrm"])
        )
        print("dm = %2.2f +%2.2f/-%2.2f mag" % (dm, e_dmp, e_dmm))

    chain_sep = chain[:, :, 0].T
    chain_th = chain[:, :, 1].T
    chain_cr = chain[:, :, 2].T

    if display:
        sep = res_corner["sep"]
        pa = res_corner["pa"]
        cr = res_corner["cr"]
        plt.figure(figsize=(5, 7))
        plt.subplot(3, 1, 1)
        plt.plot(chain_sep, color="grey", alpha=0.5)
        plt.plot(len(chain_sep), sep, marker="*", color="#0085ca", zorder=1e3)
        plt.ylabel("Separation [mas]")
        plt.subplot(3, 1, 2)
        plt.plot(chain_th, color="grey", alpha=0.5)
        plt.plot(len(chain_sep), pa, marker="*", color="#0085ca", zorder=1e3)
        plt.ylabel("PA [deg]")
        plt.subplot(3, 1, 3)
        plt.plot(chain_cr, color="grey", alpha=0.2)
        plt.plot(len(chain_sep), cr, marker="*", color="#0085ca", zorder=1e3)
        plt.xlabel("Step")
        plt.ylabel("CR")
        plt.tight_layout()
        plt.show(block=False)

    res = {
        "best": {
            "model": "binary",
            "dm": dm,
            "theta": res_corner["pa"],
            "sep": res_corner["sep"],
            "x0": 0,
            "y0": 0,
        },
        "uncer": {
            "dm_p": e_dmp,
            "dm_m": e_dmm,
            "theta_p": res_corner["delpap"],
            "theta_m": res_corner["delpam"],
            "sep_p": res_corner["delsepp"],
            "sep_m": res_corner["delsepm"],
        },
    }
    return res


def pymask_cr_limit(
    input_data,
    nsim=100,
    err_scale=1,
    extra_error_cp=0,
    ncore=1,
    cmax=500,
    nsep=60,
    ncrat=60,
    nth=30,
    smin=20,
    smax=250,
    cmin=1.0001,
):
    cpo = pymask.cpo(input_data)
    lims_data = pymask.detec_limits(
        cpo,
        threads=ncore,
        nsim=nsim,
        nsep=nsep,
        ncon=ncrat,
        nth=nth,
        smax=smax,
        cmax=cmax,
        cmin=cmin,
        smin=smin,
        err_scale=err_scale,
        extra_error=extra_error_cp,
    )

    limits = lims_data["limits"]
    seps = lims_data["seps"]
    crats = lims_data["cons"]
    crat_limits = 0 * seps

    # Loop through seps and find the highest contrast ratio that would be detectable
    for sep_ix in range(len(seps)):
        would_detec = limits[:, sep_ix] >= 0.9973
        if np.sum(would_detec) >= 1:
            threesig_lim = np.max(crats[would_detec])
            crat_limits[sep_ix] = threesig_lim  # fivesig_lim
        else:
            crat_limits[sep_ix] = 1.0
    con_limits = 2.5 * np.log10(crat_limits)

    plt.figure()
    plt.plot(seps, con_limits)
    plt.xlabel("Separation [mas]")
    plt.ylabel(r"$\Delta \mathrm{Mag}_{3\sigma}$")
    plt.title(r"PYMASK: flux ratio for 3$\sigma$ detection")
    plt.ylim(plt.ylim()[1], plt.ylim()[0])  # -- rreverse plot
    plt.tight_layout()
    res = {"r": seps, "cr_limit": con_limits, "lims_data": lims_data}
    return res
