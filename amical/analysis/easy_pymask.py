import numpy as np
from matplotlib import pyplot as plt

from amical.analysis import pymask


def pymask_grid(input_data, ngrid=40, pa_prior=[0, 360], sep_prior=[0, 100], cr_prior=[1, 150],
                err_scale=1., extra_error=0., ncore=1, verbose=False):
    cpo = pymask.cpo(input_data)
    like_grid = pymask.coarse_grid(cpo, nsep=ngrid, nth=ngrid, ncon=ngrid, thmin=pa_prior[0], thmax=pa_prior[1],
                                   smin=sep_prior[0], smax=sep_prior[1], cmin=cr_prior[0], cmax=cr_prior[1],
                                   threads=ncore, err_scale=err_scale, extra_error=extra_error, verbose=verbose)
    return like_grid


def pymask_mcmc(input_data, initial_guess, niters=1000, pa_prior=[0, 360], sep_prior=[0, 100], cr_prior=[1, 150],
                err_scale=1, extra_error=0, ncore=1, burn_in=500, walkers=100, display=True,
                verbose=True):
    cpo = pymask.cpo(input_data)
    hammer_data = pymask.hammer(cpo, ivar=initial_guess, niters=niters, model='constant', nwalcps=walkers,
                                sep_prior=sep_prior, pa_prior=pa_prior, crat_prior=cr_prior,
                                err_scale=err_scale, extra_error=extra_error, plot=display,
                                burn_in=burn_in, threads=ncore)

    res_corner = hammer_data[1]
    chain = hammer_data[0]['chain']
    if verbose:
        print('MCMC estimation')
        print('---------------')
        print('Separation = %2.1f +%2.1f/-%2.1f mas' %
              (res_corner['sep'], res_corner['delsepp'], res_corner['delsepm']))
        print('PA = %2.1f +%2.1f/-%2.1f deg' %
              (res_corner['pa'], res_corner['delpap'], res_corner['delpam']))
        print('Contrast Ratio = %2.1f +%2.1f/-%2.1f' %
              (res_corner['cr'], res_corner['delcrp'], res_corner['delcrm']))
        dm = 2.5*np.log10(res_corner['cr'])
        dmm = 2.5*np.log10(res_corner['cr']-res_corner['delcrm'])
        dmp = 2.5*np.log10(res_corner['cr']+res_corner['delcrp'])
        e_dmm = abs(dm - dmm)
        e_dmp = abs(dm - dmp)
        print('dm = %2.2f +%2.2f/-%2.2f mag' % (dm, e_dmp, e_dmm))

    chain_sep = chain[:, :, 0].T
    chain_th = chain[:, :, 1].T
    chain_cr = chain[:, :, 2].T

    if display:
        sep = res_corner['sep']
        pa = res_corner['pa']
        cr = res_corner['cr']
        plt.figure(figsize=(5, 7))
        plt.subplot(3, 1, 1)
        plt.plot(chain_sep, color='grey', alpha=.5)
        plt.hlines(sep, 0, len(chain_sep), color='#0085ca', lw=2, zorder=1e3)
        plt.ylabel('Separation [mas]')
        plt.subplot(3, 1, 2)
        plt.plot(chain_th, color='grey', alpha=.5)
        plt.hlines(pa, 0, len(chain_sep), color='#0085ca', lw=2, zorder=1e3)
        plt.ylabel('PA [deg]')
        plt.subplot(3, 1, 3)
        plt.plot(chain_cr, color='grey', alpha=.2)
        plt.hlines(cr, 0, len(chain_sep), color='#0085ca', lw=1, zorder=1e3)
        plt.xlabel('Step')
        plt.ylabel('CR')
        plt.tight_layout()
        plt.show(block=False)
    return res_corner


def pymask_cr_limit(input_data, nsim=100, err_scale=1, extra_error=0, ncore=1, cmax=500,
                    nsep=60, ncrat=60, nth=30, smax=250, ):
    cpo = pymask.cpo(input_data)
    lims_data = pymask.detec_limits(cpo, threads=ncore, nsim=nsim,
                                    nsep=nsep, ncon=ncrat, nth=nth,
                                    smax=smax, cmax=cmax,
                                    err_scale=err_scale, extra_error=extra_error)

    limits = lims_data['limits']
    seps = lims_data['seps']
    crats = lims_data['cons']
    crat_limits = 0*seps

    # Loop through seps and find the highest contrast ratio that would be detectable
    for sep_ix in range(len(seps)):
        would_detec = limits[:, sep_ix] > 0.99
        if np.sum(would_detec) > 1:
            threesig_lim = np.max(crats[would_detec])
            fivesig_lim = threesig_lim*3.3 / \
                3.  # convert to 3 sigma (normally 5)
            crat_limits[sep_ix] = fivesig_lim
        else:
            crat_limits[sep_ix] = 1.
    con_limits = 2.5*np.log10(crat_limits)

    plt.figure()
    plt.plot(seps, con_limits)
    plt.xlabel('Separation [mas]')
    plt.ylabel('$\Delta \mathrm{Mag}_{3\sigma}$')
    plt.title("PYMASK: flux ratio for 3$\sigma$ detection")
    plt.ylim(plt.ylim()[1], plt.ylim()[0])  # -- rreverse plot
    plt.tight_layout()
    res = {'r': seps,
           'cr_limit': con_limits
           }
    return res
