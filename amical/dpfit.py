"""
@author: Anthony Soulain (University of Sydney)

-------------------------------------------------------------------------
AMICAL: Aperture Masking Interferometry Calibration and Analysis Library
-------------------------------------------------------------------------

Fitting tools (developped by A. Merand).

--------------------------------------------------------------------
"""

import time

import numpy as np
import scipy.optimize

"""
IDEA: fit Y = F(X,A) where A is a dictionnary describing the
parameters of the function.

note that the items in the dictionnary should all be scalar!

author: amerand@eso.org

Tue 29 Jan 2013 17:03:21 CLST: working on adding correlations -> NOT WORKING!!!
Thu 28 Feb 2013 12:34:31 CLST: correcting leading to x2 for chi2 display
Mon  8 Apr 2013 10:51:03 BRT: alternate algorithms

http://www.rhinocerus.net/forum/lang-idl-pvwave/355826-generalized-least-squares.html
"""

verboseTime = time.time()


def leastsqFit(
    func,
    x,
    params,
    y,
    err=None,
    fitOnly=None,
    verbose=False,
    doNotFit=None,
    epsfcn=1e-7,
    ftol=1e-5,
    fullOutput=True,
    normalizedUncer=True,
    follow=None,
):
    """
    - params is a Dict containing the first guess.

    - fits 'y +- err = func(x,params)'. errors are optionnal. in case err is a
      ndarray of 2 dimensions, it is treated as the covariance of the
      errors.

      np.array([[err1**2, 0, .., 0],
                [0, err2**2, 0, .., 0],
                [0, .., 0, errN**2]]) is the equivalent of 1D errors

    - follow=[...] list of parameters to "follow" in the fit, i.e. to print in
      verbose mode

    - fitOnly is a LIST of keywords to fit. By default, it fits all
      parameters in 'params'. Alternatively, one can give a list of
      parameters not to be fitted, as 'doNotFit='

    - doNotFit has a similar purpose: for example if params={'a0':,
      'a1': 'b1':, 'b2':}, doNotFit=['a'] will result in fitting only
      'b1' and 'b2'. WARNING: if you name parameter 'A' and another one 'AA',
      you cannot use doNotFit to exclude only 'A' since 'AA' will be excluded as
      well...

    - normalizedUncer=True: the uncertainties are independent of the Chi2, in
      other words the uncertainties are scaled to the Chi2. If set to False, it
      will trust the values of the error bars: it means that if you grossely
      underestimate the data's error bars, the uncertainties of the parameters
      will also be underestimated (and vice versa).

    returns dictionary with:
    'best': bestparam,
    'uncer': uncertainties,
    'chi2': chi2_reduced,
    'model': func(x, bestparam)
    'cov': covariance matrix (normalized if normalizedUncer)
    'fitOnly': names of the columns of 'cov'
    """

    if doNotFit is None:
        doNotFit = []

    # -- fit all parameters by default
    if fitOnly is None:
        if len(doNotFit) > 0:
            fitOnly = [x for x in list(params.keys()) if x not in doNotFit]
        else:
            fitOnly = list(params.keys())
        fitOnly.sort()  # makes some display nicer

    # -- build fitted parameters vector:
    pfit = [params[k] for k in fitOnly]

    # -- built fixed parameters dict:
    pfix = {}
    for k in list(params.keys()):
        if k not in fitOnly:
            pfix[k] = params[k]
    if verbose:
        print("[dpfit] %d FITTED parameters:" % (len(fitOnly)), fitOnly)
    # -- actual fit
    plsq, cov, info, mesg, ier = scipy.optimize.leastsq(
        _fitFunc,
        pfit,
        args=(
            fitOnly,
            x,
            y,
            err,
            func,
            pfix,
            verbose,
            follow,
        ),
        full_output=True,
        epsfcn=epsfcn,
        ftol=ftol,
    )
    if isinstance(err, np.ndarray) and len(err.shape) == 2:
        print(cov)

    # -- best fit -> agregate to pfix
    for i, k in enumerate(fitOnly):
        pfix[k] = plsq[i]

    # -- reduced chi2
    model = func(x, pfix)
    tmp = _fitFunc(plsq, fitOnly, x, y, err, func, pfix)

    try:
        chi2 = (np.array(tmp) ** 2).sum()
    except Exception:
        chi2 = 0.0
        for x in tmp:
            chi2 += np.sum(x ** 2)
    reducedChi2 = chi2 / float(
        np.sum([1 if np.isscalar(i) else len(i) for i in tmp]) - len(pfit) + 1
    )
    # print(chi2, reducedChi2, float(np.sum([1 if np.isscalar(i) else
    #                                 len(i) for i in tmp])-len(pfit)+1))
    if not np.isscalar(reducedChi2):
        reducedChi2 = np.mean(reducedChi2)

    # -- uncertainties:
    uncer = {}
    for k in list(pfix.keys()):
        if k not in fitOnly:
            uncer[k] = 0  # not fitted, uncertatinties to 0
        else:
            i = fitOnly.index(k)
            if cov is None:
                uncer[k] = -1
            else:
                uncer[k] = np.sqrt(np.abs(np.diag(cov)[i]))
                if normalizedUncer:
                    uncer[k] *= np.sqrt(reducedChi2)

    if verbose:
        print("-" * 30)
        print("REDUCED CHI2=", reducedChi2)
        print("-" * 30)
        if normalizedUncer:
            print("(uncertainty normalized to data dispersion)")
        else:
            print("(uncertainty assuming error bars are correct)")
        tmp = list(pfix.keys())
        tmp.sort()
        maxLength = np.max(np.array([len(k) for k in tmp]))
        format_ = "'%s':"
        # -- write each parameter and its best fit, as well as error
        # -- writes directly a dictionnary
        print("")  # leave some space to the eye
        for ik, k in enumerate(tmp):
            padding = " " * (maxLength - len(k))
            formatS = format_ + padding
            if ik == 0:
                formatS = "{" + formatS
            if uncer[k] > 0:
                ndigit = -int(np.log10(uncer[k])) + 3
                print(formatS % k, round(pfix[k], ndigit), ",", end=" ")
                print("# +/-", round(uncer[k], ndigit))
            elif uncer[k] == 0:
                if isinstance(pfix[k], str):
                    print(formatS % k, "'" + pfix[k] + "'", ",")
                else:
                    print(formatS % k, pfix[k], ",")
            else:
                print(formatS % k, pfix[k], ",", end=" ")
                print("# +/-", uncer[k])
        print("}")  # end of the dictionnary
        try:
            if verbose > 1:
                print("-" * 3, "correlations:", "-" * 15)
                N = np.max([len(k) for k in fitOnly])
                N = min(N, 20)
                N = max(N, 5)
                sf = "%" + str(N) + "s"
                print(" " * N, end=" ")
                for k2 in fitOnly:
                    print(sf % k2, end=" ")
                print("")
                sf = "%-" + str(N) + "s"
                for k1 in fitOnly:
                    i1 = fitOnly.index(k1)
                    print(sf % k1, end=" ")
                    for k2 in fitOnly:
                        i2 = fitOnly.index(k2)
                        if k1 != k2:
                            print(
                                ("%" + str(N) + ".2f")
                                % (cov[i1, i2] / np.sqrt(cov[i1, i1] * cov[i2, i2])),
                                end=" ",
                            )
                        else:
                            print(" " * (N - 4) + "-" * 4, end=" ")
                    print("")
                print("-" * 30)
        except Exception:
            pass
    # -- result:
    if fullOutput:
        if normalizedUncer:
            try:
                cov *= reducedChi2
            except Exception:
                pass
        try:
            cor = np.sqrt(np.diag(cov))
            cor = cor[:, None] * cor[None, :]
            cor = cov / cor
        except Exception:
            cor = None

        pfix = {
            "best": pfix,
            "uncer": uncer,
            "chi2": reducedChi2,
            "model": model,
            "cov": cov,
            "fitOnly": fitOnly,
            "info": info,
            "cor": cor,
        }
    return pfix


def bootstrap(
    func,
    x,
    params,
    y,
    err=None,
    fitOnly=None,
    verbose=False,
    doNotFit=None,
    epsfcn=1e-7,
    ftol=1e-5,
    fullOutput=True,
    normalizedUncer=True,
    follow=None,
    Nboot=None,
):
    """
    bootstraping, called like leastsqFit. returns a list of fits: the first one
    is the 'normal' one, the Nboot following one are with ramdomization of data. If
    Nboot is not given, it is set to 10*len(x).
    """
    if doNotFit is None:
        doNotFit = []

    if Nboot is None:
        Nboot = 10 * len(x)
    # first fit is the "normal" one
    fits = [
        leastsqFit(
            func,
            x,
            params,
            y,
            err=err,
            fitOnly=fitOnly,
            verbose=False,
            doNotFit=doNotFit,
            epsfcn=epsfcn,
            ftol=ftol,
            fullOutput=True,
            normalizedUncer=True,
        )
    ]
    for _ in range(Nboot):
        s = np.int_(len(x) * np.random.rand(len(x)))
        fits.append(
            leastsqFit(
                func,
                x[s],
                params,
                y[s],
                err=err,
                fitOnly=fitOnly,
                verbose=False,
                doNotFit=doNotFit,
                epsfcn=epsfcn,
                ftol=ftol,
                fullOutput=True,
                normalizedUncer=True,
            )
        )
    return fits


def _fitFunc(
    pfit, pfitKeys, x, y, err=None, func=None, pfix=None, verbose=False, follow=None
):
    """
    interface to leastsq from scipy:
    - x,y,err are the data to fit: f(x) = y +- err
    - pfit is a list of the paramters
    - pfitsKeys are the keys to build the dict
    pfit and pfix (optional) and combines the two
    in 'A', in order to call F(X,A)

    in case err is a ndarray of 2 dimensions, it is treated as the
    covariance of the errors.
    np.array([[err1**2, 0, .., 0],
             [ 0, err2**2, 0, .., 0],
             [0, .., 0, errN**2]]) is the equivalent of 1D errors

    """
    global verboseTime
    params = {}
    # -- build dic from parameters to fit and their values:
    for i, k in enumerate(pfitKeys):
        params[k] = pfit[i]
    # -- complete with the non fitted parameters:
    for k in pfix:
        params[k] = pfix[k]
    if err is None:
        err = np.ones(np.array(y).shape)

    # -- compute residuals

    if type(y) == np.ndarray and type(err) == np.ndarray:
        if len(err.shape) == 2:
            # -- using correlations
            tmp = func(x, params)
            # res = np.dot(np.dot(tmp-y, linalg.inv(err)), tmp-y)
            res = np.dot(np.dot(tmp - y, err), tmp - y)
            res = np.ones(len(y)) * np.sqrt(res / len(y))
        else:
            # -- assumes y and err are a numpy array
            y = np.array(y)
            res = ((func(x, params) - y) / err).flatten()
    else:
        # much slower: this time assumes y (and the result from func) is
        # a list of things, each convertible in np.array
        res = []
        tmp = func(x, params)

        for k in range(len(y)):
            df = (np.array(tmp[k]) - np.array(y[k])) / np.array(err[k])
            try:
                res.extend(list(df))
            except Exception:
                res.append(df)

    if verbose and time.time() > (verboseTime + 1):
        verboseTime = time.time()
        print(time.asctime(), end=" ")
        try:
            chi2 = (res ** 2).sum / (len(res) - len(pfit) + 1.0)
            print("CHI2: %6.4e" % chi2, end=" ")
        except Exception:
            # list of elements
            chi2 = 0
            N = 0
            res2 = []
            for r in res:
                if np.isscalar(r):
                    chi2 += r ** 2
                    N += 1
                    res2.append(r)
                else:
                    chi2 += np.sum(np.array(r) ** 2)
                    N += len(r)
                    res2.extend(list(r))

            res = res2
            print("CHI2: %6.4e" % (chi2 / float(N - len(pfit) + 1)), end=" ")
        if follow is None:
            print("")
        else:
            try:
                print(" ".join([k + "=" + "%5.2e" % params[k] for k in follow]))
            except Exception:
                print("")
    return res
