import multiprocessing
import os

import numpy as np
from rich.progress import track

import amical
from amical.analysis import models
from amical.dpfit import leastsqFit
from amical.tools import mas2rad, roundSciDigit

err_pts_style = {
    "linestyle": "None",
    "capsize": 1,
    "ecolor": "#364f6b",
    "mec": "#364f6b",
    "marker": ".",
    "elinewidth": 0.5,
    "alpha": 1,
    "ms": 14,
}


def select_model(name):
    """Select a Fourier-space model by name.

    Parameters
    ----------
    name : str
        Model identifier.

    Returns
    -------
    callable or None
        Matching visibility-model function, or ``None`` for an unknown name.
    """
    if name == "disk":
        model = models.visUniformDisk
    elif name == "binary":
        model = models.visBinary
    elif name == "binary_res":
        model = models.visBinary_res
    elif name == "edisk":
        model = models.visEllipticalDisk
    elif name == "clumpyDebrisDisk":
        model = models.visClumpDebrisDisk
    else:
        model = None
    return model


def check_params_model(param):
    """Check whether model parameters satisfy model constraints.

    Parameters
    ----------
    param : dict
        Model parameters, including the ``model`` key.

    Returns
    -------
    tuple[bool, str]
        Whether parameters are valid and a validation message.
    """
    isValid = True
    log = ""
    if param["model"] == "edisk":
        elong = np.cos(np.deg2rad(param["incl"]))
        majorAxis = mas2rad(param["majorAxis"])
        posang = np.deg2rad(param["posang"])
        if (
            (elong < 1 and abs(1 - elong) > 4e-16)
            or (posang < 0)
            or (posang > np.pi)
            or (majorAxis < 0)
        ):
            log = "# elong > 1,\n# minorAxis > 0 mas,\n# 0 < angle < 180 deg.\n"
            isValid = False

    return isValid, log


def comput_V2(X, param, model):
    """Compute squared visibility for a model.

    Parameters
    ----------
    X : sequence of array-like
        Spatial-frequency coordinates ``(u, v, wavelength)``.
    param : dict
        Model parameters.
    model : callable
        Complex-visibility model.

    Returns
    -------
    numpy.ndarray or int
        Squared visibility, or ``2`` when model parameters are invalid.
    """
    u = X[0]
    v = X[1]
    wl = X[2]

    isValid = check_params_model(param)[0]

    if not isValid:
        V2 = 2
    else:
        V = model(u, v, wl, param)
        V2 = np.abs(V) ** 2
    return V2


def comput_CP(X, param, model):
    """Compute closure phases for a model.

    Parameters
    ----------
    X : sequence of array-like
        Baseline coordinates and wavelength ``(u1, u2, u3, v1, v2, v3, wl)``.
    param : dict
        Model parameters.
    model : callable
        Complex-visibility model.

    Returns
    -------
    numpy.ndarray
        Closure phases in degrees.
    """
    u1 = X[0]
    u2 = X[1]
    u3 = X[2]
    v1 = X[3]
    v2 = X[4]
    v3 = X[5]
    wl = X[6]

    V1 = model(u1, v1, wl, param)
    V2 = model(u2, v2, wl, param)
    V3 = model(u3, v3, wl, param)

    BS = V1 * V2 * np.conjugate(V3)
    CP = np.rad2deg(np.arctan2(BS.imag, BS.real))
    return CP


def engine_multi_proc(inputs):
    """Compute one model datum for multiprocessing.

    Parameters
    ----------
    inputs : sequence
        Model coordinates, observable type, and model parameters.

    Returns
    -------
    scalar
        Modelled observable, ``numpy.nan`` for an unknown type, or ``0`` when
        model evaluation fails.
    """
    try:
        X = inputs[0]
        typ = inputs[1]
        param = inputs[-1]

        modelname = param["model"]

        model_target = select_model(modelname)

        if typ == "V2":
            mod = comput_V2(X, param, model_target)
        elif typ == "CP":
            mod = comput_CP(X, param, model_target)
        else:
            mod = np.nan
    except Exception:
        mod = 0

    return mod


def model_standard(obs, param):
    """Compute model values for all observations serially.

    Parameters
    ----------
    obs : array-like
        Observation rows containing coordinates and observable types.
    param : dict
        Model parameters.

    Returns
    -------
    numpy.ndarray
        Model values for the observations.
    """
    modelname = param["model"]
    model_target = select_model(modelname)
    bunch = len(obs)
    res = np.zeros(bunch)
    for i in range(bunch):
        try:
            typ = obs[i][1]
            if typ == "V2":
                X = obs[i][0]
                mod = comput_V2(X, param, model_target)
            elif typ == "CP":
                X = obs[i][0]
                mod = comput_CP(X, param, model_target)
            else:
                mod = np.nan

            res[i] = np.asarray(mod).item()
        except TypeError:
            pass

    output = np.array(res)
    return output


def model_parallelized(obs, param, ncore=12):
    """Compute model values for all observations using multiprocessing.

    Parameters
    ----------
    obs : array-like
        Observation rows containing coordinates and observable types.
    param : dict
        Model parameters.
    ncore : int, default=12
        Number of worker processes.

    Returns
    -------
    numpy.ndarray
        Model values for the observations.
    """
    pool = multiprocessing.Pool(ncore)

    b = np.array([param] * len(obs))
    c = b.reshape(len(obs), 1)
    d = np.append(obs, c, axis=1)

    res = pool.map(engine_multi_proc, d)

    pool.close()
    pool.join()
    return np.array(res)


def fits2obs(
    inputdata,
    use_flag=True,
    cond_wl=False,
    wl_min=None,
    wl_max=None,
    cond_uncer=False,
    rel_max=None,
    extra_error_v2=0,
    extra_error_cp=0,
    err_scale=1,
    input_rad=False,
    verbose=True,
):
    """Convert selected OIFITS observables to fitting observations.

    Parameters
    ----------
    inputdata : str
        OIFITS file name.
    use_flag : bool, default=True
        Whether to exclude data flagged in the OIFITS file.
    cond_wl : bool, default=False
        Whether to restrict wavelengths to ``wl_min`` through ``wl_max``.
    wl_min : float or None, default=None
        Lower wavelength limit in micrometres.
    wl_max : float or None, default=None
        Upper wavelength limit in micrometres.
    cond_uncer : bool, default=False
        Whether to filter data by relative uncertainty.
    rel_max : float or None, default=None
        Maximum allowed relative uncertainty in percent.
    extra_error_v2 : float, default=0
        Additive squared-visibility uncertainty.
    extra_error_cp : float, default=0
        Additive closure-phase uncertainty.
    err_scale : float, default=1
        Multiplicative closure-phase uncertainty scale.
    input_rad : bool, default=False
        Whether closure phases and their uncertainties are expressed in radians.
    verbose : bool, default=True
        Whether to print data-selection information.

    Returns
    -------
    numpy.ndarray
        Object array of selected coordinates, observable types, values, and
        uncertainties.
    """

    data = amical.loadc(inputdata)
    nwl = len(data.wl)

    nbl = data.vis2.shape[0]
    ncp = data.cp.shape[0]

    vis2_data = data.vis2.flatten()  # * 0.97
    e_vis2_data = (data.e_vis2.flatten() ** 2 + extra_error_v2**2) ** 0.5
    flag_V2 = data.flag_vis.flatten()

    if input_rad:
        cp_data = np.rad2deg(data.cp.flatten())
        e_cp_data = np.rad2deg(
            np.sqrt(data.e_cp.flatten() ** 2 + extra_error_cp**2) * err_scale
        )
    else:
        cp_data = data.cp.flatten()
        e_cp_data = np.sqrt(data.e_cp.flatten() ** 2 + extra_error_cp**2) * err_scale

    flag_CP = data.flag_cp.flatten()

    if use_flag:
        pass
    else:
        flag_V2 = [False] * len(vis2_data)
        flag_CP = [False] * len(cp_data)

    u_data, v_data = [], []
    u1_data, v1_data, u2_data, v2_data = [], [], [], []

    for i in range(nbl):
        for _ in range(nwl):
            u_data.append(data.u[i])
            v_data.append(data.v[i])

    for i in range(ncp):
        for _ in range(nwl):
            u1_data.append(data.u1[i])
            v1_data.append(data.v1[i])
            u2_data.append(data.u2[i])
            v2_data.append(data.v2[i])

    u_data = np.array(u_data)
    v_data = np.array(v_data)

    u1_data = np.array(u1_data)
    v1_data = np.array(v1_data)
    u2_data = np.array(u2_data)
    v2_data = np.array(v2_data)

    wl_data = np.array(list(data.wl) * nbl)
    wl_data_cp = np.array(list(data.wl) * ncp)

    obs = []

    for i in range(nbl * nwl):
        if flag_V2[i] & use_flag:
            pass
        else:
            if not cond_wl:
                tmp = [u_data[i], v_data[i], wl_data[i]]
                typ = "V2"
                obser = vis2_data[i]
                err = e_vis2_data[i]
                if cond_uncer:
                    if err / obser <= rel_max * 1e-2:
                        obs.append([tmp, typ, obser, err])
                    else:
                        pass
                else:
                    obs.append([tmp, typ, obser, err])

            else:
                if (wl_data[i] >= wl_min * 1e-6) & (wl_data[i] <= wl_max * 1e-6):
                    tmp = [u_data[i], v_data[i], wl_data[i]]
                    typ = "V2"
                    obser = vis2_data[i]
                    err = e_vis2_data[i]
                    if cond_uncer:
                        if err / obser <= rel_max * 1e-2:
                            obs.append([tmp, typ, obser, err])
                        else:
                            pass
                    else:
                        obs.append([tmp, typ, obser, err])
                else:
                    pass
    N_v2_rest = len(obs)

    for i in range(ncp * nwl):
        if flag_CP[i]:
            pass
        else:
            if not cond_wl:
                tmp = [
                    u1_data[i],
                    u2_data[i],
                    (u1_data[i] + u2_data[i]),
                    v1_data[i],
                    v2_data[i],
                    (v1_data[i] + v2_data[i]),
                    wl_data_cp[i],
                ]
                typ = "CP"
                obser = cp_data[i]
                err = e_cp_data[i]
                if cond_uncer:
                    if err / obser <= rel_max * 1e-2:
                        obs.append([tmp, typ, obser, err])
                    else:
                        pass
                else:
                    obs.append([tmp, typ, obser, err])
            else:
                if (wl_data_cp[i] >= wl_min * 1e-6) & (wl_data_cp[i] <= wl_max * 1e-6):
                    tmp = [
                        u1_data[i],
                        u2_data[i],
                        (u1_data[i] + u2_data[i]),
                        v1_data[i],
                        v2_data[i],
                        (v1_data[i] + v2_data[i]),
                        wl_data_cp[i],
                    ]
                    typ = "CP"
                    obser = cp_data[i]
                    err = e_cp_data[i]
                    if cond_uncer:
                        if err / obser <= rel_max * 1e-2:
                            obs.append([tmp, typ, obser, err])
                        else:
                            pass
                    else:
                        obs.append([tmp, typ, obser, err])
                else:
                    pass

    N_cp_rest = len(obs) - N_v2_rest

    Obs = np.array(obs, dtype=object)

    if verbose:
        print(f"\nTotal # of data points: {len(Obs)} ({N_v2_rest} V2, {N_cp_rest} CP)")
        if use_flag:
            print("-> Flag in oifits files used.")
        if cond_wl:
            print(
                rf"-> Restriction on wavelenght: {wl_min:2.2f} < {chr(955)} < {wl_max:2.2f} µm"
            )
        if cond_uncer:
            print(rf"-> Restriction on uncertainties: {chr(949)} < {rel_max:2.1f} %")

    return Obs


def _normalize_err_obs(obs, verbose=False):
    """Normalize errors to give V2 and CP data equal aggregate weight.

    Parameters
    ----------
    obs : array-like
        Observation rows whose last item is an uncertainty.
    verbose : bool, default=False
        Whether to print normalization details.

    Returns
    -------
    list
        Normalized uncertainties.
    """

    errs = [o[-1] for o in obs]
    techs = [("V2"), ("CP")]
    n = [0, 0, 0, 0]
    for o in obs:
        for j, t in enumerate(techs):
            if any(x in o[1] for x in t):
                n[j] += 1
    if verbose:
        print("-" * 50)
        print("error bar normalization by observable:")
        for j, t in enumerate(techs):
            print(t, n[j], np.sqrt(float(n[j]) / len(obs) * len(n)))
        print("-" * 50)

    for i, o in enumerate(obs):
        for j, t in enumerate(techs):
            if any(x in o[1] for x in t):
                errs[i] *= np.sqrt(float(n[j]) / len(obs) * len(n))
    return errs


def compute_chi2_curve(
    obs,
    name_param,
    params,
    array_params,
    fitOnly,
    normalizeErrors=False,
    fitCP=True,
    onlyCP=False,
    ymin=0,
    ymax=3,
):
    """Compute a one-dimensional reduced-chi-square curve.

    The curve estimates pessimistic, fully correlated uncertainty for one model
    parameter.

    Parameters
    ----------
    obs : array-like
        Selected fitting observations.
    name_param : str
        Parameter for which to calculate the curve.
    params : dict
        Model parameters.
    array_params : numpy.ndarray
        Parameter values used to calculate the curve.
    fitOnly : list of str
        Parameters to fit. This list is modified in place to remove ``name_param``.
    normalizeErrors : bool, default=False
        Whether to normalize V2 and CP errors.
    fitCP : bool, default=True
        Whether to fit closure-phase data.
    onlyCP : bool, default=False
        Whether to fit only closure-phase data.
    ymin : float, default=0
        Lower normalized vertical extent for uncertainty shading.
    ymax : float, default=3
        Upper normalized vertical extent for uncertainty shading.

    Returns
    -------
    tuple[dict, float] or None
        Initial fit and chi-square-curve uncertainty, or ``None`` if the curve
        bounds are insufficient.
    """
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d

    fit = smartfit(
        obs,
        params,
        normalizeErrors=normalizeErrors,
        fitCP=fitCP,
        onlyCP=onlyCP,
        fitOnly=fitOnly,
        ftol=1e-8,
        epsfcn=1e-6,
        multiproc=False,
        verbose=False,
    )

    # fit_chi2 = fit['chi2']
    fit_theta = fit["best"][name_param]
    fit_e_theta = fit["uncer"][name_param]

    fitOnly.remove(name_param)
    l_chi2r = []
    for pr in track(array_params, description=f"Chi2 curve ({name_param}"):
        params[name_param] = pr
        lfits = smartfit(
            obs,
            params,
            normalizeErrors=True,
            fitCP=fitCP,
            onlyCP=onlyCP,
            fitOnly=fitOnly,
            ftol=1e-8,
            epsfcn=1e-6,
            multiproc=False,
            verbose=False,
        )
        l_chi2r.append(lfits["chi2"])

    n_freedom = len(fitOnly)
    n_pts = len(obs)

    l_chi2r = np.array(l_chi2r)
    l_chi2 = np.array(l_chi2r) * (n_pts - (n_freedom - 1))

    chi2r_m = l_chi2r.min()
    chi2_m = l_chi2.min()

    fitted_param = array_params[l_chi2 == chi2_m].item()

    c_left = array_params <= fitted_param
    c_right = array_params >= fitted_param

    left_curve = interp1d(l_chi2r[c_left], array_params[c_left])
    right_curve = interp1d(l_chi2r[c_right], array_params[c_right])

    try:
        left_res = left_curve(chi2r_m + 1)
        right_res = right_curve(chi2r_m + 1)
        dr1_r = abs(fitted_param - left_res)
        dr2_r = abs(fitted_param - right_res)
        dr1_r, dig = roundSciDigit(dr1_r)
        dr2_r = roundSciDigit(dr2_r)[0]
        fitted_param = float(np.round(fitted_param, dig))
        print(f"sig_chi2: {name_param} = {fitted_param} - {dr1_r} + {dr2_r}")
    except ValueError:
        print("Try to increase the parameters bounds (chi2_r).")
        return None

    dr1_r = roundSciDigit(dr1_r)[0]
    dr2_r = roundSciDigit(dr2_r)[0]

    errors_chi2 = np.mean([dr1_r, dr2_r])

    fit_e_theta, scidigit = roundSciDigit(fit_e_theta)
    fit_theta = float(np.round(fit_theta, scidigit))

    plt.figure()
    plt.plot(
        array_params[c_left], l_chi2r[c_left], color="tab:blue", lw=3, alpha=1, zorder=1
    )
    plt.plot(array_params[c_right], l_chi2r[c_right], color="tab:blue", lw=3, alpha=1)
    plt.plot(
        fit_theta,
        l_chi2r.min(),
        ".",
        color="#fc5185",
        ms=10,
        label=f"fit: {name_param}={fit_theta}±{fit_e_theta}",
    )
    plt.axvspan(
        fitted_param - dr1_r,
        fitted_param + dr2_r,
        ymin=ymin,
        ymax=ymax,
        color="#dbe4e8",
        label=rf"$\sigma_{{m1}}=$-{dr1_r}/+{dr2_r}",
    )
    plt.axvspan(
        fitted_param - fit_e_theta,
        fitted_param + fit_e_theta,
        ymin=ymin,
        ymax=ymax,
        color="#359ccb",
        label=r"$\sigma_{m2}$",
        alpha=0.3,
    )
    plt.grid(alpha=0.1, color="grey")
    plt.legend(loc="best", fontsize=9)
    plt.ylabel(r"$\chi^2_{red}$")
    plt.xlim(array_params.min(), array_params.max())
    plt.tight_layout()
    plt.show(block=False)

    return fit, errors_chi2


def plot_model(
    inputdata,
    param,
    save=False,
    outputfile=None,
    extra_error_v2=0,
    extra_error_cp=0,
    err_scale=1,
    d_freedom=3,
    v2_min=None,
    v2_max=1.1,
    cp_max=None,
    unit="m",
):
    """Plot V2 and CP model predictions, data, and residuals.

    Parameters
    ----------
    inputdata : str
        OIFITS file name.
    param : dict
        Model parameters, such as the ``"best"`` result from
        :func:`amical.candid_grid`.
    save : bool, default=False
        Whether to save the figure.
    outputfile : str or None, default=None
        Filename for the saved figure.
    extra_error_v2 : float, default=0
        Additive squared-visibility uncertainty.
    extra_error_cp : float, default=0
        Additive closure-phase uncertainty.
    err_scale : float, default=1
        Multiplicative closure-phase uncertainty scale.
    d_freedom : int, default=3
        Number of model degrees of freedom.
    v2_min : float or None, default=None
        Lower squared-visibility axis limit.
    v2_max : float, default=1.1
        Upper squared-visibility axis limit.
    cp_max : float or None, default=None
        Absolute closure-phase axis limit.
    unit : {"m", "lambda", "arcsec"}, default="m"
        Spatial-frequency unit.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, float]
        Model squared visibilities, model closure phases, and reduced chi-square.
    """
    import matplotlib.pyplot as plt

    d = amical.loadc(inputdata)

    e_vis2 = np.sqrt(d.e_vis2**2 + extra_error_v2**2)
    e_cp = np.sqrt(d.e_cp**2 + extra_error_cp**2) * err_scale

    model_target = select_model(param["model"])

    u, v, wl = d.u, d.v, d.wl

    mod_v2 = comput_V2([u, v, wl], param, model_target)

    u1, u2, v1, v2 = d.u1, d.u2, d.v1, d.v2
    u3, v3 = u1 + u2, v1 + v2

    X = [u1, u2, u3, v1, v2, v3, wl]
    mod_cp = comput_CP(X, param, model_target)

    chi2_cp = np.sum((d.cp - mod_cp) ** 2 / (e_cp) ** 2) / (len(e_cp) - (d_freedom - 1))
    chi2_vis2 = np.sum((d.vis2 - mod_v2) ** 2 / (e_vis2) ** 2) / (
        len(e_vis2) - (d_freedom - 1)
    )

    chi2 = (
        np.sum((d.cp - mod_cp) ** 2 / (e_cp) ** 2)
        + np.sum((d.vis2 - mod_v2) ** 2 / (e_vis2) ** 2)
    ) / (len(e_cp) + len(e_vis2) - (d_freedom - 1))

    res_vis2 = (mod_v2 - d.vis2) / e_vis2
    res_cp = (mod_cp - d.cp) / e_cp

    f_unit = {
        "m": 1,
        "lambda": 1 / d.wl / 1e6,
        "arcsec": 1 / d.wl / ((3600 * 180) / np.pi),
    }
    label_unit = {"m": "m", "lambda": r"M$\lambda$", "arcsec": r"arcsec$^{-1}$"}

    x_vis2 = d.bl * f_unit[unit]
    x_cp = d.bl_cp * f_unit[unit]

    fig = plt.figure(constrained_layout=True, figsize=(11, 4))
    axd = fig.subplot_mosaic(
        [["vis", "cp"], ["res_vis2", "res_cp"]],
        gridspec_kw={"width_ratios": [2, 2], "height_ratios": [3, 1]},
    )

    axd["res_vis2"].sharex(axd["vis"])
    axd["res_cp"].sharex(axd["cp"])

    axd["vis"].errorbar(x_vis2, d.vis2, yerr=e_vis2, **err_pts_style, color="#3d84a8")
    axd["vis"].plot(
        x_vis2,
        mod_v2,
        "x",
        color="#f6416c",
        zorder=100,
        ms=10,
        label=rf"model ($\chi^2_r={chi2_vis2:2.1f}$)",
    )
    axd["vis"].legend()

    if v2_min is not None:
        axd["vis"].set_ylim(v2_min, v2_max)
    axd["vis"].grid(alpha=0.2)
    axd["vis"].set_ylabel(r"V$^2$")

    axd["res_vis2"].plot(x_vis2, res_vis2, ".", color="#3d84a8")
    axd["res_vis2"].axhspan(-1, 1, alpha=0.2, color="#418fde")
    axd["res_vis2"].axhspan(-2, 2, alpha=0.2, color="#8bb8e8")
    axd["res_vis2"].axhspan(-3, 3, alpha=0.2, color="#c8d8eb")
    if res_vis2.max() > 5:
        res_mas = res_vis2.max()
    else:
        res_mas = 5
    axd["res_cp"].set_ylim(-res_mas, res_mas)
    axd["res_vis2"].set_ylim(-5, 5)
    axd["res_vis2"].set_ylabel(r"Residual [$\sigma$]")
    axd["res_vis2"].set_xlabel(f"Sp. Freq. [{label_unit[unit]}]")

    axd["cp"].errorbar(x_cp, d.cp, yerr=e_cp, **err_pts_style, color="#2ca02c")
    axd["cp"].plot(
        x_cp,
        mod_cp,
        "x",
        color="#f6416c",
        zorder=100,
        ms=10,
        label=rf"model ($\chi^2_r={chi2_cp:2.1f}$)",
    )

    if cp_max is not None:
        axd["cp"].set_ylim(-cp_max, cp_max)
    axd["cp"].grid(alpha=0.2)
    axd["cp"].set_ylabel("Closure phases [deg]")
    axd["cp"].legend()

    axd["res_cp"].plot(x_cp, res_cp, ".", color="#1e7846")
    axd["res_cp"].axhspan(-1, 1, alpha=0.3, color="#28a16c")  # f5c893
    axd["res_cp"].axhspan(-2, 2, alpha=0.2, color="#28a16c")
    axd["res_cp"].axhspan(-3, 3, alpha=0.1, color="#28a16c")
    if res_cp.max() > 5:
        res_mas = res_cp.max()
    else:
        res_mas = 5
    axd["res_cp"].set_ylim(-res_mas, res_mas)
    axd["res_cp"].set_ylabel(r"Residual [$\sigma$]")
    axd["res_cp"].set_xlabel(f"Sp. Freq. [{label_unit[unit]}]")

    if save:
        import matplotlib.pyplot as plt

        filename = os.path.basename(inputdata) + "_fit_candid.pdf"
        if outputfile is not None:
            filename = outputfile
        plt.savefig(filename, dpi=300)

    return mod_v2, mod_cp, chi2


def smartfit(
    obs,
    first_guess,
    doNotFit=None,
    fitOnly=None,
    follow=None,
    multiproc=False,
    ftol=1e-4,
    epsfcn=1e-7,
    normalizeErrors=False,
    scale_err=1,
    fitCP=True,
    onlyCP=False,
    verbose=True,
):
    """Fit V2 and CP data contained in observations.

    Parameters
    ----------
    obs : numpy.ndarray
        Selected fitting observations.
    first_guess : dict
        Initial model parameters.
    doNotFit : list of str or None, default=None
        Parameters excluded from fitting.
    fitOnly : list of str or None, default=None
        Parameters to fit; defaults to all numeric parameters.
    follow : list of str or None, default=None
        Parameters to print in verbose mode.
    multiproc : bool, default=False
        Whether to evaluate models using multiprocessing.
    ftol : float, default=0.0001
        Relative fit-tolerance threshold.
    epsfcn : float, default=1e-07
        Finite-difference step used by the optimizer.
    normalizeErrors : bool, default=False
        Whether to normalize V2 and CP errors.
    scale_err : float, default=1
        Multiplicative scale for fitting uncertainties.
    fitCP : bool, default=True
        Whether to fit closure-phase data.
    onlyCP : bool, default=False
        Whether to fit only closure-phase data.
    verbose : bool, default=True
        Whether to print fitting information.

    Returns
    -------
    dict
        Result returned by :func:`amical.dpfit.leastsqFit`.
    """

    save_obs = obs.copy()

    if onlyCP:
        cond = save_obs[:, 1] == "CP"
        obs = save_obs[cond]

    if not fitCP and not onlyCP:
        cond = save_obs[:, 1] == "V2"
        obs = save_obs[cond]

    errs = [o[-1] for o in obs]
    if normalizeErrors:
        errs = _normalize_err_obs(obs, verbose=False)

    # -- avoid fitting string parameters
    tmp = list(filter(lambda x: isinstance(first_guess[x], str), first_guess.keys()))

    if len(tmp) > 0:
        if doNotFit is None:
            doNotFit = tmp
        else:
            doNotFit.extend(tmp)
        try:
            fitOnly = list(
                filter(lambda x: not isinstance(first_guess[x], str), fitOnly)
            )
        except Exception:
            pass

    if multiproc:
        M = model_parallelized
    else:
        M = model_standard

    lfit = leastsqFit(
        M,
        obs,
        first_guess,
        [o[2] for o in obs],
        err=np.array(errs) * scale_err,
        doNotFit=doNotFit,
        fitOnly=fitOnly,
        follow=follow,
        normalizedUncer=False,
        verbose=verbose,
        ftol=ftol,
        epsfcn=epsfcn,
    )

    return lfit
