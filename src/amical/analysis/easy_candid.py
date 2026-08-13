import os

import numpy as np
from rich import print as rprint

from amical.externals import candid


def candid_grid(
    input_data: str | list[str],
    step: int = 10,
    rmin: float = 20,
    rmax: float = 400,
    diam: float = 0,
    obs: list[str] | None = None,
    extra_error_cp: float = 0,
    err_scale: float = 1,
    extra_error_v2: float = 0,
    instruments=None,
    doNotFit=None,
    ncore: int = 1,
    save: bool = False,
    outputfile: str | None = None,
    verbose: bool = False,
):
    """Fit a binary model with the CANDID analysis package.

    Parameters
    ----------
    input_data : str or list of str
        OIFITS file name or names.
    step : int, default=10
        Grid-position step.
    rmin : float, default=20
        Minimum grid separation in milliarcseconds.
    rmax : float, default=400
        Maximum grid separation in milliarcseconds.
    diam : float, default=0
        Primary-star diameter in milliarcseconds.
    obs : list of str or None, default=None
        Observables to fit. Uses ``["cp", "v2"]`` when omitted.
    extra_error_cp : float, default=0
        Additive closure-phase uncertainty.
    err_scale : float, default=1
        Multiplicative error-bar scale.
    extra_error_v2 : float, default=0
        Additive squared-visibility uncertainty.
    instruments : optional
        Instrument selection passed to CANDID.
    doNotFit : list of str or None, default=None
        Parameters excluded from fitting. Uses ``["diam*"]`` when omitted.
    ncore : int, default=1
        Number of CANDID worker processes.
    save : bool, default=False
        Whether to save the detection map.
    outputfile : str or None, default=None
        Filename for the saved detection map.
    verbose : bool, default=False
        Whether to print CANDID information.

    Returns
    -------
    dict
        Best-fit parameters, uncertainties, reduced chi-square, detection
        significance, and CANDID companion results.
    """
    from uncertainties import ufloat, umath

    if obs is None:
        obs = ["cp", "v2"]
    if doNotFit is None:
        doNotFit = ["diam*"]

    rprint("[green] | --- Start CANDID fitting --- :")
    o = candid.Open(
        input_data,
        extra_error=extra_error_cp,
        err_scale=err_scale,
        extra_error_v2=extra_error_v2,
        instruments=instruments,
    )

    o.observables = obs

    o.fitMap(
        rmax=rmax,
        rmin=rmin,
        ncore=ncore,
        fig=0,
        step=step,
        addParam={"diam*": diam},
        doNotFit=doNotFit,
        verbose=verbose,
    )

    if save:
        import matplotlib.pyplot as plt

        if isinstance(input_data, list):
            first_input = input_data[0]
        else:
            first_input = input_data
        filename = os.path.basename(first_input) + "_detection_map_candid.pdf"
        if outputfile is not None:
            filename = outputfile
        plt.savefig(filename, dpi=300)

    fit = o.bestFit["best"]
    e_fit = o.bestFit["uncer"]
    chi2 = o.bestFit["chi2"]
    nsigma = o.bestFit["nsigma"]

    f = fit["f"] / 100.0
    e_f = e_fit["f"] / 100.0
    if (e_f < 0) or (e_fit["x"] < 0) or (e_fit["y"] < 0):
        print("Warning: error dm is negative.")
        e_f = abs(e_f)
        e_fit["x"] = 0
        e_fit["y"] = 0

    f_u = ufloat(f, e_f)
    x, y = fit["x"], fit["y"]
    x_u = ufloat(x, e_fit["x"])
    y_u = ufloat(y, e_fit["y"])

    dm = -2.5 * umath.log10(f_u)
    s = (x_u**2 + y_u**2) ** 0.5
    posang = umath.atan2(x_u, y_u) * 180 / np.pi
    if posang.nominal_value < 0:
        posang = 360 + posang

    cr = 1 / f_u
    rprint(
        f"[cyan]\nResults binary fit (χ2 = {chi2:2.1f}, nσ = {nsigma:2.1f}):\n"
        "-------------------"
    )

    print(f"Sep = {s.nominal_value:2.1f} +/- {s.std_dev:2.1f} mas")
    print(f"Theta = {posang.nominal_value:2.1f} +/- {posang.std_dev:2.1f} deg")
    print(f"CR = {cr.nominal_value:2.1f} +/- {cr.std_dev:2.1f}")
    print(f"dm = {dm.nominal_value:2.2f} +/- {dm.std_dev:2.2f}")
    res = {
        "best": {
            "model": "binary_res",
            "dm": dm.nominal_value,
            "theta": posang.nominal_value,
            "sep": s.nominal_value,
            "diam": fit["diam*"],
            "x0": 0,
            "y0": 0,
        },
        "uncer": {"dm": dm.std_dev, "theta": posang.std_dev, "sep": s.std_dev},
        "chi2": chi2,
        "nsigma": nsigma,
        "comp": o.bestFit["best"],
    }

    return res


def candid_cr_limit(
    input_data: str | list[str],
    step: int = 10,
    rmin: float = 20,
    rmax: float = 400,
    extra_error_cp: float = 0,
    err_scale: float = 1,
    extra_error_v2: float = 0,
    obs=None,
    fitComp=None,
    ncore: int = 1,
    diam=None,
    methods=None,
    instruments=None,
    save: bool = False,
    outputfile=None,
):
    if obs is None:
        obs = ["cp", "v2"]
    if methods is None:
        methods = ["injection"]

    rprint("[green] | --- Start CANDID contrast limit --- :")
    o = candid.Open(
        input_data,
        extra_error=extra_error_cp,
        err_scale=err_scale,
        extra_error_v2=extra_error_v2,
        instruments=instruments,
    )
    o.observables = obs

    res = o.detectionLimit(
        rmin=rmin,
        rmax=rmax,
        step=step,
        drawMaps=True,
        fratio=1,
        methods=methods,
        removeCompanion=fitComp,
        ncore=ncore,
        diam=diam,
    )

    if save:
        import matplotlib.pyplot as plt

        if isinstance(input_data, list):
            first_input = input_data[0]
        else:
            first_input = input_data
        filename = os.path.basename(first_input) + "_lim_detection_candid.pdf"
        if outputfile is not None:
            filename = outputfile
        plt.savefig(filename, dpi=300)
    return res
