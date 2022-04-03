import os
from typing import List
from typing import Optional
from typing import Union

import numpy as np
from matplotlib import pyplot as plt
from termcolor import cprint
from uncertainties import ufloat
from uncertainties import umath

from amical.externals import candid


def candid_grid(
    input_data: Union[str, List[str]],
    step: int = 10,
    rmin: float = 20,
    rmax: float = 400,
    diam: float = 0,
    obs: Optional[List[str]] = None,
    extra_error_cp: float = 0,
    err_scale: float = 1,
    extra_error_v2: float = 0,
    instruments=None,
    doNotFit=None,
    ncore: int = 1,
    save: bool = False,
    outputfile: Optional[str] = None,
    verbose: bool = False,
):
    """This function is an user friendly interface between the users of amical
    pipeline and the CANDID analysis package (https://github.com/amerand/CANDID).

    Parameters:
    -----------
    `input_data`:
        oifits file names or list of oifits files,\n
    `step`:
        step used to compute the binary grid positions,\n
    `rmin`, `rmax`:
        Bounds of the grid [mas],\n
    `diam`:
        Stellar diameter of the primary star [mas] (default=0),\n
    `obs`:
        List of observables to be fitted (default: ['cp', 'v2']),\n
    `doNotFit`:
        Parameters not fitted (default: ['diam*']),\n
    `verbose`:
        print some informations {default: False}.

    Outputs:
    --------
    `res` {dict}:
        Dictionnary of the results ('best'), uncertainties ('uncer'),
        reduced chi2 ('chi2') and sigma detection ('nsigma').
    """
    if obs is None:
        obs = ["cp", "v2"]
    if doNotFit is None:
        doNotFit = ["diam*"]

    cprint(" | --- Start CANDID fitting --- :", "green")
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
    cprint(f"\nResults binary fit (χ2 = {chi2:2.1f}, nσ = {nsigma:2.1f}):", "cyan")
    cprint("-------------------", "cyan")

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
    input_data: Union[str, List[str]],
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

    cprint(" | --- Start CANDID contrast limit --- :", "green")
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
        if isinstance(input_data, list):
            first_input = input_data[0]
        else:
            first_input = input_data
        filename = os.path.basename(first_input) + "_lim_detection_candid.pdf"
        if outputfile is not None:
            filename = outputfile
        plt.savefig(filename, dpi=300)
    return res
