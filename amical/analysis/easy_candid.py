import numpy as np
from matplotlib import pyplot as plt
from termcolor import cprint
from uncertainties import ufloat, umath

from amical.analysis import candid


def fit_binary(input_data, step=10, rmin=20, rmax=400, diam=0, obs=['cp', 'v2'],
               doNotFit=['diam*', ], ncore=1, verbose=False):
    """ This function is an user friendly interface between the users of amical
    pipeline and the CANDID analysis package (https://github.com/amerand/CANDID).

    Parameters:
    -----------
    `input_data` {str or list}:
        oifits file names or list of oifits files,\n
    `step` {int}:
        step used to compute the binary grid positions,\n
    `rmin`, `rmax` {float}:
        Bounds of the grid [mas],\n
    `diam` {float}:
        Stellar diameter of the primary star [mas] (default=0),\n
    `obs` {list}:
        List of observables to be fitted (default: ['cp', 'v2']),\n
    `doNotFit` {list}:
        Parameters not fitted (default: ['diam*']),\n
    `verbose` {boolean}:
        print some informations {default: False}.

    Outputs:
    --------
    `res` {dict}:
        Dictionnary of the results ('best'), uncertainties ('uncer'),
        reduced chi2 ('chi2') and sigma detection ('nsigma').
    """

    cprint(' | --- Start CANDID fitting --- :', 'green')
    o = candid.Open(input_data)

    o.observables = obs

    ifig = plt.gcf().number + 1
    o.fitMap(rmax=rmax, rmin=rmin, ncore=ncore, fig=ifig,
             step=step, addParam={"diam*": diam}, doNotFit=doNotFit, verbose=verbose)

    fit = o.bestFit["best"]
    e_fit = o.bestFit["uncer"]
    chi2 = o.bestFit['chi2']
    nsigma = o.bestFit['nsigma']

    f = fit["f"] / 100.0
    e_f = e_fit["f"] / 100.0

    f_u = ufloat(f, e_f)
    x, y = fit["x"], fit["y"]
    x_u = ufloat(x, e_fit["x"])
    y_u = ufloat(y, e_fit["y"])

    dm = umath.log(1 / (f_u)) / umath.log(2.5)
    s = (x_u ** 2 + y_u ** 2) ** 0.5
    posang = ((umath.atan2(x_u, y_u)*180/np.pi))
    if posang.nominal_value < 0:
        posang = 360 + posang

    cprint("\nResults binary fit (χ2 = %2.1f, nσ = %2.1f):" %
           (chi2, nsigma), "cyan")
    cprint("-------------------", "cyan")
    print("Sep = %2.1f +/- %2.1f mas" % (s.nominal_value, s.std_dev))
    print("Theta = %2.1f +/- %2.1f deg" %
          (posang.nominal_value, posang.std_dev))
    print("dm = %2.2f +/- %2.2f" % (dm.nominal_value, dm.std_dev))
    res = {'best': {'model': 'binary',
                    'dm': dm.nominal_value,
                    'theta': posang.nominal_value,
                    'sep': s.nominal_value,
                    'x0': 0,
                    'y0': 0},
           'uncer': {'dm': dm.std_dev,
                     'theta': posang.std_dev,
                     'sep': s.std_dev},
           'chi2': chi2,
           'nsigma': nsigma,
           'p': o.bestFit["best"]
           }

    return res  # dict2class(res)


def getContrastLimit(input_data, step=10, rmin=20, rmax=400, diam=0, obs=['cp', 'v2'],
                     fitComp=None, ncore=1, methods=['injection']):
    cprint(' | --- Start CANDID contrast limit --- :', 'green')
    o = candid.Open(input_data)
    o.observables = obs

    ifig = plt.gcf().number + 1
    res = o.detectionLimit(fig=ifig, rmin=rmin, rmax=rmax, step=step, drawMaps=True,
                           fratio=1, methods=methods, removeCompanion=fitComp,
                           ncore=ncore)
    return res
