"""
Legacy CANDID and Pymask interface, computed with virgil.

AMICAL used to bundle CANDID (A. Merand & A. Gallenne) and Pymask (B. Pope &
A. Cheetham) to analyse calibrated OIFITS files. The functions below keep the
names, arguments and return values of the AMICAL wrappers around them, but do
the work with virgil (https://benjaminpope.github.io/virgil/), so that existing
scripts keep running:

- `candid_grid`: grid search and least-squares fit of a binary (primary of
  uniform-disk diameter `diam` plus a point-source companion);
- `candid_cr_limit`: detection limits (Absil et al. 2011) around the target,
  optionally after removing a fitted companion;
- `pymask_grid`: chi2 grid of a binary in separation, position angle and
  contrast ratio, on the closure phases only;
- `pymask_mcmc`: posterior of the same binary, sampled with NUTS (HMC);
- `pymask_cr_limit`: contrast limits versus separation, on the closure phases
  only.

The numbers are close to, but not identical with, those of the original
packages: CANDID's "injection" limits and Pymask's Monte Carlo limits are
replaced by the Absil limits (both give the companions that would have been
detected at 3 sigma), Pymask's emcee is replaced by NUTS, and virgil whitens
the correlated closure phases together, so uncertainties no longer need an
`err_scale` for their redundancy (and are wider than Pymask's were). For new
work, use virgil directly (see the companion search tutorial in the AMICAL
documentation).

virgil needs Python >= 3.11 and JAX: `pip install amical[virgil]`.
"""

import os
import warnings

import numpy as np
from rich import print as rprint


def _virgil():
    """Import virgil, or explain how to install it."""
    try:
        import virgil  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "The CANDID and Pymask functions of AMICAL are computed with virgil, "
            "which is not installed. Install it with `pip install amical[virgil]` "
            "(or `pip install virgil-astro`; it needs Python >= 3.11)."
        ) from e


def _as_list(input_data):
    if isinstance(input_data, list | tuple):
        return [str(f) for f in input_data]
    return [str(input_data)]


def _load(
    input_data,
    obs=("cp", "v2"),
    extra_error_cp=0.0,
    err_scale=1.0,
    extra_error_v2=0.0,
    instruments=None,
    remove=None,
):
    """Read OIFITS files into virgil, with the error conventions of CANDID and
    Pymask: closure-phase errors are increased by `extra_error_cp` (deg) in
    quadrature then multiplied by `err_scale`; V2 errors are increased by
    `extra_error_v2` in quadrature. Observables not in `obs` are left out.
    `remove` is a model whose companion is subtracted from the data, as
    CANDID's `removeCompanion`."""
    from virgil.oidata import OIData
    from virgil.oifits import read_oifits

    record = dict(read_oifits(_as_list(input_data), insname=instruments))
    d_phi = np.asarray(record["d_phi"], dtype=float)
    record["d_phi"] = np.hypot(d_phi, np.deg2rad(extra_error_cp)) * err_scale
    record["d_vis"] = np.hypot(np.asarray(record["d_vis"], float), extra_error_v2)

    if remove is not None:
        record = _subtract_companion(record, *remove)

    for name, key in (("v2", "vis_flag"), ("cp", "phi_flag")):
        if name not in obs:
            size = np.size(record["vis" if name == "v2" else "phi"])
            record[key] = np.ones(size, dtype=bool)
    return OIData(record)


def _observables(record, model):
    """V2 and closure phases (rad) of `model` on the samples of `record`."""
    cvis = np.asarray(model.model(record["u"], record["v"], record["wavel"]))
    cvis = cvis / np.asarray(
        model.model(0.0 * record["u"], 0.0 * record["v"], record["wavel"])
    )
    v2 = np.abs(cvis) ** 2
    phase = np.angle(cvis)
    i1, i2, i3 = (np.asarray(record[k]) for k in ("i_cps1", "i_cps2", "i_cps3"))
    cp = phase[i1] + phase[i2] - phase[i3]
    return v2, cp


def _subtract_companion(record, with_companion, without_companion):
    """Subtract a companion from the data, to first order (as CANDID does)."""
    v2_b, cp_b = _observables(record, with_companion)
    v2_s, cp_s = _observables(record, without_companion)
    record = dict(record)
    record["vis"] = np.asarray(record["vis"], float) - (v2_b - v2_s)
    record["phi"] = np.asarray(record["phi"], float) - np.angle(
        np.exp(1j * (cp_b - cp_s))
    )
    return record


def _chi2(model, data):
    from virgil.likelihood import whitened_residuals

    r = np.asarray(whitened_residuals(model, data))
    return float(np.sum(r**2)), r.size


def _binary(diam, dra=0.0, ddec=0.0, flux=0.0):
    """Primary (uniform disk of diameter `diam` in mas, or a point source)
    and a point-source companion."""
    from virgil.models import PointSource, System, UniformDisk

    primary = UniformDisk(diam=diam) if diam > 0 else PointSource()
    companion = PointSource(flux=flux, dra=dra, ddec=ddec)
    return System(primary=primary, companion=companion)


def _square_grid(rmax, step):
    """Grid axis of CANDID's maps: N = ceil(2 rmax / step) points in
    [-rmax, rmax] (mas)."""
    n = max(int(np.ceil(2 * rmax / step)), 2)
    return np.linspace(-rmax, rmax, n)


def _resolution_mas(data):
    """Smallest spatial scale lambda / (2 B_max) of the data, in mas."""
    b = np.hypot(np.asarray(data.u), np.asarray(data.v)).max()
    wl = np.min(np.asarray(data.wavel))
    return np.rad2deg(wl / (2 * b)) * 3.6e6


def _savefig(save, outputfile, input_data, suffix):
    if not save:
        return
    import matplotlib.pyplot as plt

    filename = outputfile
    if filename is None:
        filename = os.path.basename(_as_list(input_data)[0]) + suffix
    plt.savefig(filename, dpi=300)


def _sep_pa_dm(dra, ddec, flux, cov=None):
    """Separation (mas), position angle (deg, east of north) and magnitude
    difference of a companion, with first-order uncertainties from the
    covariance of (dra, ddec, flux)."""
    sep = np.hypot(dra, ddec)
    pa = np.rad2deg(np.arctan2(dra, ddec)) % 360
    dm = -2.5 * np.log10(flux)
    if cov is None:
        return (sep, pa, dm), (np.nan, np.nan, np.nan)
    jac = np.array(
        [
            [dra / sep, ddec / sep, 0.0],
            [np.rad2deg(ddec / sep**2), np.rad2deg(-dra / sep**2), 0.0],
            [0.0, 0.0, -2.5 / (np.log(10) * flux)],
        ]
    )
    err = np.sqrt(np.abs(np.diag(jac @ cov @ jac.T)))
    return (float(sep), float(pa), float(dm)), tuple(float(e) for e in err)


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
    """Fit a binary model to the calibrated data, CANDID style (computed with
    virgil).

    The log-likelihood of a companion is mapped on a grid of offsets between
    `rmin` and `rmax` (no coarser than `step`), and the best grid points are
    refined by least squares (``virgil.fitting.fit``). The primary is a
    uniform disk of diameter `diam`, fitted too unless ``"diam*"`` is in
    `doNotFit`. Uncertainties come from the Fisher matrix.

    Parameters
    ----------
    input_data : str or list of str
        OIFITS file name or names.
    step : int, default=10
        Grid step in milliarcseconds.
    rmin : float, default=20
        Minimum grid separation in milliarcseconds.
    rmax : float, default=400
        Maximum grid separation in milliarcseconds.
    diam : float, default=0
        Primary-star diameter in milliarcseconds.
    obs : list of str or None, default=None
        Observables to fit. Uses ``["cp", "v2"]`` when omitted.
    extra_error_cp : float, default=0
        Additional closure-phase uncertainty in degrees, added in quadrature.
    err_scale : float, default=1
        Multiplicative closure-phase error-bar scale.
    extra_error_v2 : float, default=0
        Additional squared-visibility uncertainty, added in quadrature.
    instruments : str or list of str, optional
        INSNAME(s) of the OIFITS tables to use. Uses all tables when omitted.
    doNotFit : list of str or None, default=None
        Parameters excluded from fitting. Uses ``["diam*"]`` when omitted.
    ncore : int, default=1
        Unused; kept for compatibility (virgil vectorizes the grid).
    save : bool, default=False
        Whether to save the detection map as a PDF.
    outputfile : str or None, default=None
        Filename for the saved detection map.
    verbose : bool, default=False
        Whether to print more information.

    Returns
    -------
    dict
        Best-fit parameters (``"best"``), uncertainties (``"uncer"``), reduced
        chi-square (``"chi2"``), detection significance (``"nsigma"``) and the
        companion in CANDID format (``"comp"``: x, y in mas, f in per cent,
        diam* in mas).
    """
    _virgil()
    import jax.numpy as jnp
    import numpyro.distributions as dist
    from virgil.fitting import fit
    from virgil.grid_fit import likelihood_grid
    from virgil.inference import fisher
    from virgil.limits import nsigma as _nsigma
    from virgil.plotting import plot_grid_map

    obs = ["cp", "v2"] if obs is None else obs
    doNotFit = ["diam*"] if doNotFit is None else doNotFit
    fit_diam = "diam*" not in doNotFit

    rprint("[green] | --- Start CANDID-style fitting (virgil) --- :")
    data = _load(
        input_data, obs, extra_error_cp, err_scale, extra_error_v2, instruments
    )

    axis = _square_grid(rmax, min(step, _resolution_mas(data)))
    samples = {
        "companion.dra": jnp.asarray(axis),
        "companion.ddec": jnp.asarray(axis),
        "companion.flux": jnp.asarray(10 ** np.linspace(-4, 0, 41)),
    }
    loglike = np.array(likelihood_grid(data, _binary(diam), samples))
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    rr = np.hypot(xx, yy)
    loglike[(rr < rmin) | (rr > rmax)] = -np.inf
    best_map = loglike.max(axis=2)

    # Refine the best few local maxima of the map, as CANDID fits from many
    # starting points: masking data are often multimodal.
    order = np.argsort(best_map, axis=None)[::-1]
    starts: list[tuple[float, float, float]] = []
    for k in order:
        i, j = np.unravel_index(k, best_map.shape)
        if not np.isfinite(best_map[i, j]):
            break
        if all(
            np.hypot(xx[i, j] - x, yy[i, j] - y) > 2 * _resolution_mas(data)
            for x, y, _ in starts
        ):
            flux = float(samples["companion.flux"][np.argmax(loglike[i, j])])
            starts.append((float(xx[i, j]), float(yy[i, j]), flux))
        if len(starts) == 5:
            break

    priors = {
        "companion.dra": dist.Uniform(-1.5 * rmax, 1.5 * rmax),
        "companion.ddec": dist.Uniform(-1.5 * rmax, 1.5 * rmax),
        "companion.flux": dist.Uniform(0.0, 1.0),
    }
    if fit_diam:
        priors["primary.diam"] = dist.Uniform(0.0, 4 * _resolution_mas(data))

    results = [
        fit(_binary(max(diam, 1e-3) if fit_diam else diam, x, y, f), priors, data)
        for x, y, f in starts
    ]
    result = min(results, key=lambda r: r.info["loss"])
    v = result.values
    best_diam = float(v.get("primary.diam", diam))
    dra, ddec, flux = (float(v[f"companion.{k}"]) for k in ("dra", "ddec", "flux"))

    params = ["companion.dra", "companion.ddec", "companion.flux"]
    model = _binary(best_diam, dra, ddec, flux)
    cov = np.linalg.inv(np.asarray(fisher([dra, ddec, flux], params, data, model)))
    (sep, pa, dm), (e_sep, e_pa, e_dm) = _sep_pa_dm(dra, ddec, flux, cov)

    chi2_bin, ndata = _chi2(model, data)
    chi2_single, _ = _chi2(_binary(best_diam), data)
    chi2r_bin, chi2r_single = chi2_bin / (ndata - 1), chi2_single / (ndata - 1)
    nsig = float(_nsigma(chi2r_single, chi2r_bin, ndata - 1))

    cr, e_cr = 1 / flux, cov[2, 2] ** 0.5 / flux**2
    rprint(
        f"[cyan]\nResults binary fit (χ2 = {chi2r_bin:2.1f}, nσ = {nsig:2.1f}):\n"
        "-------------------"
    )
    print(f"Sep = {sep:2.1f} +/- {e_sep:2.1f} mas")
    print(f"Theta = {pa:2.1f} +/- {e_pa:2.1f} deg")
    print(f"CR = {cr:2.1f} +/- {e_cr:2.1f}")
    print(f"dm = {dm:2.2f} +/- {e_dm:2.2f}")

    plot_grid_map(
        best_map,
        {"dra": axis, "ddec": axis},
        best={"dra": dra, "ddec": ddec},
        label="Max log-likelihood over flux",
    )
    _savefig(save, outputfile, input_data, "_detection_map_candid.pdf")

    return {
        "best": {
            "model": "binary_res",
            "dm": dm,
            "theta": pa,
            "sep": sep,
            "diam": best_diam,
            "x0": 0,
            "y0": 0,
        },
        "uncer": {"dm": e_dm, "theta": e_pa, "sep": e_sep},
        "chi2": chi2r_bin,
        "nsigma": nsig,
        "comp": {"x": dra, "y": ddec, "f": 100 * flux, "diam*": best_diam},
    }


def _radial_limit(r, dmag, rmin, rmax, width):
    """Radial profile of a limit map in CANDID's way: points sorted by
    separation, and at each one the 99th percentile of the flux limit within
    +/- width/2, in magnitudes."""
    order = np.argsort(r)
    r, dmag = r[order], dmag[order]
    keep = (r >= rmin) & (r <= rmax) & np.isfinite(dmag)
    r, dmag = r[keep], dmag[keep]
    flux = 10 ** (-0.4 * dmag)
    prof = np.array([np.percentile(flux[np.abs(r - ri) < width / 2], 99) for ri in r])
    return r, -2.5 * np.log10(prof)


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
    """Compute 3-sigma detection limits around the target, CANDID style
    (computed with virgil).

    At each position of a map of step `step` out to `rmax`, the companion flux
    ruled out at 3 sigma is found by the method of Absil et al. (2011)
    (``virgil.limits.absil_limits``). CANDID's "injection" method is not
    available: every requested method returns the Absil limits. A companion
    fitted with `candid_grid` (its ``"comp"`` entry) can be removed first with
    `fitComp`.

    Parameters
    ----------
    input_data : str or list of str
        OIFITS file name or names.
    step : int, default=10
        Map step in milliarcseconds.
    rmin : float, default=20
        Minimum separation in milliarcseconds.
    rmax : float, default=400
        Maximum separation in milliarcseconds.
    extra_error_cp : float, default=0
        Additional closure-phase uncertainty in degrees, added in quadrature.
    err_scale : float, default=1
        Multiplicative closure-phase error-bar scale.
    extra_error_v2 : float, default=0
        Additional squared-visibility uncertainty, added in quadrature.
    obs : list of str or None, default=None
        Observables to use. Defaults to ``["cp", "v2"]``.
    fitComp : dict or None, default=None
        Companion (``candid_grid(...)["comp"]``) removed from the data before
        estimating the limits.
    ncore : int, default=1
        Unused; kept for compatibility.
    diam : float or None, default=None
        Primary-star diameter in milliarcseconds.
    methods : list of str or None, default=None
        Names under which the limits are returned. Defaults to
        ``["injection"]``; all give the Absil limits.
    instruments : str or list of str, optional
        INSNAME(s) of the OIFITS tables to use. Uses all tables when omitted.
    save : bool, default=False
        Whether to save the detection-limit map as a PDF.
    outputfile : str or None, default=None
        Filename for the saved detection-limit map.

    Returns
    -------
    dict
        Separations (``"r"``, mas), one entry per method, and ``"cr_limit"``:
        the 3-sigma magnitude-difference limit (99th percentile around each
        separation).
    """
    _virgil()
    import jax.numpy as jnp
    from virgil.limits import absil_limits, flux_to_delta_mag
    from virgil.plotting import plot_grid_map

    obs = ["cp", "v2"] if obs is None else obs
    methods = ["injection"] if methods is None else methods

    rprint("[green] | --- Start CANDID-style contrast limit (virgil) --- :")
    if diam is None:
        diam = 0 if fitComp is None else fitComp.get("diam*", 0)
    remove = None
    if fitComp is not None:
        comp = (fitComp["x"], fitComp["y"], abs(fitComp["f"]) / 100.0)
        remove = (_binary(diam, *comp), _binary(diam))
    data = _load(
        input_data, obs, extra_error_cp, err_scale, extra_error_v2, instruments, remove
    )

    axis = _square_grid(rmax, step)
    samples = {
        "companion.dra": jnp.asarray(axis),
        "companion.ddec": jnp.asarray(axis),
        "companion.flux": jnp.asarray(10 ** np.linspace(-4, 0, 21)),
    }
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "absil_limits", RuntimeWarning)
        flux = np.asarray(absil_limits(data, _binary(diam), samples, sigma=3.0))
    dmag = np.asarray(flux_to_delta_mag(flux))

    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    r, cr_limit = _radial_limit(
        np.hypot(xx, yy).ravel(), dmag.ravel(), rmin, rmax, rmax / len(axis)
    )

    plot_grid_map(
        flux, {"dra": axis, "ddec": axis}, kind="limit", units="delta_mag", sigma=3.0
    )
    _savefig(save, outputfile, input_data, "_lim_detection_candid.pdf")

    res = {"r": r}
    for m in methods:
        res[m] = cr_limit
    res["cr_limit"] = cr_limit
    return res


def _polar_chi2_grid(data, seps, ths, cons):
    """chi2 of a binary over separation (mas), position angle (deg) and
    contrast ratio (primary/companion)."""
    import jax.numpy as jnp
    from virgil.grid_fit import likelihood_grid
    from virgil.likelihood import model_loglike
    from virgil.models import BinaryModelAngular

    samples = {
        "sep": jnp.asarray(seps),
        "pa": jnp.asarray(ths),
        "flux": jnp.asarray(1.0 / np.asarray(cons)),
    }
    loglike = np.asarray(likelihood_grid(data, BinaryModelAngular, samples))
    # log L = -chi2/2 + const: get the constant from one model.
    ref = BinaryModelAngular(seps[0], ths[0], 1.0 / cons[0])
    chi2_ref, ndata = _chi2(ref, data)
    const = float(model_loglike(ref, data)) + chi2_ref / 2
    return -2 * (loglike - const), ndata


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
    """Compute a chi-square map of a binary model on a regular grid, Pymask
    style (closure phases only; computed with virgil).

    Parameters
    ----------
    input_data : str
        OIFITS file name.
    ngrid : int, default=40
        Number of grid points for each of position angle, separation and
        contrast ratio (``ngrid**3`` models).
    pa_prior : list of float or None, default=None
        Position-angle bounds in degrees. Defaults to ``[0, 360]``.
    sep_prior : list of float or None, default=None
        Separation bounds in milliarcseconds. Defaults to ``[0, 100]``.
    cr_prior : list of float or None, default=None
        Contrast-ratio bounds. Defaults to ``[1, 150]``.
    err_scale : float, default=1.0
        Multiplicative error-bar scale.
    extra_error_cp : float, default=0.0
        Additional closure-phase uncertainty in degrees.
    ncore : int, default=1
        Unused; kept for compatibility.
    verbose : bool, default=False
        Whether to print more information.

    Returns
    -------
    dict
        ``"chi2"`` and ``"like"`` arrays of shape (sep, pa, cr), the axes
        ``"seps"``, ``"ths"`` and ``"cons"``, and the ``"best"`` [sep, pa, cr].
    """
    _virgil()
    pa_prior = [0, 360] if pa_prior is None else pa_prior
    sep_prior = [0, 100] if sep_prior is None else sep_prior
    cr_prior = [1, 150] if cr_prior is None else cr_prior

    data = _load(input_data, ("cp",), extra_error_cp, err_scale)
    seps = np.linspace(sep_prior[0], sep_prior[1], ngrid)
    ths = np.linspace(pa_prior[0], pa_prior[1], ngrid)
    cons = np.linspace(cr_prior[0], cr_prior[1], ngrid)
    chi2, ndata = _polar_chi2_grid(data, seps, ths, cons)

    i, j, k = np.unravel_index(np.argmin(chi2), chi2.shape)
    best = np.array([seps[i], ths[j], cons[k]])
    print(f"\nMaximum likelihood estimation (χ2={chi2.min() / ndata:2.1f}):")
    print("-------------------------------")
    print(f"Separation = {best[0]:2.2f} mas")
    print(f"PA = {best[1]:2.2f} deg")
    print(f"Contrast Ratio = {best[2]:2.1f} ({2.5 * np.log10(best[2]):2.1f} mag)\n")

    temp_chi2 = ndata * chi2 / chi2.min()
    like = np.exp(-(temp_chi2 - ndata) / 2)
    return {
        "chi2": chi2,
        "like": like,
        "seps": seps,
        "ths": ths,
        "cons": cons,
        "best": best,
    }


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
    """Sample the posterior of a binary model, Pymask style (closure phases
    only), with NUTS (numpyro and virgil).

    The priors are uniform over `sep_prior`, `pa_prior` and `cr_prior`.
    Pymask used emcee; the NUTS sampler needs no walkers.

    Parameters
    ----------
    input_data : str
        OIFITS file name.
    initial_guess : array_like
        Initial separation (mas), position angle (deg) and contrast ratio.
    niters : int, default=1000
        Number of samples after warm-up.
    pa_prior : list of float or None, default=None
        Position-angle bounds in degrees. Defaults to ``[0, 360]``.
    sep_prior : list of float or None, default=None
        Separation bounds in milliarcseconds. Defaults to 0.5 to 2 times the
        initial separation.
    cr_prior : list of float or None, default=None
        Contrast-ratio bounds. Defaults to 1 to 10 times the initial contrast
        ratio.
    err_scale : float, default=1
        Multiplicative error-bar scale.
    extra_error_cp : float, default=0
        Additional closure-phase uncertainty in degrees.
    ncore : int, default=1
        Unused; kept for compatibility.
    burn_in : int, default=500
        Number of warm-up steps.
    walkers : int, default=100
        Unused; kept for compatibility.
    display : bool, default=True
        Whether to plot the sample chains.
    verbose : bool, default=True
        Whether to show a progress bar and print the fitted parameters.

    Returns
    -------
    dict
        Best-fit binary parameters (``"best"``) and asymmetric uncertainties
        (``"uncer"``).
    """
    _virgil()
    import jax
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    from numpyro.infer.initialization import init_to_value
    from virgil.likelihood import loglike
    from virgil.models import BinaryModelAngular

    s0, pa0, cr0 = (float(x) for x in initial_guess)
    pa_prior = [0, 360] if pa_prior is None else pa_prior
    sep_prior = [0.5 * s0, 2 * s0] if sep_prior is None else sep_prior
    cr_prior = [1, 10 * cr0] if cr_prior is None else cr_prior

    data = _load(input_data, ("cp",), extra_error_cp, err_scale)
    params = ["sep", "pa", "flux"]

    def model():
        sep = numpyro.sample("sep", dist.Uniform(*sep_prior))
        pa = numpyro.sample("pa", dist.Uniform(*pa_prior))
        cr = numpyro.sample("cr", dist.Uniform(*cr_prior))
        numpyro.factor(
            "loglike", loglike([sep, pa, 1.0 / cr], params, data, BinaryModelAngular)
        )

    kernel = NUTS(
        model, init_strategy=init_to_value(values={"sep": s0, "pa": pa0, "cr": cr0})
    )
    mcmc = MCMC(
        kernel,
        num_warmup=burn_in,
        num_samples=niters,
        num_chains=1,
        progress_bar=verbose,
    )
    mcmc.run(jax.random.PRNGKey(0))
    chain = {k: np.asarray(x) for k, x in mcmc.get_samples().items()}

    q = {
        k: [float(x) for x in np.percentile(chain[k], [16, 50, 84])]
        for k in ("sep", "pa", "cr")
    }
    sep, pa, cr = q["sep"][1], q["pa"][1], q["cr"][1]
    dm = 2.5 * np.log10(cr)
    e_dmm = abs(dm - 2.5 * np.log10(q["cr"][0]))
    e_dmp = abs(2.5 * np.log10(q["cr"][2]) - dm)

    if verbose:
        print("MCMC estimation")
        print("---------------")
        print(
            f"Separation = {sep:2.1f} +{q['sep'][2] - sep:2.1f}/-{sep - q['sep'][0]:2.1f} mas"
        )
        print(f"PA = {pa:2.1f} +{q['pa'][2] - pa:2.1f}/-{pa - q['pa'][0]:2.1f} deg")
        print(
            f"Contrast Ratio = {cr:2.1f} +{q['cr'][2] - cr:2.1f}/-{cr - q['cr'][0]:2.1f}"
        )
        print(f"dm = {dm:2.2f} +{e_dmp:2.2f}/-{e_dmm:2.2f} mag")

    if display:
        import matplotlib.pyplot as plt

        labels = {"sep": "Separation [mas]", "pa": "PA [deg]", "cr": "CR"}
        plt.figure(figsize=(5, 7))
        for n, k in enumerate(("sep", "pa", "cr")):
            plt.subplot(3, 1, n + 1)
            plt.plot(chain[k], color="grey", alpha=0.5)
            plt.plot(len(chain[k]), q[k][1], marker="*", color="#0085ca", zorder=1e3)
            plt.ylabel(labels[k])
        plt.xlabel("Step")
        plt.tight_layout()
        plt.show(block=False)

    return {
        "best": {
            "model": "binary",
            "dm": dm,
            "theta": pa,
            "sep": sep,
            "x0": 0,
            "y0": 0,
        },
        "uncer": {
            "dm_p": e_dmp,
            "dm_m": e_dmm,
            "theta_p": q["pa"][2] - pa,
            "theta_m": pa - q["pa"][0],
            "sep_p": q["sep"][2] - sep,
            "sep_m": sep - q["sep"][0],
        },
        "chain": chain,
    }


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
    display=False,
):
    """Compute 3-sigma contrast limits versus separation, Pymask style
    (closure phases only; computed with virgil).

    Pymask simulated `nsim` noise realizations; here the limit is computed
    directly with the method of Absil et al. (2011) on a polar grid of `nsep`
    separations and `nth` position angles. As in Pymask, the limit at each
    separation is the one valid at every position angle (the brightest
    companion flux ruled out everywhere).

    Parameters
    ----------
    input_data : str
        OIFITS file name.
    nsim : int, default=100
        Unused; kept for compatibility.
    err_scale : float, default=1
        Multiplicative error-bar scale.
    extra_error_cp : float, default=0
        Additional closure-phase uncertainty in degrees.
    ncore : int, default=1
        Unused; kept for compatibility.
    cmax : float, default=500
        Maximum contrast ratio.
    nsep : int, default=60
        Number of separation samples.
    ncrat : int, default=60
        Unused; kept for compatibility.
    nth : int, default=30
        Number of position-angle samples.
    smin : float, default=20
        Minimum separation in milliarcseconds.
    smax : float, default=250
        Maximum separation in milliarcseconds.
    cmin : float, default=1.0001
        Minimum contrast ratio.
    display : bool, default=False
        Whether to display the contrast-limit plot.

    Returns
    -------
    dict
        Separations (``"r"``, mas), 3-sigma magnitude-difference limits
        (``"cr_limit"``), and the limit map (``"lims_data"``: ``"flux_limit"``
        of shape (sep, pa), ``"seps"``, ``"ths"``).
    """
    _virgil()
    import jax.numpy as jnp
    from virgil.limits import absil_limits
    from virgil.models import BinaryModelAngular

    data = _load(input_data, ("cp",), extra_error_cp, err_scale)
    seps = np.linspace(smin, smax, nsep)
    ths = np.linspace(0, 360, nth, endpoint=False)
    samples = {
        "sep": jnp.asarray(seps),
        "pa": jnp.asarray(ths),
        "flux": jnp.asarray(1.0 / np.geomspace(cmin, cmax, 11)),
    }
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "absil_limits", RuntimeWarning)
        flux = np.asarray(
            absil_limits(
                data,
                BinaryModelAngular,
                samples,
                sigma=3.0,
                flux_bounds=(1.0 / cmax, 1.0 / cmin),
            )
        )
    con_limits = 2.5 * np.log10(1.0 / flux.max(axis=1))

    if display:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(seps, con_limits)
        plt.xlabel("Separation [mas]")
        plt.ylabel(r"$\Delta \mathrm{Mag}_{3\sigma}$")
        plt.title(r"Flux ratio for 3$\sigma$ detection (virgil)")
        plt.ylim(plt.ylim()[1], plt.ylim()[0])  # -- reverse plot
        plt.tight_layout()

    return {
        "r": seps,
        "cr_limit": con_limits,
        "lims_data": {"flux_limit": flux, "seps": seps, "ths": ths},
    }
