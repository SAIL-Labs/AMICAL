"""
-------------------------------------------------------------------------
AMICAL: Aperture Masking Interferometry Calibration and Analysis Library
-------------------------------------------------------------------------

AMICAL's job ends at the calibrated OIFITS file. To fit models to it (binary
searches, contrast limits, posteriors), use a dedicated interferometry fitting
package. We recommend virgil (https://benjaminpope.github.io/virgil/), which
reads AMICAL's OIFITS files directly and is built on JAX, so grid searches are
fast and posteriors can be sampled with gradient-based HMC.

The CANDID and Pymask wrappers (amical.candid_grid, amical.pymask_grid, ...)
and amical.smartfit are deprecated and will be removed in a future release.

virgil needs Python >= 3.11 and JAX. Install it alongside AMICAL, or in a
separate environment (the two only need to share the OIFITS files):

    python -m pip install virgil-astro

(the distribution is `virgil-astro`; `pip install virgil` is an unrelated
package.)

This script fits the simulated binary saved by example_NIRISS.py. For the
file tests/data/test.oifits it finds a separation of about 147 mas, a position
angle of about 47 deg and a contrast of about 6 mag.

--------------------------------------------------------------------
"""

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from jax.scipy.stats import norm
from matplotlib import pyplot as plt
from virgil.fitting import fit
from virgil.grid_fit import (
    best_grid_point,
    laplace_flux_uncertainty_grid,
    likelihood_grid,
    optimized_flux_grid,
)
from virgil.limits import flux_to_delta_mag, ruffio_upperlimit
from virgil.models import BinaryModelCartesian, PointSource, System
from virgil.oidata import OIData
from virgil.plotting import plot_contrast_curve, plot_grid_map

# Your input data is an oifits file or a list of oifits files (fitted jointly).
inputdata = "Saveoifits/example_fakebinary_NIRISS.oifits"
data = OIData(inputdata)

# Note on closure phases: AMICAL saves all N(N-1)(N-2)/6 closure phases, of
# which only (N-1)(N-2)/2 are independent. Either save the independent set
# with amical.save(..., ind_hole=0), or calibrate with
# normalize_err_indep=True, or inflate the errors in the fit (`noise=` below).

# 1. Grid search for a companion
# ------------------------------
# Log-likelihood of a point-source companion over offsets (mas) and
# companion/primary flux ratios.
samples = {
    "dra": jnp.linspace(-250.0, 250.0, 101),
    "ddec": jnp.linspace(-250.0, 250.0, 101),
    "flux": 10 ** jnp.linspace(-4.0, -1.0, 31),
}
loglike = likelihood_grid(data, BinaryModelCartesian, samples)
best = best_grid_point(loglike, samples)
plot_grid_map(loglike, samples, best=best, label="Max log-likelihood over flux")

# 2. Refine the best grid point
# -----------------------------
start = System(
    primary=PointSource(),
    companion=PointSource(flux=best["flux"], dra=best["dra"], ddec=best["ddec"]),
)
priors = {
    "companion.dra": dist.Uniform(-300.0, 300.0),
    "companion.ddec": dist.Uniform(-300.0, 300.0),
    "companion.flux": dist.Uniform(0.0, 0.1),
}
result = fit(start, priors, data)
v = result.values
sep = np.hypot(v["companion.dra"], v["companion.ddec"])
pa = np.degrees(np.arctan2(v["companion.dra"], v["companion.ddec"])) % 360
dm = float(flux_to_delta_mag(v["companion.flux"]))
print(f"chi2_red = {result.info['chi2_red']:.2f}")
print(f"sep = {sep:.1f} mas, pa = {pa:.1f} deg, contrast = {dm:.2f} mag")

# If the calibrated errors look underestimated (chi2_red >> 1), fit error
# inflation terms with the model, e.g.
#   fit(start, priors, data, noise={"phi_error": dist.HalfNormal(0.05)})
# For posteriors, see virgil.likelihood.numpyro_model and the virgil binary
# search tutorial (HMC with NUTS).

# 3. Contrast limits
# ------------------
# Ruffio et al. (2018) upper limits at the 3-sigma-equivalent percentile. For
# limits on a detected system, subtract or fit the companion first.
flux = optimized_flux_grid(data, BinaryModelCartesian, samples)
sigma_flux = laplace_flux_uncertainty_grid(
    data, BinaryModelCartesian, samples, flux=flux
)
limit = ruffio_upperlimit(flux, sigma_flux, norm.cdf(3.0))
plot_contrast_curve(limit, samples, label=r"Ruffio 3$\sigma$")

plt.show(block=True)
