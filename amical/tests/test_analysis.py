import numpy as np
import pytest

import amical
from amical import candid_cr_limit
from amical.externals import pymask


@pytest.fixture()
def example_oifits(global_datadir):
    return global_datadir / "test.oifits"


def test_load_file(example_oifits):
    s = amical.load(example_oifits)
    assert isinstance(s, dict)


@pytest.fixture()
def example_oifits_no_date_obs(global_datadir):
    return global_datadir / "test_no_date_obs.oifits"


@pytest.mark.parametrize("ncore", [1, 2])
def test_candid_grid(example_oifits, ncore):
    # Name of the model
    name_model = "binary_res"
    # Candid param
    param_candid = {"rmin": 50, "rmax": 180, "step": 50, "ncore": ncore}
    # Fit
    fit = amical.candid_grid(example_oifits, **param_candid)

    model_type = fit["best"]["model"]

    # Human checked values
    true_sep, true_pa, true_dm = 147.7, 46.6, 6.0

    sep, e_sep = fit["best"]["sep"], fit["uncer"]["sep"]
    pa, e_pa = fit["best"]["theta"], fit["uncer"]["theta"]
    dm, e_dm = fit["best"]["dm"], fit["uncer"]["dm"]

    assert isinstance(fit, dict)
    assert model_type == name_model
    # Check close true value
    assert sep == pytest.approx(true_sep, 0.01)
    assert pa == pytest.approx(true_pa, 0.01)
    assert dm == pytest.approx(true_dm, 0.01)
    # Check small errors
    assert e_sep <= 0.01 * true_sep
    assert e_pa <= 0.01 * true_pa
    assert e_dm <= 0.01 * true_dm


def test_plot_model(example_oifits):
    param_candid = {"rmin": 50, "rmax": 180, "step": 50, "ncore": 1}
    fit = amical.candid_grid(example_oifits, **param_candid)
    ret = amical.plot_model(example_oifits, fit["best"])
    assert isinstance(fit, dict)
    assert len(ret) == 3


@pytest.mark.parametrize("step", [40, 60])
def test_candid_cr(example_oifits, step):
    param_candid = {"rmin": 50, "rmax": 180, "step": step, "ncore": 1}
    fit = amical.candid_grid(example_oifits, **param_candid)
    cr_candid = candid_cr_limit(example_oifits, **param_candid, fitComp=fit["comp"])
    tested_r = cr_candid["r"]
    assert isinstance(cr_candid, dict)
    assert isinstance(tested_r, np.ndarray)
    assert len(tested_r) > 1


def test_pymask_grid(example_oifits):
    pa_prior = [30, 50]
    sep_prior = [100, 200]
    cr_prior = [100, 300]
    param_pymask = {
        "pa_prior": pa_prior,
        "sep_prior": sep_prior,
        "cr_prior": cr_prior,
    }
    fit = amical.pymask_grid(str(example_oifits), **param_pymask)
    assert isinstance(fit, dict)


def test_pymask_mcmc(example_oifits):
    param_pymask = {
        "sep_prior": [100, 180],
        "pa_prior": [20, 80],
        "cr_prior": [230, 270],
        "ncore": None,
        "extra_error_cp": 0,
        "err_scale": 1,
    }

    param_mcmc = {
        "niters": 800,
        "walkers": 100,
        "initial_guess": [146, 47, 244],
        "burn_in": 100,
    }

    fit = amical.pymask_mcmc(str(example_oifits), **param_pymask, **param_mcmc)

    # Human checked values
    true_sep, true_pa, true_dm = 147.7, 46.6, 6.0

    sep, e_sep = fit["best"]["sep"], max(fit["uncer"]["sep_p"], fit["uncer"]["sep_m"])
    pa, e_pa = (
        fit["best"]["theta"],
        max(fit["uncer"]["theta_p"], fit["uncer"]["theta_m"]),
    )
    dm, e_dm = fit["best"]["dm"], max(fit["uncer"]["dm_p"], fit["uncer"]["dm_m"])

    assert isinstance(fit, dict)
    # Check close true value
    assert sep == pytest.approx(true_sep, 0.01)
    assert pa == pytest.approx(true_pa, 0.01)
    assert dm == pytest.approx(true_dm, 0.01)
    # Check small errors
    assert e_sep <= 0.01 * true_sep
    assert e_pa <= 0.01 * true_pa
    assert e_dm <= 0.01 * true_dm


def test_pymask_oifits_no_date_obs(example_oifits_no_date_obs):
    o = pymask.oifits.open(str(example_oifits_no_date_obs))
    assert isinstance(o, pymask.oifits.oifits)
