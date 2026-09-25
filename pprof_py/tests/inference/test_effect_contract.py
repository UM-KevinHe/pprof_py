"""The shared test() contract: every route of every model family returns the same schema and semantics."""
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, t as t_dist

from pprof_py import (LinearFixedEffectModel, LinearRandomEffectModel, LogisticFixedEffectModel,
                      LogisticMixedEffectModel, LogisticRandomEffectModel)
from pprof_py.inference import HUBER_RLM, PROVIDER_TEST_COLUMNS, EmpiricalNull

sig = lambda x: 1 / (1 + np.exp(-x))


def _quiet(f, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*a, **k)


@pytest.fixture(scope="module")
def models():
    rng = np.random.default_rng(99)
    m = 70
    sizes = rng.integers(20, 110, m)
    prov = np.repeat(np.arange(m), sizes)
    X = rng.normal(size=(prov.size, 2))
    g = rng.normal(0, 0.35, m)
    y = rng.binomial(1, sig(-1.4 + X @ [0.5, -0.3] + g[prov])).astype(float)
    y[prov == 5] = 0.0                                              # a zero-event provider
    df = pd.DataFrame(X, columns=["x1", "x2"]); df["y"] = y; df["provider"] = prov
    fe = LogisticFixedEffectModel(); _quiet(fe.fit, X, y, prov)
    re = LogisticRandomEffectModel(); _quiet(re.fit, df, y_var="y", x_vars=["x1", "x2"], group_var="provider", verbose=False)
    me = LogisticMixedEffectModel()
    _quiet(me.fit, df, y_var="y", x_vars=["x1", "x2"], provider_var="provider", cluster_var="provider",
           gamma_init=np.full(m, -1.4), beta_init=fe.coefficients_["beta"].ravel(), sigma_init=0.35, verbose=False)
    yl = 5.0 + X @ [1.0, -0.5] + rng.normal(0, 0.6, m)[prov] + rng.normal(0, 2.0, prov.size)
    dfl = df.assign(y=yl)
    lfe = LinearFixedEffectModel(); _quiet(lfe.fit, X, yl, prov)
    lre = LinearRandomEffectModel(); _quiet(lre.fit, dfl, y_var="y", x_vars=["x1", "x2"], group_var="provider", verbose=False)
    return {"fe": fe, "re": re, "me": me, "lfe": lfe, "lre": lre}


ROUTES = {
    "logistic_fe/poibin_exact": ("fe", dict(test_method="poibin_exact")),
    "logistic_fe/score": ("fe", dict(test_method="score")),
    "logistic_fe/wald": ("fe", dict(test_method="wald")),
    "logistic_fe/bootstrap_exact": ("fe", dict(test_method="bootstrap_exact", n_resample=1500, seed=3)),
    "logistic_re/wald": ("re", dict(test_method="wald")),
    "logistic_re/poibin_exact": ("re", dict(test_method="poibin_exact")),
    "logistic_re/resampling": ("re", dict(test_method="resampling", n_resample=1500, seed=3)),
    "logistic_me/exact": ("me", dict(test_method="exact")),
    "logistic_me/poibin_exact": ("me", dict(test_method="poibin_exact")),
    "logistic_me/resampling": ("me", dict(test_method="resampling", n_resample=1500, seed=3)),
    "linear_fe/wald": ("lfe", {}),
    "linear_re/wald": ("lre", {}),
}
WALD = [k for k in ROUTES if k.endswith("/wald")]


def _run(models, route, **extra):
    key, kwargs = ROUTES[route]
    return _quiet(models[key].test, **{**kwargs, **extra})


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_schema_and_conventions(models, route):
    res = _run(models, route)
    assert tuple(res.columns) == PROVIDER_TEST_COLUMNS and res.index.name == "provider"
    assert str(res.flag.dtype) == "Int8" and res.flag.notna().all()
    assert ((res.flag == 1) <= (res.z_adjusted > 0)).all() and ((res.flag == -1) <= (res.z_adjusted < 0)).all()
    np.testing.assert_allclose(res.p_value, 2 * norm.sf(np.abs(res.z_adjusted)), rtol=1e-12, atol=0)
    assert (res.null_value == res.attrs["reference"]).all() and res.attrs["test_method"] is not None
    assert (res.null_mean == 0).all() and (res.null_sd == 1).all()


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_providers_subset_and_empirical_null(models, route):
    full = _run(models, route)
    ids = list(full.index[:7])
    pd.testing.assert_frame_equal(_run(models, route, providers=ids), full.loc[ids])
    en = _run(models, route, null_model=EmpiricalNull.fitter(estimator=HUBER_RLM, small_group="theoretical"))
    assert en.attrs["null_model"]["kind"] == "empirical" and not np.allclose(en.null_sd, 1.0)


@pytest.mark.parametrize("route", WALD)
def test_wald_intervals_agree_with_flags(models, route):
    res = _run(models, route)
    excludes = (res.ci_lower > res.null_value) | (res.ci_upper < res.null_value)
    assert ((res.flag != 0).to_numpy() == excludes.to_numpy()).all()


def test_linear_fe_uses_student_t(models):
    lfe = models["lfe"]
    res = _run(models, "linear_fe/wald")
    df = lfe.fitted_.size - len(lfe.coefficients_["beta"]) - res.shape[0]
    tstat = (res.estimate - res.null_value) / res.se
    np.testing.assert_allclose(res.p_value, 2 * t_dist.sf(np.abs(tstat), df), rtol=1e-10)
    np.testing.assert_allclose(res.ci_upper - res.estimate, t_dist.isf(0.025, df) * res.se, rtol=1e-10)


@pytest.mark.parametrize("route", ["logistic_fe/bootstrap_exact", "logistic_re/resampling", "logistic_me/resampling"])
def test_monte_carlo_routes_are_reproducible(models, route):
    a, b = _run(models, route), _run(models, route)
    c = _run(models, route, seed=4)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a.p_value, c.p_value)


def test_one_sided_and_zero_event_providers(models):
    fe = models["fe"]
    greater = _quiet(fe.test, alternative="greater")
    assert set(greater.flag.unique()) <= {0, 1}
    exact = _quiet(fe.test, test_method="poibin_exact")
    assert np.isfinite(exact.loc[5, "z_raw"]) and exact.loc[5, "flag"] == -1   # finite statistic at the bound


def test_binomial_trials_weight_the_score_test():
    rng = np.random.default_rng(5)
    m = 30; sizes = rng.integers(15, 40, m); prov = np.repeat(np.arange(m), sizes)
    X = rng.normal(size=(prov.size, 1)); n_trials = rng.integers(1, 6, prov.size).astype(float)
    y = rng.binomial(n_trials.astype(int), sig(-0.8 + 0.4 * X[:, 0] + rng.normal(0, 0.3, m)[prov])).astype(float)
    df = pd.DataFrame({"x1": X[:, 0], "y": y, "n": n_trials, "provider": prov})
    fe = LogisticFixedEffectModel(use_dataprep=False)                   # data prep insists on 0/1 outcomes
    _quiet(fe.fit, df, x_vars=["x1"], y_var="y", n_var="n", group_var="provider")
    g0 = np.median(fe.coefficients_["gamma"].ravel())
    p0 = np.clip(sig(g0 + fe.xbeta_.ravel()), 1e-10, 1 - 1e-10); idx = np.asarray(fe.group_indices_)
    z = (np.bincount(idx, fe.outcome_) - np.bincount(idx, fe.N_ * p0)) / np.sqrt(np.bincount(idx, fe.N_ * p0 * (1 - p0)))
    np.testing.assert_allclose(_quiet(fe.test, test_method="score").z_raw, z, rtol=1e-12)
    exact = _quiet(fe.test, test_method="poibin_exact")                    # trials expanded, so counts can exceed rows
    assert exact.z_raw.notna().all()


@pytest.mark.parametrize("method", ["poibin_exact", "score"])
def test_type_one_error_under_a_global_null(method):
    rng = np.random.default_rng(11)
    m = 300; sizes = rng.integers(40, 120, m); prov = np.repeat(np.arange(m), sizes)
    X = rng.normal(size=(prov.size, 2))
    y = rng.binomial(1, sig(-1.2 + X @ [0.4, -0.3])).astype(float)
    fe = LogisticFixedEffectModel(); _quiet(fe.fit, X, y, prov)
    rate = float((_quiet(fe.test, test_method=method).flag != 0).mean())
    assert 0.02 <= rate <= 0.08


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_flags_point_in_the_direction_of_the_effect(models, route):
    res = _run(models, route)
    assert (res.estimate[res.flag == 1] > res.null_value[res.flag == 1]).all()
    assert (res.estimate[res.flag == -1] < res.null_value[res.flag == -1]).all()
    assert (res.flag != 0).sum() > 0                                    # the check is not vacuous
