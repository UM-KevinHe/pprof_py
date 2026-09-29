"""Covariate-origin invariance of Ban (C30) and of the model-based provider inference (C34), at_bound (C35), and
binomial trials in the standardized-measure limits (C36)."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pprof_py import LogisticFixedEffectModel
from pprof_py.inference import at_bound

AOH = Path(__file__).resolve().parents[1] / "data" / "aoh" / "data.csv"
KW = dict(y_var="y", x_vars=["x1", "x2"], provider_var="prov", n_var="n")


def _aoh(shift=(0, 0), **kw):
    d = pd.read_csv(AOH)
    return LogisticFixedEffectModel(use_dataprep=False, **kw).fit(d.assign(x1=d.x1 + shift[0], x2=d.x2 + shift[1]), **KW)


@pytest.mark.parametrize("shift", [(5, -3), (50, -30), (500, -300)])
def test_ban_reaches_the_maximum_at_shifted_covariates(shift):
    serbin, ban = _aoh(shift), _aoh(shift, algorithm="Ban")
    np.testing.assert_allclose(np.ravel(ban.coefficients_["beta"]), np.ravel(serbin.coefficients_["beta"]), rtol=0, atol=1e-7)
    np.testing.assert_allclose(ban.fitted_, serbin.fitted_, rtol=0, atol=1e-7)
    assert ban.algorithm.iter < 100


def test_case_mix_variance_equals_the_dense_inverse():
    f = _aoh((5, -3))
    d = pd.read_csv(AOH).assign(x1=lambda t: t.x1 + 5, x2=lambda t: t.x2 - 3)
    p = f.fitted_; q = d.n.to_numpy() * p * (1 - p); m = len(f.provider_ids_)
    X = d[["x1", "x2"]].to_numpy(); D = np.c_[np.eye(m)[f.provider_indices_], X]
    cov = np.linalg.inv(D.T @ (q[:, None] * D))
    xbar = d.n.to_numpy() @ X / d.n.sum()
    dense = np.diag(cov)[:m] + xbar @ cov[m:, m:] @ xbar + 2 * cov[:m, m:] @ xbar       # Var(gamma_k + xbar' beta)
    np.testing.assert_allclose(f.variances_["gamma_case_mix"], dense, rtol=1e-8)
    np.testing.assert_allclose(f.variances_["gamma"], np.diag(cov)[:m], rtol=1e-8)      # R's Var(gamma_k) is kept


@pytest.mark.parametrize("shift", [(5, -3), (50, -30)])
def test_model_based_provider_inference_does_not_depend_on_the_origin(shift):
    base, moved = _aoh(), _aoh(shift)
    for a, b in ((base.test(test_method="wald"), moved.test(test_method="wald")),
                 (base.test_standardized(measure="gamma"), moved.test_standardized(measure="gamma")),
                 (base.test_standardized(measure="direct_ratio"), moved.test_standardized(measure="direct_ratio")),
                 (base.test_standardized(measure="direct_rate"), moved.test_standardized(measure="direct_rate"))):
        np.testing.assert_allclose(b.z_raw, a.z_raw, rtol=0, atol=1e-7)
        np.testing.assert_array_equal(b.flag, a.flag)
    for key in ("indirect_ratio", "direct_ratio"):
        ca = base.calculate_confidence_intervals(option="SM", stdz=["indirect", "direct"], measure="ratio", test_method="wald")[key]
        cb = moved.calculate_confidence_intervals(option="SM", stdz=["indirect", "direct"], measure="ratio", test_method="wald")[key]
        np.testing.assert_allclose(cb[["ci_ratio_lower", "ci_ratio_upper"]], ca[["ci_ratio_lower", "ci_ratio_upper"]], rtol=1e-7)


def test_at_bound_finds_the_providers_without_a_finite_estimate():
    rng = np.random.default_rng(2); m = 50; n = rng.integers(8, 80, m); prov = np.repeat(np.arange(m), n)
    x = rng.normal(size=prov.size)
    y = rng.binomial(1, 1 / (1 + np.exp(-(-1.5 + rng.normal(0, .4, m)[prov] + 0.5 * x))))
    y[prov < 3] = 0; y[prov == 3] = 1
    events, sizes = np.bincount(prov, weights=y), np.bincount(prov)
    expected = np.flatnonzero((events == 0) | (events == sizes))
    for shift in (0.0, 4.0, -6.0):
        f = LogisticFixedEffectModel(use_dataprep=False, screen_providers=False).fit(
            pd.DataFrame(dict(y=y, x=x + shift, p=prov)), y_var="y", x_vars=["x"], provider_var="p")
        np.testing.assert_array_equal(np.flatnonzero(at_bound(f)), expected)


def _bernoulli_and_binomial():
    rng = np.random.default_rng(4); m = 30; prov = np.repeat(np.arange(m), rng.integers(40, 120, m))
    x = rng.normal(size=prov.size).round(0)
    y = rng.binomial(1, 1 / (1 + np.exp(-(-1.2 + rng.normal(0, .4, m)[prov] + 0.5 * x))))
    bern = pd.DataFrame(dict(y=y, x=x, p=prov)); agg = bern.groupby(["p", "x"]).y.agg(["sum", "size"]).reset_index()
    fb = LogisticFixedEffectModel(use_dataprep=False).fit(bern, y_var="y", x_vars=["x"], provider_var="p")
    fa = LogisticFixedEffectModel(use_dataprep=False).fit(agg, y_var="sum", x_vars=["x"], provider_var="p", n_var="size")
    return fb, fa


@pytest.mark.parametrize("test_method", ["wald", "score", "exact"])
def test_sm_limits_use_binomial_trials(test_method):
    fb, fa = _bernoulli_and_binomial()
    kw = dict(option="SM", stdz=["indirect", "direct"], measure=["ratio", "rate"], test_method=test_method)
    cb, ca = fb.calculate_confidence_intervals(**kw), fa.calculate_confidence_intervals(**kw)
    for key in cb:
        cols = [c for c in cb[key] if c.startswith("ci_")]
        np.testing.assert_allclose(ca[key][cols].to_numpy(float), cb[key][cols].to_numpy(float), rtol=1e-6)
    gb = fb.calculate_confidence_intervals(option="gamma", test_method=test_method)
    ga = fa.calculate_confidence_intervals(option="gamma", test_method=test_method)
    gb, ga = (next(iter(g.values())) if isinstance(g, dict) else g for g in (gb, ga))
    np.testing.assert_allclose(ga.select_dtypes("number").to_numpy(), gb.select_dtypes("number").to_numpy(), rtol=1e-6)


def test_add_providers_carries_the_case_mix_variance():
    f = _aoh()
    ids = np.asarray(f.provider_ids_)
    new = [ids[0] + 0.5, ids[-1] + 1]                                  # one between existing IDs, one after
    before = dict(zip(ids, f.variances_["gamma_case_mix"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f.add_providers(new, gamma=[0.1, -0.2], se_gamma=[0.3, 0.4])
    after = dict(zip(f.provider_ids_, f.variances_["gamma_case_mix"]))
    assert all(after[k] == v for k, v in before.items())
    np.testing.assert_allclose([after[new[0]], after[new[1]]], [0.09, 0.16])
