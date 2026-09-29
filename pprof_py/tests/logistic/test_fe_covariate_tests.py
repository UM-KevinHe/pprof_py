"""Covariate (beta) tests of LogisticFixedEffectModel: R parity, the efficient score (C32), the reduced-model refit (C33)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pprof_py import LogisticFixedEffectModel

DATA = Path(__file__).resolve().parents[1] / "data" / "logis_fe_summary_r"
X3 = ["x1", "x2", "x3"]


def _fit(d, xv=X3, **kw):
    return LogisticFixedEffectModel(use_dataprep=False).fit(d, y_var="y", x_vars=xv, provider_var="prov", **kw)


def test_wald_lr_score_match_r_summary_logis_fe():
    d, r = pd.read_csv(DATA / "cohort.csv"), pd.read_csv(DATA / "r_summary.csv")
    f = _fit(d)
    np.testing.assert_allclose([f._compute_wald_beta(j)["statistic"] for j in range(3)], r[r.test == "wald"].stat, rtol=0, atol=1e-10)
    np.testing.assert_allclose([f._compute_lr_beta(j)["statistic"] for j in range(3)], r[r.test == "lr"].stat, rtol=0, atol=1e-10)
    np.testing.assert_allclose([f._compute_score_beta(j)["statistic"] for j in range(3)], r[r.test == "score"].stat, rtol=0, atol=1e-6)


def test_score_uses_the_efficient_information():
    d = pd.read_csv(DATA / "cohort.csv")
    f = _fit(d)
    red = _fit(d, xv=["x2", "x3"])                          # the fit under beta_1 = 0
    p = red.fitted_; q = p * (1 - p); m = red.provider_indices_.max() + 1
    D = np.c_[np.eye(m)[red.provider_indices_], d[["x2", "x3", "x1"]].to_numpy()]
    info = D.T @ (q[:, None] * D)
    efficient = info[-1, -1] - info[-1, :-1] @ np.linalg.solve(info[:-1, :-1], info[:-1, -1])
    u = d.x1.to_numpy() @ (d.y.to_numpy() - p)
    np.testing.assert_allclose(f._compute_score_beta(0)["statistic"], u**2 / efficient, rtol=1e-8)


def _small_provider_cohort():
    rng = np.random.default_rng(9)
    n = rng.integers(4, 40, 60); prov = np.repeat(np.arange(60), n)
    x1, x2 = rng.normal(size=prov.size), rng.normal(size=prov.size)
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + rng.normal(0, .4, 60)[prov] + 0.3 * x1 - 0.4 * x2))))
    return pd.DataFrame(dict(y=y, x1=x1, x2=x2, prov=prov))


def _loglik(model, d, xv):
    eta = d[xv].to_numpy() @ np.ravel(model.coefficients_["beta"]) + np.ravel(model.coefficients_["gamma"])[model.provider_indices_]
    return float(np.sum(d.y.to_numpy() * eta - np.logaddexp(0, eta)))


def test_lr_and_score_refit_on_the_fitted_rows():
    d = _small_provider_cohort()                            # providers of 4-39 records; the fit keeps all of them
    f, red = _fit(d, xv=["x1", "x2"]), _fit(d, xv=["x2"])
    lr = 2 * (_loglik(f, d, ["x1", "x2"]) - _loglik(red, d, ["x2"]))
    np.testing.assert_allclose(f._compute_lr_beta(0)["statistic"], lr, rtol=1e-10)
    assert np.isfinite(f._compute_score_beta(0)["statistic"])


def test_lr_and_score_use_binomial_trials():
    d = _small_provider_cohort().assign(x1=lambda t: t.x1.round(0), x2=lambda t: t.x2.round(0))
    agg = d.groupby(["prov", "x1", "x2"]).y.agg(["sum", "size"]).reset_index()
    fb = LogisticFixedEffectModel(use_dataprep=False).fit(agg, y_var="sum", x_vars=["x1", "x2"], provider_var="prov", n_var="size")
    fe = _fit(d, xv=["x1", "x2"])
    for test in ("lr", "score"):
        np.testing.assert_allclose(getattr(fb, f"_compute_{test}_beta")(0)["statistic"],
                                   getattr(fe, f"_compute_{test}_beta")(0)["statistic"], rtol=1e-7)


def test_lr_and_score_need_another_covariate():
    d = _small_provider_cohort()
    f = _fit(d, xv=["x1"])
    for test in ("lr", "score"):
        with pytest.raises(ValueError, match="at least one other covariate"):
            getattr(f, f"_compute_{test}_beta")(0)
