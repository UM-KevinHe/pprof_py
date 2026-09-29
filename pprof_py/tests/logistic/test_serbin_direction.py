"""C27: Serbin's joint Newton direction, its covariate-origin equivariance, and its stopping diagnostics.

Serbin used to clip the provider-effect block of the Newton direction to +-2*bound.  With covariates far from 0 the
step needs the provider effects to move by about -xbar'd_beta; clipping them alone made the direction a descent
direction, backtracking shrank the step to about 1e-16, and the beta-change rule read that as convergence (a
near-null fit, log-likelihood -1522 against -1412).  R's logis_BIN_fe_prov has no such clip; Newton's method with
the median-relative clamp and the beta rule is equivariant under a shift of the covariates, so the fit must not
depend on their origin.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pprof_py import LogisticFixedEffectModel
from pprof_py.algorithms.logistic import fixed_effect as fe

DATA = Path(__file__).resolve().parents[1] / "data"
SHIFTS = [(0, 0), (50, -30), (500, -300), (2026, 0)]


def _bernoulli_rows():
    """The AOH cohort with binomial rows expanded to Bernoulli rows, as the R goldens were generated."""
    d = pd.read_csv(DATA / "aoh" / "data.csv")
    rows = d.loc[d.index.repeat(d.n)].copy()
    rows["y"] = (rows.groupby(level=0).cumcount() < rows.y).astype(int)
    return rows.sort_values("prov", kind="stable")[["y", "prov", "x1", "x2"]].reset_index(drop=True)


def _fit(df, **kw):
    return LogisticFixedEffectModel(use_dataprep=False, screen_providers=False).fit(
        df, y_var="y", x_vars=["x1", "x2"], provider_var="prov", **kw)


@pytest.mark.parametrize("shift", SHIFTS)
def test_serbin_matches_r_at_shifted_covariates(shift):
    r = pd.read_csv(DATA / "serbin_r" / "r_serbin.csv")
    r = r[(r.shift_x1 == shift[0]) & (r.shift_x2 == shift[1])]
    d = _bernoulli_rows()
    m = _fit(d.assign(x1=d.x1 + shift[0], x2=d.x2 + shift[1]), tol=1e-8, bound=10.0, backtrack=True)
    np.testing.assert_allclose(np.ravel(m.coefficients_["beta"]), r[r.param == "beta"].value, rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.ravel(m.coefficients_["gamma"]), r[r.param == "gamma"].value, rtol=0, atol=1e-8)


@pytest.mark.parametrize("backtrack", [True, False])
@pytest.mark.parametrize("shift", SHIFTS[1:])
def test_serbin_equivariant_under_covariate_shift(shift, backtrack):
    d = pd.read_csv(DATA / "aoh" / "data.csv")
    kw = dict(y_var="y", x_vars=["x1", "x2"], provider_var="prov", n_var="n", backtrack=backtrack)
    base = LogisticFixedEffectModel(use_dataprep=False).fit(d, **kw)
    moved = LogisticFixedEffectModel(use_dataprep=False).fit(d.assign(x1=d.x1 + shift[0], x2=d.x2 + shift[1]), **kw)
    beta = np.ravel(base.coefficients_["beta"])
    np.testing.assert_allclose(np.ravel(moved.coefficients_["beta"]), beta, rtol=0, atol=1e-10)
    np.testing.assert_allclose(moved.fitted_, base.fitted_, rtol=0, atol=1e-10)
    # gamma_j in the shifted parametrization is gamma_j - c'beta
    np.testing.assert_allclose(np.ravel(moved.coefficients_["gamma"]),
                               np.ravel(base.coefficients_["gamma"]) - (shift[0] * beta[0] + shift[1] * beta[1]),
                               rtol=0, atol=1e-8)
    assert moved.algorithm.iter == base.algorithm.iter


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.name == fe.logger.name and r.levelno >= logging.WARNING]


def test_serbin_no_warning_when_converged(caplog):
    d = pd.read_csv(DATA / "aoh" / "data.csv")
    with caplog.at_level(logging.WARNING, logger=fe.logger.name):
        LogisticFixedEffectModel(use_dataprep=False).fit(d, y_var="y", x_vars=["x1", "x2"], provider_var="prov", n_var="n")
    assert _warnings(caplog) == []


def test_serbin_warns_when_the_line_search_ends_the_fit(caplog, monkeypatch):
    # A descent direction (the old failure's mechanism, forced): backtracking shrinks the step to ~0.
    original = fe.SerbinAlgorithm._compute_deltas

    def reversed_direction(self, *args):
        d_gamma, d_beta = original(self, *args)
        return -d_gamma, -d_beta

    monkeypatch.setattr(fe.SerbinAlgorithm, "_compute_deltas", reversed_direction)
    d = pd.read_csv(DATA / "aoh" / "data.csv")
    with caplog.at_level(logging.WARNING, logger=fe.logger.name):
        LogisticFixedEffectModel(use_dataprep=False).fit(d, y_var="y", x_vars=["x1", "x2"], provider_var="prov", n_var="n")
    msgs = _warnings(caplog)
    assert len(msgs) == 1 and "line search shortened the last step" in msgs[0]


@pytest.mark.parametrize("backtrack", [True, False])
def test_serbin_warns_at_the_iteration_limit(caplog, backtrack):
    d = pd.read_csv(DATA / "aoh" / "data.csv")
    with caplog.at_level(logging.WARNING, logger=fe.logger.name):
        LogisticFixedEffectModel(use_dataprep=False).fit(d, y_var="y", x_vars=["x1", "x2"], provider_var="prov",
                                                         n_var="n", max_iter=1, backtrack=backtrack)
    msgs = _warnings(caplog)
    assert len(msgs) == 1 and "did not converge" in msgs[0]
