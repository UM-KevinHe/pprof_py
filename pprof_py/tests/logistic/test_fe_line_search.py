"""C3: the fixed-effect Newton algorithms' Armijo line search near the optimum.

Near the optimum a step's predicted gain falls below the log-likelihood's rounding level, where the Armijo test only
compares rounding noise; backtracking used to shrink the step to exactly 0 (about 1,400 evaluations) and the zero
step read as convergence, one Newton step short of the optimum.
"""
import numpy as np
import pandas as pd
import pytest

from pprof_py import LogisticFixedEffectModel
from pprof_py.algorithms.logistic import fixed_effect as fe


def _cohort(seed=20260921, n=6000, K=60):
    """The REV-001/REV-002 harness cohort, on which the collapse occurs."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.normal(70.0, 10.0, n), rng.normal(0.0, 1.0, n), rng.normal(0.0, 1.0, n),
                         rng.normal(0.67, 1.0, n), rng.normal(0.973, 1.0, n), rng.normal(0.0, 1.0, n),
                         rng.normal(0.0, 1.0, n), rng.binomial(1, 0.4, n).astype(float), rng.normal(0.0, 1.0, n)])
    prov = rng.integers(0, K, n)
    gamma = rng.normal(0.0, 0.3, K)
    eta = -2.45 + (X - X.mean(0)) @ np.array([0.03, 0.4, -0.3, 0.25, 0.0, 0.15, 0.0, -0.35, 0.1]) + gamma[prov]
    df = pd.DataFrame(X, columns=[f"x{j}" for j in range(9)])
    df["y"] = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    df["prov"] = prov
    return df


def _max_score(model, df):
    xv = [f"x{j}" for j in range(9)]
    X, y = df[xv].to_numpy(), df["y"].to_numpy()
    gamma = pd.Series(np.ravel(model.coefficients_["gamma"]).astype(float), index=model.provider_ids_)
    eta = gamma.reindex(df["prov"]).to_numpy() + X @ np.ravel(model.coefficients_["beta"]).astype(float)
    r = y - 1 / (1 + np.exp(-eta))
    return max(np.max(np.abs(X.T @ r)), np.max(np.abs(pd.Series(r).groupby(df["prov"].to_numpy()).sum())))


@pytest.mark.parametrize("algorithm, tol", [("Serbin", 1e-8), ("Ban", 1e-5)])
def test_fit_reaches_the_optimum(algorithm, tol):
    df = _cohort()
    m = LogisticFixedEffectModel(algorithm=algorithm).fit(df, y_var="y", x_vars=[f"x{j}" for j in range(9)],
                                                          provider_var="prov")
    assert _max_score(m, df) < tol


def test_serbin_line_search_does_not_collapse(monkeypatch):
    calls = {"n": 0}
    original = fe.BaseAlgorithm._loglikelihood

    def counting(self, gamma_obs, beta):
        calls["n"] += 1
        return original(self, gamma_obs, beta)

    monkeypatch.setattr(fe.BaseAlgorithm, "_loglikelihood", counting)
    df = _cohort()
    m = LogisticFixedEffectModel().fit(df, y_var="y", x_vars=[f"x{j}" for j in range(9)], provider_var="prov")
    assert calls["n"] <= 4 * m.algorithm.iter            # an evaluation or two per iteration, no collapse
