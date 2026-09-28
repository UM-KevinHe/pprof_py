"""Cluster-robust variance of the provider effects (C13).

``variance="robust"`` is the full sandwich of the joint (gamma, beta) fit; ``variance="robust_fixed_beta"``
treats beta as known and reproduces R's ``test_aoh`` (goldens in ``tests/data/aoh``, written by running
``test_aoh`` and ``robust_wald_gamma`` verbatim on these rows, fitted probabilities and estimates).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pprof_py import LogisticFixedEffectModel

DATA = Path(__file__).resolve().parents[1] / "data" / "aoh"


@pytest.fixture(scope="module")
def fitted():
    d = pd.read_csv(DATA / "data.csv")
    fit = LogisticFixedEffectModel(use_dataprep=False).fit(d, y_var="y", x_vars=["x1", "x2"], provider_var="prov",
                                                           n_var="n", obs_id_var="pid")
    return d, fit


def test_fixed_beta_form_matches_r_test_aoh(fitted):
    _, fit = fitted
    r = pd.read_csv(DATA / "r_test_aoh.csv")
    np.testing.assert_allclose(np.asarray(fit.coefficients_["gamma"]).ravel(), pd.read_csv(DATA / "gamma.csv").gamma,
                               rtol=0, atol=1e-10)          # the same inputs as the R run
    t = fit.test_standardized(measure="gamma", variance="robust_fixed_beta", reference="median")
    np.testing.assert_allclose(t["se"].to_numpy(), r.se_gamma.to_numpy(), rtol=1e-10)
    np.testing.assert_allclose(t["z_raw"].to_numpy(), r.stat.to_numpy(), rtol=0, atol=1e-9)
    np.testing.assert_allclose(t["p_value"].to_numpy(), r.p.to_numpy(), rtol=0, atol=1e-10)
    assert np.array_equal(t["flag"].to_numpy(dtype=int), r.flag.to_numpy())


def test_robust_gamma_is_the_full_sandwich(fitted):
    """[I^-1 M I^-1]_jj from the dense (m + p) information and the patient-clustered meat of both scores."""
    _, fit = fitted
    p = np.clip(fit.fitted_, 1e-10, 1 - 1e-10)
    q, r = fit.N_ * p * (1 - p), fit.outcome_ - fit.N_ * p
    m = len(fit.provider_ids_)
    Z = np.c_[np.eye(m)[fit.provider_indices_], fit.X]
    info_inv = np.linalg.inv(Z.T @ (q[:, None] * Z))
    key = pd.factorize(pd.Series(fit.provider_indices_).astype(str) + ":" + pd.Series(fit.obs_ids_).astype(str))[0]
    U = np.array([np.bincount(key, weights=Z[:, c] * r) for c in range(Z.shape[1])]).T
    V = info_inv @ (U.T @ U) @ info_inv
    np.testing.assert_allclose(np.asarray(fit.robust_variances_["gamma"]).ravel(), np.diag(V)[:m], rtol=1e-10)
    np.testing.assert_allclose(np.asarray(fit.robust_variances_["beta"]), V[m:, m:], rtol=1e-10)
    np.testing.assert_allclose(np.asarray(fit.robust_variances_["gamma_fixed_beta"]).ravel(),
                               np.diag(U.T @ U)[:m] / np.bincount(fit.provider_indices_, weights=q) ** 2, rtol=1e-12)


def test_robust_variance_options_are_validated(fitted):
    _, fit = fitted
    with pytest.raises(ValueError, match="robust_fixed_beta"):
        fit.test_standardized(measure="gamma", variance="sandwich")
    with pytest.raises(ValueError, match="Indirect measures"):
        fit.test_standardized(measure="indirect_ratio", variance="robust_fixed_beta")
