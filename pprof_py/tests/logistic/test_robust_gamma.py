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


@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")   # added providers have no records: 0/0
def test_add_providers_keeps_every_provider_array_aligned(fitted):
    """add_providers sorts by provider ID: every per-provider array (gamma, its model and robust variances,
    sizes) moves with its provider, and provider_indices_ is remapped, so the existing providers' measures and
    tests do not change when added IDs fall between existing ones."""
    d, _ = fitted
    d = d.assign(prov=d["prov"] * 10)                             # IDs 10, 20, ...: room for IDs in between

    def fit():
        return LogisticFixedEffectModel(use_dataprep=False).fit(d, y_var="y", x_vars=["x1", "x2"], provider_var="prov",
                                                                n_var="n", obs_id_var="pid")
    before, after = fit(), fit()
    after.add_providers([15, 155, 999], gamma=[-17.0, 17.0, -17.0], se_gamma=[0.01, 0.02, 0.03])
    ids = pd.Index(before.provider_ids_)
    per_provider = {"gamma": lambda m: m.coefficients_["gamma"], "var": lambda m: m.variances_["gamma"],
                    "size": lambda m: m.provider_sizes_,
                    **{f"robust {k}": (lambda m, k=k: m.robust_variances_[k]) for k in ("gamma", "gamma_fixed_beta")}}
    for name, get in per_provider.items():
        moved = pd.Series(np.asarray(get(after)).ravel(), index=after.provider_ids_).reindex(ids)
        np.testing.assert_array_equal(moved.to_numpy(), np.asarray(get(before)).ravel(), err_msg=name)
    added = pd.Series(np.asarray(after.robust_variances_["gamma_fixed_beta"]).ravel(), index=after.provider_ids_)
    np.testing.assert_allclose(added.loc[[15, 155, 999]].to_numpy(), [0.01**2, 0.02**2, 0.03**2])
    sm_b = before.calculate_standardized_measures(stdz="indirect", reference=0.0)["indirect"].set_index("provider_id")
    sm_a = after.calculate_standardized_measures(stdz="indirect", reference=0.0)["indirect"].set_index("provider_id")
    np.testing.assert_allclose(sm_a.loc[ids, ["observed", "expected"]].to_numpy(), sm_b.loc[ids, ["observed", "expected"]].to_numpy())
    t_b = before.test(reference=0.0)
    t_a = after.test(reference=0.0)
    np.testing.assert_allclose(t_a.loc[ids, "z_raw"].to_numpy(), t_b.loc[ids, "z_raw"].to_numpy())
