"""The patient-clustered tests ``test(test_method="robust_wald")`` and ``"robust_score"``.

Checked end to end against R pprofR f23c2889 on the reference cohorts of ``test_pprofr_clustered_reference.py``
(pprof_py's own fit, at its default tolerance), against the Python mirror at pprof_py's own fit (including a cohort
whose patient labels repeat across providers), and against ``test_standardized(variance="robust_fixed_beta")``,
whose standard error ``robust_wald`` reuses. The negative controls show that the comparison with R catches the
plausible mistakes: the full-sandwich variance, residuals at the fit for the score test, and clusters by patient alone.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from pprof_py import LogisticFixedEffectModel
from pprof_py.inference import PROVIDER_TEST_COLUMNS
from pprof_py.tests.logistic import clustered_reference as mirror
from pprof_py.tests.logistic.test_pprofr_clustered_reference import COHORT_NAMES, COHORTS, INPUTS, PROVIDERS, _read, excess

CLUSTERED = ["robust_wald", "robust_score"]
X_VARS = ["x1", "x2", "x3"]


def _fit(rows, binomial=False, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LogisticFixedEffectModel().fit(rows, y_var="Y", x_vars=X_VARS, provider_var="provider",
                                              n_var="N" if binomial else None, **kwargs)


def _test(model, method, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.test(test_method=method, **kwargs)


@pytest.fixture(scope="module")
def cohorts():
    """pprof_py fits of the R reference cohorts: (model, R's per-provider values, R's cohort row, input rows)."""
    inputs, providers, meta = _read(INPUTS), _read(PROVIDERS), _read(COHORTS).set_index("cohort")
    out = {}
    for name in COHORT_NAMES:
        rows = inputs[inputs["cohort"] == name].reset_index(drop=True)
        out[name] = (_fit(rows, bool(meta.loc[name, "binomial"]), obs_id_var="patient"),
                     providers[providers["cohort"] == name].set_index("provider"), meta.loc[name], rows)
    return out


@pytest.fixture(scope="module")
def reused_labels():
    """Patient labels restart at every provider, so "K00" is a different patient at each provider."""
    rng = np.random.default_rng(20261022)
    parts = []
    for j in range(40):
        n_patients = int(rng.integers(12, 25))
        n_rows = rng.integers(1, 6, n_patients)
        u = rng.normal(0, 0.8, n_patients)
        patient = np.repeat([f"K{k:02d}" for k in range(n_patients)], n_rows)
        x = np.column_stack([rng.normal(size=patient.size), rng.binomial(1, 0.4, patient.size), rng.uniform(-1, 1, patient.size)])
        eta = rng.normal(-0.3, 0.5) + x @ [0.5, -0.4, 0.3] + np.repeat(u, n_rows)
        parts.append(pd.DataFrame({"provider": f"Q{j:02d}", "patient": patient, "Y": rng.binomial(1, 1 / (1 + np.exp(-eta))),
                                   "x1": x[:, 0], "x2": x[:, 1], "x3": x[:, 2]}))
    return _fit(pd.concat(parts, ignore_index=True), obs_id_var="patient")


def _clusters_by_patient_alone(model):
    """The wrong clustering: one cluster per patient label, charged to the provider of its first row."""
    idx = np.asarray(model.provider_indices_).ravel()
    codes, _ = pd.factorize(np.asarray(model.obs_ids_, dtype=object))
    first = np.unique(codes, return_index=True)[1]
    return codes, idx[first]


# ---- against R, end to end ------------------------------------------------------------------------------

@pytest.mark.parametrize("name", COHORT_NAMES)
def test_robust_wald_matches_pprofr(cohorts, name):
    model, R, meta, _ = cohorts[name]
    res = _test(model, "robust_wald").reindex(R.index)
    for mine, theirs in (("estimate", "gamma"), ("se", "robust_wald_se"), ("z_raw", "robust_wald_stat"),
                         ("p_value", "robust_wald_p"), ("ci_lower", "robust_wald_ci_lower"),
                         ("ci_upper", "robust_wald_ci_upper")):
        assert excess(res[mine], R[theirs]) <= 1, mine
    assert excess(res["null_value"], np.full(len(res), meta["gamma0"])) <= 1
    np.testing.assert_array_equal(res["flag"].to_numpy(dtype=float), R["robust_wald_flag"].to_numpy(dtype=float))


@pytest.mark.parametrize("name", COHORT_NAMES)
def test_robust_score_matches_pprofr(cohorts, name):
    model, R, _, _ = cohorts[name]
    res = _test(model, "robust_score").reindex(R.index)
    assert excess(res["z_raw"], R["robust_score_stat"]) <= 1
    assert excess(res["p_value"], R["robust_score_p"]) <= 1
    np.testing.assert_array_equal(res["flag"].to_numpy(dtype=float), R["robust_score_flag"].to_numpy(dtype=float))
    assert res[["se", "ci_lower", "ci_upper"]].isna().all().all()   # no intervals for the score forms (D7)


@pytest.mark.parametrize("method", CLUSTERED)
def test_attrs_describe_the_clustering(cohorts, method):
    model, _, meta, _ = cohorts["s1_spanning"]
    res = _test(model, method)
    assert res.attrs["test_method"] == method and res.attrs["variance"] == "patient-clustered, beta fixed"
    assert (res.attrs["clusters"], res.attrs["largest_cluster_rows"]) == (meta["n_clusters"], meta["max_cluster_rows"])


# ---- against the mirror at pprof_py's own fit -----------------------------------------------------------------

def _mirror(model):
    labels = np.asarray(model.provider_ids_).astype(str)
    gamma = pd.Series(np.asarray(model.coefficients_["gamma"], dtype=float).ravel(), index=labels)
    rows = labels[np.asarray(model.provider_indices_).ravel()]
    lay = mirror.layout(rows, np.asarray(model.obs_ids_).astype(str))
    out = mirror.clustered_tests(model.outcome_, model.N_, model.X, rows, np.asarray(model.obs_ids_).astype(str),
                                 gamma.loc[lay.providers].to_numpy(), np.asarray(model.coefficients_["beta"]).ravel(),
                                 gamma0=float(np.median(gamma)), lay=lay)
    return pd.DataFrame({k: out[k] for k in ("robust_wald_stat", "robust_wald_se", "robust_score_stat")},
                        index=lay.providers).loc[labels]


@pytest.mark.parametrize("name", COHORT_NAMES + ["reused_labels"])
def test_package_matches_the_mirror(cohorts, reused_labels, name):
    model = reused_labels if name == "reused_labels" else cohorts[name][0]
    want = _mirror(model)
    rw, rs = _test(model, "robust_wald"), _test(model, "robust_score")
    assert excess(rw["z_raw"], want["robust_wald_stat"]) <= 1
    assert excess(rw["se"], want["robust_wald_se"]) <= 1
    assert excess(rs["z_raw"], want["robust_score_stat"]) <= 1


def test_reused_patient_labels_are_separate_patients(reused_labels):
    res = _test(reused_labels, "robust_score")
    pairs = pd.MultiIndex.from_arrays([np.asarray(reused_labels.provider_ids_)[reused_labels.provider_indices_],
                                       reused_labels.obs_ids_]).nunique()
    assert res.attrs["clusters"] == pairs > pd.unique(reused_labels.obs_ids_).size


@pytest.mark.parametrize("name", ["t31_binomial", "s1_spanning"])
def test_robust_wald_is_test_standardized_robust_fixed_beta(cohorts, name):
    """D3: the same numbers, bit for bit, as the standardized test of gamma with variance='robust_fixed_beta'."""
    model = cohorts[name][0]
    rw = _test(model, "robust_wald")
    ts = model.test_standardized(measure="gamma", variance="robust_fixed_beta")
    assert tuple(ts.columns) == PROVIDER_TEST_COLUMNS and ts.index.equals(rw.index)
    for col in PROVIDER_TEST_COLUMNS:
        if col != "null_group":
            assert np.array_equal(rw[col].to_numpy(dtype=float), ts[col].to_numpy(dtype=float), equal_nan=True), col


# ---- behaviour ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("method", CLUSTERED)
def test_patient_ids_are_required(cohorts, method):
    model = _fit(cohorts["t13"][3])
    with pytest.raises(ValueError, match="obs_id_var"):
        model.test(test_method=method)


@pytest.mark.parametrize("method", CLUSTERED)
def test_missing_patient_ids_are_an_error(cohorts, method):
    rows = cohorts["t13"][3].assign(pid=lambda d: d["patient"].str[1:].astype(float))
    rows.loc[[0, 5], "pid"] = np.nan
    model = _fit(rows, obs_id_var="pid")
    with pytest.raises(ValueError, match="must not be missing"):
        model.test(test_method=method)


@pytest.mark.parametrize("name", ["s1_spanning", "s2_long"])
def test_row_order_does_not_matter(cohorts, name):
    """Shuffling the rows reorders providers, patients and the rows within each patient."""
    model, _, meta, rows = cohorts[name]
    other = _fit(rows.sample(frac=1, random_state=3).reset_index(drop=True), bool(meta["binomial"]), obs_id_var="patient")
    for method in CLUSTERED:
        a, b = _test(model, method), _test(other, method).reindex(model.provider_ids_)
        assert excess(b["z_raw"], a["z_raw"]) <= 1, method


@pytest.mark.parametrize("method", CLUSTERED)
def test_providers_subset_uses_every_provider(cohorts, method):
    model = cohorts["t14"][0]
    full = _test(model, method)
    ids = list(full.index[[1, 4, 7]])
    pd.testing.assert_frame_equal(_test(model, method, providers=ids), full.loc[ids])


def test_robust_wald_intervals(cohorts):
    res = _test(cohorts["s2_long"][0], "robust_wald", level=0.9)
    q = norm.ppf(0.95)
    assert excess(res["ci_lower"], res["estimate"] - q * res["se"]) <= 1
    assert excess(res["ci_upper"], res["estimate"] + q * res["se"]) <= 1


@pytest.mark.parametrize("method", CLUSTERED)
def test_no_funnel_limits_yet(cohorts, method):
    with pytest.raises(ValueError, match="no funnel limits yet"):
        cohorts["t13"][0].funnel_limits(test_method=method)


def test_robust_wald_warns_about_providers_at_the_bound(cohorts):
    rows = cohorts["t14"][3].copy()
    rows.loc[rows["provider"] == "P003", "Y"] = 0
    model = _fit(rows, obs_id_var="patient")
    with pytest.warns(UserWarning, match="no events or only events"):
        model.test(test_method="robust_wald")


# ---- negative controls ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", COHORT_NAMES)
def test_negative_controls(cohorts, name):
    """Each variant must miss R by more than 1,000 times the tolerance; clusters by patient alone only matter where
    patients are treated at more than one provider (s1_spanning)."""
    model, R, _, _ = cohorts[name]
    order = pd.Index(np.asarray(model.provider_ids_)).get_indexer(R.index)
    res = _test(model, "robust_wald").iloc[order]
    full = np.sqrt(np.asarray(model.robust_variances_["gamma"], dtype=float).ravel())[order]   # D1's full sandwich
    assert excess((res["estimate"] - res["null_value"]) / full, R["robust_wald_stat"]) > 1e3

    idx = np.asarray(model.provider_indices_).ravel()
    m = len(model.provider_ids_)
    trials = np.ones(idx.size) if model.N_ is None else np.asarray(model.N_, dtype=float)
    fitted = np.asarray(model.outcome_, dtype=float) - trials * np.asarray(model.fitted_, dtype=float)
    cluster, cluster_provider = _clusters_by_patient_alone(model)
    at_fit = np.bincount(idx, weights=fitted, minlength=m)          # the score at the fit instead of at gamma_0
    assert excess(at_fit[order] / np.sqrt(np.asarray(R["V_fit"])), R["robust_score_stat"]) > 1e3

    score = _test(model, "robust_score").iloc[order]
    p0 = np.clip(1 / (1 + np.exp(-(score["null_value"].iloc[0] + np.asarray(model.xbeta_)))), 1e-10, 1 - 1e-10)
    r0 = np.asarray(model.outcome_, dtype=float) - trials * p0
    v_bad = np.bincount(cluster_provider, weights=np.bincount(cluster, weights=r0) ** 2, minlength=m)[order]
    err = excess(np.bincount(idx, weights=r0, minlength=m)[order] / np.sqrt(v_bad), R["robust_score_stat"])
    assert err > 1e3 if name == "s1_spanning" else err <= 1
