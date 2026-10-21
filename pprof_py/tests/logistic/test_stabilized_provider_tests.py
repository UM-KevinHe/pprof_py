"""The stabilized tests ``test(test_method="stabilized_wald")`` and ``"stabilized_score"``, and the row order behind them.

A patient's rows are taken in the order recorded at fit: by ``order_var`` (ties in input order) or, without it, in
input order with a warning. Checked end to end against R pprofR f23c2889 on the reference cohorts of
``test_pprofr_clustered_reference.py`` in both ways: the rows as R received them without ``order_var``, and the rows
shuffled with ``order_var="time"``. Also against the Python mirror at pprof_py's own fit, and, with R-hat forced to
the identity, against the beta-fixed model-based tests.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from scipy.stats import norm

from pprof_py import LogisticFixedEffectModel
from pprof_py.inference.logistic.fixed_effect.provider_tests import _patient_clusters, _stabilized_meat
from pprof_py.tests.logistic import clustered_reference as mirror
from pprof_py.tests.logistic.test_pprofr_clustered_reference import COHORT_NAMES, COHORTS, INPUTS, PROVIDERS, _read, excess
from pprof_py.tests.logistic.test_robust_provider_tests import reused_labels  # noqa: F401  (fixture)

STABILIZED = ["stabilized_wald", "stabilized_score"]
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
    """Per cohort: fits in input order and with order_var on shuffled rows; R's values; R's cohort row; the rows."""
    inputs, providers, meta = _read(INPUTS), _read(PROVIDERS), _read(COHORTS).set_index("cohort")
    out = {}
    for name in COHORT_NAMES:
        rows = inputs[inputs["cohort"] == name].reset_index(drop=True)
        binomial = bool(meta.loc[name, "binomial"])
        shuffled = rows.sample(frac=1, random_state=5).reset_index(drop=True)
        models = {"input": _fit(rows, binomial, obs_id_var="patient"),
                  "order_var": _fit(shuffled, binomial, obs_id_var="patient", order_var="time")}
        out[name] = (models, providers[providers["cohort"] == name].set_index("provider"), meta.loc[name], rows)
    return out


# ---- against R, end to end ------------------------------------------------------------------------------

@pytest.mark.parametrize("order", ["input", "order_var"])
@pytest.mark.parametrize("name", COHORT_NAMES)
def test_stabilized_tests_match_pprofr(cohorts, name, order):
    models, R, meta, _ = cohorts[name]
    sw = _test(models[order], "stabilized_wald").reindex(R.index)
    ss = _test(models[order], "stabilized_score").reindex(R.index)
    for got, want in ((sw["se"], R["stabilized_wald_se"]), (sw["z_raw"], R["stabilized_wald_stat"]),
                      (sw["p_value"], R["stabilized_wald_p"]), (ss["z_raw"], R["stabilized_score_stat"]),
                      (ss["p_value"], R["stabilized_score_p"]), (sw["null_value"], np.full(len(sw), meta["gamma0"]))):
        assert excess(got, want) <= 1, want.name
    for res, col in ((sw, "stabilized_wald_flag"), (ss, "stabilized_score_flag")):
        np.testing.assert_array_equal(res["flag"].to_numpy(dtype=float), R[col].to_numpy(dtype=float))
    source = "input position" if order == "input" else "order_var 'time'"
    for res in (sw, ss):
        assert res.attrs["row_order"] == source and res.attrs["variance"] == "patient-clustered, stabilized, beta fixed"
        assert (res.attrs["clusters"], res.attrs["largest_cluster_rows"]) == (meta["n_clusters"], meta["max_cluster_rows"])


def test_input_order_warns_and_order_var_does_not(cohorts):
    models = cohorts["s1_spanning"][0]
    for method in STABILIZED:
        with pytest.warns(UserWarning, match="input order"):
            models["input"].test(test_method=method)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            models["order_var"].test(test_method=method)
        assert not [w for w in caught if "input order" in str(w.message)]


def test_shuffled_rows_without_an_order_key_change_only_the_stabilized_tests(cohorts):
    """Negative control: the order matters only through the order key, and only for the stabilized tests."""
    _, R, meta, rows = cohorts["s2_long"]
    model = _fit(rows.sample(frac=1, random_state=5).reset_index(drop=True), True, obs_id_var="patient")
    for method in STABILIZED:
        assert excess(_test(model, method).reindex(R.index)["z_raw"], R[f"{method}_stat"]) > 1e3, method
    for method in ("robust_wald", "robust_score"):
        assert excess(_test(model, method).reindex(R.index)["z_raw"], R[f"{method}_stat"]) <= 1, method


# ---- against the mirror at pprof_py's own fit, and the identity check ---------------------------------------

def _mirror(model):
    order = np.argsort(model.order_key_)                 # each patient's rows in the recorded order
    labels = np.asarray(model.provider_ids_).astype(str)
    gamma = pd.Series(np.asarray(model.coefficients_["gamma"], dtype=float).ravel(), index=labels)
    rows = labels[np.asarray(model.provider_indices_).ravel()][order]
    patients = np.asarray(model.obs_ids_).astype(str)[order]
    lay = mirror.layout(rows, patients)
    out = mirror.clustered_tests(np.asarray(model.outcome_)[order], np.asarray(model.N_)[order], np.asarray(model.X)[order],
                                 rows, patients, gamma.loc[lay.providers].to_numpy(),
                                 np.asarray(model.coefficients_["beta"]).ravel(), gamma0=float(np.median(gamma)), lay=lay)
    return pd.DataFrame({k: out[k] for k in ("stabilized_wald_stat", "stabilized_wald_se", "stabilized_score_stat")},
                        index=lay.providers).loc[labels]


@pytest.mark.parametrize("name", COHORT_NAMES + ["reused_labels"])
def test_package_matches_the_mirror(cohorts, reused_labels, name):
    model = reused_labels if name == "reused_labels" else cohorts[name][0]["order_var"]
    want = _mirror(model)
    sw, ss = _test(model, "stabilized_wald"), _test(model, "stabilized_score")
    assert excess(sw["z_raw"], want["stabilized_wald_stat"]) <= 1
    assert excess(sw["se"], want["stabilized_wald_se"]) <= 1
    assert excess(ss["z_raw"], want["stabilized_score_stat"]) <= 1


def _ingredients(model, at):
    """Residuals, variances, clusters and provider count as test() builds them, at the fit or at gamma_0."""
    idx = np.asarray(model.provider_indices_).ravel()
    w = np.asarray(model.N_, dtype=float)
    gamma = np.asarray(model.coefficients_["gamma"], dtype=float).ravel()
    p = np.asarray(model.fitted_) if at == "fit" else expit(np.median(gamma) + np.asarray(model.xbeta_))
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return (np.asarray(model.outcome_, dtype=float) - w * p, w * p * (1 - p),
            _patient_clusters(idx, model.obs_ids_), gamma.size, idx)


@pytest.mark.parametrize("name", COHORT_NAMES)
def test_identity_rhat_gives_the_beta_fixed_model_based_tests(cohorts, name):
    """With R-hat the identity, the stabilized score test is the score test and the Wald SE is 1 / sqrt(I)."""
    model = cohorts[name][0]["order_var"]
    r, v, clusters, m, idx = _ingredients(model, "null")
    meat, _ = _stabilized_meat(r, v, clusters, model.order_key_, m, rhat=np.eye(int(clusters[2].max())))
    assert excess(np.bincount(idx, weights=r, minlength=m) / np.sqrt(meat), _test(model, "score")["z_raw"]) <= 1
    r, v, clusters, m, idx = _ingredients(model, "fit")
    meat, _ = _stabilized_meat(r, v, clusters, model.order_key_, m, rhat=np.eye(int(clusters[2].max())))
    assert excess(np.sqrt(meat), np.sqrt(np.bincount(idx, weights=v, minlength=m))) <= 1


# ---- invariance ---------------------------------------------------------------------------------------------

def test_reordering_providers_and_patients_changes_nothing(cohorts):
    """Patients and providers in another order, the rows of each (provider, patient) cluster in the same relative order."""
    models, _, meta, rows = cohorts["s1_spanning"]
    desc = {col: pd.Categorical(rows[col], categories=sorted(rows[col].unique(), reverse=True)).codes
            for col in ("provider", "patient")}
    moved = rows.iloc[np.lexsort((desc["patient"], desc["provider"]))].reset_index(drop=True)   # stable: ties keep order
    assert not moved["row_id"].equals(rows["row_id"])
    assert moved.groupby(["provider", "patient"], sort=False)["row_id"].apply(lambda r: r.is_monotonic_increasing).all()
    other = _fit(moved, obs_id_var="patient")
    for method in STABILIZED:
        assert excess(_test(other, method).reindex(models["input"].provider_ids_)["z_raw"],
                      _test(models["input"], method)["z_raw"]) <= 1, method


def test_order_key_survives_data_preparation(cohorts):
    models, _, _, rows = cohorts["s2_long"]
    plain = _fit(rows, True, obs_id_var="patient", use_dataprep=False)
    for method in STABILIZED:
        assert excess(_test(plain, method)["z_raw"], _test(models["input"], method).reindex(plain.provider_ids_)["z_raw"]) <= 1
    key = models["input"].order_key_
    assert np.array_equal(np.sort(key), np.arange(len(rows)))


def test_ties_in_order_var_keep_the_input_order(cohorts):
    models, _, _, rows = cohorts["t13"]
    tied = _fit(rows.assign(day=0), obs_id_var="patient", order_var="day")
    assert tied.order_source_ == "order_var 'day'"
    for method in STABILIZED:
        pd.testing.assert_series_equal(_test(tied, method)["z_raw"], _test(models["input"], method)["z_raw"])


# ---- behaviour ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("method", STABILIZED)
def test_patient_ids_are_required(cohorts, method):
    with pytest.raises(ValueError, match="obs_id_var"):
        _fit(cohorts["t13"][3]).test(test_method=method)


def test_order_var_must_be_a_complete_column(cohorts):
    rows = cohorts["t13"][3]
    with pytest.raises(ValueError, match="not a column"):
        _fit(rows, obs_id_var="patient", order_var="visit")
    with pytest.raises(ValueError, match="missing values"):
        _fit(rows.assign(time=rows["time"].where(rows.index != 4)), obs_id_var="patient", order_var="time")
    with pytest.raises(ValueError, match="DataFrame"):
        LogisticFixedEffectModel().fit(rows[X_VARS].to_numpy(), rows["Y"].to_numpy(), rows["provider"].to_numpy(),
                                       order_var="time")


@pytest.mark.parametrize("method", STABILIZED)
def test_at_most_5000_rows_per_patient(method):
    rng = np.random.default_rng(1)
    long = pd.DataFrame({"provider": "A", "patient": "p0", "x1": rng.normal(size=5001)})
    rest = pd.DataFrame({"provider": np.repeat(["A", "B"], 40), "patient": [f"q{i}" for i in range(80)],
                         "x1": rng.normal(size=80)})
    rows = pd.concat([long, rest], ignore_index=True).assign(Y=lambda d: rng.binomial(1, 0.3, len(d)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LogisticFixedEffectModel().fit(rows, y_var="Y", x_vars=["x1"], provider_var="provider",
                                               obs_id_var="patient")
        with pytest.raises(ValueError, match="5001 rows"):
            model.test(test_method=method)


@pytest.mark.parametrize("method", STABILIZED)
def test_providers_subset_uses_every_provider(cohorts, method):
    model = cohorts["t14"][0]["order_var"]
    full = _test(model, method)
    ids = list(full.index[[1, 4, 7]])
    pd.testing.assert_frame_equal(_test(model, method, providers=ids), full.loc[ids])


def test_intervals(cohorts):
    model = cohorts["s2_long"][0]["order_var"]
    res = _test(model, "stabilized_wald", level=0.9)
    q = norm.ppf(0.95)
    assert excess(res["ci_lower"], res["estimate"] - q * res["se"]) <= 1
    assert excess(res["ci_upper"], res["estimate"] + q * res["se"]) <= 1
    assert _test(model, "stabilized_score")[["se", "ci_lower", "ci_upper"]].isna().all().all()


@pytest.mark.parametrize("method", STABILIZED)
def test_no_funnel_limits_yet(cohorts, method):
    with pytest.raises(ValueError, match="no funnel limits yet"):
        cohorts["t13"][0]["order_var"].funnel_limits(test_method=method)


# ---- negative controls ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", COHORT_NAMES)
def test_negative_controls(cohorts, name):
    """R-hat built from the fitted residuals for the score test (D5's alternative), or divided by every cluster instead
    of those long enough, must miss R by more than 1,000 times the tolerance."""
    models, R, _, _ = cohorts[name]
    model = models["order_var"]
    order = pd.Index(np.asarray(model.provider_ids_)).get_indexer(R.index)
    r, v, clusters, m, idx = _ingredients(model, "null")
    _, rhat_fit = _stabilized_meat(*_ingredients(model, "fit")[:3], model.order_key_, m)
    meat, _ = _stabilized_meat(r, v, clusters, model.order_key_, m, rhat=rhat_fit)
    u = np.bincount(idx, weights=r, minlength=m)
    assert excess((u / np.sqrt(meat))[order], R["stabilized_score_stat"]) > 1e3
    _, rhat = _stabilized_meat(r, v, clusters, model.order_key_, m)
    n_rows = clusters[2]
    longer = np.array([(n_rows > q).sum() for q in range(rhat.shape[0])])
    over_all = rhat * longer[np.maximum.outer(np.arange(rhat.shape[0]), np.arange(rhat.shape[0]))] / n_rows.size
    meat, _ = _stabilized_meat(r, v, clusters, model.order_key_, m, rhat=over_all)
    assert excess((u / np.sqrt(meat))[order], R["stabilized_score_stat"]) > 1e3
