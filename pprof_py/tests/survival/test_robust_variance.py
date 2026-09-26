"""Tests for CoxPH's robust/clustered sandwich variance
(`robust=True` / `fit(..., cluster=...)`).

Cross-validated against two independent Python implementations,
`lifelines.CoxPHFitter(robust=True, cluster_col=...)` and
`statsmodels.duration.hazard_regression.PHReg.fit(groups=...)` --
neither is this package's primary reference (R's `survival` package is,
per docs/R_COMPATIBILITY.md), but agreement with two independently-
implemented tools, on top of the exact finite-difference identity
`statistics/residuals.py`'s own tests check for the score residuals this
is built from, is strong evidence on its own.

One real discrepancy came out of this cross-validation and is worth
recording rather than silently avoiding: `lifelines.CoxPHFitter(...,
entry_col=..., robust=True)` disagrees with this package by ~9-13% under
left truncation, even with no clustering at all. `statsmodels.PHReg(...,
entry=..., ties=...).fit(groups=...)`, tested on the exact same data,
matches this package to ~0.03-0.4% instead -- and this package's own
score residuals are independently verified correct under left truncation
via an exact (not approximate) finite-difference identity, in
`test_score_residuals.py`. The conclusion drawn (see
docs/R_COMPATIBILITY.md) is that lifelines' robust variance has a
limitation combining `entry_col` with `robust`/`cluster_col`, not that
this package is wrong -- but this is exactly the kind of mismatch this
project's own methodology says to investigate rather than shrug off, so
the tests below deliberately use statsmodels, not lifelines, for every
case involving left truncation, and lifelines only where the two
agreed.
"""
import numpy as np
import pandas as pd
import pytest

from pprof_py.models.survival.coxph import CoxPH

lifelines = pytest.importorskip("lifelines")
statsmodels_phreg = pytest.importorskip("statsmodels.duration.hazard_regression")
from lifelines import CoxPHFitter
from statsmodels.duration.hazard_regression import PHReg


def _synthetic(n=250, seed=11, p=3, left_truncated=False, max_start=0.5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta_true = np.linspace(0.5, -0.3, p)
    risk = np.exp(X @ beta_true)
    start = rng.uniform(0, max_start, n) if left_truncated else np.zeros(n)
    stop_event = start + rng.exponential(1 / risk)
    censor = start + rng.exponential(2, n)
    stop = np.minimum(stop_event, censor)
    event = (stop_event <= censor).astype(float)
    return X, start, stop, event, rng


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_robust_no_cluster_matches_lifelines(ties):
    X, start, stop, event, _ = _synthetic()
    ours = CoxPH(ties=ties, robust=True).fit(X, duration=stop, event=event)

    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    df["stop"] = stop
    df["event"] = event
    ll = CoxPHFitter(baseline_estimation_method="breslow").fit(
        df, duration_col="stop", event_col="event", robust=True,
    )
    np.testing.assert_allclose(ours.coef_, ll.params_.to_numpy(), atol=1e-4)
    np.testing.assert_allclose(ours.standard_errors_, ll.standard_errors_.to_numpy(), rtol=1e-3)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_clustered_matches_lifelines_and_statsmodels(ties):
    X, start, stop, event, rng = _synthetic(n=300)
    group = rng.integers(0, 60, X.shape[0])
    ours = CoxPH(ties=ties).fit(X, duration=stop, event=event, cluster=group)

    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    df["stop"] = stop
    df["event"] = event
    df["grp"] = group
    formula = "+".join(df.columns[: X.shape[1]])
    ll = CoxPHFitter(baseline_estimation_method="breslow").fit(
        df, duration_col="stop", event_col="event", cluster_col="grp", formula=formula,
    )
    sm_result = PHReg(stop, X, status=event, ties=ties).fit(groups=group)

    np.testing.assert_allclose(ours.coef_, ll.params_.to_numpy(), atol=1e-4)
    np.testing.assert_allclose(ours.standard_errors_, ll.standard_errors_.to_numpy(), rtol=1e-3)
    np.testing.assert_allclose(ours.standard_errors_, sm_result.bse, rtol=1e-3)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_strata_with_cluster_matches_lifelines(ties):
    """Strata alone (no truncation) is where lifelines and this package
    agree -- see module docstring for why truncation cases below use
    statsmodels instead.
    """
    X, start, stop, event, rng = _synthetic(n=250, p=2)
    strata = rng.integers(0, 3, X.shape[0])
    group = rng.integers(0, 50, X.shape[0])
    ours = CoxPH(ties=ties).fit(X, duration=stop, event=event, strata=strata, cluster=group)

    df = pd.DataFrame(X, columns=["x0", "x1"])
    df["stop"] = stop
    df["event"] = event
    df["strata"] = strata
    df["grp"] = group
    ll = CoxPHFitter(baseline_estimation_method="breslow").fit(
        df, duration_col="stop", event_col="event", strata="strata", cluster_col="grp", formula="x0+x1",
    )
    np.testing.assert_allclose(ours.coef_, ll.params_.to_numpy(), atol=1e-4)
    np.testing.assert_allclose(ours.standard_errors_, ll.standard_errors_.to_numpy(), rtol=1e-3)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_robust_with_left_truncation_matches_statsmodels_not_lifelines(ties):
    X, start, stop, event, _ = _synthetic(p=2, left_truncated=True)
    ours = CoxPH(ties=ties, robust=True).fit(X, start=start, stop=stop, event=event)
    sm_result = PHReg(stop, X, status=event, entry=start, ties=ties).fit(groups=np.arange(X.shape[0]))
    np.testing.assert_allclose(ours.standard_errors_, sm_result.bse, rtol=5e-3)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_clustered_with_left_truncation_matches_statsmodels(ties):
    X, start, stop, event, rng = _synthetic(p=2, left_truncated=True)
    group = rng.integers(0, 50, X.shape[0])
    ours = CoxPH(ties=ties).fit(X, start=start, stop=stop, event=event, cluster=group)
    sm_result = PHReg(stop, X, status=event, entry=start, ties=ties).fit(groups=group)
    np.testing.assert_allclose(ours.standard_errors_, sm_result.bse, rtol=1e-2)


def test_cluster_argument_activates_robust_even_when_constructor_says_false():
    X, start, stop, event, rng = _synthetic(n=150, p=2)
    group = rng.integers(0, 30, X.shape[0])
    m = CoxPH(robust=False).fit(X, duration=stop, event=event, cluster=group)
    assert not np.allclose(m.covariance_, m.naive_covariance_)
    assert m.n_clusters_ <= 30


def test_no_robust_no_cluster_leaves_covariance_naive():
    X, start, stop, event, _ = _synthetic(n=100, p=2)
    m = CoxPH().fit(X, duration=stop, event=event)
    np.testing.assert_array_equal(m.covariance_, m.naive_covariance_)
    assert m.n_clusters_ == m.n_obs_


def test_coef_unaffected_by_robust_or_cluster_choice():
    X, start, stop, event, rng = _synthetic(n=150, p=2)
    group = rng.integers(0, 30, X.shape[0])
    m_plain = CoxPH().fit(X, duration=stop, event=event)
    m_robust = CoxPH(robust=True).fit(X, duration=stop, event=event)
    m_cluster = CoxPH().fit(X, duration=stop, event=event, cluster=group)
    np.testing.assert_allclose(m_plain.coef_, m_robust.coef_)
    np.testing.assert_allclose(m_plain.coef_, m_cluster.coef_)


def test_cluster_wrong_length_raises():
    X, start, stop, event, _ = _synthetic(n=50, p=2)
    with pytest.raises(ValueError):
        CoxPH().fit(X, duration=stop, event=event, cluster=np.arange(49))


def test_robust_matches_real_r_auto_promoted_variance():
    """R itself promotes `fit$var`/`vcov(fit)` to the robust sandwich
    estimate whenever weights aren't plain integer replication counts,
    even with no explicit `robust=TRUE` and no `cluster=` -- see
    docs/R_COMPATIBILITY.md Section 5. Phase 1-3's own R reference run
    already captured this exact scenario (`weights_efron_naive_vs_robust.csv`,
    real R output, not a third-party approximation) even though robust
    variance was out of scope until now -- this is the strongest
    available check, so it doesn't need a new R run to confirm.
    """
    import os
    import pandas as pd

    base = os.path.join(os.path.dirname(__file__), "..", "..", "r_reference")
    data_path = os.path.join(base, "data", "weights.csv")
    result_path = os.path.join(base, "results", "weights_efron_naive_vs_robust.csv")
    if not (os.path.exists(data_path) and os.path.exists(result_path)):
        pytest.skip("r_reference/data or results not present in this checkout")

    d = pd.read_csv(data_path)
    r = pd.read_csv(result_path)
    X = d[["x1", "x2"]].to_numpy()
    model = CoxPH(ties="efron", robust=True).fit(
        X, duration=d["stop"].to_numpy(), event=d["event"].to_numpy(), sample_weight=d["weight"].to_numpy(),
    )
    np.testing.assert_allclose(model.standard_errors_, r["se_robust"].to_numpy(), rtol=1e-8)
    np.testing.assert_allclose(model.naive_covariance_.diagonal() ** 0.5, r["se_naive"].to_numpy(), rtol=1e-8)



def test_few_clusters_warn():
    import warnings as _w
    from pprof_py.inference.survival.robust import robust_covariance
    naive = np.eye(2)
    with pytest.warns(UserWarning, match="fewer than 30"):
        robust_covariance(naive, np.ones((5, 2)))
    with _w.catch_warnings():
        _w.simplefilter("error", UserWarning)
        robust_covariance(naive, np.ones((40, 2)))
