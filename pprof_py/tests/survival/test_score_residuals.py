"""Tests for `statistics/residuals.py::score_residuals`.

Two R-independent identities anchor these tests -- both exact, not
approximate, so both are checked to near machine precision rather than a
loosened tolerance:

1. `U_i(beta) == d(score(beta; weight)) / d(weight_i)`, i.e. the score
   residual is exactly the sensitivity of the total (weighted) score to
   a small perturbation of observation i's own case weight, everything
   else held fixed. This is checked by central finite differences
   against `cox_partial_likelihood`'s own score output -- the same
   engine `CoxPH.fit` optimizes -- for every observation, not a sample
   of them (a sampled check is exactly what let the bug below through
   once).
2. `sum_i weight_i * U_i(beta_hat) == total_score(beta_hat; weight)`,
   which is ~0 whenever beta_hat is the fitted MLE. Note `U_i` itself is
   NOT weight-scaled (matching R's own `residuals(fit, type="score")`
   default), so `sum_i U_i` alone is only ~0 in the unweighted case --
   an early version of these tests checked the wrong (unweighted) sum
   under sample weights and reported a false failure.

`test_brief_at_risk_window_no_events_in_it` is a permanent regression
test for a real bug: a subject who enters and is censored again with NO
event anywhere in between (their brief window is real but risk-
irrelevant, since no event's risk set ever needs to consult it) was
being folded into `denom`/`a` regardless, and then never removed, since
nothing ever marked it at-risk -- corrupting every later (smaller-time)
computation in that stratum. It surfaced only when checking every row's
finite-difference identity, not a random sample of 8; a sampled check on
the very dataset that exposed it happened not to include the affected
row.
"""
import numpy as np
import pytest

from pprof_py.models.survival.coxph import CoxPH
from pprof_py.inference.survival.residuals import score_residuals, dfbeta_residuals
from pprof_py.algorithms.survival.partial_likelihood import cox_partial_likelihood


def _assert_score_residuals_correct(X, start, stop, event, strata_codes, weight, ties, atol=1e-6):
    strata_labels = np.unique(strata_codes)
    strata_arg = strata_codes if strata_codes.max() > 0 else None
    model = CoxPH(ties=ties).fit(
        X, start=start, stop=stop, event=event, strata=strata_arg, sample_weight=weight,
    )
    beta_hat = model.coef_
    eta = X @ beta_hat
    n = X.shape[0]

    U = score_residuals(X, start, stop, event, eta, weight, strata_codes, strata_labels, ties=ties)

    # Identity 1: finite-difference sensitivity, every row.
    h = 1e-6
    for i in range(n):
        w_plus = weight.copy(); w_plus[i] += h
        w_minus = weight.copy(); w_minus[i] -= h
        _, score_plus, _ = cox_partial_likelihood(
            X, start, stop, event, beta_hat, offset=np.zeros(n), weight=w_plus, strata=strata_codes, ties=ties,
        )
        _, score_minus, _ = cox_partial_likelihood(
            X, start, stop, event, beta_hat, offset=np.zeros(n), weight=w_minus, strata=strata_codes, ties=ties,
        )
        finite_diff = (score_plus - score_minus) / (2 * h)
        np.testing.assert_allclose(
            U[i], finite_diff, atol=atol,
            err_msg=f"row {i}: score residual does not match d(score)/d(weight_i)",
        )

    # Identity 2: weighted sum reproduces the total (weighted) score, ~0 at the MLE.
    _, total_score, _ = cox_partial_likelihood(
        X, start, stop, event, beta_hat, offset=np.zeros(n), weight=weight, strata=strata_codes, ties=ties,
    )
    weighted_sum = (weight[:, None] * U).sum(axis=0)
    np.testing.assert_allclose(weighted_sum, total_score, atol=1e-6)

    return U, model


def _synthetic(n=100, seed=7, left_truncated=False, max_start=0.3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    beta_true = np.array([0.4, -0.2])
    risk = np.exp(X @ beta_true)
    if left_truncated:
        start = rng.uniform(0, max_start, n)
    else:
        start = np.zeros(n)
    stop = start + rng.exponential(1 / risk)
    censor = start + rng.exponential(2, n)
    final_stop = np.minimum(stop, censor)
    final_event = (stop <= censor).astype(float)
    return X, start, final_stop, final_event, rng


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_simple_right_censored(ties):
    X, start, stop, event, _ = _synthetic()
    _assert_score_residuals_correct(X, start, stop, event, np.zeros(100, dtype=np.int64), np.ones(100), ties)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_with_strata(ties):
    X, start, stop, event, rng = _synthetic()
    strata = rng.integers(0, 3, 100).astype(np.int64)
    _assert_score_residuals_correct(X, start, stop, event, strata, np.ones(100), ties)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_left_truncation(ties):
    """Regression coverage for the brief-at-risk-window bug: this exact
    seed/configuration reproduces it (see module docstring and
    `test_brief_at_risk_window_no_events_in_it` below for the isolated
    version).
    """
    X, start, stop, event, _ = _synthetic(left_truncated=True)
    _assert_score_residuals_correct(X, start, stop, event, np.zeros(100, dtype=np.int64), np.ones(100), ties)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_nontrivial_sample_weights(ties):
    X, start, stop, event, rng = _synthetic()
    weight = rng.uniform(0.3, 2.5, 100)
    _assert_score_residuals_correct(X, start, stop, event, np.zeros(100, dtype=np.int64), weight, ties)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_heavy_ties_strata_weights_truncation_combined(ties):
    X, start, stop, event, rng = _synthetic(left_truncated=True, max_start=0.5)
    strata = rng.integers(0, 3, 100).astype(np.int64)
    weight = rng.uniform(0.3, 2.5, 100)
    stop_ties = np.round(stop, 1)
    bad = start >= stop_ties
    stop_ties = np.where(bad, start + 0.1, stop_ties)
    _assert_score_residuals_correct(X, start, stop_ties, event, strata, weight, ties, atol=1e-5)


def test_brief_at_risk_window_no_events_in_it():
    """A subject at risk only during a window containing no event at
    all (from anyone) must contribute nothing and get a zero residual
    -- not be added to the risk-set sums without ever being removed.
    Isolated, minimal version of the bug `test_left_truncation` also
    happens to cover at a specific seed.
    """
    # subject 0: at risk (0.10, 0.20], censored, no event anywhere in that window
    # subject 1: event at 0.05 (before subject 0 even enters)
    # subject 2: event at 0.30 (after subject 0 has already left)
    # subject 3: enters at 0, censored at 1.0, at risk throughout (keeps the risk set non-degenerate)
    start = np.array([0.10, 0.0, 0.0, 0.0])
    stop = np.array([0.20, 0.05, 0.30, 1.0])
    event = np.array([0.0, 1.0, 1.0, 0.0])
    X = np.array([[1.0], [0.5], [-0.3], [0.2]])
    weight = np.ones(4)
    strata_codes = np.zeros(4, dtype=np.int64)
    strata_labels = np.array([0])

    beta = np.array([0.37])  # arbitrary, need not be the MLE for this check
    eta = X @ beta
    U = score_residuals(X, start, stop, event, eta, weight, strata_codes, strata_labels, ties="breslow")
    np.testing.assert_allclose(U[0], [0.0], atol=1e-12)

    # and the same subject's inclusion must not have corrupted anyone else's
    # residual either: compare against the identical data with subject 0
    # removed entirely (its own risk-irrelevant window shouldn't be able to
    # change anyone else's residual by construction, since dM_0(t) = 0
    # throughout it).
    U_without = score_residuals(
        X[1:], start[1:], stop[1:], event[1:], eta[1:], weight[1:],
        strata_codes[1:], strata_labels, ties="breslow",
    )
    np.testing.assert_allclose(U[1:], U_without, atol=1e-12)


def test_dfbeta_residuals_shape_and_scaling():
    X, start, stop, event, _ = _synthetic(n=40)
    weight = np.ones(40)
    strata_codes = np.zeros(40, dtype=np.int64)
    strata_labels = np.array([0])
    model = CoxPH().fit(X, start=start, stop=stop, event=event)
    eta = X @ model.coef_
    U = score_residuals(X, start, stop, event, eta, weight, strata_codes, strata_labels, ties="breslow")
    dfbeta = dfbeta_residuals(U, weight, model.covariance_)
    assert dfbeta.shape == U.shape
    np.testing.assert_allclose(dfbeta, U @ model.covariance_)
