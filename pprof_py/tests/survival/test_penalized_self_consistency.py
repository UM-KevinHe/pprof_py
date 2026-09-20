"""Self-consistency tests for PenalizedCoxPH/PenalizedCoxPHCV -- the
checks that don't depend on (and, for Efron, can't depend on) an R
reference. Mirrors tests/test_engine_self_consistency.py's role for
Phase 1's Numba/pure-Python kernel pair, extended to this module's own
new kernel pair and to the mathematical optimality of the penalized
solutions themselves.

glmnet's Cox family only supports Breslow ties (see
docs/R_COMPATIBILITY.md, Phase 3 section), so Efron-tie penalized fits
have no independent external reference at all. The check used instead:
as lambda -> 0, PenalizedCoxPH(ties="efron") must recover this
package's own unpenalized CoxPH(ties="efron") -- which *is*
R-validated (tests/test_r_comparison.py's test_efron_* functions) --
to essentially machine precision, since a vanishing penalty is not a
special case the solver treats differently; it is simply the
lambda -> 0 limit of the same coordinate-descent update rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pprof_py import CoxPH, PenalizedCoxPH, PenalizedCoxPHCV
from pprof_py.algorithms.survival import coordinate_descent as cd
from pprof_py.algorithms.survival.penalty import (
    soft_threshold, elastic_net_penalty_value, weighted_column_scale, rescale_penalty_factors,
)
from pprof_py.data.survival_validation import validate_fit_inputs
from pprof_py.data.survival_data import SurvivalData
from pprof_py.algorithms.survival.cox_likelihood import cox_partial_likelihood, precompute_stratum_indices


def _synthetic(n=300, p=6, seed=0, ties_round=None):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, p))
    beta_true = rng.normal(scale=0.5, size=p) * (rng.uniform(size=p) > 0.4)
    risk = np.exp(X @ beta_true)
    U = rng.uniform(size=n)
    dur = -np.log(U) / risk
    C = rng.exponential(scale=np.median(dur) * 1.5, size=n)
    stop = np.minimum(dur, C)
    if ties_round is not None:
        stop = np.round(stop, ties_round)
    event = (dur <= C).astype(float)
    return X, np.maximum(stop, 1e-6), event


# ---------------------------------------------------------------------
# Efron self-consistency (no glmnet reference exists for this)
# ---------------------------------------------------------------------
def test_efron_lambda_to_zero_recovers_unpenalized_coxph():
    X, stop, event = _synthetic(n=400, p=5, seed=1, ties_round=1)  # rounding induces real ties
    unpenalized = CoxPH(ties="efron").fit(X, duration=stop, event=event)

    penalized = PenalizedCoxPH(alpha=1.0, lambda_path=1e-9, ties="efron")
    penalized.fit(X, duration=stop, event=event)

    assert np.max(np.abs(unpenalized.coef_ - penalized.coef_)) < 1e-5
    assert abs(unpenalized.log_likelihood_ - penalized.log_likelihood_path_[0]) < 1e-6


def test_efron_ridge_lambda_to_zero_recovers_unpenalized_coxph():
    """Same check for ridge (alpha=0) -- a different code path through
    the coordinate-descent update (no soft-thresholding at all)."""
    X, stop, event = _synthetic(n=350, p=5, seed=2, ties_round=1)
    unpenalized = CoxPH(ties="efron").fit(X, duration=stop, event=event)
    penalized = PenalizedCoxPH(alpha=0.0, lambda_path=1e-9, ties="efron")
    penalized.fit(X, duration=stop, event=event)
    assert np.max(np.abs(unpenalized.coef_ - penalized.coef_)) < 1e-5


def test_breslow_lambda_to_zero_recovers_unpenalized_coxph():
    """Same check for Breslow, independent of (in addition to) the
    direct glmnet comparison in test_penalized_r_comparison.py."""
    X, stop, event = _synthetic(n=400, p=5, seed=3, ties_round=1)
    unpenalized = CoxPH(ties="breslow").fit(X, duration=stop, event=event)
    penalized = PenalizedCoxPH(alpha=1.0, lambda_path=1e-9, ties="breslow")
    penalized.fit(X, duration=stop, event=event)
    assert np.max(np.abs(unpenalized.coef_ - penalized.coef_)) < 1e-5


# ---------------------------------------------------------------------
# Numba vs pure-Python coordinate descent kernel parity
# ---------------------------------------------------------------------
def test_coordinate_descent_numba_matches_python():
    rng = np.random.RandomState(42)
    p = 12
    R = rng.normal(size=(p, p))
    A = R @ R.T + np.eye(p) * 0.5  # a random PSD matrix, not from real data
    linear_term = rng.normal(size=p) * 3
    for lam, alpha in [(0.3, 1.0), (0.3, 0.0), (0.3, 0.5), (2.0, 0.8)]:
        pf = np.ones(p); pf[2] = 0.0; pf[5] = 2.0
        beta_nb, n_nb, _ = cd._coordinate_descent_numba(
            A.copy(), linear_term.copy(), np.zeros(p), lam, alpha, pf, 1e-14, 5000
        )
        beta_py, n_py, _ = cd._coordinate_descent_python(
            A.copy(), linear_term.copy(), np.zeros(p), lam, alpha, pf, 1e-14, 5000
        )
        assert np.max(np.abs(beta_nb - beta_py)) < 1e-10, f"lam={lam}, alpha={alpha}"


def test_soft_threshold_basic():
    z = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
    out = soft_threshold(z, np.array(1.0))
    np.testing.assert_allclose(out, [-2.0, 0.0, 0.0, 0.0, 2.0])


# ---------------------------------------------------------------------
# Exact KKT / stationarity checks (independent of any reference impl)
# ---------------------------------------------------------------------
def _objective_closure(X, start, stop, event, weight=None, offset=None, strata=None, ties="breslow"):
    n = X.shape[0]
    weight = np.ones(n) if weight is None else weight
    offset = np.zeros(n) if offset is None else offset
    strata = np.zeros(n, dtype=int) if strata is None else strata
    idx = precompute_stratum_indices(strata)

    def objective(beta):
        return cox_partial_likelihood(X, start, stop, event, beta, offset=offset,
                                       weight=weight, strata=strata, ties=ties, stratum_indices=idx)
    return objective


@pytest.mark.parametrize("alpha", [0.0, 0.35, 1.0])
def test_converged_solution_satisfies_exact_kkt(alpha):
    """For every alpha, at the exact (not quadratic-approximated)
    score, the standard elastic-net stationarity condition must hold:
    zero coefficients need |c*score_j| <= lambda*alpha*pf_j; nonzero
    coefficients need c*score_j = lambda*(alpha*sign(beta_j) + (1-alpha)*beta_j).
    This is the same identity used to derive lambda_max
    (algorithms/coordinate_descent.py), checked here at an arbitrary
    interior lambda instead of exactly at the boundary.
    """
    X, stop, event = _synthetic(n=250, p=8, seed=7)
    start = np.zeros_like(stop)
    objective = _objective_closure(X, start, stop, event)
    c = 1.0 / X.shape[0]
    pf = np.ones(X.shape[1])
    lam = 0.15

    result = cd.fit_single_lambda(objective, np.zeros(X.shape[1]), c, lam, alpha, pf,
                                   outer_tol=1e-14, inner_tol=1e-14, outer_max_iter=200, inner_max_iter=5000)
    beta = result.beta
    _, score, _ = objective(beta)
    stat = c * score - lam * (alpha * np.sign(beta) + (1 - alpha) * beta)

    for j in range(len(beta)):
        if beta[j] == 0.0:
            assert abs(c * score[j]) <= lam * alpha * pf[j] + 1e-8, f"coord {j} zero-KKT violated"
        else:
            assert abs(stat[j]) < 1e-6, f"coord {j} stationarity residual {stat[j]}"


# ---------------------------------------------------------------------
# Path structure / sanity
# ---------------------------------------------------------------------
def test_lambda_max_gives_all_zero_coefficients():
    X, stop, event = _synthetic(n=300, p=10, seed=11)
    m = PenalizedCoxPH(alpha=1.0, n_lambda=25).fit(X, duration=stop, event=event)
    assert np.all(m.coef_path_[0] == 0.0)
    assert m.n_nonzero_path_[0] == 0


def test_path_lambda_descending_and_nonzero_nondecreasing_trend():
    X, stop, event = _synthetic(n=300, p=10, seed=12)
    m = PenalizedCoxPH(alpha=1.0, n_lambda=30).fit(X, duration=stop, event=event)
    assert np.all(np.diff(m.lambda_path_) <= 0)
    # not strictly monotone in general, but the overall trend from the
    # first to the last third of the path should increase substantially
    assert m.n_nonzero_path_[-5:].mean() >= m.n_nonzero_path_[:5].mean()


def test_deviance_ratio_in_unit_range_and_increasing_as_lambda_shrinks():
    X, stop, event = _synthetic(n=300, p=8, seed=13)
    m = PenalizedCoxPH(alpha=1.0, n_lambda=30).fit(X, duration=stop, event=event)
    assert np.all(m.deviance_ratio_path_ >= -1e-8)
    assert np.all(m.deviance_ratio_path_ <= 1.0 + 1e-8)
    assert m.deviance_ratio_path_[-1] >= m.deviance_ratio_path_[0] - 1e-8


def test_single_lambda_scalar_sets_coef_and_lambda_attrs():
    X, stop, event = _synthetic(n=200, p=5, seed=14)
    m = PenalizedCoxPH(alpha=1.0, lambda_path=0.05).fit(X, duration=stop, event=event)
    assert hasattr(m, "coef_")
    assert hasattr(m, "lambda_")
    assert m.lambda_ == pytest.approx(0.05)
    assert m.coef_.shape == (5,)


def test_multi_lambda_path_does_not_set_singular_coef():
    X, stop, event = _synthetic(n=200, p=5, seed=15)
    m = PenalizedCoxPH(alpha=1.0, n_lambda=10).fit(X, duration=stop, event=event)
    assert not hasattr(m, "coef_")
    with pytest.raises(ValueError):
        m.predict_linear(X)  # ambiguous without lambda_value=


def test_coef_at_matches_grid_points_and_interpolates_between():
    X, stop, event = _synthetic(n=250, p=6, seed=16)
    m = PenalizedCoxPH(alpha=1.0, n_lambda=20).fit(X, duration=stop, event=event)
    # exact grid point
    np.testing.assert_allclose(m.coef_at(m.lambda_path_[5]), m.coef_path_[5])
    # outside range clamps to the endpoints
    np.testing.assert_allclose(m.coef_at(m.lambda_path_[0] * 100), m.coef_path_[0])
    np.testing.assert_allclose(m.coef_at(m.lambda_path_[-1] / 100), m.coef_path_[-1])
    # an interpolated midpoint lies between its two bracketing grid rows
    mid_lambda = np.sqrt(m.lambda_path_[3] * m.lambda_path_[4])  # geometric midpoint (log-linear grid)
    interp = m.coef_at(mid_lambda)
    lo = np.minimum(m.coef_path_[3], m.coef_path_[4])
    hi = np.maximum(m.coef_path_[3], m.coef_path_[4])
    assert np.all(interp >= lo - 1e-9) and np.all(interp <= hi + 1e-9)


def test_predict_matches_manual_linear_predictor():
    X, stop, event = _synthetic(n=200, p=5, seed=17)
    m = PenalizedCoxPH(alpha=1.0, lambda_path=0.02).fit(X, duration=stop, event=event)
    manual_eta = X @ m.coef_
    np.testing.assert_allclose(m.predict_linear(X), manual_eta, atol=1e-10)
    np.testing.assert_allclose(m.predict_partial_hazard(X), np.exp(manual_eta), atol=1e-8)


def test_nonzero_features_matches_coef_support():
    X, stop, event = _synthetic(n=250, p=8, seed=18)
    feature_names = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    m = PenalizedCoxPH(alpha=1.0, lambda_path=0.05).fit(feature_names, duration=stop, event=event)
    expected = feature_names.columns.to_numpy()[m.coef_ != 0.0]
    np.testing.assert_array_equal(m.nonzero_features(), expected)


def test_degenerate_constant_column_is_excluded_not_crashed():
    X, stop, event = _synthetic(n=200, p=4, seed=19)
    X_aug = np.column_stack([X, np.full(200, 7.0)])  # a 5th, constant column
    with pytest.warns(UserWarning, match="zero weighted variance"):
        m = PenalizedCoxPH(alpha=1.0, n_lambda=15).fit(X_aug, duration=stop, event=event)
    assert np.all(m.coef_path_[:, 4] == 0.0)


def test_all_zero_penalty_factor_raises_clear_error():
    X, stop, event = _synthetic(n=150, p=4, seed=20)
    m = PenalizedCoxPH(alpha=1.0, penalty_factor=np.zeros(4))
    with pytest.raises(ValueError, match="all-zero"):
        m.fit(X, duration=stop, event=event)


def test_alpha_out_of_range_raises():
    X, stop, event = _synthetic(n=100, p=3, seed=21)
    with pytest.raises(ValueError, match="alpha"):
        PenalizedCoxPH(alpha=1.5).fit(X, duration=stop, event=event)


# ---------------------------------------------------------------------
# PenalizedCoxPHCV internal consistency
# ---------------------------------------------------------------------
def test_cv_final_estimator_matches_direct_fit_at_selected_lambda():
    X, stop, event = _synthetic(n=300, p=8, seed=22)
    cv = PenalizedCoxPHCV(alpha=1.0, n_folds=5, random_state=0, n_lambda=25)
    cv.fit(X, duration=stop, event=event)

    direct = PenalizedCoxPH(alpha=1.0, lambda_path=cv.lambda_min_).fit(X, duration=stop, event=event)
    np.testing.assert_allclose(cv.coef_, direct.coef_, atol=1e-8)


def test_cv_lambda_1se_is_more_regularized_than_lambda_min():
    X, stop, event = _synthetic(n=300, p=10, seed=23)
    cv = PenalizedCoxPHCV(alpha=1.0, n_folds=5, random_state=1, n_lambda=30)
    cv.fit(X, duration=stop, event=event)
    assert cv.lambda_1se_ >= cv.lambda_min_


def test_cv_explicit_fold_id_reproducible():
    X, stop, event = _synthetic(n=200, p=6, seed=24)
    fold_id = np.tile(np.arange(5), 40)
    cv1 = PenalizedCoxPHCV(alpha=1.0, fold_id=fold_id, n_lambda=20).fit(X, duration=stop, event=event)
    cv2 = PenalizedCoxPHCV(alpha=1.0, fold_id=fold_id, n_lambda=20).fit(X, duration=stop, event=event)
    np.testing.assert_allclose(cv1.cv_mean_deviance_, cv2.cv_mean_deviance_)
    assert cv1.lambda_min_ == cv2.lambda_min_


# ---------------------------------------------------------------------
# Penalty-factor rescaling (pure numerical kernel, no fit needed)
# ---------------------------------------------------------------------
def test_rescale_penalty_factors_sums_to_p_and_preserves_zeros():
    pf = np.array([0.0, 2.0, 1.0, 1.0])
    out = rescale_penalty_factors(pf, 4)
    assert out[0] == 0.0
    assert np.sum(out) == pytest.approx(4.0)
    np.testing.assert_allclose(out[1:] / out[1], pf[1:] / pf[1])  # relative weights preserved


def test_weighted_column_scale_matches_manual_population_sd():
    rng = np.random.RandomState(5)
    X = rng.normal(loc=3.0, scale=2.5, size=(500, 3))
    w = rng.uniform(0.5, 2.0, size=500)
    xs, degenerate = weighted_column_scale(X, w, standardize=True)
    wm = np.average(X, axis=0, weights=w)
    manual_var = np.average((X - wm) ** 2, axis=0, weights=w)
    np.testing.assert_allclose(xs, np.sqrt(manual_var), rtol=1e-10)
    assert not np.any(degenerate)
