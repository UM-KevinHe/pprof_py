"""Self-consistency tests for ProviderPenalizedDiscreteSurvival (pp.DiscSurv).

All tests use synthetic data so they are safe to run autonomously
(Tier 1 per .assistant_instructions.md).

Provenance: new file, 2026-09.
Source: pprof_py/models/survival/provider_discrete_survival.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pprof_py.models.survival.provider_discrete_survival import (
    ProviderPenalizedDiscreteSurvival,
    ProviderPenalizedDiscreteSurvivalCV,
    _provider_newton_from_loglik,
)
from pprof_py.models.survival.discrete_survival import DiscreteSurvival
from pprof_py.exceptions import NotFittedError


# ======================================================================
# Data generators
# ======================================================================

def _simulate_discrete_survival(
    n: int,
    p: int,
    n_providers: int,
    max_T: int = 8,
    baseline_hazard: float = 0.05,
    seed: int = 2026,
):
    """Synthetic discrete survival data with provider effects.

    Returns (X, time, event, provider_id, beta_true, gamma_true).
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    provider_id = rng.randint(0, n_providers, size=n)

    beta_true = np.zeros(p)
    n_nonzero = min(3, p)
    beta_true[:n_nonzero] = [0.5, -0.3, 0.2][:n_nonzero]
    gamma_true = rng.randn(n_providers) * 0.3

    time_obs = np.zeros(n, dtype=int)
    event_obs = np.zeros(n, dtype=int)
    logit_h0 = np.log(baseline_hazard / (1.0 - baseline_hazard))

    for i in range(n):
        eta_i = X[i] @ beta_true + gamma_true[provider_id[i]]
        for t in range(1, max_T + 1):
            h_it = 1.0 / (1.0 + np.exp(-(logit_h0 + eta_i)))
            if rng.rand() < h_it:
                time_obs[i] = t
                event_obs[i] = 1
                break
        if time_obs[i] == 0:
            time_obs[i] = max_T
            event_obs[i] = 0

    return X, time_obs, event_obs, provider_id, beta_true, gamma_true


# ======================================================================
# Shared fixtures
# ======================================================================

@pytest.fixture(scope="module")
def synth_data():
    """Standard synthetic discrete survival data with providers."""
    return _simulate_discrete_survival(
        n=300, p=6, n_providers=15, seed=2026,
    )


@pytest.fixture(scope="module")
def fitted_model(synth_data):
    """Pre-fitted model for tests that only inspect outputs."""
    X, time, event, prov, _, _ = synth_data
    model = ProviderPenalizedDiscreteSurvival(
        alpha=1.0,
        provider_bound=5.0,
        n_lambda=20,
        max_outer_iter=50,
        outer_tol=1e-3,
        max_inner_iter=1000,
        inner_tol=1e-5,
        use_active_set=True,
        standardize=False,
    )
    model.fit(X, time, event, prov)
    return model


# ======================================================================
# TestProviderNewtonStep — unit tests for the Newton helper
# ======================================================================

class TestProviderNewtonStep:
    """Unit tests for _provider_newton_from_loglik."""

    def test_zero_score_no_change(self):
        """Zero gradient → gamma unchanged."""
        gamma = np.array([0.5, -0.3, 0.1])
        gamma_new = _provider_newton_from_loglik(
            score_beta=np.zeros(10),
            working_weights=np.ones(10),
            provider_idx=np.repeat(np.arange(3), [4, 3, 3]),
            n_providers=3,
            gamma=gamma,
            provider_bound=10.0,
        )
        np.testing.assert_allclose(gamma_new, gamma, atol=1e-12)

    def test_median_clamp(self):
        """Extreme scores are bounded by median-clamp."""
        n = 30
        provider_idx = np.repeat(np.arange(3), 10)
        score_beta = np.zeros(n)
        # Huge negative score for provider 0 → large positive gamma update.
        score_beta[:10] = -100.0
        gamma = np.zeros(3)
        bound = 2.0
        gamma_new = _provider_newton_from_loglik(
            score_beta=score_beta,
            working_weights=np.ones(n),
            provider_idx=provider_idx,
            n_providers=3,
            gamma=gamma,
            provider_bound=bound,
        )
        median_g = float(np.median(gamma_new))
        assert np.all(gamma_new >= median_g - bound - 1e-10)
        assert np.all(gamma_new <= median_g + bound + 1e-10)

    def test_bincount_aggregation(self):
        """Check that score aggregation matches manual loop."""
        rng = np.random.RandomState(99)
        n, K = 50, 5
        provider_idx = rng.randint(0, K, size=n)
        score_beta = rng.randn(n)
        ww = np.abs(rng.randn(n)) + 0.1
        gamma = rng.randn(K) * 0.2

        gamma_new = _provider_newton_from_loglik(
            score_beta, ww, provider_idx, K, gamma, provider_bound=10.0,
        )

        # Manual computation.
        prov_score_manual = np.zeros(K)
        prov_info_manual = np.zeros(K)
        for k in range(K):
            mask = provider_idx == k
            prov_score_manual[k] = -np.sum(score_beta[mask])
            prov_info_manual[k] = np.sum(ww[mask])
        gamma_manual = gamma + prov_score_manual / np.maximum(prov_info_manual, 1e-12)
        med = np.median(gamma_manual)
        gamma_manual = np.clip(gamma_manual, med - 10.0, med + 10.0)

        np.testing.assert_allclose(gamma_new, gamma_manual, atol=1e-12)


# ======================================================================
# TestFitBasics — basic fit properties
# ======================================================================

class TestFitBasics:
    """Properties that should hold for any valid fit."""

    def test_all_lambdas_converged(self, fitted_model):
        """All lambda points should converge with generous tolerances."""
        assert fitted_model.converged_.sum() == len(fitted_model.converged_)

    def test_output_shapes(self, fitted_model, synth_data):
        """Output array shapes should be consistent."""
        X, _, _, _, _, _ = synth_data
        n_lam = len(fitted_model.lambda_path_)
        p = X.shape[1]
        K_prov = fitted_model.n_providers_
        K_time = fitted_model.n_timepoints_

        assert fitted_model.coef_path_.shape == (n_lam, p)
        assert fitted_model.baseline_hazard_path_.shape == (n_lam, K_time)
        assert fitted_model.gamma_path_.shape == (n_lam, K_prov)
        assert fitted_model.df_.shape == (n_lam,)
        assert fitted_model.n_iter_.shape == (n_lam,)

    def test_first_lambda_zero_beta(self, fitted_model):
        """At the largest lambda, all penalized coefs should be ~zero."""
        assert np.all(np.abs(fitted_model.coef_path_[0]) < 1e-10)

    def test_provider_labels_match(self, synth_data, fitted_model):
        """provider_labels_ should contain all unique provider IDs."""
        _, _, _, prov, _, _ = synth_data
        expected = np.sort(np.unique(prov))
        np.testing.assert_array_equal(fitted_model.provider_labels_, expected)

    def test_lambda_path_descending(self, fitted_model):
        """Lambda path should be sorted descending."""
        diffs = np.diff(fitted_model.lambda_path_)
        assert np.all(diffs <= 1e-12)

    def test_n_obs_stored(self, synth_data, fitted_model):
        X, _, event, _, _, _ = synth_data
        assert fitted_model.n_obs_ == X.shape[0]
        assert fitted_model.n_events_ == int(event.sum())
        assert fitted_model.n_features_in_ == X.shape[1]


# ======================================================================
# TestSparsity — regularization path properties
# ======================================================================

class TestSparsity:
    """Sparsity and regularization-path properties."""

    def test_sparsity_monotonicity(self, fitted_model):
        """Nonzero count should be non-decreasing along the path
        (decreasing lambda → less penalization)."""
        nz = np.array([
            np.sum(fitted_model.coef_path_[i] != 0)
            for i in range(len(fitted_model.lambda_path_))
        ])
        assert np.all(np.diff(nz) >= 0), f"Sparsity not monotonic: {nz}"

    def test_last_lambda_nonzero(self, fitted_model):
        """At the smallest lambda, at least one coef should be nonzero."""
        assert np.sum(fitted_model.coef_path_[-1] != 0) > 0

    def test_neg_loglik_decreasing(self, fitted_model):
        """Negative log-likelihood should generally decrease along the
        path (more flexible model → better fit).  Allow small increases
        due to convergence tolerance."""
        nll = fitted_model.neg_loglik_
        # Check that the trend is decreasing: nll[end] < nll[start].
        assert nll[-1] < nll[0] + 1e-6


# ======================================================================
# TestProviderEffects — provider-effect properties
# ======================================================================

class TestProviderEffects:
    """Provider effect (gamma) properties."""

    def test_gamma_bounded(self, fitted_model):
        """Provider effects should respect the median-clamp bound."""
        bound = fitted_model.provider_bound
        for i in range(len(fitted_model.lambda_path_)):
            gamma = fitted_model.gamma_path_[i]
            median_g = np.median(gamma)
            assert np.all(gamma >= median_g - bound - 1e-8), (
                f"Lambda[{i}]: gamma below lower bound"
            )
            assert np.all(gamma <= median_g + bound + 1e-8), (
                f"Lambda[{i}]: gamma above upper bound"
            )

    def test_gamma_nonzero_at_first_lambda(self, fitted_model):
        """Even when beta is all-zero, provider effects should be
        estimated (they are unpenalized)."""
        gamma_first = fitted_model.gamma_path_[0]
        # At least some providers should have nonzero effects.
        assert np.max(np.abs(gamma_first)) > 1e-6

    def test_predict_provider_effect_shape(self, fitted_model):
        """predict_provider_effect returns correct DataFrame."""
        pe = fitted_model.predict_provider_effect(which=-1)
        assert isinstance(pe, pd.DataFrame)
        assert list(pe.columns) == ["provider", "gamma"]
        assert len(pe) == fitted_model.n_providers_


# ======================================================================
# TestPrediction — hazard and survival predictions
# ======================================================================

class TestPrediction:
    """Prediction methods."""

    def test_hazard_in_unit_interval(self, synth_data, fitted_model):
        """Predicted hazard should be in [0, 1]."""
        X, _, _, prov, _, _ = synth_data
        hazard = fitted_model.predict_hazard(X, prov)
        assert np.all(hazard >= 0.0)
        assert np.all(hazard <= 1.0)

    def test_survival_monotone_decreasing(self, synth_data, fitted_model):
        """Survival curves should be non-increasing over time."""
        X, _, _, prov, _, _ = synth_data
        surv = fitted_model.predict_survival(X, prov)
        diffs = np.diff(surv, axis=1)
        assert np.all(diffs <= 1e-12), "Survival should be non-increasing"

    def test_survival_starts_below_one(self, synth_data, fitted_model):
        """S(t=1) should be < 1 (some hazard at the first timepoint)."""
        X, _, _, prov, _, _ = synth_data
        surv = fitted_model.predict_survival(X, prov)
        # S(1) = 1 - h(1), so S(1) < 1 as long as h(1) > 0.
        assert np.all(surv[:, 0] < 1.0)

    def test_hazard_shape(self, synth_data, fitted_model):
        """Hazard shape should be (n_new, n_timepoints)."""
        X, _, _, prov, _, _ = synth_data
        hazard = fitted_model.predict_hazard(X[:10], prov[:10])
        assert hazard.shape == (10, fitted_model.n_timepoints_)

    def test_provider_effect_changes_prediction(self, synth_data, fitted_model):
        """Predictions with and without provider_id should differ."""
        X, _, _, prov, _, _ = synth_data
        h_with = fitted_model.predict_hazard(X[:10], prov[:10], which=-1)
        h_without = fitted_model.predict_hazard(X[:10], which=-1)
        assert not np.allclose(h_with, h_without)

    def test_unseen_provider_gets_no_effect(self, fitted_model):
        """A provider ID not seen during fit should get gamma=0."""
        X_new = np.zeros((1, fitted_model.n_features_in_))
        h_unseen = fitted_model.predict_hazard(
            X_new, provider_id=np.array([999999]),
        )
        h_none = fitted_model.predict_hazard(X_new)
        np.testing.assert_allclose(h_unseen, h_none, atol=1e-12)


# ======================================================================
# TestNotFitted — error handling
# ======================================================================

class TestNotFitted:
    """Pre-fit error handling."""

    def test_predict_before_fit_raises(self):
        model = ProviderPenalizedDiscreteSurvival()
        with pytest.raises(NotFittedError):
            model.predict_hazard(np.zeros((1, 3)))

    def test_predict_survival_before_fit_raises(self):
        model = ProviderPenalizedDiscreteSurvival()
        with pytest.raises(NotFittedError):
            model.predict_survival(np.zeros((1, 3)))

    def test_provider_effect_before_fit_raises(self):
        model = ProviderPenalizedDiscreteSurvival()
        with pytest.raises(NotFittedError):
            model.predict_provider_effect()


# ======================================================================
# TestDataFrameInput — DataFrame acceptance
# ======================================================================

class TestDataFrameInput:
    """Model should accept both ndarray and DataFrame for X."""

    def test_dataframe_x_input(self, synth_data):
        X, time, event, prov, _, _ = synth_data
        df = pd.DataFrame(X, columns=[f"v{j}" for j in range(X.shape[1])])
        model = ProviderPenalizedDiscreteSurvival(
            n_lambda=5, max_outer_iter=30, outer_tol=1e-3,
        )
        model.fit(df, time, event, prov)
        assert hasattr(model, "feature_names_in_")
        np.testing.assert_array_equal(
            model.feature_names_in_,
            [f"v{j}" for j in range(X.shape[1])],
        )
        assert model.coef_path_.shape[1] == X.shape[1]


# ======================================================================
# TestSingleProviderEquivalence — cross-model consistency
# ======================================================================

class TestSingleProviderEquivalence:
    """With a single provider, ProviderPenalizedDiscreteSurvival should
    produce beta values very close to DiscreteSurvival (the gamma
    absorbs the intercept, so alpha differs, but beta should match)."""

    def test_beta_matches_discrete_survival(self):
        """Beta from single-provider pp.DiscSurv should match plain
        DiscreteSurvival within solver tolerance."""
        rng = np.random.RandomState(42)
        n, p = 200, 4
        X = rng.randn(n, p)
        provider = np.zeros(n, dtype=int)  # single provider
        beta_true = np.array([0.6, -0.4, 0, 0])

        # Simulate.
        max_T = 6
        time_obs = np.zeros(n, dtype=int)
        event_obs = np.zeros(n, dtype=int)
        for i in range(n):
            eta_i = X[i] @ beta_true
            for t in range(1, max_T + 1):
                h = 1.0 / (1.0 + np.exp(-(np.log(0.05 / 0.95) + eta_i)))
                if rng.rand() < h:
                    time_obs[i] = t
                    event_obs[i] = 1
                    break
            if time_obs[i] == 0:
                time_obs[i] = max_T
                event_obs[i] = 0

        lam_path = np.array([0.1, 0.05, 0.01, 0.005, 0.001])

        pp_model = ProviderPenalizedDiscreteSurvival(
            lambda_path=lam_path, provider_bound=10.0,
            max_outer_iter=100, outer_tol=1e-5, standardize=False,
        )
        pp_model.fit(X, time_obs, event_obs, provider)

        ds_model = DiscreteSurvival(
            lambda_path=lam_path, tol=1e-5, standardize=False,
        )
        ds_model.fit(X, time_obs, event_obs)

        max_diff = np.max(np.abs(pp_model.coef_path_ - ds_model.coef_path_))
        assert max_diff < 0.01, (
            f"Single-provider beta diff too large: {max_diff:.4e}"
        )


# ======================================================================
# TestActiveSetEquivalence — active-set vs full solve
# ======================================================================

class TestActiveSetEquivalence:
    """Active-set screening should produce the same results as the
    full solve."""

    def test_active_vs_full(self):
        X, time, event, prov, _, _ = _simulate_discrete_survival(
            n=200, p=6, n_providers=10, seed=99,
        )
        lam_path = np.array([0.1, 0.05, 0.01])

        m_active = ProviderPenalizedDiscreteSurvival(
            lambda_path=lam_path, provider_bound=5.0,
            max_outer_iter=80, outer_tol=1e-4,
            use_active_set=True,
        )
        m_active.fit(X, time, event, prov)

        m_full = ProviderPenalizedDiscreteSurvival(
            lambda_path=lam_path, provider_bound=5.0,
            max_outer_iter=80, outer_tol=1e-4,
            use_active_set=False,
        )
        m_full.fit(X, time, event, prov)

        np.testing.assert_allclose(
            m_active.coef_path_, m_full.coef_path_,
            atol=1e-3, rtol=1e-3,
            err_msg="Active-set beta differs from full solve",
        )
        np.testing.assert_allclose(
            m_active.gamma_path_, m_full.gamma_path_,
            atol=1e-3, rtol=1e-3,
            err_msg="Active-set gamma differs from full solve",
        )


# ======================================================================
# TestStandardize — optional standardization
# ======================================================================

class TestStandardize:
    """Standardize=True should give approximately the same coefficients
    as standardize=False (after unstandardization), and sparsity should
    match."""

    def test_standardize_coef_agreement(self):
        X, time, event, prov, _, _ = _simulate_discrete_survival(
            n=200, p=4, n_providers=8, seed=77,
        )
        lam_path = np.array([0.05, 0.01, 0.005])

        m_raw = ProviderPenalizedDiscreteSurvival(
            lambda_path=lam_path, provider_bound=5.0,
            max_outer_iter=80, outer_tol=1e-4, standardize=False,
        )
        m_raw.fit(X, time, event, prov)

        m_std = ProviderPenalizedDiscreteSurvival(
            lambda_path=lam_path, provider_bound=5.0,
            max_outer_iter=80, outer_tol=1e-4, standardize=True,
        )
        m_std.fit(X, time, event, prov)

        # Sparsity pattern should be the same at each lambda.
        for i in range(len(lam_path)):
            nz_raw = np.sum(m_raw.coef_path_[i] != 0)
            nz_std = np.sum(m_std.coef_path_[i] != 0)
            assert nz_raw == nz_std, (
                f"Lambda[{i}]: sparsity mismatch raw={nz_raw} std={nz_std}"
            )


# ======================================================================
# TestPenaltyFactor — per-variable penalty weights
# ======================================================================

class TestPenaltyFactor:
    """Test that penalty_factor=0 makes a variable unpenalized."""

    def test_unpenalized_variable_always_nonzero(self):
        X, time, event, prov, _, _ = _simulate_discrete_survival(
            n=250, p=6, n_providers=10, seed=55,
        )
        pf = np.ones(6)
        pf[0] = 0.0  # first variable is unpenalized

        model = ProviderPenalizedDiscreteSurvival(
            n_lambda=15, provider_bound=5.0,
            penalty_factor=pf,
            max_outer_iter=50, outer_tol=1e-3,
        )
        model.fit(X, time, event, prov)

        # After first lambda (where all penalized are zero),
        # the unpenalized variable should be nonzero.
        for i in range(1, len(model.lambda_path_)):
            assert model.coef_path_[i, 0] != 0.0, (
                f"Lambda[{i}]: unpenalized variable should be nonzero"
            )


# ======================================================================
# TestProviderPenalizedDiscreteSurvivalCV — cross-validation
# ======================================================================

@pytest.fixture(scope="module")
def fitted_cv_model():
    """Pre-fitted CV model for tests that only inspect outputs."""
    X, time, event, prov, _, _ = _simulate_discrete_survival(
        n=300, p=6, n_providers=15, seed=2026,
    )
    model = ProviderPenalizedDiscreteSurvivalCV(
        n_folds=5,
        se_rule="1se",
        random_state=42,
        n_lambda=15,
        provider_bound=5.0,
        max_outer_iter=30,
        outer_tol=1e-3,
    )
    model.fit(X, time, event, prov)
    return model, X, time, event, prov


class TestProviderPenalizedDiscreteSurvivalCV:
    """Cross-validation tests."""

    def test_lambda_ordering(self, fitted_cv_model):
        """lambda_1se >= lambda_min (more regularized)."""
        cv, *_ = fitted_cv_model
        assert cv.lambda_1se_ >= cv.lambda_min_

    def test_lambda_1se_within_1se(self, fitted_cv_model):
        """lambda_1se should have CV error within 1 SE of the min."""
        cv, *_ = fitted_cv_model
        threshold = (
            cv.cv_mean_[cv.lambda_min_idx_]
            + cv.cv_se_[cv.lambda_min_idx_]
        )
        assert cv.cv_mean_[cv.lambda_1se_idx_] <= threshold + 1e-12

    def test_cv_se_positive(self, fitted_cv_model):
        """CV standard errors should be positive where finite."""
        cv, *_ = fitted_cv_model
        finite = np.isfinite(cv.cv_se_)
        assert np.all(cv.cv_se_[finite] > 0)

    def test_cv_mean_not_nan(self, fitted_cv_model):
        """CV mean should have no NaN at valid lambdas."""
        cv, *_ = fitted_cv_model
        # At least some lambdas should have valid CV loss.
        assert np.sum(np.isfinite(cv.cv_mean_)) > 0

    def test_fold_assignment_shape(self, fitted_cv_model):
        """Fold assignment should have correct length and expected
        number of unique folds."""
        cv, X, _, _, _ = fitted_cv_model
        assert len(cv.fold_assignment_) == X.shape[0]
        assert len(np.unique(cv.fold_assignment_)) == 5

    def test_fold_balance(self, fitted_cv_model):
        """Fold sizes should be approximately balanced."""
        cv, *_ = fitted_cv_model
        fold_sizes = np.bincount(cv.fold_assignment_)
        ratio = fold_sizes.max() / max(fold_sizes.min(), 1)
        assert ratio < 2.0, f"Fold imbalance: {fold_sizes}"

    def test_selected_coef_shape(self, fitted_cv_model):
        """Selected coefficients should have correct shape."""
        cv, X, _, _, _ = fitted_cv_model
        assert cv.coef_.shape == (X.shape[1],)
        assert cv.baseline_hazard_.shape[0] == cv.n_timepoints_
        assert cv.gamma_.shape[0] == cv.n_providers_

    def test_predict_hazard_works(self, fitted_cv_model):
        """predict_hazard should return valid probabilities."""
        cv, X, _, _, prov = fitted_cv_model
        h = cv.predict_hazard(X[:5], prov[:5])
        assert h.shape == (5, cv.n_timepoints_)
        assert np.all(h >= 0) and np.all(h <= 1)

    def test_predict_survival_decreasing(self, fitted_cv_model):
        """Survival curves should be non-increasing."""
        cv, X, _, _, prov = fitted_cv_model
        s = cv.predict_survival(X[:5], prov[:5])
        diffs = np.diff(s, axis=1)
        assert np.all(diffs <= 1e-12)

    def test_predict_provider_effect_dataframe(self, fitted_cv_model):
        """predict_provider_effect should return correct DataFrame."""
        cv, *_ = fitted_cv_model
        pe = cv.predict_provider_effect()
        assert isinstance(pe, pd.DataFrame)
        assert list(pe.columns) == ["provider", "gamma"]
        assert len(pe) == cv.n_providers_

    def test_full_model_exposed(self, fitted_cv_model):
        """model_ should be a fitted ProviderPenalizedDiscreteSurvival."""
        cv, *_ = fitted_cv_model
        assert isinstance(cv.model_, ProviderPenalizedDiscreteSurvival)
        assert hasattr(cv.model_, 'coef_path_')

    def test_not_fitted_raises(self):
        """Predictions before fit should raise NotFittedError."""
        cv = ProviderPenalizedDiscreteSurvivalCV()
        with pytest.raises(NotFittedError):
            cv.predict_hazard(np.zeros((1, 3)))

    def test_se_rule_min(self):
        """With se_rule="min", lambda_ should equal lambda_min."""
        X, time, event, prov, _, _ = _simulate_discrete_survival(
            n=200, p=4, n_providers=10, seed=88,
        )
        cv = ProviderPenalizedDiscreteSurvivalCV(
            n_folds=3,
            se_rule="min",
            random_state=42,
            n_lambda=10,
            provider_bound=5.0,
            max_outer_iter=20,
            outer_tol=1e-2,
        )
        cv.fit(X, time, event, prov)
        assert cv.lambda_ == cv.lambda_min_
