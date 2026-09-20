"""Self-consistency tests for DiscreteSurvival and DiscreteSurvivalCV.

The ProviderPenalizedDiscreteSurvival/CV variants are already tested
in test_provider_discrete_survival.py; these tests cover the base
(non-provider) classes.

All tests use synthetic data (Tier 1 per .assistant_instructions.md).

Provenance: new file, 2025-07.
"""
from __future__ import annotations

import numpy as np
import pytest

from pprof_py.models.survival.discrete_survival import (
    DiscreteSurvival,
    DiscreteSurvivalCV,
)
from pprof_py.exceptions import NotFittedError


# ======================================================================
# Data generators
# ======================================================================

def _simulate_discrete(
    n=400, p=8, max_T=8, baseline_hazard=0.06, seed=2025,
):
    """Synthetic discrete-time survival data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    beta_true = np.zeros(p)
    beta_true[:3] = [0.5, -0.3, 0.2]

    logit_h0 = np.log(baseline_hazard / (1.0 - baseline_hazard))
    time_obs = np.zeros(n, dtype=int)
    event_obs = np.zeros(n, dtype=int)
    for i in range(n):
        eta_i = X[i] @ beta_true
        for t in range(1, max_T + 1):
            h_it = 1.0 / (1.0 + np.exp(-(logit_h0 + eta_i)))
            if rng.rand() < h_it:
                time_obs[i] = t
                event_obs[i] = 1
                break
        if time_obs[i] == 0:
            time_obs[i] = max_T
            event_obs[i] = 0
    return X, time_obs, event_obs, beta_true


@pytest.fixture(scope="module")
def synth_data():
    return _simulate_discrete()


# ======================================================================
# DiscreteSurvival
# ======================================================================

class TestDiscreteSurvival:
    """Self-consistency tests for the base discrete survival model."""

    def test_fit_returns_self(self, synth_data):
        X, time, event, _ = synth_data
        model = DiscreteSurvival(penalty_type="lasso", n_lambda=10)
        result = model.fit(X, time=time, event=event)
        assert result is model

    def test_coef_path_shape(self, synth_data):
        X, time, event, _ = synth_data
        model = DiscreteSurvival(penalty_type="lasso", n_lambda=15)
        model.fit(X, time=time, event=event)
        n_lambda = len(model.lambda_path_)
        assert model.coef_path_.shape == (n_lambda, X.shape[1])

    def test_alpha_path_shape(self, synth_data):
        """Baseline hazard params should have shape (n_lambda, K)."""
        X, time, event, _ = synth_data
        model = DiscreteSurvival(penalty_type="lasso", n_lambda=10)
        model.fit(X, time=time, event=event)
        n_lambda = len(model.lambda_path_)
        assert model.alpha_path_.shape[0] == n_lambda
        assert model.alpha_path_.shape[1] == model.n_timepoints_

    def test_first_lambda_all_zero(self, synth_data):
        """At lambda_max, covariate coefficients should be ~zero."""
        X, time, event, _ = synth_data
        model = DiscreteSurvival(penalty_type="lasso", n_lambda=20)
        model.fit(X, time=time, event=event)
        assert np.all(np.abs(model.coef_path_[0]) < 1e-10)

    def test_sparsity_monotonicity(self, synth_data):
        """Nonzero count should be non-decreasing along the path."""
        X, time, event, _ = synth_data
        model = DiscreteSurvival(penalty_type="lasso", n_lambda=20)
        model.fit(X, time=time, event=event)
        nz = np.array([np.sum(row != 0) for row in model.coef_path_])
        assert np.all(np.diff(nz) >= 0), f"Sparsity not monotonic: {nz}"

    def test_not_fitted_error(self):
        model = DiscreteSurvival()
        with pytest.raises(NotFittedError):
            model._check_is_fitted()

    def test_predict_hazard_shape(self, synth_data):
        """Predicted hazard should have correct shape."""
        X, time, event, _ = synth_data
        model = DiscreteSurvival(penalty_type="lasso", n_lambda=10)
        model.fit(X, time=time, event=event)
        hazard = model.predict_hazard(X, time=time)
        assert hazard.shape[0] == X.shape[0]

    def test_predict_survival_range(self, synth_data):
        """Predicted survival probabilities should be in [0, 1]."""
        X, time, event, _ = synth_data
        model = DiscreteSurvival(penalty_type="lasso", n_lambda=10)
        model.fit(X, time=time, event=event)
        surv = model.predict_survival(X, time=time)
        assert np.all(surv >= -1e-10)
        assert np.all(surv <= 1.0 + 1e-10)


# ======================================================================
# DiscreteSurvivalCV
# ======================================================================

class TestDiscreteSurvivalCV:
    """CV selection tests."""

    def test_lambda_ordering(self, synth_data):
        """lambda_1se >= lambda_min."""
        X, time, event, _ = synth_data
        cv = DiscreteSurvivalCV(
            penalty_type="lasso", n_lambda=15,
            n_folds=5, random_state=42,
        )
        cv.fit(X, time=time, event=event)
        assert cv.lambda_1se_ >= cv.lambda_min_

    def test_coef_at_selected_lambda(self, synth_data):
        """Refit model from best_model_ should have correct coef shape."""
        X, time, event, _ = synth_data
        cv = DiscreteSurvivalCV(
            penalty_type="lasso", n_lambda=15,
            n_folds=5, random_state=42,
        )
        cv.fit(X, time=time, event=event)
        # DiscreteSurvivalCV has no coef_ attribute;
        # coefficients are accessed via the refit model.
        assert hasattr(cv, "best_model_")
        assert cv.best_model_.coef_path_.shape[1] == X.shape[1]
