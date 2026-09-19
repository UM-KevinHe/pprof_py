"""Self-consistency tests for penalized linear models.

Covers PenalizedLinear, PenalizedLinearCV, and GroupLassoLinear.
All tests use synthetic data (Tier 1 per .assistant_instructions.md).

Provenance: new file, 2025-07.
"""
from __future__ import annotations

import numpy as np
import pytest

from pprof_py.models.linear.penalized import PenalizedLinear, PenalizedLinearCV
from pprof_py.models.linear.group_lasso import GroupLassoLinear
from pprof_py.exceptions import NotFittedError


# ======================================================================
# Shared fixtures
# ======================================================================

@pytest.fixture(scope="module")
def synth_data():
    """Standard synthetic continuous-outcome data for penalized linear."""
    rng = np.random.RandomState(2025)
    n, p = 300, 12
    rho = 0.4
    Sigma = rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    L = np.linalg.cholesky(Sigma)
    X = rng.randn(n, p) @ L.T
    beta_true = np.array([1.5, -1.0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y = X @ beta_true + rng.randn(n) * 1.0
    return X, y, beta_true


# ======================================================================
# PenalizedLinear
# ======================================================================

class TestPenalizedLinear:
    """Self-consistency tests for elastic-net penalized linear."""

    def test_sparsity_monotonicity(self, synth_data):
        """Nonzero count should be non-decreasing along the lambda path."""
        X, y, _ = synth_data
        model = PenalizedLinear(alpha=1.0, n_lambda=30)
        model.fit(X, y)
        nz = model.n_nonzero_path_
        assert np.all(np.diff(nz) >= 0), f"Sparsity not monotonic: {nz}"

    def test_first_lambda_all_zero(self, synth_data):
        """At lambda_max, all penalized coefficients should be ~zero."""
        X, y, _ = synth_data
        model = PenalizedLinear(alpha=1.0, n_lambda=30)
        model.fit(X, y)
        assert np.all(np.abs(model.coef_path_[0]) < 1e-12)

    def test_deviance_ratio_range(self, synth_data):
        """Deviance ratio should be in [0, 1]."""
        X, y, _ = synth_data
        model = PenalizedLinear(alpha=1.0, n_lambda=30)
        model.fit(X, y)
        assert np.all(model.deviance_ratio_path_ >= -1e-10)
        assert np.all(model.deviance_ratio_path_ <= 1.0 + 1e-10)

    def test_alpha_zero_is_ridge(self, synth_data):
        """With alpha=0 (ridge), no coefficients should be exactly zero
        at small lambda."""
        X, y, _ = synth_data
        model = PenalizedLinear(alpha=0.0, n_lambda=30)
        model.fit(X, y)
        assert np.all(model.coef_path_[-1] != 0.0)

    def test_not_fitted_error(self, synth_data):
        """Accessing fitted attributes before fit raises NotFittedError."""
        model = PenalizedLinear()
        with pytest.raises(NotFittedError):
            model._check_is_fitted()

    def test_predict_shape(self, synth_data):
        """Predictions have correct shape."""
        X, y, _ = synth_data
        model = PenalizedLinear(alpha=1.0, n_lambda=10)
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (X.shape[0],)

    def test_single_lambda(self, synth_data):
        """Fitting with a single lambda value works and sets coef_."""
        X, y, _ = synth_data
        model = PenalizedLinear(alpha=1.0, lambda_path=0.01)
        model.fit(X, y)
        assert hasattr(model, "coef_")
        assert model.coef_.shape == (X.shape[1],)


# ======================================================================
# PenalizedLinearCV
# ======================================================================

class TestPenalizedLinearCV:
    """CV selection tests."""

    def test_lambda_ordering(self, synth_data):
        """lambda_1se >= lambda_min."""
        X, y, _ = synth_data
        cv = PenalizedLinearCV(alpha=1.0, n_lambda=30, n_folds=5,
                               random_state=42)
        cv.fit(X, y)
        assert cv.lambda_1se_ >= cv.lambda_min_

    def test_cv_se_positive(self, synth_data):
        """CV standard errors should be non-negative."""
        X, y, _ = synth_data
        cv = PenalizedLinearCV(alpha=1.0, n_lambda=20, n_folds=5,
                               random_state=42)
        cv.fit(X, y)
        assert np.all(cv.cv_se_deviance_ >= 0)

    def test_predict_at_selected_lambda(self, synth_data):
        """Predictions at selected lambda have correct shape."""
        X, y, _ = synth_data
        cv = PenalizedLinearCV(alpha=1.0, n_lambda=20, n_folds=5,
                               random_state=42)
        cv.fit(X, y)
        pred = cv.predict(X)
        assert pred.shape == (X.shape[0],)


# ======================================================================
# GroupLassoLinear
# ======================================================================

class TestGroupLassoLinear:
    """Group consistency tests."""

    def test_groups_enter_together(self, synth_data):
        """Variables in the same group should enter/leave together."""
        X, y, _ = synth_data
        groups = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
        model = GroupLassoLinear(groups=groups, alpha=0.0, n_lambda=30)
        model.fit(X, y)
        for i in range(len(model.lambda_path_)):
            coef = model.coef_path_[i]
            for g in [1, 2, 3, 4]:
                mask = groups == g
                group_coefs = coef[mask]
                all_zero = np.all(np.abs(group_coefs) < 1e-12)
                any_zero = np.any(np.abs(group_coefs) < 1e-12)
                # Either all zero or all nonzero for pure group lasso.
                if all_zero:
                    continue
                assert not any_zero, (
                    f"Group {g} at lambda index {i}: partial sparsity "
                    f"in a pure group lasso (alpha=0)"
                )

    def test_sparsity_monotonicity(self, synth_data):
        """Active group count should be non-decreasing."""
        X, y, _ = synth_data
        groups = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
        model = GroupLassoLinear(groups=groups, alpha=0.0, n_lambda=30)
        model.fit(X, y)
        nz = model.n_nonzero_path_
        assert np.all(np.diff(nz) >= 0)

    def test_first_lambda_all_zero(self, synth_data):
        """At lambda_max all coefficients should be ~zero."""
        X, y, _ = synth_data
        groups = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
        model = GroupLassoLinear(groups=groups, alpha=0.0, n_lambda=30)
        model.fit(X, y)
        assert np.all(np.abs(model.coef_path_[0]) < 1e-12)
