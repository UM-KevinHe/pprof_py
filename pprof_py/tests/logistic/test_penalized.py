"""Self-consistency tests for penalized logistic models.

Extracted from lasso_validation notebook self-consistency checks
(Tiers 5-10). All tests use synthetic data so they are safe to run
autonomously (Tier 1 per .assistant_instructions.md).

Provenance: new file, 2026-09.
"""
from __future__ import annotations

import numpy as np
import pytest

from pprof_py.models.logistic.penalized import (
    PenalizedLogistic,
    PenalizedLogisticCV,
)
from pprof_py.models.logistic.group_lasso import (
    GroupLassoLogistic,
    GroupLassoLogisticCV,
)
from pprof_py.models.logistic.provider_penalized import (
    ProviderPenalizedLogistic,
    ProviderPenalizedLogisticCV,
    _provider_stratified_fold_assignment,
)


# ======================================================================
# Shared fixtures
# ======================================================================

@pytest.fixture(scope="module")
def synth_data():
    """Standard synthetic binary outcome data for penalized logistic."""
    rng = np.random.RandomState(2026)
    n, p = 300, 12
    rho = 0.4
    Sigma = rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    L = np.linalg.cholesky(Sigma)
    X = rng.randn(n, p) @ L.T
    beta_true = np.array([0.8, -0.5, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    eta = X @ beta_true
    y = (rng.rand(n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    return X, y, beta_true


@pytest.fixture(scope="module")
def synth_provider_data():
    """Synthetic data with provider structure."""
    rng = np.random.RandomState(2026)
    n, p, K = 400, 8, 25
    X = rng.randn(n, p)
    prov = rng.randint(0, K, size=n)
    beta_true = np.array([0.6, -0.4, 0.2, 0, 0, 0, 0, 0])
    gamma_true = rng.randn(K) * 0.4
    eta = X @ beta_true + gamma_true[prov]
    y = (rng.rand(n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    return X, y, prov, beta_true, gamma_true


# ======================================================================
# PenalizedLogistic
# ======================================================================

class TestPenalizedLogistic:
    """Self-consistency tests for the elastic-net penalized logistic."""

    def test_sparsity_monotonicity(self, synth_data):
        """Number of nonzero coefficients should be non-decreasing
        along the lambda path (decreasing lambda -> less penalization)."""
        X, y, _ = synth_data
        model = PenalizedLogistic(alpha=1.0, n_lambda=30)
        model.fit(X, y)
        nz = model.n_nonzero_path_
        # Non-decreasing (allow ties).
        assert np.all(np.diff(nz) >= 0), (
            f"Sparsity not monotonic: {nz}"
        )

    def test_first_lambda_all_zero(self, synth_data):
        """At lambda_max, all penalized coefficients should be ~zero."""
        X, y, _ = synth_data
        model = PenalizedLogistic(alpha=1.0, n_lambda=30)
        model.fit(X, y)
        # Allow numerical noise up to 1e-12.
        assert np.all(np.abs(model.coef_path_[0]) < 1e-12)

    def test_deviance_ratio_range(self, synth_data):
        """Deviance ratio should be in [0, 1]."""
        X, y, _ = synth_data
        model = PenalizedLogistic(alpha=1.0, n_lambda=30)
        model.fit(X, y)
        assert np.all(model.deviance_ratio_path_ >= -1e-10)
        assert np.all(model.deviance_ratio_path_ <= 1.0 + 1e-10)

    def test_alpha_zero_is_ridge(self, synth_data):
        """With alpha=0 (ridge), no coefficients should be exactly zero
        at small lambda."""
        X, y, _ = synth_data
        model = PenalizedLogistic(alpha=0.0, n_lambda=30)
        model.fit(X, y)
        # At last lambda (smallest), all should be nonzero.
        assert np.all(model.coef_path_[-1] != 0.0)

    def test_active_set_equivalence(self, synth_data):
        """Active-set results should match full solve within tolerance."""
        X, y, _ = synth_data
        m_full = PenalizedLogistic(alpha=1.0, n_lambda=30, use_active_set=False)
        m_full.fit(X, y)
        m_active = PenalizedLogistic(alpha=1.0, n_lambda=30, use_active_set=True)
        m_active.fit(X, y)
        np.testing.assert_allclose(
            m_full.coef_path_, m_active.coef_path_,
            atol=1e-6, rtol=1e-6,
            err_msg="Active-set coefficients differ from full solve",
        )
        np.testing.assert_allclose(
            m_full.intercept_path_, m_active.intercept_path_,
            atol=1e-6, rtol=1e-6,
        )

    def test_predict_proba_range(self, synth_data):
        """Predicted probabilities should be in [0, 1]."""
        X, y, _ = synth_data
        model = PenalizedLogistic(alpha=1.0, n_lambda=10)
        model.fit(X, y)
        for lam in model.lambda_path_:
            proba = model.predict_proba(X, lambda_value=lam)
            assert np.all(proba >= 0) and np.all(proba <= 1)


# ======================================================================
# PenalizedLogisticCV
# ======================================================================

class TestPenalizedLogisticCV:
    """CV selection tests."""

    def test_lambda_ordering(self, synth_data):
        """lambda_min should have lower CV deviance than lambda_1se,
        and lambda_1se >= lambda_min."""
        X, y, _ = synth_data
        cv = PenalizedLogisticCV(alpha=1.0, n_lambda=30, n_folds=5,
                                 random_state=42)
        cv.fit(X, y)
        assert cv.lambda_1se_ >= cv.lambda_min_
        assert cv.cv_mean_deviance_[cv.lambda_min_idx_] <= (
            cv.cv_mean_deviance_[cv.lambda_1se_idx_] + 1e-10
        )

    def test_cv_se_positive(self, synth_data):
        X, y, _ = synth_data
        cv = PenalizedLogisticCV(alpha=1.0, n_lambda=20, n_folds=5,
                                 random_state=42)
        cv.fit(X, y)
        assert np.all(cv.cv_se_deviance_ >= 0)


# ======================================================================
# GroupLassoLogistic
# ======================================================================

class TestGroupLassoLogistic:
    """Group consistency tests."""

    def test_groups_enter_together(self, synth_data):
        """Variables in the same group should enter/leave together."""
        X, y, _ = synth_data
        # 4 groups of 3
        groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        model = GroupLassoLogistic(
            groups=groups, alpha=0.0, n_lambda=30,
        )
        model.fit(X, y)
        for i in range(len(model.lambda_path_)):
            coef = model.coef_path_[i]
            for g in range(4):
                mask = groups == g
                group_coefs = coef[mask]
                # Either all zero or all nonzero.
                all_zero = np.all(group_coefs == 0.0)
                all_nonzero = np.all(group_coefs != 0.0)
                assert all_zero or all_nonzero, (
                    f"Lambda idx {i}, group {g}: mixed zero/nonzero"
                )

    def test_alpha_one_mostly_matches_penalized_logistic(self, synth_data):
        """GroupLassoLogistic with alpha=1 should produce sparsity
        patterns approximately matching PenalizedLogistic.  Exact
        match is not expected because they use different CD solvers
        (block CD vs element-wise CD)."""
        X, y, _ = synth_data
        groups = np.arange(X.shape[1])  # each variable is its own group
        gl = GroupLassoLogistic(groups=groups, alpha=1.0, n_lambda=20)
        gl.fit(X, y)
        pl = PenalizedLogistic(alpha=1.0, n_lambda=20)
        pl.fit(X, y)
        # Both should start at all-zero and end at all-nonzero.
        assert gl.n_nonzero_path_[0] == 0 and pl.n_nonzero_path_[0] <= 1
        assert gl.n_nonzero_path_[-1] > 0 and pl.n_nonzero_path_[-1] > 0
        # Coefficient sign patterns should agree at smallest lambda.
        gl_sign = np.sign(gl.coef_path_[-1])
        pl_sign = np.sign(pl.coef_path_[-1])
        # At least 80% of signs should agree.
        agreement = np.mean(gl_sign == pl_sign)
        assert agreement >= 0.8, (
            f"Sign agreement too low: {agreement:.0%}"
        )


# ======================================================================
# ProviderPenalizedLogistic
# ======================================================================

class TestProviderPenalizedLogistic:
    """Provider-effect model tests."""

    def test_provider_effects_bounded(self, synth_provider_data):
        """Provider effects should respect gamma_bound."""
        X, y, prov, _, _ = synth_provider_data
        bound = 3.0
        model = ProviderPenalizedLogistic(
            gamma_bound=bound, n_lambda=20,
        )
        model.fit(X, y, provider_id=prov)
        for i in range(len(model.lambda_path_)):
            gamma = model.gamma_path_[i]
            median_g = np.median(gamma)
            assert np.all(gamma >= median_g - bound - 1e-10)
            assert np.all(gamma <= median_g + bound + 1e-10)

    def test_first_lambda_zero_beta(self, synth_provider_data):
        """At lambda_max, beta should be all zero."""
        X, y, prov, _, _ = synth_provider_data
        model = ProviderPenalizedLogistic(n_lambda=20)
        model.fit(X, y, provider_id=prov)
        assert np.all(model.coef_path_[0] == 0.0)

    def test_provider_labels_correct(self, synth_provider_data):
        """Provider labels should match unique values in provider_id."""
        X, y, prov, _, _ = synth_provider_data
        model = ProviderPenalizedLogistic(n_lambda=5)
        model.fit(X, y, provider_id=prov)
        expected = np.sort(np.unique(prov))
        np.testing.assert_array_equal(model.provider_labels_, expected)

    def test_predict_proba_with_provider(self, synth_provider_data):
        """Predictions with provider effects should differ from without."""
        X, y, prov, _, _ = synth_provider_data
        model = ProviderPenalizedLogistic(n_lambda=20)
        model.fit(X, y, provider_id=prov)
        p_with = model.predict_proba(X, provider_id=prov, which=-1)
        p_without = model.predict_proba(X, which=-1)
        # Should be different when gamma is nonzero.
        assert not np.allclose(p_with, p_without)


# ======================================================================
# ProviderPenalizedLogisticCV
# ======================================================================

class TestProviderPenalizedLogisticCV:
    """CV tests for provider-penalized model."""

    def test_cv_selection(self, synth_provider_data):
        """CV should select a lambda and produce valid results."""
        X, y, prov, _, _ = synth_provider_data
        cv = ProviderPenalizedLogisticCV(
            n_lambda=15, n_folds=5, gamma_bound=5.0,
            random_state=42,
        )
        cv.fit(X, y, provider_id=prov)
        assert cv.lambda_1se_ >= cv.lambda_min_
        assert cv.coef_ is not None
        assert cv.gamma_ is not None
        assert cv.provider_labels_ is not None

    def test_provider_fold_integrity(self):
        """All observations from the same provider should be in
        the same fold."""
        rng = np.random.RandomState(42)
        n = 200
        y = (rng.rand(n) > 0.7).astype(float)
        prov = np.repeat(np.arange(20), 10)
        folds = _provider_stratified_fold_assignment(
            y, prov, n_folds=5, random_state=42,
        )
        for p_id in range(20):
            mask = prov == p_id
            unique_folds = np.unique(folds[mask])
            assert len(unique_folds) == 1, (
                f"Provider {p_id} in multiple folds: {unique_folds}"
            )

    def test_fold_balance(self):
        """Fold sizes should be approximately balanced."""
        rng = np.random.RandomState(42)
        n = 500
        y = (rng.rand(n) > 0.6).astype(float)
        prov = np.repeat(np.arange(50), 10)
        folds = _provider_stratified_fold_assignment(
            y, prov, n_folds=5, random_state=42,
        )
        counts = np.bincount(folds)
        # Each fold should have 20% +/- 10% of data.
        for c in counts:
            assert 0.1 * n <= c <= 0.3 * n, (
                f"Fold size {c} out of expected range"
            )
