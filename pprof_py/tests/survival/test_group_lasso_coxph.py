"""Self-consistency tests for GroupLassoCoxPH and GroupLassoCoxPHCV.

All tests use synthetic data (Tier 1 per .assistant_instructions.md).

Provenance: new file, 2025-07.
"""
from __future__ import annotations

import numpy as np
import pytest

from pprof_py.models.survival.group_lasso_coxph import (
    GroupLassoCoxPH,
    GroupLassoCoxPHCV,
)
from pprof_py.models.survival.coxph import CoxPH
from pprof_py.exceptions import NotFittedError


# ======================================================================
# Data generators
# ======================================================================

def _synthetic_cox(n=400, p=9, seed=2025):
    """Synthetic right-censored survival data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    beta_true = np.zeros(p)
    beta_true[:3] = [0.5, -0.3, 0.2]
    risk = np.exp(X @ beta_true)
    U = rng.uniform(size=n)
    dur = -np.log(U) / risk
    C = rng.exponential(scale=np.median(dur) * 1.5, size=n)
    stop = np.minimum(dur, C)
    event = (dur <= C).astype(float)
    return X, np.maximum(stop, 1e-6), event, beta_true


@pytest.fixture(scope="module")
def synth_data():
    return _synthetic_cox()


# ======================================================================
# GroupLassoCoxPH
# ======================================================================

class TestGroupLassoCoxPH:
    """Self-consistency tests for group lasso Cox."""

    def test_groups_enter_together(self, synth_data):
        """Variables in the same group should enter/leave together
        under pure group lasso (alpha=0)."""
        X, stop, event, _ = synth_data
        groups = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        model = GroupLassoCoxPH(
            groups=groups, alpha=0.0, n_lambda=30, ties="breslow",
        )
        model.fit(X, duration=stop, event=event)
        for i in range(len(model.lambda_path_)):
            coef = model.coef_path_[i]
            for g in [1, 2, 3]:
                mask = groups == g
                gc = coef[mask]
                all_zero = np.all(np.abs(gc) < 1e-12)
                if all_zero:
                    continue
                assert not np.any(np.abs(gc) < 1e-12), (
                    f"Group {g} at lambda idx {i}: partial sparsity "
                    f"in pure group lasso"
                )

    def test_first_lambda_all_zero(self, synth_data):
        """At lambda_max all coefficients should be ~zero."""
        X, stop, event, _ = synth_data
        groups = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        model = GroupLassoCoxPH(
            groups=groups, alpha=0.0, n_lambda=30, ties="breslow",
        )
        model.fit(X, duration=stop, event=event)
        assert np.all(np.abs(model.coef_path_[0]) < 1e-10)

    def test_deviance_ratio_range(self, synth_data):
        """Deviance ratio should be in [0, 1]."""
        X, stop, event, _ = synth_data
        groups = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        model = GroupLassoCoxPH(
            groups=groups, alpha=0.0, n_lambda=20, ties="breslow",
        )
        model.fit(X, duration=stop, event=event)
        assert np.all(model.deviance_ratio_path_ >= -1e-10)
        assert np.all(model.deviance_ratio_path_ <= 1.0 + 1e-10)

    def test_lambda_to_zero_recovers_unpenalized(self, synth_data):
        """At tiny lambda, group lasso should recover the unpenalized
        CoxPH solution."""
        X, stop, event, _ = synth_data
        unpenalized = CoxPH(ties="breslow").fit(
            X, duration=stop, event=event,
        )
        groups = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        penalized = GroupLassoCoxPH(
            groups=groups, alpha=0.0, n_lambda=1,
            lambda_path=1e-9, ties="breslow",
        )
        penalized.fit(X, duration=stop, event=event)
        np.testing.assert_allclose(
            unpenalized.coef_, penalized.coef_path_[-1],
            atol=1e-4, rtol=1e-4,
        )


# ======================================================================
# GroupLassoCoxPHCV
# ======================================================================

class TestGroupLassoCoxPHCV:
    """CV selection tests."""

    def test_lambda_ordering(self, synth_data):
        """lambda_1se >= lambda_min."""
        X, stop, event, _ = synth_data
        groups = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        cv = GroupLassoCoxPHCV(
            groups=groups, alpha=0.0, n_lambda=20,
            n_folds=5, random_state=42, ties="breslow",
        )
        cv.fit(X, duration=stop, event=event)
        assert cv.lambda_1se_ >= cv.lambda_min_

    def test_cv_se_positive(self, synth_data):
        """CV standard errors should be non-negative."""
        X, stop, event, _ = synth_data
        groups = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        cv = GroupLassoCoxPHCV(
            groups=groups, alpha=0.0, n_lambda=20,
            n_folds=5, random_state=42, ties="breslow",
        )
        cv.fit(X, duration=stop, event=event)
        assert np.all(cv.cv_se_ >= 0)
