"""Self-consistency tests for ProviderPenalizedCoxPH.

All tests use synthetic data (Tier 1 per .assistant_instructions.md).

Provenance: new file, 2025-07.
"""
from __future__ import annotations

import numpy as np
import pytest

from pprof_py.models.survival.provider_coxph import ProviderPenalizedCoxPH
from pprof_py.exceptions import NotFittedError


# ======================================================================
# Data generators
# ======================================================================

def _synthetic_provider_cox(
    n=500, p=6, n_providers=20, seed=2025,
):
    """Synthetic survival data with provider structure."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    provider_id = rng.randint(0, n_providers, size=n)
    beta_true = np.zeros(p)
    beta_true[:3] = [0.5, -0.3, 0.2]
    gamma_true = rng.randn(n_providers) * 0.3
    risk = np.exp(X @ beta_true + gamma_true[provider_id])
    U = rng.uniform(size=n)
    dur = -np.log(U) / risk
    C = rng.exponential(scale=np.median(dur) * 1.5, size=n)
    stop = np.minimum(dur, C)
    event = (dur <= C).astype(float)
    return X, np.maximum(stop, 1e-6), event, provider_id, beta_true, gamma_true


@pytest.fixture(scope="module")
def synth_data():
    return _synthetic_provider_cox()


# ======================================================================
# ProviderPenalizedCoxPH
# ======================================================================

class TestProviderPenalizedCoxPH:
    """Self-consistency tests."""

    def test_fit_returns_self(self, synth_data):
        """fit() should return the model instance."""
        X, stop, event, prov, _, _ = synth_data
        model = ProviderPenalizedCoxPH(
            penalty_type="elastic_net", alpha=1.0,
            n_lambda=10, ties="breslow",
        )
        result = model.fit(
            X, duration=stop, event=event, provider=prov,
        )
        assert result is model

    def test_provider_effects_shape(self, synth_data):
        """gamma_path_ should have shape (n_lambda, n_providers)."""
        X, stop, event, prov, _, _ = synth_data
        model = ProviderPenalizedCoxPH(
            penalty_type="elastic_net", alpha=1.0,
            n_lambda=10, ties="breslow",
        )
        model.fit(X, duration=stop, event=event, provider=prov)
        n_lambda = len(model.lambda_path_)
        n_prov = len(model.provider_labels_)
        assert model.gamma_path_.shape == (n_lambda, n_prov)

    def test_coef_path_shape(self, synth_data):
        """coef_path_ should have shape (n_lambda, p)."""
        X, stop, event, prov, _, _ = synth_data
        model = ProviderPenalizedCoxPH(
            penalty_type="elastic_net", alpha=1.0,
            n_lambda=10, ties="breslow",
        )
        model.fit(X, duration=stop, event=event, provider=prov)
        assert model.coef_path_.shape[1] == X.shape[1]

    def test_first_lambda_coef_zero(self, synth_data):
        """At lambda_max, penalized covariate coefficients should be ~zero."""
        X, stop, event, prov, _, _ = synth_data
        model = ProviderPenalizedCoxPH(
            penalty_type="elastic_net", alpha=1.0,
            n_lambda=20, ties="breslow",
        )
        model.fit(X, duration=stop, event=event, provider=prov)
        assert np.all(np.abs(model.coef_path_[0]) < 1e-10)

    def test_sparsity_monotonicity(self, synth_data):
        """Nonzero covariate count should be non-decreasing."""
        X, stop, event, prov, _, _ = synth_data
        model = ProviderPenalizedCoxPH(
            penalty_type="elastic_net", alpha=1.0,
            n_lambda=20, ties="breslow",
        )
        model.fit(X, duration=stop, event=event, provider=prov)
        nz = np.array([np.sum(row != 0) for row in model.coef_path_])
        assert np.all(np.diff(nz) >= 0), f"Sparsity not monotonic: {nz}"

    def test_provider_bound_respected(self, synth_data):
        """Provider effects should be bounded by provider_bound."""
        X, stop, event, prov, _, _ = synth_data
        bound = 5.0
        model = ProviderPenalizedCoxPH(
            penalty_type="elastic_net", alpha=1.0,
            n_lambda=10, provider_bound=bound, ties="breslow",
        )
        model.fit(X, duration=stop, event=event, provider=prov)
        median_gamma = np.median(model.gamma_path_, axis=1, keepdims=True)
        deviations = np.abs(model.gamma_path_ - median_gamma)
        assert np.all(deviations <= bound + 1e-6)

    def test_deviance_ratio_range(self, synth_data):
        """Deviance ratio should be in [0, 1]."""
        X, stop, event, prov, _, _ = synth_data
        model = ProviderPenalizedCoxPH(
            penalty_type="elastic_net", alpha=1.0,
            n_lambda=10, ties="breslow",
        )
        model.fit(X, duration=stop, event=event, provider=prov)
        assert np.all(model.deviance_ratio_path_ >= -1e-10)
        assert np.all(model.deviance_ratio_path_ <= 1.0 + 1e-10)
