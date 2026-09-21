"""Smoke tests for infrastructure modules with no dedicated test coverage.

Covers: statistics/deviance.py, data/validation.py, utils/, plotting/,
exceptions.py.

All tests use synthetic data (Tier 1 per .assistant_instructions.md).

Provenance: new file, 2025-07.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ======================================================================
# statistics/deviance.py
# ======================================================================

class TestDeviance:
    """Smoke tests for Cox deviance functions."""

    def test_saturated_log_likelihood_nonnegative_weight(self):
        """saturated_log_likelihood should return a finite float."""
        from pprof_py.statistics.deviance import saturated_log_likelihood
        stop = np.array([1.0, 2.0, 2.0, 3.0, 4.0])
        event = np.array([1, 1, 1, 0, 1])
        weight = np.ones(5)
        strata = np.zeros(5, dtype=int)
        lsat = saturated_log_likelihood(stop, event, weight, strata)
        assert np.isfinite(lsat)

    def test_saturated_log_likelihood_no_events(self):
        """With no events, saturated log-lik should be 0."""
        from pprof_py.statistics.deviance import saturated_log_likelihood
        stop = np.array([1.0, 2.0, 3.0])
        event = np.array([0, 0, 0])
        weight = np.ones(3)
        strata = np.zeros(3, dtype=int)
        assert saturated_log_likelihood(stop, event, weight, strata) == 0.0

    def test_cox_deviance_positive(self):
        """Deviance = 2*(lsat - ll) should be non-negative when ll <= lsat."""
        from pprof_py.statistics.deviance import cox_deviance
        assert cox_deviance(-10.0, -5.0) == 10.0
        assert cox_deviance(-5.0, -5.0) == 0.0

    def test_deviance_ratio_bounds(self):
        """Deviance ratio should be in [0, 1] for reasonable inputs."""
        from pprof_py.statistics.deviance import deviance_ratio
        # Perfect model: ll == lsat
        assert deviance_ratio(-5.0, -20.0, -5.0) == pytest.approx(1.0)
        # Null model: ll == ll_null
        assert deviance_ratio(-20.0, -20.0, -5.0) == pytest.approx(0.0)

    def test_deviance_ratio_zero_null_deviance(self):
        """When null deviance is 0, deviance_ratio should return 0."""
        from pprof_py.statistics.deviance import deviance_ratio
        assert deviance_ratio(-5.0, -5.0, -5.0) == 0.0


# ======================================================================
# data/validation.py
# ======================================================================

class TestDataValidation:
    """Smoke tests for data validation utilities."""

    def test_check_missingness(self):
        from pprof_py.data.validation import check_missingness
        df = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6], "y": [0, 1, 0]})
        # Should not raise (all columns clean).
        check_missingness(df, ["x1", "x2", "y"])

    def test_check_missingness_raises_on_nan(self):
        from pprof_py.data.validation import check_missingness
        df = pd.DataFrame({"x1": [1, np.nan, 3], "y": [0, 1, 0]})
        with pytest.raises(ValueError):
            check_missingness(df, ["x1", "y"])

    def test_check_variation(self):
        from pprof_py.data.validation import check_variation
        # x2 has zero variance — check_variation raises ValueError.
        df_bad = pd.DataFrame({"x1": [1, 2, 3], "x2": [1, 1, 1]})
        with pytest.raises(ValueError, match="zero variance"):
            check_variation(df_bad, ["x1", "x2"])

    def test_check_variation_no_raise(self):
        from pprof_py.data.validation import check_variation
        # All columns have variation — should not raise.
        df_ok = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6]})
        check_variation(df_ok, ["x1", "x2"])

    def test_validate_and_convert_inputs_basic(self):
        from pprof_py.data.validation import validate_and_convert_inputs
        df = pd.DataFrame({
            "provider": ["A", "A", "B", "B", "C", "C"],
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y": [0, 1, 0, 1, 0, 1],
        })
        result = validate_and_convert_inputs(
            df, x_vars=["x1"], y_var="y", group_var="provider",
        )
        assert result.y.shape[0] == 6
        assert result.X.shape == (6, 1)


# ======================================================================
# utils/
# ======================================================================

class TestUtils:
    """Smoke tests for utility functions."""

    def test_sigmoid(self):
        from pprof_py.utils import sigmoid
        assert sigmoid(0.0) == pytest.approx(0.5)
        assert 0.0 < sigmoid(-10.0) < 0.01
        assert 0.99 < sigmoid(10.0) < 1.0

    def test_sigmoid_array(self):
        from pprof_py.utils import sigmoid
        result = sigmoid(np.array([-100, 0, 100]))
        assert result.shape == (3,)
        np.testing.assert_allclose(result[1], 0.5)

    def test_proc_freq(self):
        from pprof_py.utils import proc_freq
        df = pd.DataFrame({"cat": ["A", "A", "B", "C", "C", "C"]})
        # proc_freq requires a DataFrame and a list of column names.
        proc_freq(df, ["cat"])

    def test_setup_logger(self):
        from pprof_py.utils import setup_logger
        logger = setup_logger("test_infra")
        assert logger.name == "test_infra"

    def test_safe_exp(self):
        from pprof_py.utils.numerical import safe_exp
        # Should not overflow.
        result = safe_exp(np.array([0.0, 1.0, 500.0, -500.0]))
        assert np.all(np.isfinite(result))
        assert result[0] == pytest.approx(1.0)


# ======================================================================
# plotting/ (import-only smoke tests — no display)
# ======================================================================

class TestPlotting:
    """Import and minimal-call tests for plotting functions."""

    def test_plot_caterpillar_import(self):
        from pprof_py.plotting import plot_caterpillar
        assert callable(plot_caterpillar)

    def test_plot_funnel_import(self):
        from pprof_py.plotting import plot_funnel
        assert callable(plot_funnel)

    def test_style_module_import(self):
        from pprof_py.plotting import style
        # Style module exposes color constants and remove_top_right_spines.
        assert hasattr(style, "COLOR_PRIMARY")
        assert hasattr(style, "remove_top_right_spines")


# ======================================================================
# exceptions.py
# ======================================================================

class TestExceptions:
    """Tests for the package-wide exception hierarchy."""

    def test_not_fitted_error_inherits_runtime(self):
        from pprof_py.exceptions import NotFittedError
        assert issubclass(NotFittedError, RuntimeError)

    def test_not_fitted_error_not_value_error(self):
        from pprof_py.exceptions import NotFittedError
        assert not issubclass(NotFittedError, ValueError)

    def test_not_fitted_error_catchable(self):
        from pprof_py.exceptions import NotFittedError
        with pytest.raises(NotFittedError):
            raise NotFittedError("test")
