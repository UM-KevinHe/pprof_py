"""Regression tests for CoxPH prediction-time input validation.

Addresses COX_USER_REVIEW Finding 1:
  - Predictions must not silently depend on DataFrame column order.
  - Offset must be validated for shape and finiteness.
  - Array feature count must match the fitted model.
"""
import numpy as np
import pandas as pd
import pytest

from pprof_py import CoxPH
from pprof_py.data.survival_validation import (
    SurvivalDataError,
    validate_X_predict,
    validate_predict_offset,
)


# ---------------------------------------------------------------------------
# Shared fixture: a minimal fitted CoxPH model
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fitted_cox():
    """Fit a CoxPH on 200-row synthetic data with columns [age, bmi, score]."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "age": rng.normal(60, 10, n),
        "bmi": rng.normal(25, 5, n),
        "score": rng.normal(0, 1, n),
    })
    time = rng.exponential(5, n)
    event = rng.binomial(1, 0.6, n).astype(float)
    model = CoxPH(ties="breslow")
    model.fit(df, duration=time, event=event)
    return model, df


# ===================================================================
# Column-order validation (DataFrame)
# ===================================================================
class TestColumnOrderValidation:
    """CoxPH.predict_linear must match columns against feature_names_in_."""

    def test_correct_order_unchanged(self, fitted_cox):
        model, df = fitted_cox
        pred = model.predict_linear(df.head(5))
        assert pred.shape == (5,)
        assert np.all(np.isfinite(pred))

    def test_reversed_columns_auto_reordered(self, fitted_cox):
        """Reversed column order must produce identical predictions."""
        model, df = fitted_cox
        df_normal = df.head(5)
        df_reversed = df_normal[["score", "bmi", "age"]]
        pred_normal = model.predict_linear(df_normal)
        pred_reversed = model.predict_linear(df_reversed)
        np.testing.assert_allclose(pred_reversed, pred_normal, rtol=1e-14)

    def test_shuffled_columns_auto_reordered(self, fitted_cox):
        """Any permutation of columns must produce identical predictions."""
        model, df = fitted_cox
        df_normal = df.head(5)
        df_shuffled = df_normal[["bmi", "score", "age"]]
        pred_normal = model.predict_linear(df_normal)
        pred_shuffled = model.predict_linear(df_shuffled)
        np.testing.assert_allclose(pred_shuffled, pred_normal, rtol=1e-14)

    def test_renamed_columns_rejected(self, fitted_cox):
        model, df = fitted_cox
        df_renamed = df.head(3).rename(columns={"age": "AGE"})
        with pytest.raises(ValueError, match="missing.*age.*unexpected.*AGE"):
            model.predict_linear(df_renamed)

    def test_extra_column_rejected(self, fitted_cox):
        model, df = fitted_cox
        df_extra = df.head(3).copy()
        df_extra["extra"] = 1.0
        with pytest.raises(ValueError, match="unexpected.*extra"):
            model.predict_linear(df_extra)

    def test_missing_column_rejected(self, fitted_cox):
        model, df = fitted_cox
        with pytest.raises(ValueError, match="missing.*score"):
            model.predict_linear(df[["age", "bmi"]].head(3))

    def test_duplicate_columns_rejected(self, fitted_cox):
        model, df = fitted_cox
        df_dup = df.head(3)[["age", "bmi", "age"]]
        with pytest.raises(ValueError, match="duplicate"):
            model.predict_linear(df_dup)


# ===================================================================
# Array feature-count validation
# ===================================================================
class TestArrayFeatureCount:
    """Plain arrays must have the same number of columns as the fitted model."""

    def test_correct_array_accepted(self, fitted_cox):
        model, df = fitted_cox
        X_arr = df.head(3).to_numpy()
        pred = model.predict_linear(X_arr)
        assert pred.shape == (3,)

    def test_too_many_features_rejected(self, fitted_cox):
        model, _ = fitted_cox
        with pytest.raises(ValueError, match="5 feature.*3"):
            model.predict_linear(np.zeros((3, 5)))

    def test_too_few_features_rejected(self, fitted_cox):
        model, _ = fitted_cox
        with pytest.raises(ValueError, match="2 feature.*3"):
            model.predict_linear(np.zeros((3, 2)))


# ===================================================================
# Offset validation
# ===================================================================
class TestOffsetValidation:
    """Offset must be 1-d, correct length, and finite."""

    def test_none_offset_is_zero(self, fitted_cox):
        model, df = fitted_cox
        pred_none = model.predict_linear(df.head(3), offset=None)
        pred_zero = model.predict_linear(df.head(3), offset=np.zeros(3))
        np.testing.assert_allclose(pred_none, pred_zero, rtol=1e-14)

    def test_scalar_offset_broadcast(self, fitted_cox):
        model, df = fitted_cox
        pred_scalar = model.predict_linear(df.head(3), offset=2.5)
        pred_vec = model.predict_linear(df.head(3), offset=np.full(3, 2.5))
        np.testing.assert_allclose(pred_scalar, pred_vec, rtol=1e-14)

    def test_2d_offset_rejected(self, fitted_cox):
        """A (3,1) offset must not silently broadcast to a (3,3) result."""
        model, df = fitted_cox
        with pytest.raises(ValueError, match="1-dimensional"):
            model.predict_linear(df.head(3), offset=np.zeros((3, 1)))

    def test_wrong_length_offset_rejected(self, fitted_cox):
        model, df = fitted_cox
        with pytest.raises(ValueError, match="5 element.*3 row"):
            model.predict_linear(df.head(3), offset=np.zeros(5))

    def test_nan_offset_rejected(self, fitted_cox):
        model, df = fitted_cox
        with pytest.raises(ValueError, match="NaN or infinite"):
            model.predict_linear(df.head(3), offset=np.array([1.0, np.nan, 3.0]))

    def test_inf_offset_rejected(self, fitted_cox):
        model, df = fitted_cox
        with pytest.raises(ValueError, match="NaN or infinite"):
            model.predict_linear(df.head(3), offset=np.array([1.0, np.inf, 3.0]))


# ===================================================================
# Downstream methods inherit validation
# ===================================================================
class TestDownstreamMethodsInherit:
    """predict_partial_hazard, predict, predict_cumulative_hazard, and
    predict_survival_function all delegate to predict_linear, so they
    must inherit the same validation."""

    def test_predict_partial_hazard_rejects_renamed(self, fitted_cox):
        model, df = fitted_cox
        df_bad = df.head(3).rename(columns={"age": "x"})
        with pytest.raises(ValueError, match="missing"):
            model.predict_partial_hazard(df_bad)

    def test_predict_rejects_bad_offset(self, fitted_cox):
        model, df = fitted_cox
        with pytest.raises(ValueError, match="1-dimensional"):
            model.predict(df.head(3), offset=np.zeros((3, 1)))

    def test_predict_cumulative_hazard_rejects_extra_col(self, fitted_cox):
        model, df = fitted_cox
        df_bad = df.head(3).copy()
        df_bad["extra"] = 1.0
        with pytest.raises(ValueError, match="unexpected"):
            model.predict_cumulative_hazard(df_bad)

    def test_predict_survival_function_rejects_wrong_array(self, fitted_cox):
        model, _ = fitted_cox
        with pytest.raises(ValueError, match="feature"):
            model.predict_survival_function(np.zeros((3, 5)))


# ===================================================================
# Unit tests for the helper functions directly
# ===================================================================
class TestValidateXPredictUnit:
    """Direct tests of validate_X_predict."""

    def test_array_passthrough(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = validate_X_predict(X, ["a", "b"], 2)
        np.testing.assert_array_equal(result, X)

    def test_dataframe_reorder(self):
        df = pd.DataFrame({"b": [4.0, 5.0], "a": [1.0, 2.0]})
        result = validate_X_predict(df, ["a", "b"], 2)
        # Column 0 should be "a", column 1 should be "b"
        np.testing.assert_array_equal(result[:, 0], [1.0, 2.0])
        np.testing.assert_array_equal(result[:, 1], [4.0, 5.0])

    def test_nonfinite_X_rejected(self):
        X = np.array([[1.0, np.nan]])
        with pytest.raises(SurvivalDataError, match="NaN"):
            validate_X_predict(X, ["a", "b"], 2)


class TestValidatePredictOffsetUnit:
    """Direct tests of validate_predict_offset."""

    def test_none_returns_zeros(self):
        result = validate_predict_offset(None, 5)
        np.testing.assert_array_equal(result, np.zeros(5))

    def test_scalar_broadcasts(self):
        result = validate_predict_offset(3.0, 4)
        np.testing.assert_array_equal(result, np.full(4, 3.0))

    def test_list_accepted(self):
        result = validate_predict_offset([1.0, 2.0, 3.0], 3)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])
