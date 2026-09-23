"""Empirical-null calibration for provider z-statistics.

* estimators: :func:`robust_location_scale`, :class:`MEstimator`, and presets
  (:data:`DEFAULT_ESTIMATOR`, :data:`HUBER_RLM`, :data:`BISQUARE_RLM`, :data:`MM_RLM`);
* grouping: :func:`assign_groups`;
* null models: :class:`TheoreticalNull`, :class:`FixedNull`, :class:`EmpiricalNull`.

The functions re-exported from ``_legacy`` are the previous implementation,
kept only until their callers move to the classes above.
"""
from .estimators import (BISQUARE_RLM, DEFAULT_ESTIMATOR, HUBER_RLM, MM_RLM, LocationScale, MEstimator,
                         robust_location_scale)
from .grouping import assign_groups
from .models import EmpiricalNull, EmpiricalNullWarning, FixedNull, NullModel, TheoreticalNull
from ._legacy import (assign_flags, calibrate_empirical_null, estimate_empirical_null, huber_location_scale,
                      poibin_exact_pvalue, pvalues_to_zscores, resample_pvalue)

__all__ = [
    "LocationScale", "MEstimator", "robust_location_scale",
    "DEFAULT_ESTIMATOR", "HUBER_RLM", "BISQUARE_RLM", "MM_RLM",
    "assign_groups", "NullModel", "TheoreticalNull", "FixedNull", "EmpiricalNull", "EmpiricalNullWarning",
]
