"""Empirical-null calibration for provider z-statistics.

* estimators: :func:`robust_location_scale`, :class:`MEstimator`, and presets
  (:data:`DEFAULT_ESTIMATOR`, :data:`HUBER_RLM`, :data:`BISQUARE_RLM`, :data:`MM_RLM`);
* grouping: :func:`assign_groups`;
* null models: :class:`TheoreticalNull`, :class:`FixedNull`, :class:`EmpiricalNull`.
"""
from .estimators import (BISQUARE_RLM, DEFAULT_ESTIMATOR, HUBER_RLM, MM_RLM, LocationScale, MEstimator,
                         robust_location_scale)
from .grouping import assign_groups
from .models import EmpiricalNull, EmpiricalNullWarning, FixedNull, NullModel, TheoreticalNull

__all__ = [
    "LocationScale", "MEstimator", "robust_location_scale",
    "DEFAULT_ESTIMATOR", "HUBER_RLM", "BISQUARE_RLM", "MM_RLM",
    "assign_groups", "NullModel", "TheoreticalNull", "FixedNull", "EmpiricalNull", "EmpiricalNullWarning",
]
