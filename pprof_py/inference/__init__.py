"""Reusable inference: covariance, standard errors, hypothesis tests,
confidence intervals, and empirical-null calibration. See `inference.survival`
for the CoxPH/IUR-specific inference layer.

Provider testing is organised in layers that can be used separately:

* measures and z-statistics: :class:`MeasureFrame`, :func:`z_statistic`;
* null models (:mod:`pprof_py.inference.empirical_null`): :class:`TheoreticalNull`,
  :class:`FixedNull`, :class:`EmpiricalNull`, with :func:`assign_groups` and
  :func:`robust_location_scale`;
* decisions: :func:`p_values`, :func:`flags`, :func:`intervals`, and
  :func:`provider_test`, which returns one fixed-schema table.
"""
from .zstat import IDENTITY, LOG, LOGIT, MeasureFrame, Transform, ZFrame, z_statistic
from .empirical_null import (BISQUARE_RLM, DEFAULT_ESTIMATOR, HUBER_RLM, MM_RLM, EmpiricalNull,
                             EmpiricalNullWarning, FixedNull, LocationScale, MEstimator, NullModel,
                             TheoreticalNull, assign_groups, estimate_empirical_null, huber_location_scale,
                             robust_location_scale)
from .decision import PROVIDER_TEST_COLUMNS, calibrate, flags, intervals, p_values, provider_test

__all__ = [
    "huber_location_scale", "estimate_empirical_null",
    "LocationScale", "MEstimator", "DEFAULT_ESTIMATOR", "HUBER_RLM", "BISQUARE_RLM", "MM_RLM",
    "robust_location_scale",
    "assign_groups",
    "Transform", "IDENTITY", "LOGIT", "LOG", "MeasureFrame", "ZFrame", "z_statistic",
    "NullModel", "TheoreticalNull", "FixedNull", "EmpiricalNull", "EmpiricalNullWarning",
    "calibrate", "p_values", "flags", "intervals", "provider_test", "PROVIDER_TEST_COLUMNS",
]
