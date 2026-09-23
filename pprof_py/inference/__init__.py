"""Reusable inference: covariance, standard errors, hypothesis tests,
confidence intervals, and empirical-null calibration. See `inference.survival`
for the CoxPH/IUR-specific inference layer.

Provider testing is organised in layers that can be used separately:

* measures and z-statistics: :func:`standardized_measure` (from a fitted model),
  :class:`MeasureFrame` (from any estimates), :func:`z_statistic`;
* null models (:mod:`pprof_py.inference.empirical_null`): :class:`TheoreticalNull`,
  :class:`FixedNull`, :class:`EmpiricalNull`, with :func:`assign_groups` and
  :func:`robust_location_scale`;
* decisions: :func:`p_values`, :func:`flags`, :func:`intervals`, and
  :func:`provider_test`, which returns one fixed-schema table.
"""
from .zstat import IDENTITY, LOG, LOGIT, MeasureFrame, Transform, ZFrame, t_to_z, z_statistic, z_to_t
from .empirical_null import (BISQUARE_RLM, DEFAULT_ESTIMATOR, HUBER_RLM, MM_RLM, EmpiricalNull,
                             EmpiricalNullWarning, FixedNull, LocationScale, MEstimator, NullModel,
                             TheoreticalNull, assign_groups, robust_location_scale)
from .decision import PROVIDER_TEST_COLUMNS, calibrate, flags, intervals, p_values, provider_test, resolve_null_model
from .effect_tests import bootstrap_tails, effect_test, poibin_tails, resample_tails, z_from_tails
from .standardized import MEASURES, StandardPopulation, at_bound, standardized_measure

__all__ = [
    "LocationScale", "MEstimator", "DEFAULT_ESTIMATOR", "HUBER_RLM", "BISQUARE_RLM", "MM_RLM",
    "robust_location_scale",
    "assign_groups",
    "Transform", "IDENTITY", "LOGIT", "LOG", "MeasureFrame", "ZFrame", "z_statistic",
    "NullModel", "TheoreticalNull", "FixedNull", "EmpiricalNull", "EmpiricalNullWarning",
    "calibrate", "p_values", "flags", "intervals", "provider_test", "resolve_null_model", "PROVIDER_TEST_COLUMNS",
    "effect_test", "poibin_tails", "resample_tails", "bootstrap_tails", "z_from_tails", "t_to_z", "z_to_t",
    "StandardPopulation", "standardized_measure", "at_bound", "MEASURES",
]
