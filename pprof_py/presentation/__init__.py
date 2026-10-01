"""Presentation layer: figures, tables and reports built from validated ``pprof_py`` results.

Nothing here computes statistics. Displays consume ``test()``, ``funnel_limits()`` and related results, show their
uncertainty, denominators and provenance, and export deterministically. The namespace grows round by round and is
provisional until the presentation layer is complete: so far it holds the design tokens (:class:`Theme`), the
shared :mod:`~pprof_py.presentation.formatting` functions, presentation data (:class:`ProviderProfile`), and the
funnel plot (:func:`funnel`) and the interval plot (:func:`caterpillar`), returning :class:`FigureResult` objects,
and the provider table (:func:`provider_table`, returning a :class:`TableResult`).
"""
from . import formatting
from .data import (PROFILE_COLUMNS, STATUSES, CapabilityError, CoefficientProfile, ProfileCollection,
                   ProviderProfile)
from .figures import (FigureResult, caterpillar, data_quality, flag_stability, forest, funnel, measure_agreement,
                      multi_measure, null_calibration,
                      observed_expected, provider_variation, reliability, shrinkage)
from .reports import Report
from .tables import (Column, TableResult, TableSpec, coefficient_table, data_quality_table, flag_stability_table,
                     multi_measure_table, null_calibration_table,
                     provider_table, provider_variation_table, reliability_table, shrinkage_table)
from .theme import STATUS_KEYS, Lines, StatusStyle, Theme, Typography, get_theme

__all__ = ["CapabilityError", "CoefficientProfile", "Column", "ProfileCollection", "Report", "FigureResult", "PROFILE_COLUMNS", "ProviderProfile", "STATUSES", "STATUS_KEYS",
           "Lines", "StatusStyle", "TableResult", "TableSpec", "Theme", "Typography", "caterpillar", "formatting",
           "coefficient_table", "data_quality", "data_quality_table", "flag_stability",
           "flag_stability_table", "forest", "measure_agreement", "multi_measure", "multi_measure_table", "funnel", "get_theme",
           "null_calibration", "null_calibration_table", "observed_expected", "provider_table",
           "provider_variation", "provider_variation_table", "reliability", "reliability_table",
           "shrinkage", "shrinkage_table"]
