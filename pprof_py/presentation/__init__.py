"""Presentation layer: figures, tables and reports built from validated ``pprof_py`` results.

Nothing here computes statistics. Displays consume ``test()``, ``funnel_limits()`` and related results, show their
uncertainty, denominators and provenance, and export deterministically. The namespace holds the design tokens
(:class:`Theme`), the shared :mod:`~pprof_py.presentation.formatting` functions, presentation data
(:class:`ProviderProfile` and its relatives), figures returning :class:`FigureResult` objects, tables returning
:class:`TableResult` objects, and reports (:class:`Report`).

Stability: from 0.8 the public names here follow the package's deprecation policy. Renaming or removing one, or
changing its signature or return type, takes one minor release of ``DeprecationWarning``. Figure styling may still
be refined, and ``_``-prefixed modules are private.
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
