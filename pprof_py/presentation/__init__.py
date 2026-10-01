"""Presentation layer: figures, tables and reports built from validated ``pprof_py`` results.

Nothing here computes statistics. Displays consume ``test()``, ``funnel_limits()`` and related results, show their
uncertainty, denominators and provenance, and export deterministically. The namespace grows round by round and is
provisional until the presentation layer is complete: so far it holds the design tokens (:class:`Theme`), the
shared :mod:`~pprof_py.presentation.formatting` functions, presentation data (:class:`ProviderProfile`), and the
funnel plot (:func:`funnel`) and the interval plot (:func:`caterpillar`), returning :class:`FigureResult` objects,
and the provider table (:func:`provider_table`, returning a :class:`TableResult`).
"""
from . import formatting
from .data import PROFILE_COLUMNS, STATUSES, CapabilityError, CoefficientProfile, ProviderProfile
from .figures import FigureResult, caterpillar, forest, funnel
from .tables import Column, TableResult, TableSpec, coefficient_table, provider_table
from .theme import STATUS_KEYS, Lines, StatusStyle, Theme, Typography, get_theme

__all__ = ["CapabilityError", "CoefficientProfile", "Column", "FigureResult", "PROFILE_COLUMNS", "ProviderProfile", "STATUSES", "STATUS_KEYS",
           "Lines", "StatusStyle", "TableResult", "TableSpec", "Theme", "Typography", "caterpillar", "formatting",
           "coefficient_table", "forest", "funnel", "get_theme", "provider_table"]
