"""Presentation layer: figures, tables and reports built from validated ``pprof_py`` results.

Nothing here computes statistics. Displays consume ``test()``, ``funnel_limits()`` and related results, show their
uncertainty, denominators and provenance, and export deterministically. The namespace grows round by round and is
provisional until the presentation layer is complete: so far it holds the design tokens (:class:`Theme`), the
shared :mod:`~pprof_py.presentation.formatting` functions, presentation data (:class:`ProviderProfile`), and the
funnel plot (:func:`funnel`) and the interval plot (:func:`caterpillar`), returning :class:`FigureResult` objects.
"""
from . import formatting
from .data import PROFILE_COLUMNS, STATUSES, CapabilityError, ProviderProfile
from .figures import FigureResult, caterpillar, funnel
from .theme import STATUS_KEYS, Lines, StatusStyle, Theme, Typography, get_theme

__all__ = ["CapabilityError", "FigureResult", "PROFILE_COLUMNS", "ProviderProfile", "STATUSES", "STATUS_KEYS", "Lines",
           "StatusStyle", "Theme", "Typography", "caterpillar", "formatting", "funnel", "get_theme"]
