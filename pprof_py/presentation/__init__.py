"""Presentation layer: figures, tables and reports built from validated ``pprof_py`` results.

Nothing here computes statistics. Displays consume ``test()``, ``funnel_limits()`` and related results, show their
uncertainty, denominators and provenance, and export deterministically. The namespace grows round by round and is
provisional until the presentation layer is complete: so far it holds the design tokens (:class:`Theme`), the
shared :mod:`~pprof_py.presentation.formatting` functions, and presentation data (:class:`ProviderProfile`).
"""
from . import formatting
from .data import PROFILE_COLUMNS, STATUSES, CapabilityError, ProviderProfile
from .theme import STATUS_KEYS, Lines, StatusStyle, Theme, Typography, get_theme

__all__ = ["CapabilityError", "PROFILE_COLUMNS", "ProviderProfile", "STATUSES", "STATUS_KEYS", "Lines", "StatusStyle",
           "Theme", "Typography", "formatting", "get_theme"]
