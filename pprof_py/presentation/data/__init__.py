"""Presentation data: provider-level results ready for display (spec §6).

:class:`ProviderProfile` holds one provider test with its denominators, statuses, provenance and capabilities;
displays consume profiles and raise :class:`CapabilityError` when a profile lacks what they need.
"""
from ._coefficients import COEFFICIENT_COLUMNS, CoefficientProfile
from ._profile import PROFILE_COLUMNS, STATUSES, CapabilityError, ProviderProfile

__all__ = ["COEFFICIENT_COLUMNS", "CapabilityError", "CoefficientProfile", "PROFILE_COLUMNS", "ProviderProfile",
           "STATUSES"]
