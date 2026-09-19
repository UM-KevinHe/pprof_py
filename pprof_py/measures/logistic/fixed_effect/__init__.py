"""Provider standardization and measure-specific workflows for
``LogisticFixedEffectModel``: standardized rates/ratios (direct/indirect),
provider-level confidence intervals, and provider-effect hypothesis
testing (including empirical-null calibration).

Formerly a single file; split into three concern-specific sub-modules
(standardized measures, confidence intervals, provider tests).  Mixed
into the model class so that ``models/logistic/fixed_effect.py`` stays
focused on configuration, fitting, and prediction.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable

import numpy as np

from .standardized_measures import _StandardizedMeasureMethods
from .confidence_intervals import _ConfidenceIntervalMethods
from .provider_tests import _ProviderTestMethods


@runtime_checkable
class _LogisticFEMeasuresHost(Protocol):
    """Attribute contract that ``FixedEffectMeasuresMixin`` expects from its
    host class (``LogisticFixedEffectModel``)."""

    coefficients_: Optional[Dict[str, Any]]
    variances_: Optional[Dict[str, Any]]
    robust_variances_: Optional[Dict[str, Any]]
    fitted_: Optional[np.ndarray]
    groups_: Optional[np.ndarray]
    group_indices_: Optional[np.ndarray]
    group_sizes_: Optional[np.ndarray]
    outcome_: Optional[np.ndarray]
    xbeta_: Optional[np.ndarray]
    N_: Optional[np.ndarray]
    # --- Configuration / algorithm ---
    algorithm: Any

    def _check_is_fitted(self) -> None: ...
    def calculate_standardized_measures(self, **kwargs: Any) -> dict: ...


class FixedEffectMeasuresMixin(
    _StandardizedMeasureMethods,
    _ConfidenceIntervalMethods,
    _ProviderTestMethods,
):
    """Standardized measures, provider-level CIs, and provider-effect tests
    for ``LogisticFixedEffectModel``.

    Composed from three concern-specific mixin fragments:

    * :class:`._StandardizedMeasureMethods` — ``calculate_standardized_measures``
    * :class:`._ConfidenceIntervalMethods` — ``calculate_confidence_intervals``
    * :class:`._ProviderTestMethods` — ``test``, ``test_standardized``
    """

    pass
