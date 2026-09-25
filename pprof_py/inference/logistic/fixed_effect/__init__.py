"""Inference for ``LogisticFixedEffectModel``.

``LogisticFixedEffectInferenceMixin`` combines three groups of methods:

* ``covariates``: variance estimation and covariate (beta) tests, ``summary()``;
* ``provider_tests``: provider-effect tests, ``test()`` and ``test_standardized()``;
* ``confidence_intervals``: provider-effect and standardized-measure intervals,
  ``calculate_confidence_intervals()``.
"""
from .confidence_intervals import _ConfidenceIntervalMethods
from .covariates import _CovariateInferenceMethods
from .provider_tests import _ProviderTestMethods


class LogisticFixedEffectInferenceMixin(
    _CovariateInferenceMethods,
    _ProviderTestMethods,
    _ConfidenceIntervalMethods,
):
    """Covariate inference, provider tests, and confidence intervals for
    ``LogisticFixedEffectModel``."""


__all__ = ["LogisticFixedEffectInferenceMixin"]
