"""Covariate-level statistical inference for the linear model family."""
from .fixed_effect import LinearFixedEffectInferenceMixin
from .random_effect import LinearRandomEffectInferenceMixin

__all__ = ["LinearFixedEffectInferenceMixin", "LinearRandomEffectInferenceMixin"]
