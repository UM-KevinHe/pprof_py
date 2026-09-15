"""Covariate-level statistical inference for the linear model family."""
from .fixed_effect import FixedEffectInferenceMixin
from .random_effect import RandomEffectInferenceMixin

__all__ = ["FixedEffectInferenceMixin", "RandomEffectInferenceMixin"]
