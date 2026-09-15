"""Plotting mixins for the linear model family."""
from .fixed_effect import FixedEffectPlottingMixin
from .random_effect import RandomEffectPlottingMixin

__all__ = ["FixedEffectPlottingMixin", "RandomEffectPlottingMixin"]
