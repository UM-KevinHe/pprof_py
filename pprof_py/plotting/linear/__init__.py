"""Plotting mixins for the linear model family."""
from .fixed_effect import LinearFixedEffectPlottingMixin
from .random_effect import LinearRandomEffectPlottingMixin

__all__ = ["LinearFixedEffectPlottingMixin", "LinearRandomEffectPlottingMixin"]
