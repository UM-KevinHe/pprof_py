"""Reusable inference: covariance, standard errors, hypothesis tests,
confidence intervals, and empirical-null calibration. See `inference.survival`
for the CoxPH/IUR-specific inference layer."""
from .empirical_null import huber_location_scale, estimate_empirical_null

__all__ = ["huber_location_scale", "estimate_empirical_null"]
