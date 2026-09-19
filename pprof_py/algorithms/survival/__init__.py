"""Numerical algorithms behind survival estimators: risk-set sweeps, tie
methods (Breslow/Efron/exact), the partial-likelihood engine,
Newton-Raphson optimization, penalized regression (CD + penalty),
provider-effect updates, and discrete-time survival.
"""
from .cox_likelihood import cox_partial_likelihood, precompute_stratum_indices
from .optimization import newton_raphson
from .ties import TieMethod, BreslowTies, EfronTies, ExactTies

__all__ = [
    "cox_partial_likelihood",
    "precompute_stratum_indices",
    "newton_raphson",
    "TieMethod",
    "BreslowTies",
    "EfronTies",
    "ExactTies",
]
