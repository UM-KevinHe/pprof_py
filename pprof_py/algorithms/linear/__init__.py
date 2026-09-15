"""Numerical kernels for linear model family."""
from .fixed_effect import (
    preprocess_groups,
    calculate_residuals,
    compute_model_statistics,
    construct_block_diag_matrix,
    perform_weighted_least_squares,
)

__all__ = [
    "preprocess_groups",
    "calculate_residuals",
    "compute_model_statistics",
    "construct_block_diag_matrix",
    "perform_weighted_least_squares",
]
