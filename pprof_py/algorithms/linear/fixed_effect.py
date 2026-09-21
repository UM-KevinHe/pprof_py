"""Numerical kernels for the linear fixed-effect model.

Implements the closed-form weighted-least-squares fit used by
``LinearFixedEffectModel``: group means are projected out via a
block-diagonal demeaning matrix, then beta is solved directly --
no iteration, which is why this model has no ``max_iter`` / ``tol``.

Kept separate from ``models/linear/fixed_effect.py`` so that the
model class stays focused on configuration and the public
fit/predict API -- the same split used for inference, measures,
and plotting.

All functions are stateless: NumPy arrays in, NumPy arrays out,
no access to model state.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import block_diag


def construct_block_diag_matrix(group_sizes: np.ndarray) -> np.ndarray:
    """Construct a block diagonal matrix for demeaning group effects.

    Parameters
    ----------
    group_sizes : np.ndarray, shape (n_groups,)
        Number of samples in each group.

    Returns
    -------
    np.ndarray
        Block diagonal matrix for demeaning group effects.
    """
    Q_blocks = [np.eye(n) - np.ones((n, n)) / n for n in group_sizes]
    return block_diag(*Q_blocks)


class GroupDemeaner:
    """Within-group demeaning operator Q = I - P_groups, applied via group indices.

    Row-order independent and O(n) memory (replaces the dense block-diagonal matrix,
    which is only correct when rows are sorted by group)."""

    def __init__(self, group_indices: np.ndarray):
        self.gi = np.asarray(group_indices)
        self.m = int(self.gi.max()) + 1 if self.gi.size else 0
        self.sizes = np.bincount(self.gi, minlength=self.m).astype(float)

    def __matmul__(self, A):
        A = np.asarray(A, dtype=float)
        one_d = A.ndim == 1
        A2 = A[:, None] if one_d else A
        out = np.empty_like(A2)
        for j in range(A2.shape[1]):
            means = np.bincount(self.gi, weights=A2[:, j], minlength=self.m) / self.sizes
            out[:, j] = A2[:, j] - means[self.gi]
        return out[:, 0] if one_d else out


def preprocess_groups(
    X: np.ndarray, y: np.ndarray, group_indices: np.ndarray, n_groups: int
) -> tuple:
    """Preprocess groups to construct the block diagonal matrix and compute group means.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Design matrix.
    y : np.ndarray, shape (n_samples,)
        Response variable.
    group_indices : np.ndarray, shape (n_samples,)
        Group indices for each sample.
    n_groups : int
        Number of groups.

    Returns
    -------
    tuple
        Q : np.ndarray
            Block diagonal matrix for demeaning group effects.
        y_means : np.ndarray, shape (n_groups,)
            Mean of the response variable for each group.
        X_means : np.ndarray, shape (n_groups, n_features)
            Mean of predictors for each group.
    """
    group_sizes = np.bincount(group_indices)
    Q = GroupDemeaner(group_indices)
    # Compute group means for y and X:
    y_means = np.array([np.mean(y[group_indices == g]) for g in range(n_groups)])
    X_means = np.array([np.mean(X[group_indices == g], axis=0) for g in range(n_groups)])
    return Q, y_means, X_means


def perform_weighted_least_squares(
    Q: np.ndarray, X: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Perform weighted least squares to estimate coefficients.

    Parameters
    ----------
    Q : np.ndarray, shape (n_samples, n_samples)
        Block diagonal matrix for demeaning group effects.
    X : np.ndarray, shape (n_samples, n_features)
        Design matrix.
    y : np.ndarray, shape (n_samples,)
        Response variable.

    Returns
    -------
    np.ndarray, shape (n_features, 1)
        Estimated regression coefficients.
    """
    QX = Q @ X
    Qy = Q @ y.reshape(-1, 1)
    beta = np.linalg.solve(QX.T @ QX, QX.T @ Qy)
    return beta


def calculate_residuals(
    xbeta: np.ndarray, gamma: np.ndarray, y: np.ndarray,
    group_indices: np.ndarray,
) -> tuple:
    """Calculate predictions and residuals.

    Computes the predicted values by adding the group-level fixed effects to
    the linear predictor (xbeta) and computes the residuals by subtracting
    the predictions from the observed response values.

    Parameters
    ----------
    xbeta : np.ndarray, shape (n_samples,)
        The linear predictor values (X @ beta) for each sample.
    gamma : np.ndarray, shape (n_groups, 1)
        Fixed effects for each group.
    y : np.ndarray, shape (n_samples,)
        Observed response variable.
    group_indices : np.ndarray, shape (n_samples,)
        Group indices for each sample.

    Returns
    -------
    tuple
        predictions : np.ndarray, shape (n_samples,)
            The predicted response values.
        residuals : np.ndarray, shape (n_samples,)
            Residuals (observed minus predicted).
    """
    gamma_obs = gamma[group_indices].flatten()
    predictions = xbeta.flatten() + gamma_obs
    residuals = y - predictions
    return predictions, residuals


def compute_model_statistics(
    residuals: np.ndarray, n_samples: int, n_groups: int, n_features: int
) -> tuple:
    """Compute model statistics including AIC and BIC.

    Parameters
    ----------
    residuals : np.ndarray, shape (n_samples,)
        Residuals from the model.
    n_samples : int
        Total number of samples.
    n_groups : int
        Number of groups.
    n_features : int
        Number of predictors.

    Returns
    -------
    tuple
        aic : float
            Akaike Information Criterion.
        bic : float
            Bayesian Information Criterion.
    """
    residual_sum_squares = np.sum(residuals**2)
    log_likelihood = (
        -n_samples / 2 * np.log(2 * np.pi)
        - n_samples / 2 * np.log(residual_sum_squares / n_samples)
        - residual_sum_squares / (2 * residual_sum_squares / n_samples)
    )
    aic = -2 * log_likelihood + 2 * (n_groups + n_features + 1)
    bic = -2 * log_likelihood + (n_groups + n_features + 1) * np.log(n_samples)
    return aic, bic
