"""Penalized linear regression: elastic net (glmnet-style).

Analogous to ``glmnet(family="gaussian")``.  The Gaussian case is
the simplest: the Hessian is constant (no IRLS needed), so a single
proximal-Newton step per lambda converges in one outer iteration.
"""
from __future__ import annotations

import logging
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from ...algorithms.penalty import (
    weighted_column_scale,
    rescale_penalty_factors,
)
from ...algorithms.coordinate_descent import (
    compute_lambda_max,
    build_lambda_sequence,
    fit_regularization_path,
)
from ...algorithms.linear.likelihood import (
    build_linear_objective,
    linear_null_score,
    linear_deviance,
    linear_null_deviance,
    linear_intercept_update,
)
from ...exceptions import NotFittedError, DegenerateFeatureWarning

logger = logging.getLogger(__name__)


def _resolve_lambda_path(
    lambda_path: Optional[np.ndarray],
    lambda_max: float,
    lambda_min_ratio: Optional[float],
    n_lambda: int,
    p_fit: int,
    n_obs: int,
) -> tuple:
    if lambda_path is None:
        if lambda_min_ratio is not None:
            ratio = lambda_min_ratio
        else:
            ratio = 1e-2 if n_obs < p_fit else 1e-4
        if lambda_max == 0.0:
            return np.zeros(int(n_lambda), dtype=np.float64), float(ratio)
        return build_lambda_sequence(lambda_max, ratio, n_lambda), float(ratio)
    if np.isscalar(lambda_path):
        return np.array([float(lambda_path)], dtype=np.float64), (
            float(lambda_min_ratio) if lambda_min_ratio is not None
            else (1e-2 if n_obs < p_fit else 1e-4)
        )
    values = np.asarray(lambda_path, dtype=np.float64)
    return np.sort(values)[::-1], (
        float(lambda_min_ratio) if lambda_min_ratio is not None
        else (1e-2 if n_obs < p_fit else 1e-4)
    )


class PenalizedLinear(BaseEstimator):
    """Elastic-net-penalized linear (Gaussian) regression.

    Matches ``glmnet(family="gaussian")`` in lambda sequence,
    standardization, and convergence conventions.

    Parameters
    ----------
    alpha : float, default=1.0
        Elastic net mixing: 1.0 = lasso, 0.0 = ridge.
    n_lambda : int, default=100
        Number of lambda values on the path.
    lambda_min_ratio : float or None
        Ratio of smallest to largest lambda.  Default: 1e-2 if
        n < p, 1e-4 otherwise.
    lambda_path : array-like or None
        User-supplied lambda sequence (overrides n_lambda).
    penalty_factor : array-like or None
        Per-feature penalty multipliers, shape (p,).
    standardize : bool, default=True
        Standardize features before fitting.
    fit_intercept : bool, default=True
        Fit an intercept term.
    max_outer_iter : int, default=100
        Maximum outer (Newton) iterations per lambda.
    outer_tol : float, default=1e-9
        Outer convergence tolerance.
    max_inner_iter : int, default=1000
        Maximum inner (CD) iterations per outer step.
    inner_tol : float, default=1e-10
        Inner convergence tolerance.
    use_active_set : bool, default=False
        Use the active-set strategy to accelerate coordinate descent.

    Attributes (after fit)
    ----------------------
    coef_path_ : ndarray, shape (n_lambda, p)
        Coefficient path.
    intercept_path_ : ndarray, shape (n_lambda,)
        Intercept path.
    lambda_path_ : ndarray, shape (n_lambda,)
        Lambda values used.
    lambda_max_ : float
        Computed lambda_max.
    lambda_min_ratio_ : float
        Lambda min ratio used.
    deviance_path_ : ndarray, shape (n_lambda,)
        Weighted RSS (Gaussian deviance) at each lambda.
    deviance_ratio_path_ : ndarray, shape (n_lambda,)
        Fraction of null deviance explained.
    null_deviance_ : float
        Null model deviance.
    log_likelihood_path_ : ndarray, shape (n_lambda,)
        Log-likelihood at each lambda.
    converged_path_ : ndarray of bool, shape (n_lambda,)
        Whether the outer loop converged at each lambda.
    n_iter_path_ : ndarray of int, shape (n_lambda,)
        Number of outer iterations used at each lambda.
    n_nonzero_path_ : ndarray of int, shape (n_lambda,)
        Number of nonzero coefficients at each lambda.
    n_obs_ : int
        Number of observations.
    n_features_in_ : int
        Number of features.
    feature_names_in_ : ndarray of str
        Feature names.
    column_scale_ : ndarray, shape (p,)
        Column standard deviations used for standardization.
    penalty_factor_ : ndarray, shape (p,)
        Penalty factors used.
    coef_ : ndarray, shape (p,)
        Coefficients (only set when a single lambda is fitted).
    intercept_ : float
        Intercept (only set when a single lambda is fitted).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        n_lambda: int = 100,
        lambda_min_ratio: Optional[float] = None,
        lambda_path: Optional[np.ndarray] = None,
        penalty_factor: Optional[np.ndarray] = None,
        standardize: bool = True,
        fit_intercept: bool = True,
        max_outer_iter: int = 100,
        outer_tol: float = 1e-9,
        max_inner_iter: int = 1000,
        inner_tol: float = 1e-10,
        use_active_set: bool = False,
    ):
        """Penalized linear regression path (R: ``glmnet(family='gaussian')``)."""
        self.alpha = alpha
        self.n_lambda = n_lambda
        self.lambda_min_ratio = lambda_min_ratio
        self.lambda_path = lambda_path
        self.penalty_factor = penalty_factor
        self.standardize = standardize
        self.fit_intercept = fit_intercept
        self.max_outer_iter = max_outer_iter
        self.outer_tol = outer_tol
        self.max_inner_iter = max_inner_iter
        self.inner_tol = inner_tol
        self.use_active_set = use_active_set

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> "PenalizedLinear":
        """Fit the penalized linear regression path.

        Parameters
        ----------
        X : ndarray or DataFrame, shape (n_samples, n_features)
            Feature matrix.
        y : ndarray, shape (n_samples,)
            Continuous response.
        sample_weight : ndarray or None, shape (n_samples,)
            Observation weights. Default: equal weights.
        offset : ndarray or None, shape (n_samples,)
            Offset term added to the linear predictor.

        Returns
        -------
        self
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns.tolist())
            X = X.values
        else:
            self.feature_names_in_ = np.array(
                [f"x{j}" for j in range(X.shape[1])]
            )
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, p_full = X.shape
        self.n_obs_ = n
        self.n_features_in_ = p_full

        if sample_weight is None:
            weight = np.ones(n, dtype=np.float64)
        else:
            weight = np.asarray(sample_weight, dtype=np.float64)
        if offset is not None:
            offset = np.asarray(offset, dtype=np.float64)

        # Column standardization.
        xs_full, degenerate = weighted_column_scale(
            X, weight, standardize=self.standardize,
        )
        if np.any(degenerate):
            warnings.warn(
                f"{int(np.sum(degenerate))} degenerate feature(s) excluded.",
                category=DegenerateFeatureWarning, stacklevel=2,
            )
        fit_cols = ~degenerate
        if not np.any(fit_cols):
            raise ValueError("All predictors have zero weighted variance")
        X_fit = X[:, fit_cols] / xs_full[fit_cols]
        p_fit = X_fit.shape[1]

        # Penalty factors.
        if self.penalty_factor is not None:
            pf_full = np.asarray(self.penalty_factor, dtype=np.float64)
        else:
            pf_full = np.ones(p_full, dtype=np.float64)
        pf_fit = rescale_penalty_factors(pf_full[fit_cols], p_fit)

        c = 1.0 / float(np.sum(weight))

        # Null score and lambda_max.
        score_null, intercept_null = linear_null_score(
            X_fit, y, weight, offset=offset,
            fit_intercept=self.fit_intercept,
        )
        lam_max = compute_lambda_max(score_null, c, pf_fit, self.alpha)
        lambda_sequence, lam_min_ratio = _resolve_lambda_path(
            self.lambda_path, lam_max, self.lambda_min_ratio,
            self.n_lambda, p_fit, n,
        )

        # Objective with intercept tracking.
        intercept = intercept_null if self.fit_intercept else 0.0
        intercepts = []

        def objective_fn(beta):  # noqa: D401
            """Gaussian log-likelihood, score, and information at *beta*."""
            nonlocal intercept
            eta = X_fit @ beta + intercept
            if offset is not None:
                eta = eta + offset
            if self.fit_intercept:
                intercept += linear_intercept_update(y, eta, weight)
                eta = X_fit @ beta + intercept
                if offset is not None:
                    eta = eta + offset
            from ...algorithms.linear.likelihood import (
                linear_loglik, linear_score, linear_information,
            )
            ll = linear_loglik(y, eta, weight)
            sc = linear_score(X_fit, y, eta, weight)
            info = linear_information(X_fit, weight)
            return ll, sc, info

        beta = np.zeros(p_fit)
        results = []
        for lam_val in lambda_sequence:
            result = fit_regularization_path(
                objective_fn, p_fit, c, self.alpha,
                np.array([lam_val]), pf_fit,
                beta_warm_start=beta,
                outer_max_iter=self.max_outer_iter,
                outer_tol=self.outer_tol,
                inner_max_iter=self.max_inner_iter,
                inner_tol=self.inner_tol,
                use_active_set=self.use_active_set,
            )[0]
            results.append(result)
            beta = result.beta
            intercepts.append(intercept)

        # Store results.
        n_lam = len(results)
        coef_path_fit = np.array(
            [r.beta / xs_full[fit_cols] for r in results]
        )
        coef_path = np.zeros((n_lam, p_full))
        coef_path[:, fit_cols] = coef_path_fit

        self.coef_path_ = coef_path
        self.intercept_path_ = np.array(intercepts)
        self.lambda_path_ = np.asarray(lambda_sequence, dtype=np.float64)
        self.lambda_max_ = float(lam_max)
        self.lambda_min_ratio_ = float(lam_min_ratio)
        self.log_likelihood_path_ = np.array(
            [r.log_likelihood for r in results]
        )
        self.converged_path_ = np.array([r.converged for r in results])
        self.n_iter_path_ = np.array([r.n_outer_iter for r in results])
        self.n_nonzero_path_ = np.array(
            [int(np.sum(row != 0.0)) for row in coef_path]
        )
        self.column_scale_ = xs_full
        self.penalty_factor_ = pf_full

        null_dev = linear_null_deviance(y, weight)
        deviances = np.array([linear_deviance(
            y, X[:, fit_cols] @ (r.beta / xs_full[fit_cols]) + intercepts[i]
            + (offset if offset is not None else 0.0), weight,
        ) for i, r in enumerate(results)])
        self.deviance_path_ = deviances
        self.null_deviance_ = null_dev
        self.deviance_ratio_path_ = np.where(
            null_dev > 0, 1.0 - deviances / null_dev, 0.0,
        )

        if n_lam == 1:
            self.coef_ = coef_path[0]
            self.intercept_ = intercepts[0]

        return self

    def _check_is_fitted(self):
        if not hasattr(self, "coef_path_"):
            raise NotFittedError(
                f"This {type(self).__name__} is not fitted yet."
            )

    def coef_at(self, lambda_value: float) -> np.ndarray:
        """Coefficients at an arbitrary lambda."""
        self._check_is_fitted()
        lam_path = self.lambda_path_
        if lambda_value >= lam_path[0]:
            return self.coef_path_[0].copy()
        if lambda_value <= lam_path[-1]:
            return self.coef_path_[-1].copy()
        log_lam = np.log(lam_path)
        log_val = np.log(lambda_value)
        idx = np.searchsorted(-log_lam, -log_val) - 1
        idx = max(0, min(idx, len(lam_path) - 2))
        frac = (log_val - log_lam[idx]) / (log_lam[idx + 1] - log_lam[idx])
        return (1.0 - frac) * self.coef_path_[idx] + frac * self.coef_path_[idx + 1]

    def intercept_at(self, lambda_value: float) -> float:
        """Intercept at an arbitrary lambda (log-lambda interpolation)."""
        self._check_is_fitted()
        lam_path = self.lambda_path_
        if lambda_value >= lam_path[0]:
            return float(self.intercept_path_[0])
        if lambda_value <= lam_path[-1]:
            return float(self.intercept_path_[-1])
        log_lam = np.log(lam_path)
        log_val = np.log(lambda_value)
        idx = np.searchsorted(-log_lam, -log_val) - 1
        idx = max(0, min(idx, len(lam_path) - 2))
        frac = (log_val - log_lam[idx]) / (log_lam[idx + 1] - log_lam[idx])
        return float(
            (1.0 - frac) * self.intercept_path_[idx]
            + frac * self.intercept_path_[idx + 1]
        )

    def predict(self, X, lambda_value=None):
        """Predicted values."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        if lambda_value is None:
            lambda_value = self.lambda_path_[-1]
        coef = self.coef_at(lambda_value)
        intercept = self.intercept_at(lambda_value) if self.fit_intercept else 0.0
        return X @ coef + intercept

    def summary(self, which=-1):
        """Coefficient summary at a given path index.

        Parameters
        ----------
        which : int, default -1
            Index into the lambda path.

        Returns
        -------
        DataFrame
        """
        self._check_is_fitted()
        coef = self.coef_path_[which]
        return pd.DataFrame({
            "feature": self.feature_names_in_,
            "coef": coef,
            "nonzero": coef != 0,
        })


class PenalizedLinearCV(BaseEstimator):
    """Cross-validated penalized linear regression.

    Analogous to ``cv.glmnet(family="gaussian")``.

    Parameters
    ----------
    alpha : float, default=1.0
        Elastic net mixing: 1.0 = lasso, 0.0 = ridge.
    n_lambda : int, default=100
        Number of lambda values.
    lambda_min_ratio : float or None
        Ratio of smallest to largest lambda.
    lambda_path : array-like or None
        User-supplied lambda sequence.
    penalty_factor : array-like or None
        Per-feature penalty multipliers.
    standardize : bool, default=True
    fit_intercept : bool, default=True
    n_folds : int, default=10
    fold_id : array-like or None
        User-supplied fold assignments (overrides n_folds).
    use_1se : bool, default=True
        If True, use lambda.1se; else lambda.min.
    random_state : int or None
    max_outer_iter : int, default=100
    outer_tol : float, default=1e-9
    max_inner_iter : int, default=1000
    inner_tol : float, default=1e-10
    use_active_set : bool, default=False
        Use the active-set strategy to accelerate coordinate descent.

    Attributes (after fit)
    ----------------------
    lambda_min_ : float
        Lambda that minimizes CV deviance.
    lambda_1se_ : float
        Largest lambda within 1 SE of the minimum.
    lambda_ : float
        Selected lambda (lambda_1se_ if use_1se, else lambda_min_).
    cv_mean_deviance_ : ndarray, shape (n_lambda,)
        Mean cross-validated deviance (weighted RSS) at each lambda.
    cv_std_deviance_ : ndarray, shape (n_lambda,)
        Standard deviation of CV deviance across folds.
    cv_se_deviance_ : ndarray, shape (n_lambda,)
        Standard error of CV deviance.
    model_ : PenalizedLinear
        Full-data model fitted at the selected lambda path.
    coef_ : ndarray, shape (p,)
        Coefficients at the selected lambda.
    intercept_ : float
        Intercept at the selected lambda.
    coef_path_ : ndarray, shape (n_lambda, p)
        Full coefficient path from the full-data model.
    intercept_path_ : ndarray, shape (n_lambda,)
        Full intercept path from the full-data model.
    lambda_path_ : ndarray, shape (n_lambda,)
        Lambda values used.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        n_lambda: int = 100,
        lambda_min_ratio: Optional[float] = None,
        lambda_path: Optional[np.ndarray] = None,
        penalty_factor: Optional[np.ndarray] = None,
        standardize: bool = True,
        fit_intercept: bool = True,
        n_folds: int = 10,
        fold_id: Optional[np.ndarray] = None,
        use_1se: bool = True,
        random_state: Optional[int] = None,
        max_outer_iter: int = 100,
        outer_tol: float = 1e-9,
        max_inner_iter: int = 1000,
        inner_tol: float = 1e-10,
        use_active_set: bool = False,
    ):
        """Cross-validated penalized linear (R: ``cv.glmnet(family='gaussian')``)."""
        self.alpha = alpha
        self.n_lambda = n_lambda
        self.lambda_min_ratio = lambda_min_ratio
        self.lambda_path = lambda_path
        self.penalty_factor = penalty_factor
        self.standardize = standardize
        self.fit_intercept = fit_intercept
        self.n_folds = n_folds
        self.fold_id = fold_id
        self.use_1se = use_1se
        self.random_state = random_state
        self.max_outer_iter = max_outer_iter
        self.outer_tol = outer_tol
        self.max_inner_iter = max_inner_iter
        self.inner_tol = inner_tol
        self.use_active_set = use_active_set

    def fit(self, X, y, sample_weight=None, offset=None):
        """Fit CV to select lambda."""
        full_model = PenalizedLinear(
            alpha=self.alpha, n_lambda=self.n_lambda,
            lambda_min_ratio=self.lambda_min_ratio,
            lambda_path=self.lambda_path,
            penalty_factor=self.penalty_factor,
            standardize=self.standardize,
            fit_intercept=self.fit_intercept,
            max_outer_iter=self.max_outer_iter,
            outer_tol=self.outer_tol,
            max_inner_iter=self.max_inner_iter,
            inner_tol=self.inner_tol,
            use_active_set=self.use_active_set,
        )
        full_model.fit(X, y, sample_weight=sample_weight, offset=offset)
        lambda_path = full_model.lambda_path_
        n_lambda = len(lambda_path)

        y_arr = np.asarray(y, dtype=np.float64).ravel()
        n = len(y_arr)
        X_arr = np.asarray(X, dtype=np.float64) if not isinstance(X, np.ndarray) else X
        if sample_weight is None:
            weight = np.ones(n, dtype=np.float64)
        else:
            weight = np.asarray(sample_weight, dtype=np.float64)

        if self.fold_id is not None:
            fold_id = np.asarray(self.fold_id, dtype=np.intp)
        else:
            rng = np.random.RandomState(self.random_state)
            fold_id = rng.randint(0, self.n_folds, size=n)
        n_folds = int(fold_id.max()) + 1

        cv_deviance = np.full((n_folds, n_lambda), np.nan)
        for k in range(n_folds):
            train = fold_id != k
            val = fold_id == k
            fold_model = PenalizedLinear(
                alpha=self.alpha, lambda_path=lambda_path,
                penalty_factor=self.penalty_factor,
                standardize=self.standardize,
                fit_intercept=self.fit_intercept,
                max_outer_iter=self.max_outer_iter,
                outer_tol=self.outer_tol,
                max_inner_iter=self.max_inner_iter,
                inner_tol=self.inner_tol,
                use_active_set=self.use_active_set,
            )
            fold_model.fit(
                X_arr[train], y_arr[train],
                sample_weight=weight[train],
                offset=offset[train] if offset is not None else None,
            )
            for j in range(n_lambda):
                coef_j = fold_model.coef_path_[j]
                intercept_j = fold_model.intercept_path_[j]
                pred = X_arr[val] @ coef_j + intercept_j
                if offset is not None:
                    pred = pred + offset[val]
                cv_deviance[k, j] = float(
                    np.sum(weight[val] * (y_arr[val] - pred)**2)
                )

        cv_mean = np.nanmean(cv_deviance, axis=0)
        cv_std = np.nanstd(cv_deviance, axis=0, ddof=1)
        cv_se = cv_std / np.sqrt(n_folds)
        idx_min = int(np.nanargmin(cv_mean))
        threshold = cv_mean[idx_min] + cv_se[idx_min]
        candidates = np.where(cv_mean <= threshold)[0]
        idx_1se = int(candidates[0])

        self.cv_mean_deviance_ = cv_mean
        self.cv_std_deviance_ = cv_std
        self.cv_se_deviance_ = cv_se
        self.lambda_min_ = float(lambda_path[idx_min])
        self.lambda_1se_ = float(lambda_path[idx_1se])
        self.lambda_ = float(
            lambda_path[idx_1se] if self.use_1se else lambda_path[idx_min]
        )
        self.model_ = full_model
        self.coef_ = full_model.coef_at(self.lambda_)
        self.intercept_ = full_model.intercept_at(self.lambda_)
        self.coef_path_ = full_model.coef_path_
        self.intercept_path_ = full_model.intercept_path_
        self.lambda_path_ = full_model.lambda_path_
        return self

    def _check_is_fitted(self):
        if not hasattr(self, "model_"):
            raise NotFittedError(
                f"This {type(self).__name__} is not fitted yet. "
                f"Call 'fit' before using this estimator."
            )

    def predict(self, X, lambda_value=None):
        """Predict at the selected lambda.

        Parameters
        ----------
        X : DataFrame or ndarray, shape (n_new, n_features)
        lambda_value : float or None

        Returns
        -------
        ndarray, shape (n_new,)
        """
        self._check_is_fitted()
        if lambda_value is None:
            lambda_value = self.lambda_
        return self.model_.predict(X, lambda_value)
