"""Penalized logistic regression: elastic net (glmnet-style).

Analogous to ``glmnet(family="binomial")``.  Uses the unified
proximal-Newton + coordinate-descent solver from
``algorithms/coordinate_descent.py`` with the logistic likelihood
engine from ``algorithms/logistic/likelihood.py``.
"""
from __future__ import annotations

import logging
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from ...base import ProviderModel

from ...algorithms.penalty import (
    weighted_column_scale,
    weighted_column_center_scale,
    rescale_penalty_factors,
)
from ...algorithms.coordinate_descent import (
    compute_lambda_max,
    build_lambda_sequence,
    fit_regularization_path,
)
from ...algorithms.logistic.likelihood import (
    build_logistic_objective,
    logistic_null_score,
    logistic_unpenalized_null_fit,
    logistic_loglik,
    logistic_deviance,
    logistic_null_deviance,
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
    """Build or validate the lambda sequence."""
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


def _stratified_fold_assignment(
    y: np.ndarray,
    n_folds: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Event-stratified CV fold assignment."""
    rng = np.random.RandomState(random_state)
    n = len(y)
    fold = np.empty(n, dtype=np.intp)
    for stratum_mask in [y == 1, y == 0]:
        idx = np.where(stratum_mask)[0]
        if len(idx) == 0:
            continue
        assignments = np.tile(
            np.arange(n_folds), len(idx) // n_folds + 1
        )[:len(idx)]
        fold[idx] = rng.permutation(assignments)
    return fold


# ======================================================================
# PenalizedLogistic
# ======================================================================

class PenalizedLogistic(ProviderModel):
    """Elastic-net-penalized logistic regression.

    Fits the regularization path over a grid of lambda values using
    the proximal-Newton + cyclic coordinate descent algorithm.  Matches
    ``glmnet(family="binomial")`` in lambda sequence, standardization,
    and convergence conventions.

    Parameters
    ----------
    alpha : float, default=1.0
        Elastic net mixing (1 = lasso, 0 = ridge).
    n_lambda : int, default=100
        Number of lambda values on the auto-generated grid.
    lambda_min_ratio : float or None
        Ratio of lambda_min to lambda_max.  Default: 1e-2 if n < p,
        else 1e-4.
    lambda_path : array-like or None
        User-supplied lambda sequence (overrides auto grid).
    penalty_factor : array-like or None
        Per-variable penalty weights (rescaled to sum to p).
    standardize : bool, default=True
        Standardize columns by weighted population std.
    fit_intercept : bool, default=True
        Whether to fit an unpenalized intercept.
    max_outer_iter : int, default=100
        Maximum outer (Newton/IRLS) iterations per lambda.
    outer_tol : float, default=1e-9
        Outer convergence tolerance.
    max_inner_iter : int, default=1000
        Maximum inner (CD) iterations per outer step.
    inner_tol : float, default=1e-10
        Inner convergence tolerance.
    use_active_set : bool, default=False
        Use the active-set strategy to accelerate coordinate descent.
        When True, only variables with nonzero coefficients are updated
        in most inner iterations.

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
        Deviance (-2 * log-likelihood) at each lambda.
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
    lambda_ : float
        Lambda value (only set when a single lambda is fitted).
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
        """Penalized logistic regression path (R: ``glmnet(family='binomial')``)."""
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
    ) -> "PenalizedLogistic":
        """Fit the penalized logistic regression path.

        Parameters
        ----------
        X : ndarray or DataFrame, shape (n, p)
            Design matrix.
        y : ndarray, shape (n,)
            Binary outcome (0 or 1).
        sample_weight : ndarray or None, shape (n,)
            Per-observation weights.
        offset : ndarray or None, shape (n,)
            Fixed offset.

        Returns
        -------
        self
        """
        # --- Validate inputs ---
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

        # Validate y.
        unique_y = np.unique(y)
        if not np.all(np.isin(unique_y, [0.0, 1.0])):
            raise ValueError("y must contain only 0 and 1 for logistic regression")

        # --- Column standardization ---
        # Center as well as scale when an intercept is fitted: with a free
        # intercept this is an exact reparameterization (lambda_max and the
        # coefficient path are invariant) that removes the information-
        # diagonal inflation an uncentered large-mean/sd column causes.
        # Without an intercept there is nothing to absorb the shift, so the
        # uncentered path is kept -- see weighted_column_center_scale.
        xm_full, xs_full, degenerate = weighted_column_center_scale(
            X, weight, standardize=self.standardize
        )
        if not self.fit_intercept:
            xm_full = np.zeros_like(xm_full)
        if np.any(degenerate):
            warnings.warn(
                f"{int(np.sum(degenerate))} feature(s) have ~zero weighted "
                "variance; excluding from penalized fit.",
                category=DegenerateFeatureWarning, stacklevel=2,
            )
        fit_cols = ~degenerate
        if not np.any(fit_cols):
            raise ValueError("All predictors have zero weighted variance")
        X_fit = (X[:, fit_cols] - xm_full[fit_cols]) / xs_full[fit_cols]
        p_fit = X_fit.shape[1]

        # --- Penalty factors ---
        pf_input = self.penalty_factor
        if pf_input is not None:
            pf_full = np.asarray(pf_input, dtype=np.float64)
            if pf_full.shape != (p_full,):
                raise ValueError(
                    f"penalty_factor must have shape ({p_full},), got {pf_full.shape}"
                )
        else:
            pf_full = np.ones(p_full, dtype=np.float64)
        pf_fit = rescale_penalty_factors(pf_full[fit_cols], p_fit)

        # --- Build objective ---
        c = 1.0 / float(np.sum(weight))

        # REV-001: the null point for lambda_max is not beta=0 everywhere --
        # it is "penalized coefficients at 0, unpenalized ones at their own
        # MLE".  Fitting them here both corrects lambda_max and gives the
        # path a warm start whose unpenalized coefficients are already right
        # at the top of the path (mirrors the R reference's SerBIN.residuals).
        beta_null, score_null, intercept_null = logistic_unpenalized_null_fit(
            X_fit, y, weight, unpenalized=(pf_fit == 0.0), offset=offset,
            fit_intercept=self.fit_intercept,
        )

        lam_max = compute_lambda_max(
            score_null, c, pf_fit, self.alpha,
        )
        lambda_sequence, lam_min_ratio = _resolve_lambda_path(
            self.lambda_path, lam_max, self.lambda_min_ratio,
            self.n_lambda, p_fit, n,
        )

        # --- Fit the path ---
        # Intercept is handled inside the path: at each lambda, after
        # the CD update for beta, we do one Newton step for the intercept.
        # This matches glmnet's approach of cycling the intercept with
        # the CD coordinates.
        #
        # Implementation: we include a wrapper objective that updates
        # the intercept as part of the objective evaluation.
        intercept = intercept_null if self.fit_intercept else 0.0
        intercepts = []

        if self.fit_intercept:
            # Build objective with intercept baked into offset.
            from ...algorithms.logistic.likelihood import (
                logistic_intercept_update,
                _safe_expit,
            )

            def objective_with_intercept(beta):  # noqa: D401
                """Log-likelihood, score, and information with intercept update."""
                nonlocal intercept
                eta = X_fit @ beta + intercept
                if offset is not None:
                    eta = eta + offset
                # Newton step for intercept.
                intercept += logistic_intercept_update(y, eta, weight)
                # Rebuild eta with updated intercept.
                eta = X_fit @ beta + intercept
                if offset is not None:
                    eta = eta + offset
                # Compute objective components.
                from ...algorithms.logistic.likelihood import (
                    logistic_loglik as ll_fn,
                    logistic_score as sc_fn,
                    logistic_information as info_fn,
                )
                ll = ll_fn(y, eta, weight)
                sc = sc_fn(X_fit, y, eta, weight)
                info = info_fn(X_fit, eta, weight)
                return ll, sc, info

            objective_fn = objective_with_intercept
        else:
            objective_fn = build_logistic_objective(
                X_fit, y, weight, offset=offset,
            )

        # Warm-start path with intercept tracking.
        beta = beta_null.copy()
        results = []
        for lam_val in lambda_sequence:
            result = fit_regularization_path(
                objective_fn, p_fit, c, self.alpha,
                np.array([lam_val]),
                pf_fit,
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

        # --- Store results ---
        n_lam = len(results)
        coef_path_fit = np.array(
            [r.beta / xs_full[fit_cols] for r in results]
        )
        coef_path = np.zeros((n_lam, p_full))
        coef_path[:, fit_cols] = coef_path_fit

        # Intercept back-transform: the fitted intercept is in centered
        # coordinates, so shift it back into the original units of X.
        # (No-op when fit_intercept=False, where xm_full is all zeros.)
        intercept_path = np.array(intercepts, dtype=np.float64) - (
            coef_path @ xm_full
        )

        self.coef_path_ = coef_path
        self.intercept_path_ = intercept_path
        self.lambda_path_ = np.asarray(lambda_sequence, dtype=np.float64)
        self.lambda_max_ = float(lam_max)
        self.lambda_min_ratio_ = float(lam_min_ratio)
        self.log_likelihood_path_ = np.array(
            [r.log_likelihood for r in results]
        )
        self.converged_path_ = np.array([r.converged for r in results])
        # Distance from stationarity at each path point.  When
        # ``converged_path_`` is False this says how far off the point is,
        # which the boolean alone cannot.
        self.kkt_violation_path_ = np.array(
            [getattr(r, "kkt_violation", float("nan")) for r in results]
        )
        self.n_iter_path_ = np.array([r.n_outer_iter for r in results])
        self.n_nonzero_path_ = np.array(
            [int(np.sum(row != 0.0)) for row in coef_path]
        )
        self.column_scale_ = xs_full
        self.column_center_ = xm_full
        self.penalty_factor_ = pf_full
        self._excluded_features_ = degenerate

        # Deviance and deviance ratio.
        null_dev = logistic_null_deviance(y, weight)
        deviances = np.array(
            [-2.0 * r.log_likelihood for r in results]
        )
        self.deviance_path_ = deviances
        self.null_deviance_ = null_dev
        self.deviance_ratio_path_ = np.where(
            null_dev > 0, 1.0 - deviances / null_dev, 0.0
        )

        if n_lam == 1:
            self.coef_ = coef_path[0]
            self.intercept_ = float(intercept_path[0])
            self.lambda_ = float(lambda_sequence[0])

        return self

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "coef_path_"):
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet."
            )

    def coef_at(self, lambda_value: float) -> np.ndarray:
        """Coefficients at an arbitrary lambda (linear interpolation in log-lambda).

        Parameters
        ----------
        lambda_value : float

        Returns
        -------
        ndarray, shape (p,)
        """
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
        """Intercept at an arbitrary lambda."""
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

    def predict_proba(
        self, X: np.ndarray, lambda_value: Optional[float] = None,
    ) -> np.ndarray:
        """Predicted probabilities.

        Parameters
        ----------
        X : ndarray, shape (n_new, p)
        lambda_value : float or None
            If None, uses the last lambda in the path.

        Returns
        -------
        ndarray, shape (n_new,)
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        if lambda_value is None:
            lambda_value = self.lambda_path_[-1]
        coef = self.coef_at(lambda_value)
        intercept = self.intercept_at(lambda_value) if self.fit_intercept else 0.0
        eta = X @ coef + intercept
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))

    def predict(self, X: np.ndarray, lambda_value: Optional[float] = None,
                threshold: float = 0.5) -> np.ndarray:
        """Binary predictions."""
        return (self.predict_proba(X, lambda_value) >= threshold).astype(int)

    def summary(self, which: int = -1) -> pd.DataFrame:
        """Summary DataFrame for one lambda index."""
        self._check_is_fitted()
        coef = self.coef_path_[which]
        lam = self.lambda_path_[which]
        names = self.feature_names_in_
        df = pd.DataFrame({
            "feature": names,
            "coef": coef,
            "nonzero": coef != 0,
        })
        df.attrs["lambda"] = lam
        df.attrs["deviance_ratio"] = self.deviance_ratio_path_[which]
        df.attrs["n_nonzero"] = int(np.sum(coef != 0))
        return df


# ======================================================================
# PenalizedLogisticCV
# ======================================================================

class PenalizedLogisticCV(ProviderModel):
    """Cross-validated penalized logistic regression.

    Fits the elastic-net regularization path on the full data, then
    selects lambda.min or lambda.1se via k-fold CV on binomial
    deviance.

    Parameters
    ----------
    alpha : float, default=1.0
    n_lambda : int, default=100
    lambda_min_ratio : float or None
    lambda_path : array-like or None
    penalty_factor : array-like or None
    standardize : bool, default=True
    fit_intercept : bool, default=True
    n_folds : int, default=10
    fold_id : array-like or None
        User-supplied fold assignments (overrides n_folds).
    se_rule : {"1se", "min"}, default="1se"
        If True, use lambda.1se; else lambda.min.
    random_state : int or None
    max_outer_iter, outer_tol, max_inner_iter, inner_tol

    Attributes (after fit)
    ----------------------
    lambda_min_ : float
        Lambda that minimizes CV deviance.
    lambda_1se_ : float
        Largest lambda within 1 SE of the minimum.
    lambda_ : float
        Selected lambda (lambda_1se_ if se_rule == "1se", else lambda_min_).
    cv_mean_deviance_ : ndarray, shape (n_lambda,)
        Mean cross-validated deviance at each lambda.
    cv_std_deviance_ : ndarray, shape (n_lambda,)
        Standard deviation of CV deviance across folds.
    cv_se_deviance_ : ndarray, shape (n_lambda,)
        Standard error of CV deviance.
    lambda_min_idx_ : int
        Index of lambda_min_ in lambda_path_.
    lambda_1se_idx_ : int
        Index of lambda_1se_ in lambda_path_.
    model_ : PenalizedLogistic
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
        se_rule: str = "1se",
        random_state: Optional[int] = None,
        max_outer_iter: int = 100,
        outer_tol: float = 1e-9,
        max_inner_iter: int = 1000,
        inner_tol: float = 1e-10,
        use_active_set: bool = False,
    ):
        """Cross-validated penalized logistic (R: ``cv.glmnet(family='binomial')``)."""
        self.alpha = alpha
        self.n_lambda = n_lambda
        self.lambda_min_ratio = lambda_min_ratio
        self.lambda_path = lambda_path
        self.penalty_factor = penalty_factor
        self.standardize = standardize
        self.fit_intercept = fit_intercept
        self.n_folds = n_folds
        self.fold_id = fold_id
        self.se_rule = se_rule
        self.random_state = random_state
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
    ) -> "PenalizedLogisticCV":
        """Fit CV to select lambda, then refit on full data."""
        if self.se_rule not in ("min", "1se"):
            raise ValueError(f"se_rule must be 'min' or '1se', got {self.se_rule!r}")
        # --- Fit full-data model to get lambda path ---
        full_model = PenalizedLogistic(
            alpha=self.alpha,
            n_lambda=self.n_lambda,
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

        # --- Fold assignment ---
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        n = len(y_arr)
        if self.fold_id is not None:
            fold_id = np.asarray(self.fold_id, dtype=np.intp)
        else:
            fold_id = _stratified_fold_assignment(
                y_arr, self.n_folds, random_state=self.random_state,
            )
        n_folds = int(fold_id.max()) + 1

        # --- CV deviance ---
        X_arr = np.asarray(X, dtype=np.float64) if not isinstance(X, np.ndarray) else X
        if sample_weight is None:
            weight = np.ones(n, dtype=np.float64)
        else:
            weight = np.asarray(sample_weight, dtype=np.float64)

        cv_deviance = np.full((n_folds, n_lambda), np.nan)
        for k in range(n_folds):
            train_mask = fold_id != k
            val_mask = fold_id == k

            fold_model = PenalizedLogistic(
                alpha=self.alpha,
                lambda_path=lambda_path,
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
                X_arr[train_mask], y_arr[train_mask],
                sample_weight=weight[train_mask],
                offset=offset[train_mask] if offset is not None else None,
            )

            for j in range(n_lambda):
                coef_j = fold_model.coef_path_[j]
                intercept_j = fold_model.intercept_path_[j]
                eta_val = X_arr[val_mask] @ coef_j + intercept_j
                if offset is not None:
                    eta_val += offset[val_mask]
                cv_deviance[k, j] = logistic_deviance(
                    y_arr[val_mask], eta_val, weight[val_mask],
                )

        # --- Select lambda ---
        cv_mean = np.nanmean(cv_deviance, axis=0)
        cv_std = np.nanstd(cv_deviance, axis=0, ddof=1)
        # Correct for fold-size weighting.
        cv_se = cv_std / np.sqrt(n_folds)

        idx_min = int(np.nanargmin(cv_mean))
        lambda_min = lambda_path[idx_min]

        # lambda.1se: largest lambda within 1 SE of min.
        threshold = cv_mean[idx_min] + cv_se[idx_min]
        candidates = np.where(cv_mean <= threshold)[0]
        idx_1se = int(candidates[0])  # first (largest lambda)
        lambda_1se = lambda_path[idx_1se]

        self.cv_mean_deviance_ = cv_mean
        self.cv_std_deviance_ = cv_std
        self.cv_se_deviance_ = cv_se
        self.lambda_min_ = float(lambda_min)
        self.lambda_1se_ = float(lambda_1se)
        self.lambda_ = float(lambda_1se if (self.se_rule == "1se") else lambda_min)
        self.lambda_min_idx_ = idx_min
        self.lambda_1se_idx_ = idx_1se

        # Store full model.
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

    def predict_proba(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        lambda_value: Optional[float] = None,
    ) -> np.ndarray:
        """Predicted probabilities at the selected lambda.

        Parameters
        ----------
        X : ndarray or DataFrame, shape (n_samples, n_features)
        lambda_value : float or None
            Lambda at which to predict. Default: the CV-selected lambda.

        Returns
        -------
        ndarray, shape (n_samples,)
            Predicted probabilities in [0, 1].
        """
        self._check_is_fitted()
        if lambda_value is None:
            lambda_value = self.lambda_
        return self.model_.predict_proba(X, lambda_value)

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        lambda_value: Optional[float] = None,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Binary predictions at the selected lambda.

        Parameters
        ----------
        X : ndarray or DataFrame, shape (n_samples, n_features)
        lambda_value : float or None
            Lambda at which to predict. Default: the CV-selected lambda.
        threshold : float, default=0.5
            Classification threshold.

        Returns
        -------
        ndarray of int, shape (n_samples,)
        """
        self._check_is_fitted()
        return (self.predict_proba(X, lambda_value) >= threshold).astype(int)
