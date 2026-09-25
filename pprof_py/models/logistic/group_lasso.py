"""Group lasso / sparse group lasso penalized logistic regression.

Matches R ``grplasso::grp.lasso()`` for binomial family.  Uses the
unified proximal-Newton + block-CD solver from
``algorithms/coordinate_descent.py``.
"""
from __future__ import annotations

import logging
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from ...base import ProviderModel

from ...algorithms.penalty import (
    within_group_orthogonalize,
    unorthogonalize_coefs,
    weighted_column_scale,
    weighted_column_center_scale,
    rescale_penalty_factors,
    validate_groups,
    rescale_group_multipliers,
    compute_group_indices,
)
from ...algorithms.coordinate_descent import (
    compute_group_lambda_max,
    build_lambda_sequence,
    fit_group_regularization_path,
)
from ...algorithms.logistic.likelihood import (
    build_logistic_objective,
    logistic_null_score,
    logistic_unpenalized_null_fit,
    logistic_loglik,
    logistic_deviance,
    logistic_null_deviance,
    logistic_intercept_update,
)
from ...exceptions import NotFittedError
from .penalized import DegenerateFeatureWarning, _resolve_lambda_path, _stratified_fold_assignment

logger = logging.getLogger(__name__)


class GroupLassoLogistic(ProviderModel):
    """Group lasso / sparse group lasso penalized logistic regression.

    Fits the regularization path over a grid of lambda values using
    the proximal-Newton + block-CD algorithm.  Matches R ``grp.lasso``
    for binomial family.

    Parameters
    ----------
    groups : array-like, shape (p,)
        Integer group labels (0 = unpenalized, positive = penalized).
    alpha : float, default=0.0
        Sparse group lasso mixing (0 = pure group, 1 = pure lasso).
    n_lambda : int, default=100
    lambda_min_ratio : float or None
    lambda_path : array-like or None
    penalty_factor : array-like or None
        Per-variable penalty factors for the element-wise L1 part.
    group_multiplier : array-like or None
        Per-group penalty multipliers.  Default: sqrt(group_size).
    standardize : bool, default=True
    fit_intercept : bool, default=True
    use_active_set : bool, default=True
    max_outer_iter : int, default=100
    outer_tol : float, default=1e-9
        Outer-loop convergence tolerance.  Note that the group block
        solver frequently fails to reach stationarity at this tolerance
        on correlated within-group designs; ``converged_path_`` reports
        that honestly rather than declaring success.  Inspect
        ``kkt_violation_path_`` for the actual distance from
        stationarity at each path point.
    max_inner_iter : int, default=1000
    inner_tol : float, default=1e-10

    Attributes (after fit)
    ----------------------
    coef_path_ : ndarray, shape (n_lambda, p)
    intercept_path_ : ndarray, shape (n_lambda,)
    lambda_path_ : ndarray, shape (n_lambda,)
    active_groups_path_ : ndarray of bool, shape (n_lambda, n_groups)
    deviance_ratio_path_ : ndarray, shape (n_lambda,)
    """

    def __init__(
        self,
        groups: np.ndarray,
        alpha: float = 0.0,
        n_lambda: int = 100,
        lambda_min_ratio: Optional[float] = None,
        lambda_path: Optional[np.ndarray] = None,
        penalty_factor: Optional[np.ndarray] = None,
        group_multiplier: Optional[np.ndarray] = None,
        standardize: bool = True,
        orthogonalize: bool = True,
        fit_intercept: bool = True,
        use_active_set: bool = True,
        max_outer_iter: int = 100,
        outer_tol: float = 1e-9,
        max_inner_iter: int = 1000,
        inner_tol: float = 1e-10,
    ):
        """Group lasso logistic regression path (R: ``grplasso``)."""
        self.groups = groups
        self.alpha = alpha
        self.n_lambda = n_lambda
        self.lambda_min_ratio = lambda_min_ratio
        self.lambda_path = lambda_path
        self.penalty_factor = penalty_factor
        self.group_multiplier = group_multiplier
        self.standardize = standardize
        self.orthogonalize = orthogonalize
        self.fit_intercept = fit_intercept
        self.use_active_set = use_active_set
        self.max_outer_iter = max_outer_iter
        self.outer_tol = outer_tol
        self.max_inner_iter = max_inner_iter
        self.inner_tol = inner_tol

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> "GroupLassoLogistic":
        """Fit the group lasso logistic path.

        Parameters
        ----------
        X : ndarray or DataFrame, shape (n, p)
        y : ndarray, shape (n,)
        sample_weight, offset : ndarray or None

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

        # Validate groups.
        groups_full, group_sizes, n_groups = validate_groups(
            np.asarray(self.groups), p_full,
        )
        group_weights = rescale_group_multipliers(
            self.group_multiplier, group_sizes, n_groups,
        )
        self.groups_ = groups_full
        self.group_sizes_ = group_sizes
        self.n_groups_ = n_groups
        self.group_weights_ = group_weights

        # Column standardization.
        # Center as well as scale when an intercept is fitted: with a free
        # intercept this is an exact reparameterization (lambda_max and the
        # coefficient path are invariant) that removes the information-
        # diagonal inflation an uncentered large-mean/sd column causes.
        xm_full, xs_full, degenerate = weighted_column_center_scale(
            X, weight, standardize=self.standardize,
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
        X_fit = (X[:, fit_cols] - xm_full[fit_cols]) / xs_full[fit_cols]


        p_fit = X_fit.shape[1]
        groups_fit = groups_full[fit_cols]

        # Revalidate groups after column exclusion.
        groups_fit, group_sizes_fit, n_groups_fit = validate_groups(
            groups_fit, p_fit,
        )
        gw_fit = rescale_group_multipliers(
            None, group_sizes_fit, n_groups_fit,
        )  # recompute default sqrt(size) for the reduced set
        if self.group_multiplier is not None:
            # User-supplied multipliers: keep them but they must still
            # map correctly.  For now, use sqrt(size) for safety.
            gw_fit = rescale_group_multipliers(
                None, group_sizes_fit, n_groups_fit,
            )

        # Penalty factors.
        pf_input = self.penalty_factor
        if pf_input is not None:
            pf_full = np.asarray(pf_input, dtype=np.float64)
        else:
            pf_full = np.ones(p_full, dtype=np.float64)
        pf_fit = rescale_penalty_factors(pf_full[fit_cols], p_fit)

        c = 1.0 / float(np.sum(weight))

        # C2: within-group orthogonalization (R/grplasso convention).
        # Each penalized group is mapped so that X_g' diag(w) X_g / sum(w) = I,
        # which is what makes the block coordinate update exact -- see the
        # scale-correction note in the group CD kernel.  The unpenalized
        # pseudo-group (label 0) is left untouched, as in R.
        #
        # NOTE: this changes the ESTIMATOR, not just the algorithm.  In the
        # original coordinates the penalty becomes
        #     lam * sqrt(K_g) * sqrt( beta_g' (X_g' W X_g / sum w) beta_g )
        # i.e. the standardized group lasso (Simon & Tibshirani 2012), which
        # is what R/grplasso fits.  Set orthogonalize=False only to reproduce
        # pre-change results; that path is NOT a validated alternative.
        QL_blocks = None
        if self.orthogonalize:
            X_fit, QL_blocks = within_group_orthogonalize(
                X_fit, groups_fit, weight,
            )

        # REV-001: the null point for lambda_max is not beta=0 everywhere --
        # it is "penalized coefficients at 0, unpenalized ones at their own
        # MLE".  For the group lasso the unpenalized set is the group==0
        # pseudo-group.  Fitting it here corrects lambda_max and gives the
        # path a warm start whose unpenalized coefficients are already right
        # at the top of the path (mirrors the R reference's SerBIN.residuals).
        beta_null, score_null, intercept_null = logistic_unpenalized_null_fit(
            X_fit, y, weight,
            unpenalized=((groups_fit == 0) | (pf_fit == 0.0)),
            offset=offset, fit_intercept=self.fit_intercept,
        )

        lam_max = compute_group_lambda_max(
            score_null, c, groups_fit, gw_fit, pf_fit, self.alpha,
        )
        lambda_sequence, lam_min_ratio = _resolve_lambda_path(
            self.lambda_path, lam_max, self.lambda_min_ratio,
            self.n_lambda, p_fit, n,
        )

        # C3 is only well-posed once C2 has made A_gg proportional to I.
        majorize = bool(self.orthogonalize)

        # Objective with intercept tracking.
        intercept = intercept_null if self.fit_intercept else 0.0
        intercepts = []

        def objective_fn(beta):  # noqa: D401
            """Log-likelihood, score, and information at *beta*."""
            nonlocal intercept
            eta = X_fit @ beta + intercept
            if offset is not None:
                eta = eta + offset
            if self.fit_intercept:
                intercept += logistic_intercept_update(y, eta, weight)
                eta = X_fit @ beta + intercept
                if offset is not None:
                    eta = eta + offset
            from ...algorithms.logistic.likelihood import (
                logistic_loglik as ll_fn,
                logistic_score as sc_fn,
                logistic_information as info_fn,
            )
            ll = ll_fn(y, eta, weight)
            sc = sc_fn(X_fit, y, eta, weight)
            if majorize:
                # C3: fixed v = 1/4 majorizer (R/grplasso convention).
                # p(1-p) <= 1/4, so this is a quadratic upper bound on the
                # negative log-likelihood: block CD on it descends
                # monotonically and needs no step-halving.  Combined with
                # C2 it makes A_gg = v*I exactly and constant across IRLS
                # iterations, which is what the kernel's scale correction
                # needs in order to be exact.
                info = 0.25 * (X_fit.T * weight) @ X_fit
            else:
                info = info_fn(X_fit, eta, weight)
            return ll, sc, info

        # Fit path lambda by lambda for intercept tracking.
        beta = beta_null.copy()
        results = []
        for lam_val in lambda_sequence:
            result = fit_group_regularization_path(
                objective_fn, p_fit, c, self.alpha,
                np.array([lam_val]),
                groups_fit, gw_fit, pf_fit, n_groups_fit,
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
            [
                (unorthogonalize_coefs(r.beta, groups_fit, QL_blocks)
                 if QL_blocks is not None else r.beta) / xs_full[fit_cols]
                for r in results
            ]
        )
        coef_path = np.zeros((n_lam, p_full))
        coef_path[:, fit_cols] = coef_path_fit

        # Intercept back-transform into the original units of X.
        intercept_path = np.array(intercepts, dtype=np.float64) - (
            coef_path @ xm_full
        )
        intercepts = list(intercept_path)

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
        # Snap solver noise to exact zero before counting nonzeros
        # (ISSUE-013: FP noise ~1e-10 at lambda_max from IRLS/intercept).
        _snap = max(getattr(self, 'inner_tol', 1e-10) * 100, 1e-8)
        self.n_nonzero_path_ = np.array(
            [int(np.sum(np.abs(row) > _snap)) for row in coef_path]
        )
        self.active_groups_path_ = np.array(
            [r.active_groups for r in results]
        )
        self.group_norms_path_ = np.array(
            [r.group_norms for r in results]
        )
        self.column_scale_ = xs_full
        self.column_center_ = xm_full
        self.penalty_factor_ = pf_full

        null_dev = logistic_null_deviance(y, weight)
        deviances = np.array([-2.0 * r.log_likelihood for r in results])
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
                f"This {type(self).__name__} instance is not fitted yet."
            )

    def active_group_labels(self, which: int = -1) -> np.ndarray:
        """Which groups are nonzero at a given lambda index.

        Parameters
        ----------
        which : int
            Lambda index.

        Returns
        -------
        ndarray of int
            Canonical group labels that are active.
        """
        self._check_is_fitted()
        active = self.active_groups_path_[which]
        return np.where(active)[0] + 1  # 1-indexed labels

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

    def predict_proba(self, X, lambda_value=None):
        """Predicted probabilities."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        if lambda_value is None:
            lam = self.lambda_path_[-1]
        else:
            lam = lambda_value
        coef = self.coef_at(lam)
        intercept = self.intercept_path_[-1] if not hasattr(self, 'intercept_') else 0.0
        # Interpolate intercept similarly.
        idx = max(0, min(
            int(np.searchsorted(-np.log(self.lambda_path_), -np.log(lam))) - 1,
            len(self.lambda_path_) - 2,
        ))
        if lambda_value is not None and len(self.lambda_path_) > 1:
            log_lam = np.log(self.lambda_path_)
            log_val = np.log(max(lam, self.lambda_path_[-1]))
            frac = (log_val - log_lam[idx]) / (log_lam[idx + 1] - log_lam[idx]) if log_lam[idx + 1] != log_lam[idx] else 0.0
            intercept = (1.0 - frac) * self.intercept_path_[idx] + frac * self.intercept_path_[idx + 1]
        else:
            intercept = self.intercept_path_[-1]
        eta = X @ coef + intercept
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))

    def predict(self, X, lambda_value=None, threshold=0.5):
        """Binary predictions."""
        return (self.predict_proba(X, lambda_value) >= threshold).astype(int)


class GroupLassoLogisticCV(ProviderModel):
    """Cross-validated group lasso logistic regression.

    Fits the group lasso path, selects lambda via CV binomial deviance.

    Parameters
    ----------
    groups : array-like, shape (p,)
    alpha : float, default=0.0
    n_lambda, lambda_min_ratio, lambda_path, penalty_factor,
    group_multiplier, standardize, fit_intercept, use_active_set
    n_folds : int, default=10
    fold_id : array-like or None
    use_1se : bool, default=True
    random_state : int or None
    max_outer_iter, outer_tol, max_inner_iter, inner_tol
    """

    def __init__(
        self,
        groups: np.ndarray,
        alpha: float = 0.0,
        n_lambda: int = 100,
        lambda_min_ratio: Optional[float] = None,
        lambda_path: Optional[np.ndarray] = None,
        penalty_factor: Optional[np.ndarray] = None,
        group_multiplier: Optional[np.ndarray] = None,
        standardize: bool = True,
        orthogonalize: bool = True,
        fit_intercept: bool = True,
        use_active_set: bool = True,
        n_folds: int = 10,
        fold_id: Optional[np.ndarray] = None,
        use_1se: bool = True,
        random_state: Optional[int] = None,
        max_outer_iter: int = 100,
        outer_tol: float = 1e-9,
        max_inner_iter: int = 1000,
        inner_tol: float = 1e-10,
    ):
        """Cross-validated group lasso logistic."""
        self.groups = groups
        self.alpha = alpha
        self.n_lambda = n_lambda
        self.lambda_min_ratio = lambda_min_ratio
        self.lambda_path = lambda_path
        self.penalty_factor = penalty_factor
        self.group_multiplier = group_multiplier
        self.standardize = standardize
        self.orthogonalize = orthogonalize
        self.fit_intercept = fit_intercept
        self.use_active_set = use_active_set
        self.n_folds = n_folds
        self.fold_id = fold_id
        self.use_1se = use_1se
        self.random_state = random_state
        self.max_outer_iter = max_outer_iter
        self.outer_tol = outer_tol
        self.max_inner_iter = max_inner_iter
        self.inner_tol = inner_tol

    def fit(self, X, y, sample_weight=None, offset=None):
        """Fit CV to select lambda, then refit on full data."""
        full_model = GroupLassoLogistic(
            groups=self.groups, alpha=self.alpha,
            n_lambda=self.n_lambda,
            lambda_min_ratio=self.lambda_min_ratio,
            lambda_path=self.lambda_path,
            penalty_factor=self.penalty_factor,
            group_multiplier=self.group_multiplier,
            standardize=self.standardize,
            orthogonalize=self.orthogonalize,
            fit_intercept=self.fit_intercept,
            use_active_set=self.use_active_set,
            max_outer_iter=self.max_outer_iter,
            outer_tol=self.outer_tol,
            max_inner_iter=self.max_inner_iter,
            inner_tol=self.inner_tol,
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
            fold_id = _stratified_fold_assignment(
                y_arr, self.n_folds, random_state=self.random_state,
            )
        n_folds = int(fold_id.max()) + 1

        cv_deviance = np.full((n_folds, n_lambda), np.nan)
        for k in range(n_folds):
            train = fold_id != k
            val = fold_id == k
            fold_model = GroupLassoLogistic(
                groups=self.groups, alpha=self.alpha,
                lambda_path=lambda_path,
                penalty_factor=self.penalty_factor,
                group_multiplier=self.group_multiplier,
                standardize=self.standardize,
            orthogonalize=self.orthogonalize,
                fit_intercept=self.fit_intercept,
                use_active_set=self.use_active_set,
                max_outer_iter=self.max_outer_iter,
                outer_tol=self.outer_tol,
                max_inner_iter=self.max_inner_iter,
                inner_tol=self.inner_tol,
            )
            fold_model.fit(
                X_arr[train], y_arr[train],
                sample_weight=weight[train],
            )
            for j in range(n_lambda):
                coef_j = fold_model.coef_path_[j]
                intercept_j = fold_model.intercept_path_[j]
                eta_val = X_arr[val] @ coef_j + intercept_j
                cv_deviance[k, j] = logistic_deviance(
                    y_arr[val], eta_val, weight[val],
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
        self.coef_path_ = full_model.coef_path_
        self.lambda_path_ = full_model.lambda_path_
        return self

    def predict_proba(self, X, lambda_value=None):
        """Predicted probabilities at the selected lambda.

        Parameters
        ----------
        X : DataFrame or ndarray, shape (n_new, n_features)
        lambda_value : float or None

        Returns
        -------
        ndarray, shape (n_new,)
        """
        if lambda_value is None:
            lambda_value = self.lambda_
        return self.model_.predict_proba(X, lambda_value)

    def predict(self, X, lambda_value=None, threshold=0.5):
        """Binary class predictions at the selected lambda.

        Parameters
        ----------
        X : DataFrame or ndarray, shape (n_new, n_features)
        lambda_value : float or None
        threshold : float, default 0.5

        Returns
        -------
        ndarray, shape (n_new,)
        """
        return (self.predict_proba(X, lambda_value) >= threshold).astype(int)
