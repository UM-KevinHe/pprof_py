"""Group lasso penalized linear regression.
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
)
from ...algorithms.coordinate_descent import (
    compute_group_lambda_max,
    build_lambda_sequence,
    fit_group_regularization_path,
)
from ...algorithms.linear.likelihood import (
    build_linear_objective,
    linear_null_score,
    linear_unpenalized_null_fit,
    linear_deviance,
    linear_null_deviance,
    linear_intercept_update,
)
from ...exceptions import NotFittedError
from .penalized import DegenerateFeatureWarning, _resolve_lambda_path

logger = logging.getLogger(__name__)


class GroupLassoLinear(ProviderModel):
    """Group lasso / sparse group lasso penalized linear regression.

    Parameters
    ----------
    groups : array-like, shape (p,)
    alpha : float, default=0.0
    n_lambda : int, default=100
    lambda_min_ratio : float or None
    lambda_path : array-like or None
    penalty_factor : array-like or None
    group_multiplier : array-like or None
    standardize : bool, default=True
    fit_intercept : bool, default=True
    use_active_set : bool, default=True
        Use the active-set strategy to accelerate coordinate descent.
    max_outer_iter : int, default=100
        Maximum outer (proximal Newton) iterations per lambda.
    outer_tol : float, default=1e-9
        Outer-loop convergence tolerance.  Note that the group block
        solver frequently fails to reach stationarity at this tolerance
        on correlated within-group designs; ``converged_path_`` reports
        that honestly rather than declaring success.  Inspect
        ``kkt_violation_path_`` for the actual distance from
        stationarity at each path point.
        Outer convergence tolerance.
    max_inner_iter : int, default=1000
        Maximum inner (CD) iterations per outer step.
    inner_tol : float, default=1e-10
        Inner convergence tolerance.
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
        """Group lasso linear regression path."""
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

    def fit(self, X, y, sample_weight=None, offset=None):
        """Fit the group lasso linear path."""
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

        groups_full, group_sizes, n_groups = validate_groups(
            np.asarray(self.groups), p_full,
        )
        group_weights = rescale_group_multipliers(
            self.group_multiplier, group_sizes, n_groups,
        )

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
                f"{int(np.sum(degenerate))} degenerate feature(s) excluded.",
                category=DegenerateFeatureWarning, stacklevel=2,
            )
        fit_cols = ~degenerate
        X_fit = (X[:, fit_cols] - xm_full[fit_cols]) / xs_full[fit_cols]


        p_fit = X_fit.shape[1]
        groups_fit = groups_full[fit_cols]
        groups_fit, group_sizes_fit, n_groups_fit = validate_groups(
            groups_fit, p_fit,
        )
        gw_fit = rescale_group_multipliers(
            None, group_sizes_fit, n_groups_fit,
        )

        if self.penalty_factor is not None:
            pf_full = np.asarray(self.penalty_factor, dtype=np.float64)
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
        beta_null, score_null, intercept_null = linear_unpenalized_null_fit(
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

        # Intercept back-transform into the original units of X.  This must
        # happen before the deviance block below, which rebuilds eta from the
        # *uncentered* X and so needs an intercept in matching units.
        intercept_path = np.array(intercepts, dtype=np.float64) - (
            coef_path @ xm_full
        )
        intercepts = list(intercept_path)

        self.coef_path_ = coef_path
        self.intercept_path_ = intercept_path
        self.lambda_path_ = np.asarray(lambda_sequence, dtype=np.float64)
        self.lambda_max_ = float(lam_max)
        self.converged_path_ = np.array([r.converged for r in results])
        # REV-005 (also noted): parity with GroupLassoLogistic and
        # PenalizedLinear, which the ISSUE-003 fix did not reach here.
        self.n_iter_path_ = np.array([r.n_outer_iter for r in results])
        # Distance from stationarity at each path point.  When
        # ``converged_path_`` is False this says how far off the point is,
        # which the boolean alone cannot.
        self.kkt_violation_path_ = np.array(
            [getattr(r, "kkt_violation", float("nan")) for r in results]
        )
        # Snap solver noise to exact zero before counting nonzeros
        # (ISSUE-013: FP noise ~1e-10 at lambda_max from IRLS/intercept).
        _snap = max(getattr(self, 'inner_tol', 1e-10) * 100, 1e-8)
        self.n_nonzero_path_ = np.array(
            [int(np.sum(np.abs(row) > _snap)) for row in coef_path]
        )
        self.active_groups_path_ = np.array(
            [r.active_groups for r in results]
        )
        self.column_scale_ = xs_full
        self.column_center_ = xm_full

        null_dev = linear_null_deviance(y, weight)
        # Use the back-transformed coefficients: r.beta lives in the
        # orthogonalized space when C2 is active, so it cannot be paired
        # with the original X here.
        deviances = np.array([linear_deviance(
            y, X[:, fit_cols] @ coef_path_fit[i] + intercepts[i]
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
            raise NotFittedError(f"{type(self).__name__} is not fitted.")

    def coef_at(self, lambda_value):
        """Interpolated coefficient vector at *lambda_value*.

        Parameters
        ----------
        lambda_value : float

        Returns
        -------
        ndarray, shape (n_features,)
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
        """Predict responses at a given lambda.

        Parameters
        ----------
        X : DataFrame or ndarray, shape (n_new, n_features)
        lambda_value : float or None

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
        return X @ coef + intercept

    def active_group_labels(self, which=-1):
        """1-indexed labels of active (nonzero) groups at path step *which*.

        Parameters
        ----------
        which : int, default -1

        Returns
        -------
        ndarray
        """
        self._check_is_fitted()
        return np.where(self.active_groups_path_[which])[0] + 1
