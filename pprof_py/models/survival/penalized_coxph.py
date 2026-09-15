"""User-facing penalized Cox estimators: `PenalizedCoxPH` (a
regularization path, analogous to R's `glmnet(family="cox")`) and
`PenalizedCoxPHCV` (k-fold cross-validated lambda selection, analogous
to `cv.glmnet(family="cox")`).

Both are built the same way `CoxPHSelector` (Phase 2) is built on top
of `CoxPH`: they reuse `data/validation.py` and `data/survival_data.py`
untouched, and reuse `algorithms/partial_likelihood.py` (via a closure
identical in shape to `CoxPH.fit`'s `objective(beta)`) for every
likelihood/score/information evaluation. The new numerical work
(standardization, the penalty, coordinate descent, the lambda path,
cross-validation deviance) lives in `algorithms/penalty.py`,
`algorithms/coordinate_descent.py`, and `statistics/deviance.py`; this
module's job is orchestration, validation, and presenting results in
the same estimator style as `CoxPH` -- see docs/ARCHITECTURE.md and
docs/R_COMPATIBILITY.md (Phase 3 sections) for the conventions matched
against glmnet 4.1-8.

Naming note: `alpha` here is glmnet's elastic-net mixing parameter
(0 = ridge, 1 = lasso), NOT scikit-learn's `ElasticNet.alpha` (overall
strength) -- this package validates against R's `survival`/`glmnet`,
so its penalized-regression vocabulary follows glmnet, not sklearn's
`linear_model` module. `lambda` is a Python keyword, so the
regularization-strength argument is `lambda_path` (a full sequence, a
single scalar, or `None` for glmnet's auto-generated grid); the fitted
grid actually used is `lambda_path_`.
"""
from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from ...data.survival_validation import validate_fit_inputs, validate_X
from ...data.survival_data import SurvivalData
from ...algorithms.survival.cox_likelihood import cox_partial_likelihood, precompute_stratum_indices
from ...algorithms.survival.penalty import weighted_column_scale, rescale_penalty_factors
from ...algorithms.survival.coordinate_descent import (
    fit_regularization_path,
    compute_lambda_max,
    build_lambda_sequence,
)
from ...algorithms.survival.ties import TieMethod
from ...inference.survival.deviance import saturated_log_likelihood, cox_deviance, deviance_ratio
from ...utils.numerical import safe_exp
from .coxph import CoxPH, NotFittedError


class DegenerateFeatureWarning(UserWarning):
    """Warning emitted when a predictor has numerically zero weighted variance."""


def _validate_common_parameters(
    alpha, n_lambda, lambda_min_ratio, lambda_path, standardize,
    max_outer_iter, outer_tol, max_inner_iter, inner_tol, fit_intercept,
):
    if not np.isfinite(alpha) or not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if isinstance(n_lambda, (bool, np.bool_)) or int(n_lambda) != n_lambda or int(n_lambda) < 1:
        raise ValueError(f"n_lambda must be a positive integer, got {n_lambda!r}")
    if lambda_min_ratio is not None:
        ratio = float(lambda_min_ratio)
        if not np.isfinite(ratio) or not (0.0 < ratio <= 1.0):
            raise ValueError(
                f"lambda_min_ratio must be finite and in (0, 1], got {lambda_min_ratio!r}"
            )
    if lambda_path is not None:
        values = np.atleast_1d(np.asarray(lambda_path, dtype=np.float64))
        if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("lambda_path must contain finite, strictly positive values")
        if np.unique(values).size != values.size:
            raise ValueError("lambda_path must not contain duplicate values")
    if not isinstance(standardize, (bool, np.bool_)):
        raise ValueError("standardize must be a boolean")
    if isinstance(max_outer_iter, (bool, np.bool_)) or int(max_outer_iter) != max_outer_iter or int(max_outer_iter) < 1:
        raise ValueError(f"max_outer_iter must be a positive integer, got {max_outer_iter!r}")
    if not np.isfinite(outer_tol) or float(outer_tol) <= 0:
        raise ValueError(f"outer_tol must be finite and > 0, got {outer_tol!r}")
    if isinstance(max_inner_iter, (bool, np.bool_)) or int(max_inner_iter) != max_inner_iter or int(max_inner_iter) < 1:
        raise ValueError(f"max_inner_iter must be a positive integer, got {max_inner_iter!r}")
    if not np.isfinite(inner_tol) or float(inner_tol) <= 0:
        raise ValueError(f"inner_tol must be finite and > 0, got {inner_tol!r}")
    if fit_intercept:
        raise ValueError("Cox proportional hazards regression does not support an intercept")


def _resolve_lambda_path(lambda_path, lambda_max, lambda_min_ratio, n_lambda, p_fit, n_obs):
    if lambda_path is None:
        if lambda_min_ratio is not None:
            ratio = lambda_min_ratio
        else:
            ratio = 1e-2 if n_obs < p_fit else 1e-4
        if lambda_max < 0 or not np.isfinite(lambda_max):
            raise ValueError(f"lambda_max must be finite and non-negative, got {lambda_max!r}")
        if lambda_max == 0.0:
            # The null score is exactly zero, so beta=0 is already the optimum
            # for every positive lambda. Keep a zero path as a representational
            # edge case rather than constructing log(0). No positive lambda can
            # be inferred from lambda_max in this situation.
            return np.zeros(int(n_lambda), dtype=np.float64), float(ratio)
        return build_lambda_sequence(lambda_max, ratio, n_lambda), float(ratio)

    if np.isscalar(lambda_path):
        return np.array([float(lambda_path)], dtype=np.float64), (
            float(lambda_min_ratio) if lambda_min_ratio is not None else (1e-2 if n_obs < p_fit else 1e-4)
        )
    values = np.asarray(lambda_path, dtype=np.float64)
    return np.sort(values)[::-1], (
        float(lambda_min_ratio) if lambda_min_ratio is not None else (1e-2 if n_obs < p_fit else 1e-4)
    )


class PenalizedCoxPH(BaseEstimator):
    """Elastic-net-penalized Cox Proportional Hazards regression, fit
    by proximal Newton + coordinate descent over a lambda path.

    Reuses the exact same (Breslow/Efron) partial-likelihood engine as
    `CoxPH`, so every non-penalization capability -- strata, offset,
    sample weights, start/stop (left-truncated) data -- carries over
    unchanged. See docs/R_COMPATIBILITY.md, Phase 3 section, for the
    numerical comparison against the package's pinned glmnet reference
    version. Current glmnet releases support additional Cox options;
    compatibility claims here refer to the explicitly pinned reference
    used by the regression suite. Efron-tie penalized fits use the same
    engine and are validated by self-consistency against this
    package's own (R-validated) unpenalized `CoxPH(ties="efron")` in
    the lambda -> 0 limit, since no independent Efron+penalty
    reference implementation is available.

    Parameters
    ----------
    alpha : float, default 1.0
        Elastic-net mixing parameter in [0, 1], glmnet's convention:
        0 = ridge, 1 = lasso (the default), values in between = elastic
        net. NOT scikit-learn's `ElasticNet.alpha` -- see module
        docstring.
    n_lambda : int, default 100
        Number of lambda values in the auto-generated path (glmnet's
        `nlambda`). Ignored if `lambda_path` is given.
    lambda_min_ratio : float, optional
        Smallest lambda in the auto-generated path, as a fraction of
        `lambda_max_`. Defaults to glmnet's own rule: 1e-2 if
        n_obs < n_features, else 1e-4. Ignored if `lambda_path` is given.
    lambda_path : float, sequence of float, or None, default None
        None auto-generates the path (see `n_lambda`,
        `lambda_min_ratio`). A single float fits exactly that one
        lambda. A sequence fits exactly those lambda values (sorted
        descending internally for warm-starting; `lambda_path_`
        reflects the order actually fit).
    penalty_factor : array-like of shape (n_features,), optional
        Per-feature multiplier on the penalty (glmnet convention): 0
        marks an always-unpenalized feature (always in the model,
        never shrunk); values are rescaled internally to sum to
        n_features, matching glmnet. Defaults to all-ones (every
        feature penalized equally).
    standardize : bool, default True
        Divide each column of X by its weighted population standard
        deviation before fitting so the penalty is comparable across
        features measured in different units, then divide the fitted
        coefficients back by the same scale (glmnet's default
        behavior; does not affect an unpenalized fit but changes which
        coefficients an L1/L2 penalty shrinks first).
    ties, fit_intercept, max_outer_iter, outer_tol, max_inner_iter, inner_tol :
        See `CoxPH` for `ties`/`fit_intercept`; the remaining four
        control the proximal-Newton outer loop and coordinate-descent
        inner loop (`algorithms/coordinate_descent.py`).

    Attributes (set by `fit`)
    -------------------------
    coef_path_ : ndarray, shape (n_lambda_, n_features)
        Fitted coefficients (original units) at every lambda in
        `lambda_path_`, in the same order.
    lambda_path_ : ndarray, shape (n_lambda_,)
        The lambda values actually fit, descending.
    lambda_max_ : float
        Smallest lambda at which every penalized coefficient is 0
        (see `algorithms/coordinate_descent.py::compute_lambda_max`).
        0 if every feature is unpenalized.
    log_likelihood_path_, deviance_ratio_path_, n_nonzero_path_ : ndarray, shape (n_lambda_,)
        Per-lambda partial log-likelihood, glmnet-style deviance ratio
        (`statistics/deviance.py`), and count of exactly-nonzero
        coefficients (glmnet's `df`).
    log_likelihood_null_ : float
        Partial log-likelihood at beta=0 (all features), the
        deviance-ratio denominator.
    column_scale_ : ndarray, shape (n_features,)
        The `xs` divisor applied to each column before fitting
        (all 1s if `standardize=False`).
    coef_, lambda_ : ndarray / float
        Only set when the resolved `lambda_path_` has exactly one
        value (a single explicit `lambda_path` scalar, or a
        user-supplied length-1 sequence) -- the natural case of "just
        fit one penalized model". Use `coef_at()` or index
        `coef_path_` directly otherwise.
    n_obs_, n_events_, n_features_in_, feature_names_in_, converged_path_, n_iter_path_ :
        See `CoxPH` for the analogous non-path attributes.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        n_lambda: int = 100,
        lambda_min_ratio: Optional[float] = None,
        lambda_path: Optional[Union[float, "np.ndarray"]] = None,
        penalty_factor: Optional[np.ndarray] = None,
        standardize: bool = True,
        ties: Union[str, TieMethod] = "breslow",
        fit_intercept: bool = False,
        max_outer_iter: int = 100,
        outer_tol: float = 1e-9,
        max_inner_iter: int = 1000,
        inner_tol: float = 1e-10,
    ):
        self.alpha = alpha
        self.n_lambda = n_lambda
        self.lambda_min_ratio = lambda_min_ratio
        self.lambda_path = lambda_path
        self.penalty_factor = penalty_factor
        self.standardize = standardize
        self.ties = ties
        self.fit_intercept = fit_intercept
        self.max_outer_iter = max_outer_iter
        self.outer_tol = outer_tol
        self.max_inner_iter = max_inner_iter
        self.inner_tol = inner_tol

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(
        self,
        X,
        duration=None,
        event=None,
        start=None,
        stop=None,
        strata=None,
        offset=None,
        sample_weight=None,
    ) -> "PenalizedCoxPH":
        _validate_common_parameters(
            self.alpha, self.n_lambda, self.lambda_min_ratio, self.lambda_path,
            self.standardize, self.max_outer_iter, self.outer_tol,
            self.max_inner_iter, self.inner_tol, self.fit_intercept,
        )

        clean = validate_fit_inputs(
            X, duration=duration, event=event, start=start, stop=stop,
            strata=strata, offset=offset, sample_weight=sample_weight,
        )
        data = SurvivalData(**clean)

        feature_names = list(data.feature_names)
        p_full = data.n_features

        penalty_factor_full = (
            np.ones(p_full, dtype=np.float64)
            if self.penalty_factor is None
            else np.asarray(self.penalty_factor, dtype=np.float64)
        )
        if penalty_factor_full.shape != (p_full,):
            raise ValueError(
                f"penalty_factor must have shape ({p_full},), got {penalty_factor_full.shape}"
            )
        if not np.all(np.isfinite(penalty_factor_full)):
            raise ValueError("penalty_factor must contain only finite values")
        if np.any(penalty_factor_full < 0):
            raise ValueError("penalty_factor must be non-negative")

        xs_full, degenerate = weighted_column_scale(data.X, data.weight, standardize=self.standardize)
        if np.any(degenerate):
            warnings.warn(
                f"{int(np.sum(degenerate))} feature(s) have ~zero weighted variance and "
                "contribute no information to the partial likelihood regardless of their "
                "coefficient; excluding them from the penalized fit (coefficient fixed at 0): "
                f"{list(np.array(feature_names)[degenerate])}",
                category=DegenerateFeatureWarning, stacklevel=2,
            )
        fit_cols = ~degenerate
        if not np.any(fit_cols):
            raise ValueError("All predictors have zero weighted variance; no usable predictors remain")
        X_full = data.X
        X_fit = X_full[:, fit_cols] / xs_full[fit_cols]
        pf_fit = rescale_penalty_factors(penalty_factor_full[fit_cols], int(fit_cols.sum()))
        p_fit = X_fit.shape[1]

        _stratum_idx = precompute_stratum_indices(data.strata_codes)

        def objective(beta_fit):
            return cox_partial_likelihood(
                X_fit, data.start, data.stop, data.event, beta_fit,
                offset=data.offset, weight=data.weight, strata=data.strata_codes,
                ties=self.ties, stratum_indices=_stratum_idx,
            )

        c = 1.0 / float(np.sum(data.weight))

        # Null point for lambda_max: beta=0 in the common case, or (if
        # some features are unpenalized) the point where those
        # features sit at their own unpenalized MLE and every
        # penalized feature is 0 -- matches glmnet's
        # get_cox_lambda_max, which pre-fits the unpenalized variables
        # via survival::coxph for exactly this reason.
        beta_null = np.zeros(p_fit)
        always_unpenalized = pf_fit == 0.0
        if np.any(always_unpenalized) and not np.all(always_unpenalized):
            restricted = CoxPH(ties=self.ties, max_iter=self.max_outer_iter, eps=self.outer_tol)
            restricted.fit(
                X_fit[:, always_unpenalized], duration=None, event=data.event,
                start=data.start, stop=data.stop, strata=data.strata_codes,
                offset=data.offset, sample_weight=data.weight,
            )
            beta_null[always_unpenalized] = restricted.coef_

        _, score_null, _ = objective(beta_null)
        lambda_max = (
            0.0 if np.all(always_unpenalized)
            else compute_lambda_max(score_null, c, pf_fit, self.alpha)
        )

        lambda_sequence, lambda_min_ratio = _resolve_lambda_path(
            self.lambda_path, lambda_max, self.lambda_min_ratio, self.n_lambda, p_fit, data.n_obs
        )

        results = fit_regularization_path(
            objective, p_fit, c, self.alpha, lambda_sequence, pf_fit,
            beta_warm_start=beta_null,
            outer_max_iter=self.max_outer_iter, outer_tol=self.outer_tol,
            inner_max_iter=self.max_inner_iter, inner_tol=self.inner_tol,
        )

        n_lam = len(results)
        coef_path_fit = np.array([r.beta / xs_full[fit_cols] for r in results])  # unscale
        coef_path = np.zeros((n_lam, p_full))
        coef_path[:, fit_cols] = coef_path_fit

        self.coef_path_ = coef_path
        self.lambda_path_ = np.asarray(lambda_sequence, dtype=np.float64)
        self.lambda_max_ = float(lambda_max)
        self.lambda_min_ratio_ = float(lambda_min_ratio)
        self.log_likelihood_path_ = np.array([r.log_likelihood for r in results])
        self.converged_path_ = np.array([r.converged for r in results])
        self.n_iter_path_ = np.array([r.n_outer_iter for r in results])
        self.n_nonzero_path_ = np.array([int(np.sum(row != 0.0)) for row in coef_path])
        self.column_scale_ = xs_full
        self.penalty_factor_ = penalty_factor_full
        self._excluded_features_ = degenerate

        null_loglik, _, _ = cox_partial_likelihood(
            X_full, data.start, data.stop, data.event, np.zeros(p_full),
            offset=data.offset, weight=data.weight, strata=data.strata_codes,
            ties=self.ties, stratum_indices=_stratum_idx,
        )
        self.log_likelihood_null_ = null_loglik
        lsat = saturated_log_likelihood(data.stop, data.event, data.weight, data.strata_codes)
        self._lsat_ = lsat
        self.deviance_ratio_path_ = np.array(
            [deviance_ratio(ll, null_loglik, lsat) for ll in self.log_likelihood_path_]
        )

        self.n_obs_ = data.n_obs
        self.n_events_ = int(np.sum(data.event))
        self.n_features_in_ = p_full
        self.feature_names_in_ = np.array(feature_names)

        if n_lam == 1:
            self.coef_ = coef_path[0]
            self.lambda_ = float(self.lambda_path_[0])

        return self

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "coef_path_"):
            raise NotFittedError("This PenalizedCoxPH instance is not fitted yet. Call `fit` first.")

    # ------------------------------------------------------------------
    # Coefficient lookup
    # ------------------------------------------------------------------
    def coef_at(self, lambda_value: float) -> np.ndarray:
        """Coefficients at an arbitrary lambda, linearly interpolated
        in log(lambda) between the two bracketing fitted grid points
        (glmnet's `lambda.interp` convention); clamped to the nearest
        endpoint if `lambda_value` is outside the fitted range.
        """
        self._check_is_fitted()
        grid = self.lambda_path_
        lam = float(lambda_value)
        if not np.isfinite(lam) or lam <= 0:
            raise ValueError(f"lambda_value must be finite and strictly positive, got {lambda_value!r}")
        if np.any(grid <= 0):
            raise ValueError("coef_at() is undefined for a path containing non-positive lambda values")
        if len(grid) == 1:
            return self.coef_path_[0]
        log_grid = np.log(grid)
        if lam >= grid[0]:
            return self.coef_path_[0]
        if lam <= grid[-1]:
            return self.coef_path_[-1]
        log_lam = np.log(lam)
        # grid is descending; find the bracketing pair
        upper_idx = np.searchsorted(-log_grid, -log_lam) - 1
        upper_idx = int(np.clip(upper_idx, 0, len(grid) - 2))
        lower_idx = upper_idx + 1
        l_up, l_lo = log_grid[upper_idx], log_grid[lower_idx]
        frac = 0.0 if l_up == l_lo else (log_lam - l_lo) / (l_up - l_lo)
        return frac * self.coef_path_[upper_idx] + (1 - frac) * self.coef_path_[lower_idx]

    def nonzero_features(self, lambda_value: Optional[float] = None) -> np.ndarray:
        """Feature names with a nonzero coefficient at `lambda_value`
        (defaults to `self.lambda_` when the fit resolved to a single
        lambda)."""
        self._check_is_fitted()
        if lambda_value is None:
            if not hasattr(self, "coef_"):
                raise ValueError(
                    "lambda_value must be specified when multiple lambda values were fitted"
                )
            coef = self.coef_
        else:
            coef = self.coef_at(lambda_value)
        return self.feature_names_in_[coef != 0.0]

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _resolve_coef(self, lambda_value: Optional[float]) -> np.ndarray:
        self._check_is_fitted()
        if lambda_value is not None:
            return self.coef_at(lambda_value)
        if hasattr(self, "coef_"):
            return self.coef_
        raise ValueError(
            "This fit produced a path with more than one lambda; pass `lambda_value=` "
            "to select which point on the path to use for prediction."
        )

    def predict_linear(self, X, offset=None, lambda_value: Optional[float] = None) -> np.ndarray:
        coef = self._resolve_coef(lambda_value)
        X_arr, _ = validate_X(X)
        offset_arr = np.zeros(X_arr.shape[0]) if offset is None else np.asarray(offset, dtype=np.float64)
        return X_arr @ coef + offset_arr

    def predict_partial_hazard(self, X, offset=None, lambda_value: Optional[float] = None) -> np.ndarray:
        return safe_exp(self.predict_linear(X, offset=offset, lambda_value=lambda_value))

    def predict(self, X, offset=None, lambda_value: Optional[float] = None) -> np.ndarray:
        return self.predict_partial_hazard(X, offset=offset, lambda_value=lambda_value)

    def summary(self) -> pd.DataFrame:
        """Per-lambda path summary analogous to glmnet's printed
        `glmnet` object table (lambda, df, %dev)."""
        self._check_is_fitted()
        return pd.DataFrame(
            {
                "lambda": self.lambda_path_,
                "n_nonzero": self.n_nonzero_path_,
                "deviance_ratio": self.deviance_ratio_path_,
                "log_likelihood": self.log_likelihood_path_,
                "converged": self.converged_path_,
            }
        )


class PenalizedCoxPHCV(BaseEstimator):
    """K-fold cross-validated lambda selection for `PenalizedCoxPH`,
    analogous to R's `cv.glmnet(family="cox")`.

    Fits the full-data path once to establish the lambda grid
    (`lambda_path_`), then, for each fold, fits `PenalizedCoxPH` on the
    training-minus-fold data *at exactly those same lambda values*
    (rather than letting each fold auto-generate and interpolate onto
    the master grid, as glmnet does internally) -- a simplification
    enabled by `PenalizedCoxPH.fit(lambda_path=...)` accepting an
    explicit sequence, and at least as accurate as interpolation since
    each fold is fit exactly at the lambda of interest. The
    per-fold-per-lambda deviance uses the Verweij & Van Houwelingen
    (1993) grouped method, matching glmnet's default (`grouped=TRUE`)
    -- see `statistics/deviance.py` and docs/R_COMPATIBILITY.md.

    Parameters
    ----------
    n_folds : int, default 10
        Number of cross-validation folds (glmnet's `nfolds`).
    fold_id : array-like of shape (n_obs,), optional
        Explicit fold assignment (1..n_folds or 0..n_folds-1), matching
        glmnet's `foldid` -- primarily useful for reproducing an exact
        R comparison with the same fold assignment.
    random_state : int, optional
        Used to randomly assign folds when `fold_id` is not given.
    select : {"lambda_min", "lambda_1se"}, default "lambda_min"
        Which cross-validated lambda `final_estimator_`/`coef_` uses.
    Remaining parameters are passed through to the underlying
    `PenalizedCoxPH` fits -- see that class.

    Attributes (set by `fit`)
    -------------------------
    lambda_path_, cv_mean_deviance_, cv_se_deviance_ : ndarray, shape (n_lambda_,)
        The fitted lambda grid and, per lambda, the cross-validated
        mean (and standard error of the mean, across folds) deviance
        per event -- glmnet's `cvm`/`cvsd`.
    lambda_min_, lambda_1se_ : float
        The lambda with minimum cross-validated deviance, and the
        largest lambda within one standard error of that minimum
        (glmnet's 1-SE rule).
    final_estimator_ : PenalizedCoxPH
        Fit on the *full* data at `lambda_min_` or `lambda_1se_` (per
        `select`).
    coef_ : ndarray
        `final_estimator_.coef_`.
    fold_id_ : ndarray, shape (n_obs,)
        The fold assignment actually used.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        n_lambda: int = 100,
        lambda_min_ratio: Optional[float] = None,
        lambda_path: Optional[Union[float, "np.ndarray"]] = None,
        penalty_factor: Optional[np.ndarray] = None,
        standardize: bool = True,
        ties: Union[str, TieMethod] = "breslow",
        fit_intercept: bool = False,
        n_folds: int = 10,
        fold_id: Optional[np.ndarray] = None,
        random_state: Optional[int] = None,
        select: str = "lambda_min",
        max_outer_iter: int = 100,
        outer_tol: float = 1e-9,
        max_inner_iter: int = 1000,
        inner_tol: float = 1e-10,
    ):
        self.alpha = alpha
        self.n_lambda = n_lambda
        self.lambda_min_ratio = lambda_min_ratio
        self.lambda_path = lambda_path
        self.penalty_factor = penalty_factor
        self.standardize = standardize
        self.ties = ties
        self.fit_intercept = fit_intercept
        self.n_folds = n_folds
        self.fold_id = fold_id
        self.random_state = random_state
        self.select = select
        self.max_outer_iter = max_outer_iter
        self.outer_tol = outer_tol
        self.max_inner_iter = max_inner_iter
        self.inner_tol = inner_tol

    def _base_kwargs(self) -> dict:
        return dict(
            alpha=self.alpha, penalty_factor=self.penalty_factor, standardize=self.standardize,
            ties=self.ties, fit_intercept=self.fit_intercept,
            max_outer_iter=self.max_outer_iter, outer_tol=self.outer_tol,
            max_inner_iter=self.max_inner_iter, inner_tol=self.inner_tol,
        )

    def fit(
        self,
        X,
        duration=None,
        event=None,
        start=None,
        stop=None,
        strata=None,
        offset=None,
        sample_weight=None,
    ) -> "PenalizedCoxPHCV":
        _validate_common_parameters(
            self.alpha, self.n_lambda, self.lambda_min_ratio, self.lambda_path,
            self.standardize, self.max_outer_iter, self.outer_tol,
            self.max_inner_iter, self.inner_tol, self.fit_intercept,
        )
        if self.select not in ("lambda_min", "lambda_1se"):
            raise ValueError(f"select must be 'lambda_min' or 'lambda_1se', got {self.select!r}")
        if self.n_lambda < 2 and self.lambda_path is None:
            raise ValueError("n_lambda must be at least 2 for cross-validation")
        if self.lambda_path is not None and np.atleast_1d(self.lambda_path).size < 2:
            raise ValueError("lambda_path must contain at least 2 values for cross-validation")
        if isinstance(self.n_folds, (bool, np.bool_)) or int(self.n_folds) != self.n_folds or int(self.n_folds) < 3:
            raise ValueError("n_folds must be an integer >= 3")

        clean = validate_fit_inputs(
            X, duration=duration, event=event, start=start, stop=stop,
            strata=strata, offset=offset, sample_weight=sample_weight,
        )
        data = SurvivalData(**clean)
        n = data.n_obs

        full_fit = PenalizedCoxPH(
            n_lambda=self.n_lambda, lambda_min_ratio=self.lambda_min_ratio,
            lambda_path=self.lambda_path, **self._base_kwargs(),
        )
        full_fit.fit(
            data.X, event=data.event, start=data.start, stop=data.stop,
            strata=data.strata_codes, offset=data.offset, sample_weight=data.weight,
        )
        lambda_grid = full_fit.lambda_path_
        n_lam = len(lambda_grid)

        if self.fold_id is not None:
            fold_id_raw = np.asarray(self.fold_id)
            if fold_id_raw.ndim != 1 or fold_id_raw.shape[0] != n:
                raise ValueError(f"fold_id must be a one-dimensional array of length {n}")
            if not np.all(np.isfinite(fold_id_raw.astype(np.float64, copy=False))):
                raise ValueError("fold_id must contain finite values")
            if not np.all(np.equal(fold_id_raw, np.floor(fold_id_raw))):
                raise ValueError("fold_id values must be integers")
            fold_id = fold_id_raw.astype(np.int64)
            fold_labels = np.unique(fold_id)
            # When fold_id is supplied, it determines the actual number of
            # folds, as in glmnet. Accept either 0..K-1 or 1..K labels and
            # require contiguous labels so there cannot be an implicit
            # empty fold. `n_folds` controls generated folds only.
            k = fold_labels.size
            if not (np.array_equal(fold_labels, np.arange(k)) or
                    np.array_equal(fold_labels, np.arange(1, k + 1))):
                raise ValueError("fold_id must use contiguous labels 0..K-1 or 1..K")
        else:
            rng = np.random.RandomState(self.random_state)
            base = np.tile(np.arange(self.n_folds), int(np.ceil(n / self.n_folds)))[:n]
            fold_id = rng.permutation(base)
            fold_labels = np.arange(self.n_folds)

        self.fold_id_ = fold_id
        cvraw = np.full((len(fold_labels), n_lam), np.nan, dtype=np.float64)
        fold_weight = np.zeros(len(fold_labels), dtype=np.float64)
        fold_event_weight = np.zeros(len(fold_labels), dtype=np.float64)

        # Full-data quantities are independent of the held-out fold.
        lsat_full = saturated_log_likelihood(data.stop, data.event, data.weight, data.strata_codes)
        _stratum_idx_full = precompute_stratum_indices(data.strata_codes)

        for i, fold in enumerate(fold_labels):
            held_out = fold_id == fold
            train = ~held_out
            fold_weight[i] = float(np.sum(data.weight[held_out]))
            fold_event_weight[i] = float(np.sum(data.weight[held_out] * data.event[held_out]))
            if fold_event_weight[i] <= 0.0:
                raise ValueError(f"CV fold {fold!r} contains no positive event weight")

            fold_fit = PenalizedCoxPH(lambda_path=lambda_grid, **self._base_kwargs())
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DegenerateFeatureWarning)
                fold_fit.fit(
                    data.X[train], event=data.event[train], start=data.start[train],
                    stop=data.stop[train], strata=data.strata_codes[train],
                    offset=data.offset[train], sample_weight=data.weight[train],
                )

            if not np.all(fold_fit.converged_path_):
                bad = np.flatnonzero(~fold_fit.converged_path_)
                warnings.warn(
                    f"Penalized Cox fold {fold!r} did not converge at lambda indices "
                    f"{bad.tolist()}; using the last iterate (matches glmnet behavior)",
                    stacklevel=2,
                )

            lsat_train = saturated_log_likelihood(
                data.stop[train], data.event[train], data.weight[train], data.strata_codes[train]
            )
            _stratum_idx_train = precompute_stratum_indices(data.strata_codes[train])
            for j in range(n_lam):
                beta_j = fold_fit.coef_path_[j]
                loglik_full, _, _ = cox_partial_likelihood(
                    data.X, data.start, data.stop, data.event, beta_j,
                    offset=data.offset, weight=data.weight, strata=data.strata_codes,
                    ties=self.ties, stratum_indices=_stratum_idx_full,
                )
                loglik_train, _, _ = cox_partial_likelihood(
                    data.X[train], data.start[train], data.stop[train], data.event[train], beta_j,
                    offset=data.offset[train], weight=data.weight[train], strata=data.strata_codes[train],
                    ties=self.ties, stratum_indices=_stratum_idx_train,
                )
                dev_full = cox_deviance(loglik_full, lsat_full)
                dev_train = cox_deviance(loglik_train, lsat_train)
                cvraw[i, j] = dev_full - dev_train
            # The pinned glmnet 4.1-8 Cox reference used by this package's
            # regression suite expresses the grouped deviance on an event-
            # weighted scale. Normalizing by held-out event weight reproduces
            # those stored cv.glmnet values; `fold_weight` remains available
            # as the observation-weight sum for diagnostics.
            cvraw[i, :] /= fold_event_weight[i]

        valid = np.isfinite(cvraw)
        n_folds_used = np.sum(valid, axis=0)
        cvm = np.full(n_lam, np.nan, dtype=np.float64)
        cvsd = np.full(n_lam, np.nan, dtype=np.float64)
        for j in range(n_lam):
            ok = valid[:, j]
            if np.sum(ok) == 0:
                continue
            wj = fold_event_weight[ok]
            yj = cvraw[ok, j]
            wsum = np.sum(wj)
            cvm[j] = np.sum(wj * yj) / wsum
            n_ok = int(np.sum(ok))
            if n_ok > 1:
                # Match glmnet::cvstats: weighted population variance divided
                # by (N - 1), where N is the number of usable folds.
                cv_var = np.sum(wj * (yj - cvm[j]) ** 2) / wsum
                cvsd[j] = np.sqrt(cv_var / (n_ok - 1))
            else:
                # glmnet drops lambdas whose CV standard error is NA. We
                # cannot estimate a standard error from a single fold.
                cvsd[j] = np.nan

        self.lambda_path_ = lambda_grid
        self.cv_mean_deviance_ = cvm
        self.cv_se_deviance_ = cvsd
        self.cv_n_folds_ = n_folds_used

        valid_lambda = np.isfinite(cvm) & np.isfinite(cvsd)
        if not np.any(valid_lambda):
            raise RuntimeError("Cross-validation produced no valid lambda values")
        min_idx = int(np.nanargmin(np.where(valid_lambda, cvm, np.nan)))
        self.lambda_min_ = float(lambda_grid[min_idx])
        within_1se = valid_lambda & (cvm <= cvm[min_idx] + cvsd[min_idx])
        # lambda_grid is descending, so the *first* True is the largest
        # lambda within one SE of the minimum.
        self.lambda_1se_ = float(lambda_grid[np.flatnonzero(within_1se)[0]])

        chosen_lambda = self.lambda_min_ if self.select == "lambda_min" else self.lambda_1se_
        self.final_estimator_ = PenalizedCoxPH(lambda_path=chosen_lambda, **self._base_kwargs())
        self.final_estimator_.fit(
            data.X, event=data.event, start=data.start, stop=data.stop,
            strata=data.strata_codes, offset=data.offset, sample_weight=data.weight,
        )
        self.coef_ = self.final_estimator_.coef_
        self.n_obs_ = data.n_obs
        self.n_events_ = int(np.sum(data.event))
        self.feature_names_in_ = full_fit.feature_names_in_
        self.n_nonzero_path_ = full_fit.n_nonzero_path_
        self.full_fit_ = full_fit

        return self

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "final_estimator_"):
            raise NotFittedError("This PenalizedCoxPHCV instance is not fitted yet. Call `fit` first.")

    def predict_linear(self, X, offset=None) -> np.ndarray:
        self._check_is_fitted()
        return self.final_estimator_.predict_linear(X, offset=offset)

    def predict_partial_hazard(self, X, offset=None) -> np.ndarray:
        self._check_is_fitted()
        return self.final_estimator_.predict_partial_hazard(X, offset=offset)

    def predict(self, X, offset=None) -> np.ndarray:
        return self.predict_partial_hazard(X, offset=offset)

    def summary(self) -> pd.DataFrame:
        """Per-lambda cross-validation summary analogous to R's
        `print(cv.glmnet_fit)` / plotting data."""
        self._check_is_fitted()
        return pd.DataFrame(
            {
                "lambda": self.lambda_path_,
                "n_nonzero": self.n_nonzero_path_,
                "cv_mean_deviance": self.cv_mean_deviance_,
                "cv_se_deviance": self.cv_se_deviance_,
            }
        )
