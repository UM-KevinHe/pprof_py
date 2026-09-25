"""Discrete-time survival estimators with penalized regression.

This module provides ``DiscreteSurvival`` (regularization path) and
``DiscreteSurvivalCV`` (cross-validated lambda selection) for discrete-
time survival models with a logistic link.

Architecture
------------
The discrete-time survival model expresses the conditional hazard as::

    logit(h(t_k | Z_i)) = alpha_k + gamma_i + Z_i @ beta

where ``alpha_k`` are baseline hazard parameters (one per distinct
timepoint), ``gamma_i`` are optional provider effects (unpenalized),
and ``beta`` are covariate coefficients (penalized).

The model supports three penalty types:

* ``'lasso'`` — individual L1 penalty
* ``'group_lasso'`` — group L2 penalty
* ``'sparse_group_lasso'`` — combined L1 + group L2

When ``provider`` is supplied, the model uses the two-layer
architecture from ``grplasso``: provider effects are updated via
Newton steps in the outer layer, covariates via IRLS + coordinate
descent in the inner layer.

See Also
--------
algorithms.discrete_survival : Numerical kernels.
PenalizedCoxPH : Continuous-time penalized Cox PH.
ProviderPenalizedCoxPH : Continuous-time Cox with provider effects.

References
----------
.. [1] He, K., Kalbfleisch, J., Li, Y., et al. (2013). Evaluating
   hospital readmission rates in dialysis facilities; adjusting for
   hospital effects. *Lifetime Data Analysis*, 19, 490-512.
.. [2] Shao, Y. & He, K. (2026). grplasso R package.
"""
from __future__ import annotations

import logging
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from ...base import ProviderModel

from ...algorithms.survival.discrete_survival import (
    discretize_times,
    initialize_baseline_hazard,
    compute_n_at_risk,
    compute_discrete_lambda_max,
    discrete_residuals,
    discrete_loglik,
    fit_discrete_regularization_path,
    fit_single_lambda_discrete,
    person_period_expand,
    predict_discrete_hazard,
    predict_survival_probability,
    DiscreteFitResult,
)
from ...algorithms.survival.penalty import (
    weighted_column_scale,
    rescale_penalty_factors,
)
from ...algorithms.survival.coordinate_descent import build_lambda_sequence
from .coxph import NotFittedError

logger = logging.getLogger(__name__)


# ======================================================================
# Validation helpers
# ======================================================================

def _validate_discrete_params(
    penalty_type, n_lambda, lambda_min_ratio, lambda_path,
    standardize, max_iter, tol, bound,
):
    """Validate constructor parameters."""
    valid_penalties = ('lasso', 'group_lasso', 'sparse_group_lasso')
    if penalty_type not in valid_penalties:
        raise ValueError(
            f"penalty_type must be one of {valid_penalties}, got {penalty_type!r}"
        )
    if isinstance(n_lambda, (bool, np.bool_)) or int(n_lambda) != n_lambda or int(n_lambda) < 1:
        raise ValueError(f"n_lambda must be a positive integer, got {n_lambda!r}")
    if lambda_min_ratio is not None:
        ratio = float(lambda_min_ratio)
        if not np.isfinite(ratio) or not (0.0 < ratio <= 1.0):
            raise ValueError(
                f"lambda_min_ratio must be in (0, 1], got {lambda_min_ratio!r}"
            )
    if lambda_path is not None:
        values = np.atleast_1d(np.asarray(lambda_path, dtype=np.float64))
        if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("lambda_path must contain finite, strictly positive values")
    if not isinstance(standardize, (bool, np.bool_)):
        raise TypeError(f"standardize must be bool, got {type(standardize).__name__}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    if tol <= 0:
        raise ValueError(f"tol must be > 0, got {tol}")
    if bound <= 0:
        raise ValueError(f"bound must be > 0, got {bound}")


# ======================================================================
# DiscreteSurvival — Regularization path
# ======================================================================

class DiscreteSurvival(ProviderModel):
    """Penalized discrete-time survival model.

    Fits a regularization path of discrete-time survival models with
    a logistic link and L1 penalty on covariate coefficients.  Baseline
    hazard parameters (one per discrete timepoint) are estimated via
    Newton updates at each iteration.

    Parameters
    ----------
    penalty_type : str, default 'lasso'
        One of ``'lasso'``, ``'group_lasso'``, ``'sparse_group_lasso'``.
    groups : array-like or None, default None
        Group labels per feature (required for group/sparse group).
        0 = unpenalized.
    alpha : float, default 1.0
        Sparse group mixing: 0 = pure group lasso, 1 = lasso.
        Only used when ``penalty_type='sparse_group_lasso'``.
    group_multiplier : array-like or None, default None
        Per-group multipliers; defaults to sqrt(group_size).
    penalty_factor : array-like or None, default None
        Per-feature penalty factors.  Rescaled to sum to p.
    n_lambda : int, default 100
        Number of lambda values in auto grid.
    lambda_min_ratio : float or None, default None
        Ratio of min/max lambda (auto: 1e-4).
    lambda_path : array-like or None, default None
        User-supplied lambda sequence (overrides auto grid).
    standardize : bool, default True
        Standardize features before fitting.
    method : str, default 'Newton'
        ``'Newton'`` (exact IRLS weights) or ``'MM'``
        (diagonal majorization, v=0.25).
    bound : float, default 10.0
        Bounding constraint for baseline hazard parameters.
    backtrack : bool, default False
        Armijo backtracking for baseline hazard updates.
    use_active_set : bool, default True
        Active-set screening for beta.
    max_iter : int, default 10000
        Maximum iterations per lambda.
    tol : float, default 1e-4
        Convergence tolerance.
    nvar_max : int or None, default None
        Maximum selected variables (early stop).

    Attributes
    ----------
    coef_path_ : ndarray, shape (n_lambda, p)
        Coefficients along the path.
    alpha_path_ : ndarray, shape (n_lambda, K)
        Baseline hazard parameters along the path.
    lambda_path_ : ndarray, shape (n_lambda,)
        Lambda values used.
    lambda_max_ : float
        Smallest lambda zeroing all penalized coefficients.
    df_ : ndarray, shape (n_lambda,)
        Effective degrees of freedom.
    n_iter_ : ndarray of int, shape (n_lambda,)
        Iterations per lambda.
    converged_ : ndarray of bool, shape (n_lambda,)
        Convergence per lambda.
    timepoint_map_ : ndarray, shape (K,)
        Mapping from integer codes to original timepoints.
    n_timepoints_ : int
        Number of distinct timepoints.

    See Also
    --------
    DiscreteSurvivalCV : Cross-validated lambda selection.
    """

    def __init__(
        self,
        penalty_type: str = 'lasso',
        groups=None,
        alpha: float = 1.0,
        group_multiplier=None,
        penalty_factor=None,
        n_lambda: int = 100,
        lambda_min_ratio: Optional[float] = None,
        lambda_path=None,
        standardize: bool = True,
        method: str = 'Newton',
        bound: float = 10.0,
        backtrack: bool = False,
        use_active_set: bool = True,
        max_iter: int = 10000,
        tol: float = 1e-4,
        nvar_max: Optional[int] = None,
    ):
        """Discrete-time survival with penalized regression."""
        self.penalty_type = penalty_type
        self.groups = groups
        self.alpha = alpha
        self.group_multiplier = group_multiplier
        self.penalty_factor = penalty_factor
        self.n_lambda = n_lambda
        self.lambda_min_ratio = lambda_min_ratio
        self.lambda_path = lambda_path
        self.standardize = standardize
        self.method = method
        self.bound = bound
        self.backtrack = backtrack
        self.use_active_set = use_active_set
        self.max_iter = max_iter
        self.tol = tol
        self.nvar_max = nvar_max

    def fit(
        self,
        X,
        time,
        event,
        sample_weight=None,
    ) -> "DiscreteSurvival":
        """Fit the discrete-time survival regularization path.

        Parameters
        ----------
        X : DataFrame or ndarray, shape (n, p)
            Covariate matrix.
        time : Series or ndarray, shape (n,)
            Follow-up times (discrete or continuous; discretized
            internally to unique sorted values).
        event : Series or ndarray, shape (n,)
            Event indicator (1 = event, 0 = censored).
        sample_weight : Series or ndarray or None
            Per-observation weights.

        Returns
        -------
        self
        """
        _validate_discrete_params(
            self.penalty_type, self.n_lambda, self.lambda_min_ratio,
            self.lambda_path, self.standardize, self.max_iter,
            self.tol, self.bound,
        )

        # --- Coerce inputs ---
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns)
            X_np = X.values.astype(np.float64)
        else:
            X_np = np.asarray(X, dtype=np.float64)
            self.feature_names_in_ = np.array(
                [f"x{j}" for j in range(X_np.shape[1])]
            )
        time_np = np.asarray(time, dtype=np.float64)
        event_np = np.asarray(event, dtype=np.float64)
        n, p = X_np.shape

        if sample_weight is not None:
            weight = np.asarray(sample_weight, dtype=np.float64)
        else:
            weight = np.ones(n, dtype=np.float64)

        # --- Discretize times ---
        time_int, timepoint_map, n_events = discretize_times(time_np, event_np)
        K = len(timepoint_map)
        self.timepoint_map_ = timepoint_map
        self.n_timepoints_ = K

        # --- Standardize ---
        # Bug fix (2026-09-18): weighted_column_scale returns (xs, degenerate),
        # not (centers, scales).  The old code subtracted pop_sd from X
        # instead of dividing.  Now correctly: center by weighted mean,
        # scale by weighted population sd — matching R's standardize.Z.
        if self.standardize:
            xs, degenerate = weighted_column_scale(X_np, weight)
            w_sum = weight.sum()
            xm = (weight / w_sum) @ X_np           # weighted column means
            xs_safe = np.where(xs == 0, 1.0, xs)   # avoid /0 for degenerate cols
            X_fit = (X_np - xm) / xs_safe
            self._centers = xm
            self._scales = xs
        else:
            X_fit = X_np.copy()
            self._centers = np.zeros(p)
            self._scales = np.ones(p)

        # --- Penalty factor ---
        if self.penalty_factor is not None:
            pf = np.asarray(self.penalty_factor, dtype=np.float64)
            if len(pf) != p:
                raise ValueError(
                    f"penalty_factor length ({len(pf)}) != n_features ({p})"
                )
            pf = rescale_penalty_factors(pf)
        else:
            pf = np.ones(p, dtype=np.float64)

        # --- Initialize baseline hazard from KM ---
        n_at_risk = compute_n_at_risk(time_int, K)
        alpha_init = initialize_baseline_hazard(n_events, n_at_risk)

        # --- Compute residuals at null and lambda_max ---
        r_null = discrete_residuals(time_int, event_np, alpha_init, np.zeros(n))
        lambda_max = compute_discrete_lambda_max(r_null, X_fit, pf)
        self.lambda_max_ = lambda_max

        # --- Build lambda sequence ---
        if self.lambda_path is not None:
            lam_seq = np.sort(np.atleast_1d(
                np.asarray(self.lambda_path, dtype=np.float64)
            ))[::-1]
        else:
            ratio = self.lambda_min_ratio if self.lambda_min_ratio else 1e-4
            lam_seq = np.exp(
                np.linspace(
                    np.log(lambda_max + 1e-5),
                    np.log(ratio * lambda_max),
                    self.n_lambda,
                )
            )

        use_mm = self.method.upper() == 'MM'

        # --- Fit path ---
        results = fit_discrete_regularization_path(
            X_fit, time_int, event_np, n_events,
            lambda_sequence=lam_seq,
            penalty_factor=pf,
            alpha_init=alpha_init,
            use_mm=use_mm,
            bound=self.bound,
            backtrack=self.backtrack,
            tol=self.tol,
            max_iter=self.max_iter,
            active_set=self.use_active_set,
            nvar_max=self.nvar_max,
        )

        # --- Store results ---
        n_lam = len(results)
        self.lambda_path_ = lam_seq[:n_lam]
        self.coef_path_ = np.zeros((n_lam, p), dtype=np.float64)
        self.alpha_path_ = np.zeros((n_lam, K), dtype=np.float64)
        self.df_ = np.zeros(n_lam, dtype=np.float64)
        self.n_iter_ = np.zeros(n_lam, dtype=np.int64)
        self.converged_ = np.zeros(n_lam, dtype=bool)
        self.neg_loglik_ = np.zeros(n_lam, dtype=np.float64)

        for i, res in enumerate(results):
            # Unstandardize coefficients: beta_orig = beta_std / scale
            scales_safe = np.where(self._scales == 0, 1.0, self._scales)
            beta_orig = res.beta / scales_safe
            self.coef_path_[i] = beta_orig
            # Adjust alpha for centering: alpha_k_orig = alpha_k - mean @ beta_orig
            self.alpha_path_[i] = res.alpha - self._centers @ beta_orig
            self.df_[i] = res.df
            self.n_iter_[i] = res.n_iter
            self.converged_[i] = res.converged
            self.neg_loglik_[i] = res.neg_loglik

        self.n_features_in_ = p
        self.n_obs_ = n
        self.n_events_ = int(np.sum(event_np))
        self._time_int = time_int
        self._event = event_np
        self._weight = weight
        self._pf = pf

        return self

    def _check_is_fitted(self):
        if not hasattr(self, 'coef_path_'):
            raise NotFittedError(
                "DiscreteSurvival is not fitted. Call `fit` first."
            )

    def coef_at(self, lambda_value: float) -> np.ndarray:
        """Coefficients at an arbitrary lambda via linear interpolation."""
        self._check_is_fitted()
        lam = self.lambda_path_
        idx = np.interp(
            lambda_value,
            lam[::-1],
            np.arange(len(lam), dtype=np.float64)[::-1],
        )
        lo = int(np.floor(idx))
        hi = int(np.ceil(idx))
        lo = max(0, min(lo, len(lam) - 1))
        hi = max(0, min(hi, len(lam) - 1))
        if lo == hi:
            return self.coef_path_[lo]
        w = idx - lo
        return (1.0 - w) * self.coef_path_[lo] + w * self.coef_path_[hi]

    def predict(
        self,
        X,
        time=None,
        lambda_value=None,
        which: Optional[int] = None,
        type: str = 'link',
    ):
        """Predict linear predictor, hazard, or survival.

        Parameters
        ----------
        X : DataFrame or ndarray
        time : array-like or None
            Required when *type* is ``'hazard'`` or ``'survival'``.
            Per-subject follow-up time (same scale as training).
        lambda_value : float or None
            Query at a specific lambda.
        which : int or None
            Index into ``lambda_path_``.
        type : str
            ``'link'`` (linear predictor X @ beta), ``'hazard'``
            (person-period hazard, requires *time*), or
            ``'survival'`` (cumulative survival, requires *time*).

        Returns
        -------
        ndarray
        """
        self._check_is_fitted()
        if type in ('hazard', 'survival') and time is None:
            raise ValueError(
                f"type={type!r} requires a 'time' array; pass "
                f"time= or use predict_hazard()/predict_survival()."
            )
        if type == 'hazard':
            return self.predict_hazard(
                X, time, lambda_value=lambda_value, which=which,
            )
        if type == 'survival':
            return self.predict_survival(
                X, time, lambda_value=lambda_value, which=which,
            )

        if isinstance(X, pd.DataFrame):
            X_np = X.values.astype(np.float64)
        else:
            X_np = np.asarray(X, dtype=np.float64)

        if lambda_value is not None:
            coef = self.coef_at(lambda_value)
        elif which is not None:
            coef = self.coef_path_[which]
        else:
            coef = self.coef_path_[-1]

        eta = X_np @ coef
        if type == 'link':
            return eta
        else:
            raise ValueError(
                f"type={type!r} not recognised; expected "
                f"'link', 'hazard', or 'survival'."
            )

    def predict_hazard(
        self,
        X,
        time,
        lambda_value=None,
        which: Optional[int] = None,
    ) -> np.ndarray:
        """Predicted hazard probabilities in person-period (long) format.

        Returns a 1-D array whose length equals ``sum(time_int)`` after
        discretizing each subject's *time* into integer codes via
        ``timepoint_map_``.  This is distinct from
        ``ProviderPenalizedDiscreteSurvival.predict_hazard()``, which
        returns a 2-D ``(n, K)`` wide-format array instead (see
        ISSUE-023 in ``CODE_ISSUES.md``).

        Parameters
        ----------
        X : DataFrame or ndarray, shape ``(n, p)``
        time : array-like, shape ``(n,)``
            Follow-up time for each subject.
        lambda_value : float or None
        which : int or None

        Returns
        -------
        ndarray, shape ``(sum(time_int),)``
        """
        self._check_is_fitted()
        if isinstance(X, pd.DataFrame):
            X_np = X.values.astype(np.float64)
        else:
            X_np = np.asarray(X, dtype=np.float64)
        time_np = np.asarray(time, dtype=np.float64)

        # ISSUE-019 fix: look up query times against the fitted
        # timepoint mapping rather than re-discretizing locally.
        time_int = np.searchsorted(self.timepoint_map_, time_np) + 1
        # Clip to valid range [1, K].
        K = len(self.timepoint_map_)
        time_int = np.clip(time_int, 1, K)

        if lambda_value is not None:
            coef = self.coef_at(lambda_value)
            # Find nearest alpha
            idx = np.argmin(np.abs(self.lambda_path_ - lambda_value))
            alpha = self.alpha_path_[idx]
        elif which is not None:
            coef = self.coef_path_[which]
            alpha = self.alpha_path_[which]
        else:
            coef = self.coef_path_[-1]
            alpha = self.alpha_path_[-1]

        eta = X_np @ coef
        return predict_discrete_hazard(alpha, eta, time_int)

    def predict_survival(
        self,
        X,
        time,
        lambda_value=None,
        which: Optional[int] = None,
    ) -> np.ndarray:
        """Predicted survival probabilities S(T_i | Z_i).

        Returns a 1-D person-period array (same convention as
        ``predict_hazard``; see ISSUE-023 for the shape difference
        vs. the provider-aware class).
        """
        self._check_is_fitted()
        if isinstance(X, pd.DataFrame):
            X_np = X.values.astype(np.float64)
        else:
            X_np = np.asarray(X, dtype=np.float64)
        time_np = np.asarray(time, dtype=np.float64)

        # ISSUE-019 fix: same as predict_hazard — use fitted mapping.
        time_int = np.searchsorted(self.timepoint_map_, time_np) + 1
        K = len(self.timepoint_map_)
        time_int = np.clip(time_int, 1, K)

        if lambda_value is not None:
            coef = self.coef_at(lambda_value)
            idx = np.argmin(np.abs(self.lambda_path_ - lambda_value))
            alpha = self.alpha_path_[idx]
        elif which is not None:
            coef = self.coef_path_[which]
            alpha = self.alpha_path_[which]
        else:
            coef = self.coef_path_[-1]
            alpha = self.alpha_path_[-1]

        eta = X_np @ coef
        return predict_survival_probability(alpha, eta, time_int)

    def summary(self, which: int = -1) -> pd.DataFrame:
        """Summary table at a given lambda index."""
        self._check_is_fitted()
        coef = self.coef_path_[which]
        return pd.DataFrame({
            'coef': coef,
            'nonzero': coef != 0,
        }, index=self.feature_names_in_)


# ======================================================================
# DiscreteSurvivalCV — Cross-validated lambda selection
# ======================================================================

class DiscreteSurvivalCV(ProviderModel):
    """Cross-validated discrete-time survival model.

    Performs k-fold cross-validation to select the optimal lambda.
    Implements the timepoint coverage check from ``grplasso``'s
    ``cv.DiscSurv``: ensures every training fold contains all
    distinct timepoints.

    Parameters
    ----------
    n_folds : int, default 10
        Number of cross-validation folds.
    se_rule : str, default '1se'
        Lambda selection rule: ``'min'`` (minimum CV error) or
        ``'1se'`` (largest lambda within 1 SE of minimum).
    random_state : int or None, default None
        Random seed for fold assignment.
    max_fold_retries : int, default 100
        Maximum retries for event-stratified fold assignment with
        timepoint coverage.
    **kwargs
        Passed to ``DiscreteSurvival``.

    Attributes
    ----------
    lambda_min_ : float
        Lambda with minimum mean CV error.
    lambda_1se_ : float
        Largest lambda within 1 SE of the minimum.
    cv_mean_ : ndarray, shape (n_lambda,)
        Mean CV error per lambda.
    cv_se_ : ndarray, shape (n_lambda,)
        Standard error of CV error per lambda.
    model_ : DiscreteSurvival
        Full-data fit.
    fold_assignment_ : ndarray of int, shape (n,)
    """

    def __init__(
        self,
        n_folds: int = 10,
        se_rule: str = '1se',
        random_state=None,
        max_fold_retries: int = 100,
        **kwargs,
    ):
        """Cross-validated discrete-time survival."""
        self.n_folds = n_folds
        self.se_rule = se_rule
        self.random_state = random_state
        self.max_fold_retries = max_fold_retries
        self._model_kwargs = kwargs

    def fit(
        self,
        X,
        time,
        event,
        sample_weight=None,
    ) -> "DiscreteSurvivalCV":
        """Fit with cross-validation.

        Parameters
        ----------
        X, time, event, sample_weight
            Same as ``DiscreteSurvival.fit``.

        Returns
        -------
        self
        """
        if self.se_rule not in ("min", "1se"):
            raise ValueError(f"se_rule must be 'min' or '1se', got {self.se_rule!r}")
        # Coerce
        if isinstance(X, pd.DataFrame):
            X_np = X.values.astype(np.float64)
        else:
            X_np = np.asarray(X, dtype=np.float64)
        time_np = np.asarray(time, dtype=np.float64)
        event_np = np.asarray(event, dtype=np.float64)
        n = len(event_np)

        # Discretize to check timepoints
        time_int, timepoint_map, n_events = discretize_times(time_np, event_np)
        K = len(timepoint_map)

        # --- Event-stratified fold assignment with timepoint coverage ---
        rng = np.random.RandomState(self.random_state)
        fold = self._assign_folds(
            event_np, time_int, K, n, rng,
        )
        self.fold_assignment_ = fold

        # --- Fit full model ---
        full_model = DiscreteSurvival(**self._model_kwargs)
        full_model.fit(X, time, event, sample_weight=sample_weight)
        self.model_ = full_model
        lambda_seq = full_model.lambda_path_
        n_lambda = len(lambda_seq)

        # --- Cross-validation ---
        # Collect per-fold loss on expanded person-period data
        # Loss: binary cross-entropy deviance
        # -2 * [y * log(p) + (1-y) * log(1-p)] per expanded obs
        fold_losses = []

        for fold_idx in range(self.n_folds):
            train_mask = fold != fold_idx
            test_mask = fold == fold_idx

            # Fit on training data
            if isinstance(X, pd.DataFrame):
                X_train = X.iloc[train_mask]
                X_test = X.iloc[test_mask]
            else:
                X_train = X_np[train_mask]
                X_test = X_np[test_mask]
            time_train = time_np[train_mask]
            event_train = event_np[train_mask]
            time_test = time_np[test_mask]
            event_test = event_np[test_mask]

            wt_train = None
            if sample_weight is not None:
                wt_train = np.asarray(sample_weight, dtype=np.float64)[train_mask]

            fold_model = DiscreteSurvival(
                lambda_path=lambda_seq,
                **{k: v for k, v in self._model_kwargs.items()
                   if k not in ('lambda_path', 'n_lambda', 'lambda_min_ratio')}
            )
            fold_model.fit(X_train, time_train, event_train, sample_weight=wt_train)

            # Predict on test: for each lambda, compute loss
            n_fold_lam = len(fold_model.lambda_path_)
            # ISSUE-019 (CV path): use fold_model's fitted mapping.
            time_int_test = np.searchsorted(
                fold_model.timepoint_map_, time_test,
            ) + 1
            K_fold = len(fold_model.timepoint_map_)
            time_int_test = np.clip(time_int_test, 1, K_fold)
            pp = person_period_expand(time_int_test, event_test)
            y_pp = pp['y']
            N_pp = len(y_pp)

            losses = np.full(n_lambda, np.nan)
            if isinstance(X_test, pd.DataFrame):
                X_test_np = X_test.values.astype(np.float64)
            else:
                X_test_np = np.asarray(X_test, dtype=np.float64)

            for l_idx in range(min(n_fold_lam, n_lambda)):
                coef = fold_model.coef_path_[l_idx]
                alpha_bh = fold_model.alpha_path_[l_idx]
                eta_test = X_test_np @ coef

                # Compute predicted hazard on expanded data
                p_hat = predict_discrete_hazard(alpha_bh, eta_test, time_int_test)
                # Clamp
                p_hat = np.clip(p_hat, 1e-5, 1.0 - 1e-5)
                # Binary cross-entropy deviance
                loss = -2.0 * (
                    y_pp * np.log(p_hat)
                    + (1.0 - y_pp) * np.log(1.0 - p_hat)
                )
                losses[l_idx] = np.mean(loss)

            fold_losses.append(losses)

        # --- Aggregate CV results ---
        loss_matrix = np.array(fold_losses)  # (n_folds, n_lambda)

        # Eliminate saturated lambdas (any fold is NaN)
        valid = np.all(np.isfinite(loss_matrix), axis=0)
        if not np.any(valid):
            raise RuntimeError(
                "All lambda values produced non-finite CV losses. "
                "Consider reducing n_lambda or increasing lambda_min_ratio."
            )

        cv_mean = np.full(n_lambda, np.nan)
        cv_se = np.full(n_lambda, np.nan)
        cv_mean[valid] = np.mean(loss_matrix[:, valid], axis=0)
        cv_se[valid] = np.std(loss_matrix[:, valid], axis=0, ddof=1) / np.sqrt(self.n_folds)

        self.cv_mean_ = cv_mean
        self.cv_se_ = cv_se

        # Lambda selection
        valid_idx = np.where(valid)[0]
        min_idx = valid_idx[np.argmin(cv_mean[valid])]
        self.lambda_min_ = lambda_seq[min_idx]

        # REV-012: lambda_min_ and lambda_1se_ are both properties of the CV
        # curve and are reported regardless of se_rule; se_rule only decides
        # which of them the reported model is evaluated at (see the
        # ``rule == '1se'`` branches below).  Previously lambda_1se_ was
        # overwritten with lambda_min_ whenever se_rule='min', so a single fit
        # could not compare the two rules and the attribute contradicted its
        # own docstring.  This matches PenalizedCoxPHCV / PenalizedLogisticCV.
        threshold = cv_mean[min_idx] + cv_se[min_idx]
        # Largest lambda (earliest index) whose mean CV error
        # is within 1 SE of the minimum
        candidates = valid_idx[cv_mean[valid_idx] <= threshold]
        self.lambda_1se_ = lambda_seq[candidates[0]]

        return self

    def _assign_folds(
        self,
        event: np.ndarray,
        time_int: np.ndarray,
        K: int,
        n: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Event-stratified fold assignment with timepoint coverage.

        Ensures every training fold (complement of each test fold)
        contains all K distinct timepoints.  Retries up to
        ``max_fold_retries`` times (following ``grplasso``'s
        ``cv.DiscSurv``).
        """
        for attempt in range(self.max_fold_retries):
            # Event-stratified assignment
            idx_event = np.where(event == 1)[0]
            idx_cens = np.where(event == 0)[0]

            fold = np.empty(n, dtype=np.int64)

            # Assign event observations
            perm_e = rng.permutation(len(idx_event))
            for i, pos in enumerate(perm_e):
                fold[idx_event[pos]] = i % self.n_folds

            # Assign censored observations
            perm_c = rng.permutation(len(idx_cens))
            for i, pos in enumerate(perm_c):
                fold[idx_cens[pos]] = i % self.n_folds

            # Check timepoint coverage in every training fold
            coverage_ok = True
            for f in range(self.n_folds):
                train_mask = fold != f
                n_unique = len(np.unique(time_int[train_mask]))
                if n_unique < K:
                    coverage_ok = False
                    break

            if coverage_ok:
                return fold

        raise RuntimeError(
            f"Could not find a fold assignment where every training fold "
            f"contains all {K} distinct timepoints after "
            f"{self.max_fold_retries} attempts. Consider merging adjacent "
            f"timepoints or reducing n_folds."
        )

    def _check_is_fitted(self):
        if not hasattr(self, 'model_'):
            raise NotFittedError(
                "DiscreteSurvivalCV is not fitted. Call `fit` first."
            )

    def coef_at(self, lambda_value: float) -> np.ndarray:
        """Coefficients at an arbitrary lambda."""
        self._check_is_fitted()
        return self.model_.coef_at(lambda_value)

    def predict(self, X, rule: Optional[str] = None, type: str = 'link'):
        """Predict using the selected lambda.

        Parameters
        ----------
        X : DataFrame or ndarray
        rule : {'min', '1se'} or None, default None
            Which CV lambda to evaluate at.  ``None`` uses the estimator's
            own ``se_rule``, so ``se_rule`` alone decides the reported
            model (REV-012).
        type : str
            ``'link'``.
        """
        self._check_is_fitted()
        rule = self.se_rule if rule is None else rule
        lam = self.lambda_1se_ if rule == '1se' else self.lambda_min_
        return self.model_.predict(X, lambda_value=lam, type=type)

    def summary(self, rule: Optional[str] = None) -> pd.DataFrame:
        """Summary at the selected lambda.

        ``rule=None`` (default) uses the estimator's ``se_rule``.
        """
        self._check_is_fitted()
        rule = self.se_rule if rule is None else rule
        lam = self.lambda_1se_ if rule == '1se' else self.lambda_min_
        idx = np.argmin(np.abs(self.model_.lambda_path_ - lam))
        result = self.model_.summary(which=idx)
        result.attrs['lambda'] = lam
        result.attrs['rule'] = rule
        result.attrs['cv_mean'] = self.cv_mean_[idx]
        result.attrs['cv_se'] = self.cv_se_[idx]
        return result
