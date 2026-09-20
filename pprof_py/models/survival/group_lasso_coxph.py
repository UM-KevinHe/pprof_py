"""User-facing group-lasso penalized Cox estimators: ``GroupLassoCoxPH``
(a regularization path for group / sparse-group lasso) and
``GroupLassoCoxPHCV`` (k-fold cross-validated lambda selection).

Built on the same (Breslow / Efron) partial-likelihood engine as
``CoxPH`` and ``PenalizedCoxPH``.  The numerical work (group
penalties, block coordinate descent, group lambda_max) lives in
``algorithms/penalty.py`` and ``algorithms/coordinate_descent.py``;
this module handles orchestration, validation, and presenting results.

Naming: ``alpha`` here is the *sparse-group-lasso mixing parameter*
(0 = pure group lasso, 1 = standard lasso).  This is conceptually
different from ``PenalizedCoxPH.alpha`` (glmnet elastic-net mixing:
0 = ridge, 1 = lasso), though both happen to be called "alpha".

References
----------
Yuan, M. & Lin, Y. (2006). Model selection and estimation in
regression with grouped variables. *JRSS-B*, 68(1), 49--67.

Simon, N., Friedman, J., Hastie, T. & Tibshirani, R. (2013). A
sparse-group lasso. *JCGS*, 22(2), 231--245.

Shao, Y. & He, K. (2026). grplasso R package.
"""
from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from ...data.survival_validation import validate_fit_inputs
from ...data.survival_data import SurvivalData
from ...algorithms.survival.cox_likelihood import (
    cox_partial_likelihood, precompute_stratum_indices,
)
from ...algorithms.survival.penalty import (
    validate_groups, rescale_group_multipliers, compute_group_indices,
)
from ...algorithms.survival.coordinate_descent import (
    compute_group_lambda_max, build_lambda_sequence,
    fit_group_regularization_path,
)
from ...algorithms.survival.ties import TieMethod
from ...statistics.deviance import (
    saturated_log_likelihood, cox_deviance, deviance_ratio,
)
from .coxph import CoxPH, NotFittedError
from .penalized_coxph import (
    _PenalizedCoxPHBase,
    _PenalizedCoxPHCVBase,
    _resolve_lambda_path,
    DegenerateFeatureWarning,
)
from ...statistics.deviance import bootstrap_cv_se


# ------------------------------------------------------------------
# Group-specific validation
# ------------------------------------------------------------------

def _validate_group_parameters(
    alpha, groups, n_lambda, lambda_min_ratio, lambda_path,
    standardize, orthogonalize, method, use_active_set,
    max_outer_iter, outer_tol, max_inner_iter, inner_tol,
    fit_intercept,
):
    """Validate constructor parameters for GroupLassoCoxPH."""
    if not np.isfinite(alpha) or not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if groups is None:
        raise ValueError("groups must be provided (array of integer group labels)")
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
    if not isinstance(standardize, (bool, np.bool_)):
        raise ValueError("standardize must be a boolean")
    if not isinstance(orthogonalize, (bool, np.bool_)):
        raise ValueError("orthogonalize must be a boolean")
    if method not in ("proximal_newton", "MM"):
        raise ValueError(f"method must be 'proximal_newton' or 'MM', got {method!r}")
    if not isinstance(use_active_set, (bool, np.bool_)):
        raise ValueError("use_active_set must be a boolean")
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


# ------------------------------------------------------------------
# GroupLassoCoxPH
# ------------------------------------------------------------------

class GroupLassoCoxPH(_PenalizedCoxPHBase, BaseEstimator):
    """Group lasso / sparse group lasso penalized Cox regression.

    Fits a regularization path for the penalty:

        lambda * [(1-alpha) * sum_g m_g * ||beta_g||_2
                  + alpha * sum_j pf_j * |beta_j|]

    where ``m_g`` are per-group multipliers (default: ``sqrt(group_size)``)
    and ``pf_j`` are per-feature penalty factors for the L1 part.

    Reuses the exact same (Breslow / Efron) partial-likelihood engine
    as ``CoxPH`` and ``PenalizedCoxPH``; all non-penalization
    capabilities -- strata, offset, sample weights, start-stop data --
    carry over unchanged.

    Parameters
    ----------
    groups : array-like of int, shape (n_features,)
        Group label per feature.  Use 0 for unpenalized features;
        positive integers for penalized groups.  Features sharing a
        label belong to the same group.  Must be contiguous when
        sorted (after internal remapping to 1..G).
    alpha : float, default 0.0
        Sparse group lasso mixing parameter:
        * ``alpha=0.0``: pure group lasso (entire groups in/out).
        * ``0 < alpha < 1``: sparse group lasso (group selection
          plus within-group sparsity).
        * ``alpha=1.0``: standard lasso (groups ignored).
    group_multiplier : array-like or None, default None
        Per-group penalty multiplier ``m_g``.  If ``None``, defaults
        to ``sqrt(group_size)`` (Yuan & Lin 2006).  Shape
        ``(n_penalized_groups,)``.
    penalty_factor : array-like or None, default None
        Per-feature penalty factor for the L1 part.  If ``None``, all
        1s (rescaled to sum to ``n_features``).  Only active when
        ``alpha > 0``.
    n_lambda : int, default 100
    lambda_min_ratio : float or None, default None
        Ratio of lambda_min to lambda_max.  Defaults to 1e-2 if
        ``n < p``, else 1e-4 (glmnet convention).
    lambda_path : array-like or None, default None
        User-supplied lambda sequence.  Overrides ``n_lambda`` and
        ``lambda_min_ratio``.
    standardize : bool, default True
        Standardize features before fitting; coefficients returned on
        original scale.
    orthogonalize : bool, default False
        Within-group SVD orthogonalization (recommended for
        ``method='MM'``).  Currently a placeholder; ``method='MM'``
        is not yet implemented.
    method : str, default 'proximal_newton'
        Optimization method: ``'proximal_newton'`` (exact Hessian) or
        ``'MM'`` (diagonal majorization, not yet implemented).
    ties : str, default 'breslow'
        Tie-handling method: ``'breslow'`` or ``'efron'``.
    use_active_set : bool, default False
        When True, the proximal-Newton inner CD solver uses active-set
        screening to skip groups whose coefficients are zero.  This
        can speed up large problems (many groups, most inactive) but
        may yield slightly different sparsity patterns at intermediate
        lambda values due to proximal-Newton path dependence.
    max_outer_iter : int, default 100
    outer_tol : float, default 1e-9
    max_inner_iter : int, default 1000
    inner_tol : float, default 1e-10
    fit_intercept : bool, default False
        Must be ``False`` (Cox PH has no intercept).

    Attributes (set by ``fit``)
    ---------------------------
    coef_path_ : ndarray, shape (n_lambda_, n_features)
        Coefficient matrix along the regularization path.
    lambda_path_ : ndarray, shape (n_lambda_,)
        Lambda values actually used (descending).
    lambda_max_ : float
    log_likelihood_path_ : ndarray, shape (n_lambda_,)
    deviance_ratio_path_ : ndarray, shape (n_lambda_,)
    n_nonzero_path_ : ndarray of int, shape (n_lambda_,)
    converged_path_ : ndarray of bool, shape (n_lambda_,)
    n_iter_path_ : ndarray of int, shape (n_lambda_,)
    group_norms_ : ndarray, shape (n_lambda_, n_groups)
        ``||beta_g||_2`` per group at each lambda.
    active_groups_ : list of ndarray of bool
        Boolean mask of active groups at each lambda.
    df_path_ : ndarray, shape (n_lambda_,)
        Degrees of freedom (number of nonzero coefficients) per lambda.
    groups_ : ndarray of int, shape (n_features,)
        Canonicalized group labels (after validation).
    group_sizes_ : ndarray of int
    n_groups_ : int
    group_weights_ : ndarray, shape (n_groups_,)
    column_scale_ : ndarray, shape (n_features,)
    """

    def __init__(
        self,
        groups,
        alpha: float = 0.0,
        group_multiplier=None,
        penalty_factor=None,
        n_lambda: int = 100,
        lambda_min_ratio: Optional[float] = None,
        lambda_path=None,
        standardize: bool = True,
        orthogonalize: bool = False,
        method: str = "proximal_newton",
        ties: Union[str, TieMethod] = "breslow",
        use_active_set: bool = False,
        max_outer_iter: int = 100,
        outer_tol: float = 1e-9,
        max_inner_iter: int = 1000,
        inner_tol: float = 1e-10,
        fit_intercept: bool = False,
    ):
        """Group lasso Cox regularization path (R: ``grplasso``)."""
        self.groups = groups
        self.alpha = alpha
        self.group_multiplier = group_multiplier
        self.penalty_factor = penalty_factor
        self.n_lambda = n_lambda
        self.lambda_min_ratio = lambda_min_ratio
        self.lambda_path = lambda_path
        self.standardize = standardize
        self.orthogonalize = orthogonalize
        self.method = method
        self.ties = ties
        self.use_active_set = use_active_set
        self.max_outer_iter = max_outer_iter
        self.outer_tol = outer_tol
        self.max_inner_iter = max_inner_iter
        self.inner_tol = inner_tol
        self.fit_intercept = fit_intercept

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
    ) -> "GroupLassoCoxPH":
        """Fit the group lasso regularization path.

        Parameters
        ----------
        X : DataFrame or ndarray, shape (n_samples, n_features)
        duration : Series or ndarray or None
            Follow-up time (right-censored data).
        event : Series or ndarray
            Event indicator (1 = event, 0 = censored).
        start, stop : Series or ndarray or None
            For start-stop (left-truncated) data.
        strata : Series or ndarray or None
        offset, sample_weight : Series or ndarray or None

        Returns
        -------
        self
        """
        _validate_group_parameters(
            self.alpha, self.groups, self.n_lambda,
            self.lambda_min_ratio, self.lambda_path,
            self.standardize, self.orthogonalize, self.method,
            self.use_active_set, self.max_outer_iter, self.outer_tol,
            self.max_inner_iter, self.inner_tol, self.fit_intercept,
        )
        if self.method == "MM":
            raise NotImplementedError(
                "method='MM' (diagonal majorization) is not yet "
                "implemented; use method='proximal_newton'"
            )

        # --- Common data prep (shared with PenalizedCoxPH) ---
        prep = self._prepare_fit_data(
            X, duration, event, start, stop, strata, offset,
            sample_weight,
        )

        # --- Group validation ---
        groups_raw = np.asarray(self.groups, dtype=np.float64)
        if groups_raw.shape != (prep.p_full,):
            raise ValueError(
                f"groups must have shape ({prep.p_full},), "
                f"got {groups_raw.shape}"
            )
        groups, group_sizes, n_groups = validate_groups(
            groups_raw, prep.p_full,
        )
        group_weights = rescale_group_multipliers(
            self.group_multiplier, group_sizes, n_groups,
        )

        # Remap groups for the reduced feature set (after
        # degenerate-column removal).
        groups_fit = groups[prep.fit_cols]
        groups_fit, group_sizes_fit, n_groups_fit = validate_groups(
            groups_fit, prep.p_fit,
        )
        group_weights_fit = rescale_group_multipliers(
            None if self.group_multiplier is None
            else group_weights,
            group_sizes_fit, n_groups_fit,
        )
        gs_arr, ge_arr = compute_group_indices(
            groups_fit, n_groups_fit,
        )

        # --- Null point for lambda_max ---
        # Group-specific: unpenalized features are those with
        # group label 0 (not pf == 0 as in elastic net).
        beta_null = np.zeros(prep.p_fit)
        always_unpenalized = groups_fit == 0
        if (
            np.any(always_unpenalized)
            and not np.all(always_unpenalized)
        ):
            restricted = CoxPH(
                ties=self.ties,
                max_iter=self.max_outer_iter,
                eps=self.outer_tol,
            )
            restricted.fit(
                prep.X_fit[:, always_unpenalized],
                duration=None, event=prep.data.event,
                start=prep.data.start, stop=prep.data.stop,
                strata=prep.data.strata_codes,
                offset=prep.data.offset,
                sample_weight=prep.data.weight,
            )
            beta_null[always_unpenalized] = restricted.coef_

        _, score_null, _ = prep.objective_fn(beta_null)
        lambda_max = (
            0.0 if np.all(always_unpenalized)
            else compute_group_lambda_max(
                score_null, prep.c, groups_fit,
                group_weights_fit, prep.pf_fit, self.alpha,
            )
        )

        lambda_sequence, lambda_min_ratio = _resolve_lambda_path(
            self.lambda_path, lambda_max, self.lambda_min_ratio,
            self.n_lambda, prep.p_fit, prep.data.n_obs,
            lambda_pad=1e-5,
        )

        # --- Fit the group lasso path ---
        results = fit_group_regularization_path(
            prep.objective_fn, prep.p_fit, prep.c, self.alpha,
            lambda_sequence, groups_fit, group_weights_fit,
            prep.pf_fit, n_groups_fit,
            beta_warm_start=beta_null,
            outer_max_iter=self.max_outer_iter,
            outer_tol=self.outer_tol,
            inner_max_iter=self.max_inner_iter,
            inner_tol=self.inner_tol,
            use_active_set=self.use_active_set,
        )

        # --- Common result storage (from base) ---
        self._store_path_results(
            results, prep, lambda_sequence, lambda_max,
            lambda_min_ratio,
        )

        # --- Group-specific result attributes ---
        self.group_norms_ = np.array(
            [r.group_norms for r in results],
        )
        self.active_groups_ = [r.active_groups for r in results]
        self.df_path_ = np.array([r.df for r in results])
        self.groups_ = groups
        self.group_sizes_ = group_sizes
        self.n_groups_ = n_groups
        self.group_weights_ = group_weights

        return self

    # ------------------------------------------------------------------
    # Group-specific queries
    # ------------------------------------------------------------------
    def active_group_labels(
        self, lambda_value: Optional[float] = None,
        which: Optional[int] = None,
    ) -> np.ndarray:
        """Return labels of active (nonzero-norm) groups at a given lambda.

        Parameters
        ----------
        lambda_value : float or None
            Query lambda (nearest grid point).  Ignored if ``which``
            is given.
        which : int or None
            Direct index into ``lambda_path_``.
        """
        self._check_is_fitted()
        if which is not None:
            idx = int(which)
        elif lambda_value is not None:
            idx = int(
                np.argmin(np.abs(self.lambda_path_ - lambda_value))
            )
        elif hasattr(self, "lambda_"):
            idx = 0
        else:
            raise ValueError(
                "Specify lambda_value or which for multi-lambda fits"
            )
        active = self.active_groups_[idx]
        return np.arange(1, self.n_groups_ + 1)[active]

    # Override base summary to include group-specific columns.
    def summary(self) -> pd.DataFrame:
        """Per-lambda path summary with group-specific columns."""
        self._check_is_fitted()
        return pd.DataFrame({
            "lambda": self.lambda_path_,
            "n_nonzero": self.n_nonzero_path_,
            "n_active_groups": [
                int(np.sum(ag)) for ag in self.active_groups_
            ],
            "deviance_ratio": self.deviance_ratio_path_,
            "log_likelihood": self.log_likelihood_path_,
            "converged": self.converged_path_,
        })

    # _check_is_fitted, coef_at, nonzero_features, _resolve_coef,
    # predict_linear, predict_partial_hazard, predict are inherited
    # from _PenalizedCoxPHBase.


# ------------------------------------------------------------------
# GroupLassoCoxPHCV
# ------------------------------------------------------------------

class GroupLassoCoxPHCV(_PenalizedCoxPHCVBase, BaseEstimator):
    """Cross-validated group lasso Cox regression.

    Performs k-fold cross-validation over the lambda path to select
    the optimal regularization strength.

    Improvements over ``PenalizedCoxPHCV``:

    * Event-stratified fold assignment (from ``grplasso``).
    * Saturated-lambda elimination (discard lambdas where any fold
      produced non-finite deviance).
    * Strata coverage check (warn when any stratum is missing from a
      training fold).

    Parameters
    ----------
    groups, alpha, group_multiplier, penalty_factor, ...:
        Same as ``GroupLassoCoxPH``.
    n_folds : int, default 10
    fold_id : array-like or None, default None
        Explicit fold assignment (0..K-1 or 1..K).
    se_method : str, default 'analytical'
        ``'analytical'`` (glmnet-style weighted CV SE) or
        ``'bootstrap'`` (Breslow ties only).
    random_state : int or None, default None
    select : str, default 'lambda_min'
        ``'lambda_min'`` or ``'lambda_1se'``.

    Attributes (set by ``fit``)
    ---------------------------
    lambda_path_ : ndarray, shape (n_lambda_,)
    cv_mean_deviance_ : ndarray, shape (n_lambda_,)
    cv_se_deviance_ : ndarray, shape (n_lambda_,)
    lambda_min_ : float
    lambda_1se_ : float
    final_estimator_ : GroupLassoCoxPH
    coef_ : ndarray
    fold_id_ : ndarray, shape (n_obs,)
    full_fit_ : GroupLassoCoxPH
    """

    def __init__(
        self,
        groups,
        alpha: float = 0.0,
        group_multiplier=None,
        penalty_factor=None,
        n_lambda: int = 100,
        lambda_min_ratio: Optional[float] = None,
        lambda_path=None,
        standardize: bool = True,
        orthogonalize: bool = False,
        method: str = "proximal_newton",
        ties: Union[str, TieMethod] = "breslow",
        use_active_set: bool = False,
        max_outer_iter: int = 100,
        outer_tol: float = 1e-9,
        max_inner_iter: int = 1000,
        inner_tol: float = 1e-10,
        fit_intercept: bool = False,
        n_folds: int = 10,
        fold_id=None,
        se_method: str = "analytical",
        n_bootstrap: int = 100,
        random_state: Optional[int] = None,
        select: str = "lambda_min",
    ):
        """Cross-validated group lasso Cox."""
        self.groups = groups
        self.alpha = alpha
        self.group_multiplier = group_multiplier
        self.penalty_factor = penalty_factor
        self.n_lambda = n_lambda
        self.lambda_min_ratio = lambda_min_ratio
        self.lambda_path = lambda_path
        self.standardize = standardize
        self.orthogonalize = orthogonalize
        self.method = method
        self.ties = ties
        self.use_active_set = use_active_set
        self.max_outer_iter = max_outer_iter
        self.outer_tol = outer_tol
        self.max_inner_iter = max_inner_iter
        self.inner_tol = inner_tol
        self.fit_intercept = fit_intercept
        self.n_folds = n_folds
        self.fold_id = fold_id
        self.se_method = se_method
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.select = select

    def _base_kwargs(self) -> dict:
        """Constructor kwargs forwarded to ``GroupLassoCoxPH``."""
        return dict(
            groups=self.groups, alpha=self.alpha,
            group_multiplier=self.group_multiplier,
            penalty_factor=self.penalty_factor,
            standardize=self.standardize,
            orthogonalize=self.orthogonalize,
            method=self.method, ties=self.ties,
            use_active_set=self.use_active_set,
            fit_intercept=self.fit_intercept,
            max_outer_iter=self.max_outer_iter,
            outer_tol=self.outer_tol,
            max_inner_iter=self.max_inner_iter,
            inner_tol=self.inner_tol,
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
    ) -> "GroupLassoCoxPHCV":
        """Fit the full path, then cross-validate to select lambda."""
        if self.select not in ("lambda_min", "lambda_1se"):
            raise ValueError(
                f"select must be 'lambda_min' or 'lambda_1se', "
                f"got {self.select!r}"
            )
        if self.se_method not in ("analytical", "bootstrap"):
            raise ValueError(
                f"se_method must be 'analytical' or 'bootstrap', "
                f"got {self.se_method!r}"
            )
        if self.se_method == "bootstrap":
            ties_name = (
                self.ties
                if isinstance(self.ties, str)
                else str(self.ties).lower()
            )
            if ties_name != "breslow":
                raise NotImplementedError(
                    "se_method='bootstrap' only supports "
                    "ties='breslow'"
                )
        if self.n_lambda < 2 and self.lambda_path is None:
            raise ValueError(
                "n_lambda must be >= 2 for cross-validation"
            )
        if self.lambda_path is not None and np.atleast_1d(self.lambda_path).size < 2:
            raise ValueError(
                "lambda_path must contain >= 2 values for cross-validation"
            )
        if isinstance(self.n_folds, (bool, np.bool_)) or int(self.n_folds) != self.n_folds or int(self.n_folds) < 3:
            raise ValueError("n_folds must be an integer >= 3")

        # --- Data ---
        clean = validate_fit_inputs(
            X, duration=duration, event=event, start=start, stop=stop,
            strata=strata, offset=offset, sample_weight=sample_weight,
        )
        data = SurvivalData(**clean)
        n = data.n_obs

        # --- Full-data fit ---
        full_fit = GroupLassoCoxPH(
            n_lambda=self.n_lambda,
            lambda_min_ratio=self.lambda_min_ratio,
            lambda_path=self.lambda_path,
            **self._base_kwargs(),
        )
        full_fit.fit(
            data.X, event=data.event, start=data.start,
            stop=data.stop, strata=data.strata_codes,
            offset=data.offset, sample_weight=data.weight,
        )
        lambda_grid = full_fit.lambda_path_
        n_lam = len(lambda_grid)

        # --- Event-stratified fold assignment (from base) ---
        fold_id, fold_labels = self._assign_folds(
            data.event, n, self.n_folds, self.fold_id,
            self.random_state,
        )
        self.fold_id_ = fold_id

        # --- Strata coverage check ---
        unique_strata = np.unique(data.strata_codes)
        for fold in fold_labels:
            train_strata = np.unique(data.strata_codes[fold_id != fold])
            if len(train_strata) < len(unique_strata):
                missing = set(unique_strata) - set(train_strata)
                warnings.warn(
                    f"Fold {fold}: strata {missing} are absent from "
                    f"the training set. Results may be unreliable.",
                    stacklevel=2,
                )

        # --- Per-fold deviance ---
        cvraw = np.full(
            (len(fold_labels), n_lam), np.nan, dtype=np.float64,
        )
        fold_event_weight = np.zeros(
            len(fold_labels), dtype=np.float64,
        )
        # Out-of-fold eta for bootstrap SE (§4.3.2).
        use_bootstrap = self.se_method == "bootstrap"
        eta_hat = (
            np.full((n, n_lam), np.nan, dtype=np.float64)
            if use_bootstrap
            else None
        )

        lsat_full = saturated_log_likelihood(
            data.stop, data.event, data.weight, data.strata_codes,
        )
        _si_full = precompute_stratum_indices(data.strata_codes)

        for i, fold in enumerate(fold_labels):
            held_out = fold_id == fold
            train = ~held_out
            fold_event_weight[i] = float(
                np.sum(data.weight[held_out] * data.event[held_out])
            )
            if fold_event_weight[i] <= 0.0:
                raise ValueError(
                    f"CV fold {fold!r} contains no positive event weight"
                )

            fold_fit = GroupLassoCoxPH(
                lambda_path=lambda_grid, **self._base_kwargs(),
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=DegenerateFeatureWarning,
                )
                fold_fit.fit(
                    data.X[train], event=data.event[train],
                    start=data.start[train], stop=data.stop[train],
                    strata=data.strata_codes[train],
                    offset=data.offset[train],
                    sample_weight=data.weight[train],
                )

            if not np.all(fold_fit.converged_path_):
                bad = np.flatnonzero(~fold_fit.converged_path_)
                warnings.warn(
                    f"Group lasso fold {fold!r} did not converge at "
                    f"lambda indices {bad.tolist()}",
                    stacklevel=2,
                )

            lsat_train = saturated_log_likelihood(
                data.stop[train], data.event[train],
                data.weight[train], data.strata_codes[train],
            )
            _si_train = precompute_stratum_indices(
                data.strata_codes[train],
            )
            for j in range(n_lam):
                beta_j = fold_fit.coef_path_[j]
                loglik_full, _, _ = cox_partial_likelihood(
                    data.X, data.start, data.stop, data.event, beta_j,
                    offset=data.offset, weight=data.weight,
                    strata=data.strata_codes, ties=self.ties,
                    stratum_indices=_si_full,
                )
                loglik_train, _, _ = cox_partial_likelihood(
                    data.X[train], data.start[train],
                    data.stop[train], data.event[train], beta_j,
                    offset=data.offset[train], weight=data.weight[train],
                    strata=data.strata_codes[train], ties=self.ties,
                    stratum_indices=_si_train,
                )
                dev_full = cox_deviance(loglik_full, lsat_full)
                dev_train = cox_deviance(loglik_train, lsat_train)
                raw = dev_full - dev_train
                # Saturated-lambda elimination: if non-finite, leave NaN.
                if np.isfinite(raw):
                    cvraw[i, j] = raw

            cvraw[i, :] /= fold_event_weight[i]

            # Collect out-of-fold linear predictors for bootstrap SE.
            if use_bootstrap:
                for j in range(n_lam):
                    beta_j = fold_fit.coef_path_[j]
                    eta_hat[held_out, j] = (
                        data.X[held_out] @ beta_j
                        + data.offset[held_out]
                    )

        # --- CV statistics with saturated-lambda elimination ---
        cvm, cvsd, n_folds_used = self._compute_cv_statistics(
            cvraw, fold_event_weight,
        )

        # Bootstrap SE override (§4.3.2).
        if use_bootstrap:
            cvsd_boot = bootstrap_cv_se(
                eta_hat, data.start, data.stop, data.event,
                data.weight, data.strata_codes,
                n_bootstrap=self.n_bootstrap,
                random_state=self.random_state,
                ties=self.ties,
            )
            valid = np.isfinite(cvm)
            cvsd[valid] = cvsd_boot[valid]

        self.lambda_path_ = lambda_grid
        self.cv_mean_deviance_ = cvm
        self.cv_se_deviance_ = cvsd
        self.cv_n_folds_ = n_folds_used

        # --- Lambda selection (min + 1-SE rule) ---
        self.lambda_min_, self.lambda_1se_ = self._select_lambda(
            lambda_grid, cvm, cvsd,
        )

        # --- Final estimator at selected lambda ---
        chosen_lambda = (
            self.lambda_min_ if self.select == "lambda_min"
            else self.lambda_1se_
        )
        self.final_estimator_ = GroupLassoCoxPH(
            lambda_path=chosen_lambda, **self._base_kwargs(),
        )
        self.final_estimator_.fit(
            data.X, event=data.event, start=data.start,
            stop=data.stop, strata=data.strata_codes,
            offset=data.offset, sample_weight=data.weight,
        )
        self.coef_ = self.final_estimator_.coef_
        self.n_obs_ = data.n_obs
        self.n_events_ = int(np.sum(data.event))
        self.feature_names_in_ = full_fit.feature_names_in_
        self.n_nonzero_path_ = full_fit.n_nonzero_path_
        self.full_fit_ = full_fit

        return self

    # _check_is_fitted, predict_linear, predict_partial_hazard,
    # predict, summary are inherited from _PenalizedCoxPHCVBase.
