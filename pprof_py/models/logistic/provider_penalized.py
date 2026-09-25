"""Two-layer provider-penalized logistic regression.

Matches R ``grplasso::pp.lasso()`` for binomial family.  The two-layer
architecture: unpenalized provider gamma_k + penalized covariate beta.
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
    validate_groups,
    rescale_group_multipliers,
    compute_group_indices,
    elastic_net_penalty_value,
    sparse_group_lasso_penalty_value,
)
from ...algorithms.coordinate_descent import (
    compute_lambda_max,
    compute_group_lambda_max,
    build_lambda_sequence,
    solve_penalized_quadratic,
    solve_sparse_group_penalized_quadratic,
    PenalizedFitResult,
    GroupPenalizedFitResult,
)
from ...algorithms.logistic.likelihood import (
    logistic_loglik,
    logistic_score,
    logistic_information,
    logistic_deviance,
    logistic_null_deviance,
    logistic_null_score,
    logistic_unpenalized_null_fit,
    logistic_intercept_update,
    _safe_expit,
)
from ...algorithms.logistic.provider_effects import (
    compute_provider_indices,
    logistic_provider_newton_step,
    provider_bound_clamp,
)
from ...exceptions import NotFittedError
from .penalized import (
    DegenerateFeatureWarning, _resolve_lambda_path,
    _stratified_fold_assignment,
)

logger = logging.getLogger(__name__)


def _resolve_provider_alias(provider_id, provider):
    """Resolve ``provider_id`` / ``provider`` alias pair.

    Both names refer to the same concept (per-observation provider
    identifiers).  Accepts either one; raises if both are supplied.
    """
    if provider_id is not None and provider is not None:
        raise ValueError(
            "Cannot specify both 'provider_id' and 'provider'; they "
            "are aliases for the same argument."
        )
    return provider_id if provider_id is not None else provider


class ProviderPenalizedLogistic(ProviderModel):
    """Two-layer provider-penalized logistic regression.

    Unpenalized provider effects gamma_k (bounded by median-clamp)
    combined with penalized covariate effects beta (elastic net,
    group lasso, or sparse group lasso).

    Matches R ``pp.lasso`` for binomial family.

    Parameters
    ----------
    penalty_type : str, default='elastic_net'
        One of 'elastic_net', 'group_lasso', 'sparse_group_lasso'.
    alpha : float, default=1.0
        Elastic net mixing / sparse group mixing.
    groups : array-like or None
        Group labels (required for group_lasso/sparse_group_lasso).
    group_multiplier : array-like or None
    gamma_bound : float, default=10.0
        Maximum provider-effect deviation from median.  Alias:
        ``provider_bound`` (accepted for cross-family consistency
        with ``ProviderPenalizedCoxPH``).
    n_lambda : int, default=100
    lambda_min_ratio : float or None
    lambda_path : array-like or None
    penalty_factor : array-like or None
    standardize : bool, default=True
    fit_intercept : bool, default=True
    max_outer_iter : int, default=100
    outer_tol : float, default=1e-7
        Convergence tolerance for the outer β solver loop.  Provider
        models use 1e-7 (looser than the 1e-9 default in non-provider
        penalized models) because the two-layer γ–β alternation
        provides additional implicit convergence pressure.
    max_inner_iter : int, default=1000
    inner_tol : float, default=1e-10
    provider_max_iter : int, default=10
        Maximum provider-effect Newton steps per outer iteration.
        Alias: ``max_provider_iter``.
    provider_tol : float or None, default=None
        Convergence tolerance for the provider-effect Newton loop.
        When *None*, falls back to ``outer_tol``.  Matches the
        dedicated ``provider_tol`` parameter on
        ``ProviderPenalizedCoxPH``.

    Attributes (after fit)
    ----------------------
    coef_path_ : ndarray, shape (n_lambda, p)
    intercept_path_ : ndarray, shape (n_lambda,)
    gamma_path_ : ndarray, shape (n_lambda, K)
    lambda_path_ : ndarray, shape (n_lambda,)
    provider_labels_ : ndarray, shape (K,)
    """

    def __init__(
        self,
        penalty_type: str = "elastic_net",
        alpha: float = 1.0,
        groups: Optional[np.ndarray] = None,
        group_multiplier: Optional[np.ndarray] = None,
        gamma_bound: float = 10.0,
        n_lambda: int = 100,
        lambda_min_ratio: Optional[float] = None,
        lambda_path: Optional[np.ndarray] = None,
        penalty_factor: Optional[np.ndarray] = None,
        standardize: bool = True,
        fit_intercept: bool = True,
        max_outer_iter: int = 100,
        outer_tol: float = 1e-7,
        max_inner_iter: int = 1000,
        inner_tol: float = 1e-10,
        provider_max_iter: int = 10,
        # --- Cross-family aliases (ISSUE-016, -017, -018) ---
        provider_bound: Optional[float] = None,
        max_provider_iter: Optional[int] = None,
        provider_tol: Optional[float] = None,
    ):
        """Two-layer provider + penalized-covariate logistic."""
        self.penalty_type = penalty_type
        self.alpha = alpha
        self.groups = groups
        self.group_multiplier = group_multiplier
        # ISSUE-016: accept provider_bound as alias for gamma_bound
        self.gamma_bound = (
            provider_bound if provider_bound is not None else gamma_bound
        )
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
        # ISSUE-017: accept max_provider_iter as alias for provider_max_iter
        self.provider_max_iter = (
            max_provider_iter if max_provider_iter is not None
            else provider_max_iter
        )
        # ISSUE-018: dedicated provider-effect convergence tolerance
        self.provider_tol = provider_tol

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        provider_id: np.ndarray = None,
        sample_weight: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
        *,
        provider: Optional[np.ndarray] = None,
    ) -> "ProviderPenalizedLogistic":
        """Fit the two-layer provider-penalized logistic model.

        Parameters
        ----------
        X : ndarray or DataFrame, shape (n, p)
        y : ndarray, shape (n,)
        provider_id : ndarray, shape (n,)
            Provider identifiers.  Alias: ``provider``.
        sample_weight, offset : ndarray or None
        provider : ndarray or None
            Alias for *provider_id* (cross-family consistency with
            ``ProviderPenalizedCoxPH``).

        Returns
        -------
        self
        """
        provider_id = _resolve_provider_alias(provider_id, provider)
        if provider_id is None:
            raise ValueError(
                "provider_id (or provider=) must be provided "
                "(per-observation provider identifiers)."
            )
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
        else:
            offset = np.zeros(n, dtype=np.float64)

        # Provider indexing.
        prov_idx, prov_labels, n_providers = compute_provider_indices(
            np.asarray(provider_id),
        )
        self.provider_labels_ = prov_labels
        self.n_providers_ = n_providers

        # Column standardization.
        # Center as well as scale when an intercept is fitted: with a free
        # intercept this is an exact reparameterization (lambda_max and the
        # coefficient path are invariant) that removes the information-
        # diagonal inflation an uncentered large-mean/sd column causes.
        # The correction goes to the scalar intercept only, leaving the
        # provider effects gamma -- the model's actual deliverable -- alone.
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

        # Penalty factors.
        pf_input = self.penalty_factor
        if pf_input is not None:
            pf_full = np.asarray(pf_input, dtype=np.float64)
        else:
            pf_full = np.ones(p_full, dtype=np.float64)
        pf_fit = rescale_penalty_factors(pf_full[fit_cols], p_fit)

        # Group setup (for group/sparse group lasso penalty types).
        use_groups = self.penalty_type in ("group_lasso", "sparse_group_lasso")
        if use_groups:
            if self.groups is None:
                raise ValueError(
                    f"groups must be provided for penalty_type='{self.penalty_type}'"
                )
            groups_full = np.asarray(self.groups)
            groups_fit = groups_full[fit_cols]
            groups_fit, group_sizes_fit, n_groups_fit = validate_groups(
                groups_fit, p_fit,
            )
            gw_fit = rescale_group_multipliers(
                None, group_sizes_fit, n_groups_fit,
            )
            gs_arr, ge_arr = compute_group_indices(groups_fit, n_groups_fit)
        else:
            groups_fit = None
            n_groups_fit = 0
            gw_fit = None
            gs_arr = ge_arr = None

        c = 1.0 / float(np.sum(weight))

        # REV-001: the null point for lambda_max is not beta=0 everywhere --
        # it is "penalized coefficients at 0, unpenalized ones at their own
        # MLE".  Fitting them here corrects lambda_max and gives the path a
        # warm start whose unpenalized coefficients are already right at the
        # top of the path (mirrors the R reference's SerBIN.residuals).
        unpen_mask = (pf_fit == 0.0)
        if use_groups:
            unpen_mask = unpen_mask | (groups_fit == 0)
        beta_null, score_null, intercept_null = logistic_unpenalized_null_fit(
            X_fit, y, weight, unpenalized=unpen_mask, offset=offset,
            fit_intercept=self.fit_intercept,
        )

        if use_groups:
            lam_max = compute_group_lambda_max(
                score_null, c, groups_fit, gw_fit, pf_fit, self.alpha,
            )
        else:
            lam_max = compute_lambda_max(
                score_null, c, pf_fit, self.alpha,
            )

        lambda_sequence, lam_min_ratio = _resolve_lambda_path(
            self.lambda_path, lam_max, self.lambda_min_ratio,
            self.n_lambda, p_fit, n,
        )

        # --- Two-layer path fitting ---
        intercept = intercept_null if self.fit_intercept else 0.0
        beta = beta_null.copy()
        gamma = np.zeros(n_providers, dtype=np.float64)

        coef_results = []
        intercept_results = []
        gamma_results = []
        loglik_results = []
        converged_results = []
        n_iter_results = []

        for lam_val in lambda_sequence:
            lam = float(lam_val)
            converged = False
            n_outer = 0

            for outer_iter in range(1, self.max_outer_iter + 1):
                n_outer = outer_iter

                # Step 1: Provider-effect Newton steps.
                for _prov_iter in range(self.provider_max_iter):
                    eta = X_fit @ beta + gamma[prov_idx] + offset + intercept
                    gamma_new = logistic_provider_newton_step(
                        y, eta, weight, prov_idx, n_providers,
                        gamma, gamma_bound=self.gamma_bound,
                    )
                    gamma_change = float(np.max(np.abs(gamma_new - gamma)))
                    gamma = gamma_new
                    _ptol = (
                        self.provider_tol if self.provider_tol is not None
                        else self.outer_tol
                    )
                    if gamma_change < _ptol:
                        break

                # Update intercept.
                if self.fit_intercept:
                    eta = X_fit @ beta + gamma[prov_idx] + offset + intercept
                    from ...algorithms.logistic.likelihood import (
                        logistic_intercept_update,
                    )
                    intercept += logistic_intercept_update(y, eta, weight)

                # Step 2: Build objective for beta (with fixed gamma, intercept).
                eta_base = gamma[prov_idx] + offset + intercept

                def objective_fn(b):  # noqa: D401
                    """Log-likelihood, score, and information at *b* (fixed gamma)."""
                    eta = X_fit @ b + eta_base
                    ll = logistic_loglik(y, eta, weight)
                    sc = logistic_score(X_fit, y, eta, weight)
                    info = logistic_information(X_fit, eta, weight)
                    return ll, sc, info

                loglik, score, info = objective_fn(beta)
                A = c * info
                g_score = c * score
                linear_term = A @ beta + g_score

                # Step 3: Solve penalized WLS for beta.
                if use_groups:
                    beta_new, n_inner, _ = \
                        solve_sparse_group_penalized_quadratic(
                            A, linear_term, beta, lam, self.alpha,
                            pf_fit, gs_arr, ge_arr, gw_fit,
                            tol=self.inner_tol,
                            max_iter=self.max_inner_iter,
                        )
                else:
                    beta_new, n_inner, _ = solve_penalized_quadratic(
                        A, linear_term, beta, lam, self.alpha, pf_fit,
                        tol=self.inner_tol, max_iter=self.max_inner_iter,
                    )

                # Step 4: Convergence check.
                beta_change = float(
                    np.max(np.abs(beta_new - beta) / np.maximum(1.0, np.abs(beta)))
                )
                beta = beta_new

                if beta_change < self.outer_tol:
                    converged = True
                    break

            # Store results for this lambda.
            eta_final = X_fit @ beta + gamma[prov_idx] + offset + intercept
            ll_final = logistic_loglik(y, eta_final, weight)

            coef_results.append(beta / xs_full[fit_cols])
            intercept_results.append(intercept)
            gamma_results.append(gamma.copy())
            loglik_results.append(ll_final)
            converged_results.append(converged)
            n_iter_results.append(n_outer)

        # --- Store fitted attributes ---
        n_lam = len(lambda_sequence)
        coef_path = np.zeros((n_lam, p_full))
        for i, coef_fit in enumerate(coef_results):
            coef_path[i, fit_cols] = coef_fit

        # Intercept back-transform into the original units of X.  gamma is
        # untouched: the whole shift is absorbed by the scalar intercept.
        intercept_path = np.array(intercept_results, dtype=np.float64) - (
            coef_path @ xm_full
        )
        intercept_results = list(intercept_path)

        self.coef_path_ = coef_path
        self.intercept_path_ = intercept_path
        self.gamma_path_ = np.array(gamma_results)
        self.lambda_path_ = np.asarray(lambda_sequence, dtype=np.float64)
        self.lambda_max_ = float(lam_max)
        self.lambda_min_ratio_ = float(lam_min_ratio)
        self.log_likelihood_path_ = np.array(loglik_results)
        self.converged_path_ = np.array(converged_results)
        self.n_iter_path_ = np.array(n_iter_results)
        self.n_nonzero_path_ = np.array(
            [int(np.sum(row != 0.0)) for row in coef_path]
        )
        self.column_scale_ = xs_full
        self.column_center_ = xm_full
        self.penalty_factor_ = pf_full
        self._excluded_features_ = degenerate
        self._provider_idx_ = prov_idx

        null_dev = logistic_null_deviance(y, weight)
        deviances = np.array([-2.0 * ll for ll in loglik_results])
        self.deviance_path_ = deviances
        self.null_deviance_ = null_dev
        self.deviance_ratio_path_ = np.where(
            null_dev > 0, 1.0 - deviances / null_dev, 0.0,
        )

        if n_lam == 1:
            self.coef_ = coef_path[0]
            self.intercept_ = intercept_results[0]
            self.gamma_ = gamma_results[0]

        return self

    def _check_is_fitted(self):
        if not hasattr(self, "coef_path_"):
            raise NotFittedError(
                f"This {type(self).__name__} is not fitted yet."
            )

    def predict_provider_effect(self, which: int = -1) -> pd.DataFrame:
        """Provider effects at a given lambda index.

        Parameters
        ----------
        which : int

        Returns
        -------
        DataFrame with columns 'provider' and 'gamma'.
        """
        self._check_is_fitted()
        gamma = self.gamma_path_[which]
        return pd.DataFrame({
            "provider": self.provider_labels_,
            "gamma": gamma,
        })

    def predict_proba(
        self, X, provider_id=None, lambda_value=None, which: int = -1,
        *, provider=None,
    ):
        """Predicted probabilities.

        Parameters
        ----------
        X : ndarray, shape (n_new, p)
        provider_id : ndarray or None, shape (n_new,)
            Alias: ``provider``.
        lambda_value : float or None
            If given, selects the nearest ``lambda_path_`` entry
            (overrides *which* when *which* is at its default).
        which : int
            Integer index into the path.

        Returns
        -------
        ndarray, shape (n_new,)
        """
        self._check_is_fitted()
        provider_id = _resolve_provider_alias(provider_id, provider)
        # ISSUE-015: wire lambda_value to a which index.
        if lambda_value is not None and which == -1:
            which = int(
                np.argmin(np.abs(self.lambda_path_ - lambda_value))
            )
        X = np.asarray(X, dtype=np.float64)
        coef = self.coef_path_[which]
        intercept = self.intercept_path_[which]
        eta = X @ coef + intercept

        if provider_id is not None:
            gamma = self.gamma_path_[which]
            # Map provider_id to indices.
            label_to_idx = {
                lab: i for i, lab in enumerate(self.provider_labels_)
            }
            prov_idx = np.array(
                [label_to_idx.get(pid, -1) for pid in provider_id]
            )
            valid = prov_idx >= 0
            eta[valid] += gamma[prov_idx[valid]]

        return 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))

    def predict(self, X, provider_id=None, lambda_value=None,
                which=-1, threshold=0.5):
        """Binary predictions."""
        return (
            self.predict_proba(X, provider_id, lambda_value, which)
            >= threshold
        ).astype(int)


# ======================================================================
# Provider-aware fold assignment
# ======================================================================

def _provider_stratified_fold_assignment(
    y: np.ndarray,
    provider_id: np.ndarray,
    n_folds: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Provider-aware event-stratified CV fold assignment.

    All observations from the same provider are assigned to the same
    fold.  Providers are sorted by size (largest first) and distributed
    round-robin across folds to achieve approximate balance, then
    shuffled randomly within each fold-assignment batch.

    If fewer providers than folds exist, falls back to observation-level
    stratified assignment (same as ``PenalizedLogisticCV``).

    Parameters
    ----------
    y : ndarray, shape (n,)
        Binary outcome.
    provider_id : ndarray, shape (n,)
        Provider labels.
    n_folds : int
    random_state : int or None

    Returns
    -------
    fold : ndarray of int, shape (n,)
        Fold assignment per observation (0-indexed).
    """
    rng = np.random.RandomState(random_state)
    n = len(y)

    # Unique providers and their observation counts.
    unique_provs, inverse = np.unique(provider_id, return_inverse=True)
    n_provs = len(unique_provs)

    if n_provs < n_folds:
        logger.warning(
            "Fewer providers (%d) than folds (%d); falling back to "
            "observation-level stratified folds.",
            n_provs, n_folds,
        )
        return _stratified_fold_assignment(y, n_folds, random_state)

    # Compute provider-level event rate for stratified assignment.
    prov_sizes = np.bincount(inverse)
    prov_events = np.bincount(inverse, weights=y)
    prov_rate = prov_events / np.maximum(prov_sizes, 1)

    # Sort providers: high event-rate first, then by size descending.
    # This helps balance outcome prevalence across folds.
    sort_key = -(prov_rate * 1000 + prov_sizes)  # composite key
    sorted_prov_idx = np.argsort(sort_key)

    # Round-robin assignment with random permutation within chunks.
    prov_fold = np.empty(n_provs, dtype=np.intp)
    n_chunks = (n_provs + n_folds - 1) // n_folds
    for chunk in range(n_chunks):
        start = chunk * n_folds
        end = min(start + n_folds, n_provs)
        chunk_size = end - start
        prov_fold[sorted_prov_idx[start:end]] = rng.permutation(chunk_size)

    # Map provider fold to observation fold.
    fold = prov_fold[inverse]
    return fold


# ======================================================================
# ProviderPenalizedLogisticCV
# ======================================================================

class ProviderPenalizedLogisticCV(ProviderModel):
    """Cross-validated two-layer provider-penalized logistic regression.

    Fits the full regularization path on all data, then selects
    lambda.min or lambda.1se via k-fold cross-validation on binomial
    deviance.  Folds are assigned at the **provider level** so that
    all observations from a provider fall into the same fold,
    preserving the two-layer structure.

    Parameters
    ----------
    penalty_type : str, default='elastic_net'
        One of 'elastic_net', 'group_lasso', 'sparse_group_lasso'.
    alpha : float, default=1.0
    groups : array-like or None
    group_multiplier : array-like or None
    gamma_bound : float, default=10.0
    n_lambda : int, default=100
    lambda_min_ratio : float or None
    lambda_path : array-like or None
    penalty_factor : array-like or None
    standardize : bool, default=True
    fit_intercept : bool, default=True
    max_outer_iter : int, default=100
    outer_tol : float, default=1e-7
        See ``ProviderPenalizedLogistic`` for rationale.
    max_inner_iter : int, default=1000
    inner_tol : float, default=1e-10
    provider_max_iter : int, default=10
    n_folds : int, default=10
    fold_id : array-like or None
        User-supplied fold assignments (overrides n_folds).
    use_1se : bool, default=True
        If True, select lambda.1se; else lambda.min.
    random_state : int or None

    Attributes (after fit)
    ----------------------
    lambda_min_ : float
    lambda_1se_ : float
    lambda_ : float
        Selected lambda.
    cv_mean_deviance_ : ndarray, shape (n_lambda,)
    cv_se_deviance_ : ndarray, shape (n_lambda,)
    model_ : ProviderPenalizedLogistic
        Full-data model.
    coef_ : ndarray, shape (p,)
        Coefficients at the selected lambda.
    intercept_ : float
    gamma_ : ndarray, shape (K,)
        Provider effects at the selected lambda.
    """

    def __init__(
        self,
        penalty_type: str = "elastic_net",
        alpha: float = 1.0,
        groups: Optional[np.ndarray] = None,
        group_multiplier: Optional[np.ndarray] = None,
        gamma_bound: float = 10.0,
        n_lambda: int = 100,
        lambda_min_ratio: Optional[float] = None,
        lambda_path: Optional[np.ndarray] = None,
        penalty_factor: Optional[np.ndarray] = None,
        standardize: bool = True,
        fit_intercept: bool = True,
        max_outer_iter: int = 100,
        outer_tol: float = 1e-7,
        max_inner_iter: int = 1000,
        inner_tol: float = 1e-10,
        provider_max_iter: int = 10,
        n_folds: int = 10,
        fold_id: Optional[np.ndarray] = None,
        use_1se: bool = True,
        random_state: Optional[int] = None,
        # --- Cross-family aliases (ISSUE-016, -017, -018) ---
        provider_bound: Optional[float] = None,
        max_provider_iter: Optional[int] = None,
        provider_tol: Optional[float] = None,
    ):
        """Cross-validated provider-penalized logistic."""
        self.penalty_type = penalty_type
        self.alpha = alpha
        self.groups = groups
        self.group_multiplier = group_multiplier
        self.gamma_bound = (
            provider_bound if provider_bound is not None else gamma_bound
        )
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
        self.provider_max_iter = (
            max_provider_iter if max_provider_iter is not None
            else provider_max_iter
        )
        self.n_folds = n_folds
        self.fold_id = fold_id
        self.use_1se = use_1se
        self.random_state = random_state
        self.provider_tol = provider_tol

    def _model_kwargs(self) -> dict:
        """Shared keyword arguments for ProviderPenalizedLogistic."""
        return dict(
            penalty_type=self.penalty_type,
            alpha=self.alpha,
            groups=self.groups,
            group_multiplier=self.group_multiplier,
            gamma_bound=self.gamma_bound,
            penalty_factor=self.penalty_factor,
            standardize=self.standardize,
            fit_intercept=self.fit_intercept,
            max_outer_iter=self.max_outer_iter,
            outer_tol=self.outer_tol,
            max_inner_iter=self.max_inner_iter,
            inner_tol=self.inner_tol,
            provider_max_iter=self.provider_max_iter,
            provider_tol=self.provider_tol,
        )

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        provider_id: np.ndarray = None,
        sample_weight: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
        *,
        provider: Optional[np.ndarray] = None,
    ) -> "ProviderPenalizedLogisticCV":
        """Fit CV to select lambda, then refit on full data.

        Parameters
        ----------
        X : ndarray or DataFrame, shape (n, p)
        y : ndarray, shape (n,)
        provider_id : ndarray, shape (n,)
        sample_weight, offset : ndarray or None

        Returns
        -------
        self
        """
        provider_id = _resolve_provider_alias(provider_id, provider)
        if provider_id is None:
            raise ValueError(
                "provider_id (or provider=) must be provided "
                "(per-observation provider identifiers)."
            )
        # --- Fit full-data model to get the lambda path ---
        full_model = ProviderPenalizedLogistic(
            n_lambda=self.n_lambda,
            lambda_min_ratio=self.lambda_min_ratio,
            lambda_path=self.lambda_path,
            **self._model_kwargs(),
        )
        full_model.fit(
            X, y, provider_id,
            sample_weight=sample_weight, offset=offset,
        )
        lambda_path = full_model.lambda_path_
        n_lambda = len(lambda_path)

        # --- Fold assignment (provider-stratified) ---
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        prov_arr = np.asarray(provider_id)
        n = len(y_arr)

        if self.fold_id is not None:
            fold_id = np.asarray(self.fold_id, dtype=np.intp)
        else:
            fold_id = _provider_stratified_fold_assignment(
                y_arr, prov_arr, self.n_folds,
                random_state=self.random_state,
            )
        n_folds = int(fold_id.max()) + 1
        self.fold_assignment_ = fold_id

        # --- Prepare arrays ---
        X_arr = (
            np.asarray(X.values if isinstance(X, pd.DataFrame) else X,
                       dtype=np.float64)
        )
        if sample_weight is None:
            weight = np.ones(n, dtype=np.float64)
        else:
            weight = np.asarray(sample_weight, dtype=np.float64)
        if offset is not None:
            offset_arr = np.asarray(offset, dtype=np.float64)
        else:
            offset_arr = None

        # --- Cross-validation ---
        cv_deviance = np.full((n_folds, n_lambda), np.nan)

        for k in range(n_folds):
            train_mask = fold_id != k
            val_mask = fold_id == k

            # Fit fold model at the full-data lambda path.
            fold_model = ProviderPenalizedLogistic(
                lambda_path=lambda_path,
                **self._model_kwargs(),
            )
            fold_offset_train = (
                offset_arr[train_mask] if offset_arr is not None else None
            )
            fold_model.fit(
                X_arr[train_mask], y_arr[train_mask],
                prov_arr[train_mask],
                sample_weight=weight[train_mask],
                offset=fold_offset_train,
            )

            # Evaluate on validation fold.
            for j in range(n_lambda):
                coef_j = fold_model.coef_path_[j]
                intercept_j = fold_model.intercept_path_[j]
                gamma_j = fold_model.gamma_path_[j]

                # Linear predictor for validation observations.
                eta_val = X_arr[val_mask] @ coef_j + intercept_j
                if offset_arr is not None:
                    eta_val += offset_arr[val_mask]

                # Add provider effects for validation providers.
                # Providers unseen in training get gamma=0.
                label_to_idx = {
                    lab: i for i, lab in enumerate(
                        fold_model.provider_labels_
                    )
                }
                for row_idx, pid in enumerate(prov_arr[val_mask]):
                    pidx = label_to_idx.get(pid, -1)
                    if pidx >= 0:
                        eta_val[row_idx] += gamma_j[pidx]

                cv_deviance[k, j] = logistic_deviance(
                    y_arr[val_mask], eta_val, weight[val_mask],
                )

        # --- Lambda selection ---
        cv_mean = np.nanmean(cv_deviance, axis=0)
        cv_std = np.nanstd(cv_deviance, axis=0, ddof=1)
        cv_se = cv_std / np.sqrt(n_folds)

        idx_min = int(np.nanargmin(cv_mean))
        lambda_min = lambda_path[idx_min]

        # lambda.1se: largest lambda within 1 SE of min.
        threshold = cv_mean[idx_min] + cv_se[idx_min]
        candidates = np.where(cv_mean <= threshold)[0]
        idx_1se = int(candidates[0])
        lambda_1se = lambda_path[idx_1se]

        # --- Store results ---
        self.cv_mean_deviance_ = cv_mean
        self.cv_std_deviance_ = cv_std
        self.cv_se_deviance_ = cv_se
        self.lambda_min_ = float(lambda_min)
        self.lambda_1se_ = float(lambda_1se)
        self.lambda_min_idx_ = idx_min
        self.lambda_1se_idx_ = idx_1se
        self.lambda_ = float(lambda_1se if self.use_1se else lambda_min)

        idx_selected = idx_1se if self.use_1se else idx_min
        self.model_ = full_model
        self.coef_ = full_model.coef_path_[idx_selected]
        self.intercept_ = full_model.intercept_path_[idx_selected]
        self.gamma_ = full_model.gamma_path_[idx_selected]
        self.coef_path_ = full_model.coef_path_
        self.intercept_path_ = full_model.intercept_path_
        self.gamma_path_ = full_model.gamma_path_
        self.lambda_path_ = full_model.lambda_path_
        self.provider_labels_ = full_model.provider_labels_

        return self

    def predict_proba(
        self, X, provider_id=None, lambda_value=None, which=None,
        *, provider=None,
    ):
        """Predicted probabilities at the selected lambda."""
        provider_id = _resolve_provider_alias(provider_id, provider)
        if which is None:
            # ISSUE-015: honour lambda_value when which is unset.
            if lambda_value is not None:
                which = int(
                    np.argmin(np.abs(self.lambda_path_ - lambda_value))
                )
            else:
                which = (
                    self.lambda_1se_idx_ if self.use_1se
                    else self.lambda_min_idx_
                )
        return self.model_.predict_proba(X, provider_id, which=which)

    def predict(
        self, X, provider_id=None, lambda_value=None,
        which=None, threshold=0.5, *, provider=None,
    ):
        """Binary predictions at the selected lambda."""
        provider_id = _resolve_provider_alias(provider_id, provider)
        return (
            self.predict_proba(X, provider_id, lambda_value, which)
            >= threshold
        ).astype(int)

    def predict_provider_effect(self, which=None) -> pd.DataFrame:
        """Provider effects at the selected lambda."""
        if which is None:
            which = self.lambda_1se_idx_ if self.use_1se else self.lambda_min_idx_
        return self.model_.predict_provider_effect(which=which)
