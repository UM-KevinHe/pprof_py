"""Provider-effect penalized Cox PH estimator.

Implements the two-layer architecture from the ``grplasso`` package:
provider effects γ are explicit unpenalized intercepts updated in the
outer layer via one-step Newton, while covariate coefficients β are
updated in the inner layer via the existing penalized proximal-Newton
solvers (elastic net or group / sparse group lasso).

Model
-----
The linear predictor for observation j is::

    η_j = γ_{provider(j)} + X_j @ β + offset_j

with a **shared** baseline hazard across all providers (optionally
stratified by an external variable via ``strata``, but NOT by
provider).  This makes γ_i identifiable from the Cox partial
likelihood: it captures the provider-specific log-hazard ratio
relative to the common baseline, after risk-adjusting for covariates.

The penalized objective is::

    J(γ, β) = −c · ℓ(γ, β) + λ · penalty(β)

where ``c = 1 / sum(weight)`` and ``penalty`` is one of elastic net,
group lasso, or sparse group lasso.  Provider effects are never
penalized.

Optimization
-----------
For each λ in the path:

1. Warm-start (γ, β) from the previous λ's converged solution.
2. Alternate:
   a. Update γ (one Newton step per provider, then median-clamp).
   b. Update β (full proximal-Newton solve with γ[provider] as
      additional offset).
3. Converge when both max|Δγ| and max relative |Δβ| are small.

References
----------
He, K., Kalbfleisch, J., Li, Y., et al. (2013). Evaluating hospital
readmission rates in dialysis facilities; adjusting for hospital
effects.  *Lifetime Data Analysis*, 19, 490--512.

Shao, Y. & He, K. (2026). grplasso R package.
"""
from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from ...base import ProviderModel

from ...data.survival_validation import validate_X
from ...algorithms.survival.cox_likelihood import cox_partial_likelihood
from ...algorithms.survival.penalty import (
    validate_groups,
    rescale_group_multipliers,
)
from ...algorithms.survival.coordinate_descent import (PenalizedFitResult, fit_single_lambda, compute_lambda_max)
from ...algorithms.coordinate_descent import (
    GroupPenalizedFitResult, fit_single_lambda_group, compute_group_lambda_max,
    add_unpenalized_block, drop_unpenalized_block, _compute_group_kkt_violation,
)
from ...algorithms.penalty import fit_group_multipliers, unorthogonalize_coefs, resolve_penalty_alpha, unpenalized_columns
from ...algorithms.survival.provider_effects import (
    validate_provider_ids,
    compute_provider_scores,
    provider_newton_step,
)
from ...algorithms.survival.ties import TieMethod
from .penalized_coxph import _PenalizedCoxPHBase, _resolve_lambda_path

logger = logging.getLogger(__name__)

# ======================================================================
# Validation
# ======================================================================

def _validate_provider_parameters(
    penalty_type, alpha, provider_bound, provider_max_iter,
    provider_tol, groups, n_lambda, lambda_min_ratio, lambda_path,
    standardize, max_outer_iter, outer_tol, max_inner_iter, inner_tol,
    fit_intercept,
):
    """Validate constructor parameters for ProviderPenalizedCoxPH.

    Returns the mixing parameter the penalty type fits with.
    """
    alpha = resolve_penalty_alpha(penalty_type, alpha)
    if penalty_type in ("group_lasso", "sparse_group_lasso") and groups is None:
        raise ValueError(
            f"groups must be provided when penalty_type={penalty_type!r}"
        )
    if not np.isfinite(provider_bound) or float(provider_bound) <= 0:
        raise ValueError(
            f"provider_bound must be finite and > 0, got {provider_bound!r}"
        )
    if (
        isinstance(provider_max_iter, (bool, np.bool_))
        or int(provider_max_iter) != provider_max_iter
        or int(provider_max_iter) < 1
    ):
        raise ValueError(
            f"provider_max_iter must be a positive integer, "
            f"got {provider_max_iter!r}"
        )
    if not np.isfinite(provider_tol) or float(provider_tol) <= 0:
        raise ValueError(
            f"provider_tol must be finite and > 0, got {provider_tol!r}"
        )
    # --- Common parameters (same as PenalizedCoxPH) ---
    if (
        isinstance(n_lambda, (bool, np.bool_))
        or int(n_lambda) != n_lambda
        or int(n_lambda) < 1
    ):
        raise ValueError(
            f"n_lambda must be a positive integer, got {n_lambda!r}"
        )
    if lambda_min_ratio is not None:
        ratio = float(lambda_min_ratio)
        if not np.isfinite(ratio) or not (0.0 < ratio <= 1.0):
            raise ValueError(
                f"lambda_min_ratio must be in (0, 1], "
                f"got {lambda_min_ratio!r}"
            )
    if lambda_path is not None:
        values = np.atleast_1d(np.asarray(lambda_path, dtype=np.float64))
        if (
            values.size == 0
            or not np.all(np.isfinite(values))
            or np.any(values <= 0)
        ):
            raise ValueError(
                "lambda_path must contain finite, strictly positive values"
            )
    if not isinstance(standardize, (bool, np.bool_)):
        raise ValueError("standardize must be a boolean")
    if (
        isinstance(max_outer_iter, (bool, np.bool_))
        or int(max_outer_iter) != max_outer_iter
        or int(max_outer_iter) < 1
    ):
        raise ValueError(
            f"max_outer_iter must be a positive integer, "
            f"got {max_outer_iter!r}"
        )
    if not np.isfinite(outer_tol) or float(outer_tol) <= 0:
        raise ValueError(
            f"outer_tol must be finite and > 0, got {outer_tol!r}"
        )
    if (
        isinstance(max_inner_iter, (bool, np.bool_))
        or int(max_inner_iter) != max_inner_iter
        or int(max_inner_iter) < 1
    ):
        raise ValueError(
            f"max_inner_iter must be a positive integer, "
            f"got {max_inner_iter!r}"
        )
    if not np.isfinite(inner_tol) or float(inner_tol) <= 0:
        raise ValueError(
            f"inner_tol must be finite and > 0, got {inner_tol!r}"
        )
    if fit_intercept:
        raise ValueError(
            "Cox proportional hazards regression does not support "
            "an intercept"
        )
    return alpha


# ======================================================================
# ProviderPenalizedCoxPH
# ======================================================================

class ProviderPenalizedCoxPH(_PenalizedCoxPHBase, ProviderModel):
    """Penalized Cox PH with explicit provider effects.

    Minimizes the penalized negative log-partial-likelihood with
    provider-specific intercepts (γ) treated as unpenalized
    first-class parameters::

        J(γ, β) = −c · ℓ(γ, β) + λ · penalty(β)

    The linear predictor for observation j is::

        η_j = γ_{provider(j)} + X_j @ β + offset_j

    The optimization uses a two-layer architecture:

    * **Outer layer:** update provider effects γ via one-step Newton
      (``algorithms/provider_effects.py``), with median-bounding.
    * **Inner layer:** update covariate coefficients β via the
      penalized proximal-Newton solver from
      ``algorithms/coordinate_descent.py``.

    Both γ and β are warm-started from the previous λ's converged
    solution.

    Parameters
    ----------
    penalty_type : str, default ``'elastic_net'``
        ``'elastic_net'``, ``'group_lasso'``, or
        ``'sparse_group_lasso'``.
    alpha : float or None, default None
        Mixing parameter.  ``'elastic_net'``: 0 = ridge, 1 = lasso (glmnet
        convention; ``None`` means 1).  ``'group_lasso'``: the pure group
        lasso; only ``None`` or 0 is accepted.  ``'sparse_group_lasso'``:
        required; 0 = group lasso, 1 = lasso.
    groups : array-like or None, default None
        Group labels per feature (required for group penalties).
    group_multiplier : array-like or None, default None
        Per-group penalty multiplier; ``None`` = ``sqrt(group_size)``.
    penalty_factor : array-like or None, default None
    n_lambda : int, default 100
    lambda_min_ratio : float or None, default None
    lambda_path : float, sequence, or None, default None
    standardize : bool, default True
    orthogonalize : bool, default True
        Group penalties only.  Orthogonalize each penalized group within
        itself (on its weighted, centered columns), so the group penalty is
        the standardized group lasso ``GroupLassoCoxPH`` and R's ``grplasso``
        fit; ``False`` fits the plain group lasso on the standardized columns.
    ties : str, default ``'breslow'``
    fit_intercept : bool, default False
    provider_bound : float, default 10.0
        Clamp γ to ``median(γ) ± bound``.
    provider_backtrack : bool, default False
        Placeholder for backtracking line search on γ.
    provider_max_iter : int, default 20
        Maximum two-layer (γ–β) alternation iterations per λ.
    provider_tol : float, default 1e-6
        Convergence tolerance on ``max|Δγ|``.
    max_outer_iter : int, default 100
        Max proximal-Newton outer iterations for the β solver.
    outer_tol : float, default 1e-7
        Convergence tolerance for the outer β solver loop.  Provider
        models use 1e-7 (looser than the 1e-9 default in non-provider
        penalized models) because the two-layer γ–β alternation
        provides additional implicit convergence pressure.
    max_inner_iter : int, default 1000
    inner_tol : float, default 1e-10

    Attributes (set by ``fit``)
    ---------------------------
    coef_path_ : ndarray, shape ``(n_lambda_, n_features)``
        Covariate coefficients β (original units) at each λ.
    gamma_path_ : ndarray, shape ``(n_lambda_, n_providers_)``
        Provider effects γ at each λ.
    lambda_path_ : ndarray, shape ``(n_lambda_,)``
    lambda_max_ : float
    provider_labels_ : ndarray
        Sorted unique provider labels from ``fit``.
    n_providers_ : int
    n_provider_iter_path_ : ndarray of int, shape ``(n_lambda_,)``
        Two-layer iterations per λ.
    provider_converged_path_ : ndarray of bool, shape ``(n_lambda_,)``
    log_likelihood_path_, deviance_ratio_path_, n_nonzero_path_ :
        From ``_PenalizedCoxPHBase``.
    groups_, n_groups_, group_weights_, kkt_violation_path_ : (group penalties only)
    alpha_ : float
        The mixing parameter used (see ``alpha``).
    gamma_, coef_, lambda_ : set when ``lambda_path_`` has length 1.
    """

    def __init__(
        self,
        penalty_type: str = "elastic_net",
        alpha: Optional[float] = None,
        groups=None,
        group_multiplier=None,
        penalty_factor=None,
        n_lambda: int = 100,
        lambda_min_ratio: Optional[float] = None,
        lambda_path=None,
        standardize: bool = True,
        orthogonalize: bool = True,
        ties: Union[str, TieMethod] = "breslow",
        fit_intercept: bool = False,
        provider_bound: float = 10.0,
        provider_backtrack: bool = False,
        provider_max_iter: int = 20,
        provider_tol: float = 1e-6,
        max_outer_iter: int = 100,
        outer_tol: float = 1e-7,
        max_inner_iter: int = 1000,
        inner_tol: float = 1e-10,
    ):
        """Two-layer provider + penalized-covariate Cox."""
        self.penalty_type = penalty_type
        self.alpha = alpha
        self.groups = groups
        self.group_multiplier = group_multiplier
        self.penalty_factor = penalty_factor
        self.n_lambda = n_lambda
        self.lambda_min_ratio = lambda_min_ratio
        self.lambda_path = lambda_path
        self.standardize = standardize
        self.orthogonalize = orthogonalize
        self.ties = ties
        self.fit_intercept = fit_intercept
        self.provider_bound = provider_bound
        self.provider_backtrack = provider_backtrack
        self.provider_max_iter = provider_max_iter
        self.provider_tol = provider_tol
        self.max_outer_iter = max_outer_iter
        self.outer_tol = outer_tol
        self.max_inner_iter = max_inner_iter
        self.inner_tol = inner_tol

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_provider_null_point(self, prep, X_design, provider_idx,
                                     n_providers, unpenalized):
        """Null point for ``lambda_max``: penalized β at 0, γ (and any
        unpenalized β) at their joint MLE.

        ``unpenalized`` marks the unpenalized columns: ``pf == 0`` for the
        elastic net, group 0 for the group penalties.  Returns
        ``(beta_null, gamma_null)``.  Alternates the provider Newton layer
        with the restricted unpenalized β fit, which reduces to a pure γ fit
        when every column is penalized (the common case).
        """
        beta_null = np.zeros(prep.p_fit, dtype=np.float64)
        gamma_null = np.zeros(n_providers, dtype=np.float64)
        always_unpen = np.asarray(unpenalized, dtype=bool)
        has_unpen = bool(np.any(always_unpen)) and not bool(
            np.all(always_unpen)
        )

        n_rounds = self.provider_max_iter if not has_unpen else max(
            2, min(self.provider_max_iter, 25)
        )
        for _round in range(n_rounds):
            # γ layer, given current β.
            eta = (
                gamma_null[provider_idx]
                + X_design @ beta_null
                + prep.data.offset
            )
            score_gamma, info_gamma = compute_provider_scores(
                eta, prep.data.event, prep.data.weight,
                prep.data.start, prep.data.stop,
                provider_idx, n_providers,
                strata_codes=prep.data.strata_codes,
                stratum_indices=prep.stratum_idx,
            )
            gamma_result = provider_newton_step(
                gamma_null.copy(), score_gamma, info_gamma,
                bound=self.provider_bound,
                tol=self.provider_tol,
            )
            gamma_null = gamma_result.gamma
            if not has_unpen:
                if gamma_result.converged:
                    break
                continue

            # Unpenalized β layer, given current γ.
            obj = self._build_beta_objective(
                prep, X_design, gamma_null, provider_idx,
            )
            _ll, score, info = obj(beta_null)
            idx = np.flatnonzero(always_unpen)
            H = info[np.ix_(idx, idx)]
            g = score[idx]
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(H, g, rcond=None)[0]
            beta_null[idx] += step
            if (gamma_result.converged
                    and float(np.max(np.abs(step))) < self.outer_tol):
                break

        return beta_null, gamma_null

    def _build_beta_objective(self, prep, X_design, gamma, provider_idx):
        """Build β objective closure with current γ as offset.

        The γ contribution enters as an additional offset so that the
        existing penalized solvers can be reused without modification.
        ``X_design`` is the fitted design (orthogonalized for group
        penalties with ``orthogonalize``).
        """
        effective_offset = prep.data.offset + gamma[provider_idx]
        ties = self.ties
        si = prep.stratum_idx

        def objective_fn(beta):  # noqa: D401
            """Partial log-likelihood, score, and information at *beta*."""
            return cox_partial_likelihood(
                X_design, prep.data.start, prep.data.stop,
                prep.data.event, beta,
                offset=effective_offset,
                weight=prep.data.weight,
                strata=prep.data.strata_codes,
                ties=ties, stratum_indices=si,
            )

        return objective_fn

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
        *,
        provider_id=None,
    ) -> "ProviderPenalizedCoxPH":
        """Fit the two-layer penalized Cox model.

        Parameters
        ----------
        X : DataFrame or ndarray, shape ``(n, p)``
        duration, event, start, stop, strata, offset, sample_weight :
            Standard survival data (see ``PenalizedCoxPH``).
        provider_id : array-like, shape ``(n,)``
            Provider identifier per observation.  Must have >= 2
            distinct values.

        Returns
        -------
        self
        """
        provider = provider_id
        if provider is None:
            raise ValueError(
                "provider_id must be provided (per-observation provider "
                "identifier array)"
            )

        alpha = _validate_provider_parameters(
            self.penalty_type, self.alpha,
            self.provider_bound, self.provider_max_iter,
            self.provider_tol, self.groups,
            self.n_lambda, self.lambda_min_ratio, self.lambda_path,
            self.standardize, self.max_outer_iter, self.outer_tol,
            self.max_inner_iter, self.inner_tol, self.fit_intercept,
        )
        self.alpha_ = alpha

        # --- Common data preparation (base mixin) ---
        prep = self._prepare_fit_data(
            X, duration, event, start, stop, strata, offset,
            sample_weight,
        )

        # --- Provider validation ---
        provider_arr = np.asarray(provider)
        unique_labels = np.unique(provider_arr)
        provider_idx, n_providers = validate_provider_ids(
            provider_arr, prep.data.n_obs,
        )

        # --- Penalty-specific setup ---
        use_groups = self.penalty_type != "elastic_net"
        X_design, QL_blocks = prep.X_fit, None
        groups_fit = group_weights_fit = None
        n_groups_fit = 0
        groups_full = group_sizes = group_weights = None
        n_groups = 0

        if use_groups:
            groups_raw = np.asarray(self.groups, dtype=np.float64)
            if groups_raw.shape != (prep.p_full,):
                raise ValueError(
                    f"groups must have shape ({prep.p_full},), "
                    f"got {groups_raw.shape}"
                )
            groups_full, group_sizes, n_groups = validate_groups(
                groups_raw, prep.p_full,
            )
            group_weights = rescale_group_multipliers(
                self.group_multiplier, group_sizes, n_groups,
            )
            # Remap for reduced feature set (after degenerate removal).
            groups_fit, _sizes_fit, n_groups_fit = validate_groups(
                groups_full[prep.fit_cols], prep.p_fit,
            )
            group_weights_fit = fit_group_multipliers(
                self.group_multiplier, groups_full, prep.fit_cols,
            )
            # C8b: the standardized group lasso (groups orthogonalized on
            # their centered columns), as GroupLassoCoxPH and R fit it.
            X_design, _objective, QL_blocks = self._group_design(
                prep, groups_fit,
            )
            # Unpenalized (group 0) columns join the block solver as a block
            # with no penalty (ISSUE-010).
            (blocks, gs_arr, ge_arr, gw_blocks, pf_blocks, n_blocks,
             has_unpen) = add_unpenalized_block(
                groups_fit, n_groups_fit, group_weights_fit, prep.pf_fit,
            )
            unpenalized = unpenalized_columns(
                groups_fit, group_weights_fit, prep.pf_fit, alpha,
            )
        else:
            unpenalized = prep.pf_fit == 0.0

        # --- Null point (β_null, γ=γ̂) ---
        # REV-003: the inherited PenalizedCoxPH null point evaluates the
        # score at γ=0, which is not the null point of *this* model.  With
        # the provider effects at zero the score is too small and the
        # resulting lambda_max does not zero the penalized coefficients, so
        # the first path point is not actually null.  Fit γ (alternating
        # with the unpenalized β when there are any) before taking the
        # score.  The R reference does the same thing via stratification:
        # its set.lambda.cox builds the null residual from per-provider risk
        # sets rather than pooled ones.
        beta_null, gamma_null = self._compute_provider_null_point(
            prep, X_design, provider_idx, n_providers, unpenalized,
        )
        objective_null = self._build_beta_objective(
            prep, X_design, gamma_null, provider_idx,
        )
        _ll_null, score_null, _info_null = objective_null(beta_null)

        # --- Lambda max ---
        if not use_groups:
            lambda_max = (
                0.0 if np.all(unpenalized)
                else compute_lambda_max(
                    score_null, prep.c, prep.pf_fit, alpha,
                )
            )
        else:
            lambda_max = (
                0.0 if np.all(unpenalized)
                else compute_group_lambda_max(
                    score_null, prep.c, groups_fit,
                    group_weights_fit, prep.pf_fit, alpha,
                )
            )

        lambda_sequence, lambda_min_ratio = _resolve_lambda_path(
            self.lambda_path, lambda_max, self.lambda_min_ratio,
            self.n_lambda, prep.p_fit, prep.data.n_obs,
        )

        # ============================================================
        # Two-layer path fitting
        # ============================================================
        gamma = gamma_null.copy()
        beta = beta_null.copy()

        # lambda_max is the smallest lambda at which the null point solves
        # the problem (for the elastic net only when alpha >= 1e-3, the floor
        # compute_lambda_max divides by), so at or above it the null point is
        # stored as is: iterating from it only adds rounding noise to its
        # exact zeros.  (A generated path starts at exp(log(lambda_max)),
        # which can fall an ulp below it.)
        null_is_solution = (use_groups or alpha >= 1e-3) and lambda_max > 0
        lam_null = lambda_max * (1.0 - 1e-12)
        common_null = dict(
            beta=beta_null.copy(), log_likelihood=_ll_null,
            objective_value=-prep.c * _ll_null, n_outer_iter=0,
            n_inner_iter_total=0, converged=True,
            message="null point (lambda >= lambda_max)",
            information=_info_null,
        )

        results = []
        gamma_path = []
        n_provider_iter = []
        provider_converged_list = []

        for lam_idx, lam in enumerate(lambda_sequence):
            lam_val = float(lam)
            if null_is_solution and lam_val >= lam_null:
                if use_groups:
                    null_result = GroupPenalizedFitResult(
                        **common_null,
                        active_groups=np.zeros(n_groups_fit, dtype=bool),
                        group_norms=np.zeros(n_groups_fit),
                        df=float(np.sum(beta_null != 0.0)),
                        kkt_violation=_compute_group_kkt_violation(
                            prep.c * score_null, beta_null, lam_val, alpha,
                            blocks, gw_blocks, pf_blocks, n_blocks,
                        ),
                    )
                else:
                    null_result = PenalizedFitResult(**common_null)
                results.append(null_result)
                gamma_path.append(gamma_null.copy())
                n_provider_iter.append(0)
                provider_converged_list.append(True)
                continue
            converged_two_layer = False
            prov_iters = 0

            for prov_iter in range(1, self.provider_max_iter + 1):
                prov_iters = prov_iter

                # --- Step 1: update γ given current β ---
                eta = (
                    gamma[provider_idx]
                    + X_design @ beta
                    + prep.data.offset
                )
                score_gamma, info_gamma = compute_provider_scores(
                    eta, prep.data.event, prep.data.weight,
                    prep.data.start, prep.data.stop,
                    provider_idx, n_providers,
                    strata_codes=prep.data.strata_codes,
                    stratum_indices=prep.stratum_idx,
                )
                gamma_result = provider_newton_step(
                    gamma.copy(), score_gamma, info_gamma,
                    bound=self.provider_bound,
                    tol=self.provider_tol,
                )
                gamma = gamma_result.gamma

                # --- Step 2: update β given current γ ---
                objective_fn_beta = self._build_beta_objective(
                    prep, X_design, gamma, provider_idx,
                )

                if not use_groups:
                    beta_result = fit_single_lambda(
                        objective_fn_beta, beta, prep.c, lam_val,
                        alpha, prep.pf_fit,
                        outer_max_iter=self.max_outer_iter,
                        outer_tol=self.outer_tol,
                        inner_max_iter=self.max_inner_iter,
                        inner_tol=self.inner_tol,
                    )
                else:
                    beta_result = fit_single_lambda_group(
                        objective_fn_beta, beta, prep.c, lam_val,
                        alpha, pf_blocks,
                        blocks, gw_blocks,
                        gs_arr, ge_arr, n_blocks,
                        outer_max_iter=self.max_outer_iter,
                        outer_tol=self.outer_tol,
                        inner_max_iter=self.max_inner_iter,
                        inner_tol=self.inner_tol,
                    )
                    if has_unpen:
                        beta_result = drop_unpenalized_block(beta_result)

                beta_change = (
                    float(np.max(
                        np.abs(beta_result.beta - beta)
                        / np.maximum(1.0, np.abs(beta))
                    ))
                    if prep.p_fit > 0
                    else 0.0
                )
                beta = beta_result.beta

                # --- Step 3: check two-layer convergence ---
                if (
                    gamma_result.max_change < self.provider_tol
                    and beta_change < self.outer_tol
                ):
                    converged_two_layer = True
                    break

            if not converged_two_layer:
                logger.info(
                    "\u03bb[%d]=%.4e: two-layer loop did not converge "
                    "in %d iterations (\u03b3_change=%.2e, "
                    "\u03b2_change=%.2e)",
                    lam_idx, lam_val, prov_iters,
                    gamma_result.max_change, beta_change,
                )

            results.append(beta_result)
            gamma_path.append(gamma.copy())
            n_provider_iter.append(prov_iters)
            provider_converged_list.append(converged_two_layer)

        # ============================================================
        # Store results
        # ============================================================
        stored = results if QL_blocks is None else [
            r._replace(beta=unorthogonalize_coefs(r.beta, groups_fit, QL_blocks))
            for r in results
        ]
        self._store_path_results(
            stored, prep, lambda_sequence, lambda_max,
            lambda_min_ratio,
        )

        # Override converged_path_: require both β AND γ convergence.
        self.converged_path_ = (
            self.converged_path_
            & np.array(provider_converged_list)
        )

        # Provider-specific attributes.
        self.gamma_path_ = np.array(gamma_path)
        self.provider_labels_ = unique_labels
        self.provider_idx_ = provider_idx
        self.n_providers_ = n_providers
        self.n_provider_iter_path_ = np.array(n_provider_iter)
        self.provider_converged_path_ = np.array(
            provider_converged_list,
        )

        # Group-specific attributes (group penalties only).
        if use_groups:
            self.groups_ = groups_full
            self.group_sizes_ = group_sizes
            self.n_groups_ = n_groups
            self.group_weights_ = group_weights
            self.group_norms_ = np.array(
                [r.group_norms for r in results],
            )
            self.active_groups_ = [
                r.active_groups for r in results
            ]
            self.df_path_ = np.array(
                [r.df for r in results],
            )
            self.kkt_violation_path_ = np.array(
                [r.kkt_violation for r in results],
            )

        # Convenience: single-lambda case.
        if len(gamma_path) == 1:
            self.gamma_ = gamma_path[0]

        return self

    # ------------------------------------------------------------------
    # Provider-aware prediction
    # ------------------------------------------------------------------
    def predict_linear_with_provider(
        self,
        X,
        provider_id=None,
        offset=None,
        lambda_value: Optional[float] = None,
    ) -> np.ndarray:
        """Linear predictor including provider effects.

        Returns ``γ_{provider(j)} + X_j @ β + offset_j``.

        Parameters
        ----------
        X : array-like, shape ``(n, p)``
        provider_id : array-like, shape ``(n,)``
            Provider labels (must be among those seen during ``fit``).

        offset : array-like or None
        lambda_value : float or None

        Returns
        -------
        ndarray, shape ``(n,)``
        """
        self._check_is_fitted()
        provider = provider_id
        if provider is None:
            raise ValueError("provider (or provider_id=) is required.")
        coef = self._resolve_coef(lambda_value)
        gamma = self._resolve_gamma(lambda_value)
        X_arr, _ = validate_X(X)
        n = X_arr.shape[0]

        offset_arr = (
            np.zeros(n)
            if offset is None
            else np.asarray(offset, dtype=np.float64)
        )

        provider_arr = np.asarray(provider)
        prov_idx = np.searchsorted(
            self.provider_labels_, provider_arr,
        )
        # Validate: all labels must be known.
        valid = (
            (prov_idx < len(self.provider_labels_))
            & (self.provider_labels_[np.minimum(
                prov_idx, len(self.provider_labels_) - 1,
            )] == provider_arr)
        )
        if not np.all(valid):
            unknown = set(provider_arr[~valid])
            raise ValueError(
                f"Unknown provider labels: {unknown}. Only providers "
                f"seen during fit() are supported."
            )

        return gamma[prov_idx] + X_arr @ coef + offset_arr

    def predict_provider_effect(
        self,
        provider_id=None,
        lambda_value: Optional[float] = None,
    ) -> np.ndarray:
        """Estimated provider effect γ_i.

        Parameters
        ----------
        provider_id : array-like or None
            Provider labels to query.  ``None`` returns all providers
            in ``provider_labels_`` order.
        lambda_value : float or None

        Returns
        -------
        ndarray, shape ``(n_query,)``
        """
        self._check_is_fitted()
        provider = provider_id
        gamma = self._resolve_gamma(lambda_value)
        if provider is None:
            return gamma
        provider_arr = np.asarray(provider)
        prov_idx = np.searchsorted(
            self.provider_labels_, provider_arr,
        )
        valid = (
            (prov_idx < len(self.provider_labels_))
            & (self.provider_labels_[np.minimum(
                prov_idx, len(self.provider_labels_) - 1,
            )] == provider_arr)
        )
        if not np.all(valid):
            unknown = set(provider_arr[~valid])
            raise ValueError(
                f"Unknown provider labels: {unknown}"
            )
        return gamma[prov_idx]

    def _resolve_gamma(
        self, lambda_value: Optional[float],
    ) -> np.ndarray:
        """Resolve γ at a lambda value (nearest grid point)."""
        self._check_is_fitted()
        if lambda_value is None:
            if hasattr(self, "gamma_"):
                return self.gamma_
            raise ValueError(
                "lambda_value must be specified for multi-lambda fits"
            )
        idx = int(
            np.argmin(np.abs(self.lambda_path_ - lambda_value))
        )
        return self.gamma_path_[idx]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self) -> pd.DataFrame:
        """Per-lambda path summary with provider convergence info."""
        self._check_is_fitted()
        df = pd.DataFrame({
            "lambda": self.lambda_path_,
            "n_nonzero": self.n_nonzero_path_,
            "deviance_ratio": self.deviance_ratio_path_,
            "log_likelihood": self.log_likelihood_path_,
            "converged": self.converged_path_,
            "n_provider_iter": self.n_provider_iter_path_,
            "provider_converged": self.provider_converged_path_,
        })
        return df
