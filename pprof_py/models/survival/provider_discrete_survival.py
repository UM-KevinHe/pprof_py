"""Provider-penalized discrete-time survival model.

Matches R ``grplasso::pp.DiscSurv()`` for provider-profiling with
time-to-event outcomes discretized into intervals.

Three-layer architecture:
  1. Provider effects gamma_k (unpenalized, median-clamp bounded)
  2. Baseline hazard alpha_t  (unpenalized, logit-scale time effects)
  3. Covariate effects beta    (penalized: elastic net or group lasso)

The linear predictor for observation i at time t is:

    eta_{i,t} = gamma_{prov(i)} + alpha_t + X_i @ beta

R reference: grplasso/R/pp_DiscSurv.R, grplasso/src/pp_DiscSurv_lasso.cpp.

Provenance: new file, 2026-09. Pattern drawn from:
  - models/logistic/provider_penalized.py  (two-layer gamma + beta)
  - models/survival/discrete_survival.py   (baseline hazard + beta)
  - models/survival/provider_coxph.py      (provider + Cox)
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
    rescale_penalty_factors,
    validate_groups,
    rescale_group_multipliers,
    compute_group_indices,
)
from ...algorithms.survival.discrete_survival import (
    discretize_times,
    initialize_baseline_hazard,
    compute_n_at_risk,
    baseline_hazard_update,
    discrete_loglik,
    compute_discrete_lambda_max,
    discrete_coordinate_descent_step,
    discrete_residuals,
    compute_working_response,
    person_period_expand,
    predict_discrete_hazard,
)
from ...algorithms.logistic.provider_effects import (
    compute_provider_indices,
)
from ...exceptions import NotFittedError

logger = logging.getLogger(__name__)


# ======================================================================
# Provider Newton step for discrete survival likelihood
# ======================================================================

def _provider_newton_from_loglik(
    score_beta: np.ndarray,
    working_weights: np.ndarray,
    provider_idx: np.ndarray,
    n_providers: int,
    gamma: np.ndarray,
    provider_bound: float = 10.0,
) -> np.ndarray:
    """One Newton step for provider effects using discrete_loglik outputs.

    Uses the subject-level outputs of ``discrete_loglik`` directly,
    avoiding the need for person-period expansion.

    The per-provider score and information are:

        score_k = - sum_{i in k} score_beta_i
                = sum_{i in k} (delta_i - sum_t p_{it})

        info_k  = sum_{i in k} working_weights_i
                = sum_{i in k} sum_t p_{it} * (1 - p_{it})

    where ``score_beta`` from ``discrete_loglik`` is the gradient of
    the *negative* log-likelihood, hence the sign flip.

    Parameters
    ----------
    score_beta : ndarray, shape (n,)
        Per-subject gradient from ``discrete_loglik``
        (``sum_k p_{ik} - delta_i``).
    working_weights : ndarray, shape (n,)
        Per-subject Newton working weights from ``discrete_loglik``.
    provider_idx : ndarray of int, shape (n,)
        Provider index per observation (0..K-1).
    n_providers : int
    gamma : ndarray, shape (n_providers,)
        Current provider effects.
    provider_bound : float
        Maximum deviation from median.

    Returns
    -------
    gamma_new : ndarray, shape (n_providers,)
    """
    # Aggregate per-provider: O(n) via bincount.
    prov_score = -np.bincount(
        provider_idx, weights=score_beta, minlength=n_providers,
    ).astype(np.float64)
    prov_info = np.bincount(
        provider_idx, weights=working_weights, minlength=n_providers,
    ).astype(np.float64)

    # Newton step with median-clamp bounding.
    info_safe = np.maximum(prov_info, 1e-12)
    gamma_new = gamma + prov_score / info_safe

    # Median-clamp.
    median_gamma = float(np.median(gamma_new))
    gamma_new = np.clip(
        gamma_new,
        median_gamma - provider_bound,
        median_gamma + provider_bound,
    )
    return gamma_new


# ======================================================================
# ProviderPenalizedDiscreteSurvival
# ======================================================================

class ProviderPenalizedDiscreteSurvival(ProviderModel):
    """Provider-penalized discrete-time survival model.

    Three-layer architecture matching R ``pp.DiscSurv``:

    * **Layer 1 (provider):** unpenalized gamma_k per provider,
      bounded by median-clamp.
    * **Layer 2 (baseline hazard):** unpenalized alpha_t per
      discrete time point (logit scale).
    * **Layer 3 (covariates):** penalized beta via elastic net
      coordinate descent on the person-period expanded data.

    Parameters
    ----------
    alpha : float, default=1.0
        Elastic net mixing (1 = lasso, 0 = ridge).
    provider_bound : float, default=10.0
        Maximum provider effect deviation from median.
    n_lambda : int, default=100
    lambda_min_ratio : float or None
    lambda_path : array-like or None
    penalty_factor : array-like or None
    standardize : bool, default=False
        Whether to standardize covariates.  Default False to match
        R ``pp.DiscSurv`` which comments: "standardize = TRUE may
        cause problems in transforming gamma and alpha back."
    max_outer_iter : int, default=100
        Maximum three-layer alternation iterations per lambda.
    outer_tol : float, default=1e-7
        Convergence tolerance for the three-layer alternation loop.
        Provider models use 1e-7 (looser than the 1e-9 default in
        non-provider penalized models) because the multi-layer
        alternation provides additional implicit convergence pressure.
    max_inner_iter : int, default=10000
    inner_tol : float, default=1e-7
    provider_max_iter : int, default=10
    baseline_max_iter : int, default=10
        Maximum Newton iterations for baseline hazard per outer step.
    use_active_set : bool, default=True
        Active-set screening for the CD solver.

    Attributes (after fit)
    ----------------------
    coef_path_ : ndarray, shape (n_lambda, p)
    baseline_hazard_path_ : ndarray, shape (n_lambda, n_timepoints)
    gamma_path_ : ndarray, shape (n_lambda, K)
    lambda_path_ : ndarray, shape (n_lambda,)
    provider_labels_ : ndarray, shape (K,)
    time_points_ : ndarray
    """

    def __init__(
        self,
        alpha: float = 1.0,
        provider_bound: float = 10.0,
        n_lambda: int = 100,
        lambda_min_ratio: Optional[float] = None,
        lambda_path: Optional[np.ndarray] = None,
        penalty_factor: Optional[np.ndarray] = None,
        standardize: bool = False,
        max_outer_iter: int = 100,
        outer_tol: float = 1e-7,
        max_inner_iter: int = 10000,
        inner_tol: float = 1e-7,
        provider_max_iter: int = 10,
        baseline_max_iter: int = 10,
        use_active_set: bool = True,
    ):
        """Two-layer provider + penalized discrete survival."""
        self.alpha = alpha
        self.provider_bound = provider_bound
        self.n_lambda = n_lambda
        self.lambda_min_ratio = lambda_min_ratio
        self.lambda_path = lambda_path
        self.penalty_factor = penalty_factor
        self.standardize = standardize
        self.max_outer_iter = max_outer_iter
        self.outer_tol = outer_tol
        self.max_inner_iter = max_inner_iter
        self.inner_tol = inner_tol
        self.provider_max_iter = provider_max_iter
        self.baseline_max_iter = baseline_max_iter
        self.use_active_set = use_active_set

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        time: np.ndarray,
        event: np.ndarray,
        provider_id: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "ProviderPenalizedDiscreteSurvival":
        """Fit the three-layer provider-penalized discrete survival model.

        Parameters
        ----------
        X : ndarray or DataFrame, shape (n, p)
            Covariate matrix.
        time : ndarray, shape (n,)
            Discrete event/censoring times.
        event : ndarray, shape (n,)
            Event indicator (1 = event, 0 = censored).
        provider_id : ndarray, shape (n,)
            Provider identifiers.
        sample_weight : ndarray or None, shape (n,)

        Returns
        -------
        self

        Algorithm
        ---------
        For each lambda in the path (warm-started):

        1. **Provider step:** Newton update for gamma_k using the
           person-period discrete-survival score and information,
           with median-clamp bounding.

        2. **Baseline hazard step:** Newton update for alpha_t
           using ``baseline_hazard_update()`` from
           ``algorithms/survival/discrete_survival.py``.

        3. **Covariate step:** Penalized CD update for beta on the
           person-period expanded working response, using
           ``discrete_coordinate_descent_step()``.

        4. Convergence check: max |delta_beta| + max |delta_gamma|
           + max |delta_alpha| < outer_tol.
        """
        # =============================================================
        # 1. Input validation and feature names
        # =============================================================
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

        # =============================================================
        # 2. Discretize times
        # =============================================================
        time_int, timepoint_map, n_events = discretize_times(
            time_np, event_np,
        )
        K = len(timepoint_map)
        self.time_points_ = timepoint_map
        self.n_timepoints_ = K

        # =============================================================
        # 3. Provider indexing
        # =============================================================
        provider_idx, provider_labels, n_providers = (
            compute_provider_indices(np.asarray(provider_id))
        )
        self.provider_labels_ = provider_labels
        self.n_providers_ = n_providers

        # =============================================================
        # 4. Column standardization (optional)
        # =============================================================
        if self.standardize:
            xs, _degen = weighted_column_scale(X_np, weight)
            w_sum = weight.sum()
            xm = (weight / w_sum) @ X_np
            xs_safe = np.where(xs == 0, 1.0, xs)
            X_fit = (X_np - xm) / xs_safe
            self._centers = xm
            self._scales = xs
        else:
            X_fit = X_np.copy()
            self._centers = np.zeros(p)
            self._scales = np.ones(p)

        # =============================================================
        # 5. Penalty factors
        # =============================================================
        if self.penalty_factor is not None:
            pf = np.asarray(self.penalty_factor, dtype=np.float64)
            if len(pf) != p:
                raise ValueError(
                    f"penalty_factor length ({len(pf)}) != n_features ({p})"
                )
            pf = rescale_penalty_factors(pf, p)
        else:
            pf = np.ones(p, dtype=np.float64)

        # =============================================================
        # 6. Initialize baseline hazard from KM estimator
        # =============================================================
        n_at_risk = compute_n_at_risk(time_int, K)
        alpha = initialize_baseline_hazard(n_events, n_at_risk)
        beta = np.zeros(p, dtype=np.float64)
        gamma = np.zeros(n_providers, dtype=np.float64)
        # eta = X @ beta + gamma[provider_idx]  (alpha is separate)
        eta = np.zeros(n, dtype=np.float64)

        # =============================================================
        # 7. Converge gamma + alpha at beta=0 for lambda_max
        # =============================================================
        # R's set.lambda.pp_DiscSurv runs a preliminary model with
        # gamma treated as dummy-coded regressors to converge the
        # null model. We approximate this by iterating gamma + alpha
        # updates at beta=0 until convergence.
        for _null_iter in range(50):
            # Gamma update
            ll_null = discrete_loglik(
                time_int, event_np, alpha, eta,
            )
            gamma_new = _provider_newton_from_loglik(
                ll_null.score_beta, ll_null.working_weights,
                provider_idx, n_providers, gamma, self.provider_bound,
            )
            eta += (gamma_new - gamma)[provider_idx]
            gamma_change = np.max(np.abs(gamma_new - gamma))
            gamma = gamma_new

            # Alpha update
            alpha_new = baseline_hazard_update(
                alpha, time_int, event_np, eta,
                n_events, bound=self.provider_bound,
            )
            alpha_change = np.max(np.abs(alpha_new - alpha))
            alpha = alpha_new

            if max(gamma_change, alpha_change) < self.outer_tol:
                break

        # Compute residuals and lambda_max at the converged null.
        r_null = discrete_residuals(time_int, event_np, alpha, eta)
        lambda_max = compute_discrete_lambda_max(r_null, X_fit, pf)
        self.lambda_max_ = lambda_max

        # =============================================================
        # 8. Build lambda sequence
        # =============================================================
        if self.lambda_path is not None:
            lam_seq = np.sort(np.atleast_1d(
                np.asarray(self.lambda_path, dtype=np.float64)
            ))[::-1]
        else:
            ratio = (
                self.lambda_min_ratio if self.lambda_min_ratio else 1e-4
            )
            lam_seq = np.exp(np.linspace(
                np.log(lambda_max + 1e-5),
                np.log(ratio * lambda_max),
                self.n_lambda,
            ))

        # =============================================================
        # 9. Three-layer warm-started path loop
        # =============================================================
        n_lam = len(lam_seq)
        coef_path = np.zeros((n_lam, p), dtype=np.float64)
        alpha_path = np.zeros((n_lam, K), dtype=np.float64)
        gamma_path = np.zeros((n_lam, n_providers), dtype=np.float64)
        df_path = np.zeros(n_lam, dtype=np.float64)
        n_iter_path = np.zeros(n_lam, dtype=np.int64)
        converged_path = np.zeros(n_lam, dtype=bool)
        neg_loglik_path = np.zeros(n_lam, dtype=np.float64)

        # Active set initialization.
        if self.use_active_set:
            active = pf == 0  # unpenalized always active
            pen_idx = np.where(pf > 0)[0]
            if len(pen_idx) > 0:
                active[pen_idx[0]] = True
        else:
            active = np.ones(p, dtype=bool)

        for l_idx, lam in enumerate(lam_seq):
            converged = False
            n_iter = 0

            for outer_iter in range(self.max_outer_iter):
                n_iter += 1
                old_beta = beta.copy()
                old_gamma = gamma.copy()
                old_alpha = alpha.copy()

                # --- Layer 1: Provider gamma update ---
                ll_res = discrete_loglik(
                    time_int, event_np, alpha, eta,
                )
                gamma_new = _provider_newton_from_loglik(
                    ll_res.score_beta, ll_res.working_weights,
                    provider_idx, n_providers, gamma, self.provider_bound,
                )
                # Update eta for gamma change.
                eta += (gamma_new - gamma)[provider_idx]
                gamma = gamma_new

                # --- Layer 2: Baseline hazard alpha update ---
                alpha = baseline_hazard_update(
                    alpha, time_int, event_np, eta,
                    n_events, bound=self.provider_bound,
                )

                # --- Layer 3: Penalized beta update (IRLS + CD) ---
                ll_res = discrete_loglik(
                    time_int, event_np, alpha, eta,
                )
                wr = compute_working_response(
                    ll_res.score_beta, ll_res.working_weights,
                )

                if self.use_active_set:
                    active_idx = np.where(active)[0]
                    X_active = X_fit[:, active_idx]
                    pf_active = pf[active_idx]
                    beta_active = beta[active_idx].copy()

                    beta_active, max_change, df = (
                        discrete_coordinate_descent_step(
                            X_active, wr, ll_res.working_weights,
                            beta_active, lam, pf_active,
                            tol=self.inner_tol,
                            max_iter=self.max_inner_iter,
                        )
                    )

                    # Write back and update eta incrementally.
                    for ii, col_i in enumerate(active_idx):
                        delta_j = beta_active[ii] - beta[col_i]
                        if delta_j != 0:
                            eta += delta_j * X_fit[:, col_i]
                        beta[col_i] = beta_active[ii]
                else:
                    old_eta = eta.copy()
                    # Remove beta contribution, recompute after.
                    beta, max_change, df = (
                        discrete_coordinate_descent_step(
                            X_fit, wr, ll_res.working_weights,
                            beta, lam, pf,
                            tol=self.inner_tol,
                            max_iter=self.max_inner_iter,
                        )
                    )
                    # Recompute eta = X @ beta + gamma[provider_idx]
                    eta = X_fit @ beta + gamma[provider_idx]

                # --- Convergence check ---
                max_beta_change = (
                    np.max(np.abs(beta - old_beta)) if p > 0 else 0.0
                )
                max_gamma_change = np.max(np.abs(gamma - old_gamma))
                max_alpha_change = np.max(np.abs(alpha - old_alpha))
                max_all = max(max_beta_change, max_gamma_change,
                              max_alpha_change)

                if max_all < self.outer_tol:
                    if self.use_active_set:
                        # Check KKT on inactive variables.
                        inactive_idx = np.where(
                            ~active & (pf > 0)
                        )[0]
                        if len(inactive_idx) == 0:
                            converged = True
                            break
                        # Screen inactive variables.
                        ll_check = discrete_loglik(
                            time_int, event_np, alpha, eta,
                        )
                        wr_check = compute_working_response(
                            ll_check.score_beta,
                            ll_check.working_weights,
                        )
                        n_added = 0
                        for j_inact in inactive_idx:
                            wXj = ll_check.working_weights * X_fit[
                                :, j_inact
                            ]
                            denom = np.dot(wXj, X_fit[:, j_inact])
                            if denom == 0:
                                continue
                            z_j = np.dot(wXj, wr_check) / denom
                            threshold = (
                                n * lam * pf[j_inact] / denom
                            )
                            if abs(z_j) > threshold:
                                active[j_inact] = True
                                n_added += 1
                        if n_added == 0:
                            converged = True
                            break
                    else:
                        converged = True
                        break

            # --- Unstandardize and store ---
            scales_safe = np.where(
                self._scales == 0, 1.0, self._scales,
            )
            beta_orig = beta / scales_safe
            alpha_adj = alpha - self._centers @ beta_orig

            coef_path[l_idx] = beta_orig
            alpha_path[l_idx] = alpha_adj
            gamma_path[l_idx] = gamma
            df_path[l_idx] = df
            n_iter_path[l_idx] = n_iter
            converged_path[l_idx] = converged
            neg_loglik_path[l_idx] = discrete_loglik(
                time_int, event_np, alpha, eta,
            ).neg_loglik

            # Update active set for next lambda.
            if self.use_active_set:
                active = (beta != 0) | (pf == 0)
                if not np.any(active):
                    active[:] = True

            logger.debug(
                "lambda[%d]=%.4e: n_iter=%d, converged=%s, "
                "nnz=%d, max_gamma=%.4f",
                l_idx, lam, n_iter, converged,
                int(np.sum(beta != 0)),
                float(np.max(np.abs(gamma))),
            )

        # =============================================================
        # 10. Store fitted attributes
        # =============================================================
        self.coef_path_ = coef_path
        self.baseline_hazard_path_ = alpha_path
        self.gamma_path_ = gamma_path
        self.lambda_path_ = lam_seq
        self.df_ = df_path
        self.n_iter_ = n_iter_path
        self.converged_ = converged_path
        self.neg_loglik_ = neg_loglik_path
        self.n_features_in_ = p
        self.n_obs_ = n
        self.n_events_ = int(np.sum(event_np))

        return self

    def _check_is_fitted(self):
        if not hasattr(self, "coef_path_"):
            raise NotFittedError(
                f"This {type(self).__name__} is not fitted yet."
            )

    def predict_provider_effect(self, which: int = -1) -> pd.DataFrame:
        """Provider effects at a given lambda index."""
        self._check_is_fitted()
        gamma = self.gamma_path_[which]
        return pd.DataFrame({
            "provider_id": self.provider_labels_,
            "gamma": gamma,
        })

    def predict_hazard(
        self, X, provider_id=None, which: int = -1,
    ) -> np.ndarray:
        """Predicted conditional hazard at each time point.

        Returns
        -------
        ndarray, shape (n_new, n_timepoints)
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        coef = self.coef_path_[which]
        alpha = self.baseline_hazard_path_[which]
        n_new = X.shape[0]
        n_t = len(alpha)

        # eta_{i,t} = alpha_t + X_i @ beta
        eta_base = X @ coef  # shape (n_new,)
        eta = np.outer(np.ones(n_new), alpha) + eta_base[:, np.newaxis]

        # Add provider effects.
        if provider_id is not None:
            gamma = self.gamma_path_[which]
            label_to_idx = {
                lab: i for i, lab in enumerate(self.provider_labels_)
            }
            for row_idx, pid in enumerate(provider_id):
                pidx = label_to_idx.get(pid, -1)
                if pidx >= 0:
                    eta[row_idx, :] += gamma[pidx]

        return 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))

    def predict_survival(
        self, X, provider_id=None, which: int = -1,
    ) -> np.ndarray:
        """Predicted survival probability at each time point.

        S(t) = prod_{s<=t} (1 - h(s))

        Returns
        -------
        ndarray, shape (n_new, n_timepoints)
        """
        hazard = self.predict_hazard(X, provider_id, which)
        return np.cumprod(1.0 - hazard, axis=1)


# ======================================================================
# Cross-validated ProviderPenalizedDiscreteSurvival
# ======================================================================

class ProviderPenalizedDiscreteSurvivalCV(ProviderModel):
    """Cross-validated provider-penalized discrete-time survival model.

    Fits the full regularization path on all data, then selects
    ``lambda_min`` or ``lambda_1se`` via k-fold cross-validation on
    person-period binary cross-entropy deviance.

    Fold assignment uses **event-stratified** assignment with a
    timepoint-coverage check, matching R ``cv.pp.DiscSurv``.
    Optionally, a user-supplied ``fold_id`` can enforce
    provider-stratified folds.

    R reference
    -----------
    ``grplasso/R/cv.ppDiscSurv.R``

    Loss function
    -------------
    Per person-period binary cross-entropy deviance, matching
    ``loss.Disc.Surv`` in R::

        L_{ik} = -2 * [y_{ik} * log(p_{ik}) + (1-y_{ik}) * log(1-p_{ik})]

    where ``p_{ik} = expit(alpha_k + X_i @ beta + gamma_{prov(i)})``.

    Each fold produces a mean loss across its expanded person-periods,
    then mean and SE are computed across folds.

    Parameters
    ----------
    n_folds : int, default=10
    se_rule : {"1se", "min"}, default="1se"
        ``"1se"`` selects ``lambda_1se_``; ``"min"`` selects ``lambda_min_``.
    random_state : int or None
    max_fold_retries : int, default=100
        Maximum retries for event-stratified fold assignment with
        timepoint coverage.
    fold_id : array-like or None
        User-supplied fold assignments (0-indexed).  Overrides
        ``n_folds`` and the internal assignment logic.
    **kwargs
        Forwarded to ``ProviderPenalizedDiscreteSurvival``
        (e.g., ``alpha``, ``provider_bound``, ``n_lambda``,
        ``penalty_factor``, ``standardize``, ``outer_tol``, etc.).
        ``lambda_path``, ``n_lambda``, ``lambda_min_ratio`` are only
        used for the full-data fit; fold models inherit the full-data
        lambda sequence.

    Attributes (after fit)
    ----------------------
    lambda_min_ : float
        Lambda with minimum mean CV error.
    lambda_1se_ : float
        Largest lambda within 1 SE of the minimum.
    lambda_ : float
        Selected lambda (``lambda_1se_`` if ``se_rule="1se"`` else
        ``lambda_min_``).
    lambda_min_idx_ : int
    lambda_1se_idx_ : int
    cv_mean_ : ndarray, shape (n_lambda,)
        Mean CV error per lambda.
    cv_se_ : ndarray, shape (n_lambda,)
        Standard error of CV error per lambda.
    model_ : ProviderPenalizedDiscreteSurvival
        Full-data fit.
    coef_ : ndarray, shape (p,)
        Coefficients at the selected lambda.
    baseline_hazard_ : ndarray, shape (K_time,)
        Baseline hazard at the selected lambda.
    gamma_ : ndarray, shape (K_prov,)
        Provider effects at the selected lambda.
    coef_path_ : ndarray
    baseline_hazard_path_ : ndarray
    gamma_path_ : ndarray
    lambda_path_ : ndarray
    provider_labels_ : ndarray
    fold_assignment_ : ndarray of int, shape (n,)
    """

    def __init__(
        self,
        n_folds: int = 10,
        se_rule: str = "1se",
        random_state: Optional[int] = None,
        max_fold_retries: int = 100,
        fold_id: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """Cross-validated provider-penalized discrete survival."""
        self.n_folds = n_folds
        self.se_rule = se_rule
        self.random_state = random_state
        self.max_fold_retries = max_fold_retries
        self.fold_id = fold_id
        self._ds_kwargs = kwargs

    def _model_kwargs(self, exclude_lambda: bool = False) -> dict:
        """Shared kwargs for ProviderPenalizedDiscreteSurvival.

        Parameters
        ----------
        exclude_lambda : bool
            If True, strip ``lambda_path``, ``n_lambda``, and
            ``lambda_min_ratio`` (used for fold models that inherit
            the full-data path).
        """
        kw = dict(self._ds_kwargs)
        if exclude_lambda:
            for key in ('lambda_path', 'n_lambda', 'lambda_min_ratio'):
                kw.pop(key, None)
        return kw

    # ------------------------------------------------------------------
    # Fold assignment
    # ------------------------------------------------------------------

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
        ``max_fold_retries`` times, matching ``grplasso``'s
        ``cv.pp.DiscSurv``.
        """
        for _attempt in range(self.max_fold_retries):
            # Event-stratified: proportional events / censored per fold.
            idx_event = np.where(event == 1)[0]
            idx_cens = np.where(event == 0)[0]

            fold = np.empty(n, dtype=np.int64)

            perm_e = rng.permutation(len(idx_event))
            for i, pos in enumerate(perm_e):
                fold[idx_event[pos]] = i % self.n_folds

            perm_c = rng.permutation(len(idx_cens))
            for i, pos in enumerate(perm_c):
                fold[idx_cens[pos]] = i % self.n_folds

            # Check timepoint coverage in every training fold.
            coverage_ok = True
            for f in range(self.n_folds):
                train_mask = fold != f
                if len(np.unique(time_int[train_mask])) < K:
                    coverage_ok = False
                    break

            if coverage_ok:
                return fold

        raise RuntimeError(
            f"Could not find a fold assignment where every training "
            f"fold contains all {K} distinct timepoints after "
            f"{self.max_fold_retries} attempts. Consider merging "
            f"adjacent timepoints or reducing n_folds."
        )

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        time: np.ndarray,
        event: np.ndarray,
        provider_id: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "ProviderPenalizedDiscreteSurvivalCV":
        """Fit CV to select lambda, then expose the full-data model.

        Parameters
        ----------
        X : ndarray or DataFrame, shape (n, p)
        time : ndarray, shape (n,)
        event : ndarray, shape (n,)
        provider_id : ndarray, shape (n,)
        sample_weight : ndarray or None, shape (n,)

        Returns
        -------
        self
        """
        if self.se_rule not in ("min", "1se"):
            raise ValueError(f"se_rule must be 'min' or '1se', got {self.se_rule!r}")
        # =============================================================
        # 1. Fit full-data model to get the lambda path
        # =============================================================
        full_model = ProviderPenalizedDiscreteSurvival(
            **self._model_kwargs(exclude_lambda=False)
        )
        full_model.fit(
            X, time, event, provider_id,
            sample_weight=sample_weight,
        )
        lambda_path = full_model.lambda_path_
        n_lambda = len(lambda_path)

        # =============================================================
        # 2. Prepare arrays and discretize once on full data
        # =============================================================
        if isinstance(X, pd.DataFrame):
            X_np = X.values.astype(np.float64)
        else:
            X_np = np.asarray(X, dtype=np.float64)
        time_np = np.asarray(time, dtype=np.float64)
        event_np = np.asarray(event, dtype=np.float64)
        prov_arr = np.asarray(provider_id)
        n = len(event_np)

        if sample_weight is not None:
            weight = np.asarray(sample_weight, dtype=np.float64)
        else:
            weight = np.ones(n, dtype=np.float64)

        # Discretize once — used for fold assignment and for mapping
        # validation times to the correct integer codes.
        time_int_full, timepoint_map, _ = discretize_times(
            time_np, event_np,
        )
        K = len(timepoint_map)

        # =============================================================
        # 3. Fold assignment
        # =============================================================
        if self.fold_id is not None:
            fold = np.asarray(self.fold_id, dtype=np.int64)
        else:
            rng = np.random.RandomState(self.random_state)
            fold = self._assign_folds(
                event_np, time_int_full, K, n, rng,
            )
        n_folds_actual = int(fold.max()) + 1
        self.fold_assignment_ = fold

        # =============================================================
        # 4. Cross-validation loop
        # =============================================================
        fold_losses = []

        for k in range(n_folds_actual):
            train_mask = fold != k
            val_mask = fold == k

            # --- Fit fold model at the full-data lambda path ---
            fold_model = ProviderPenalizedDiscreteSurvival(
                lambda_path=lambda_path,
                **self._model_kwargs(exclude_lambda=True),
            )
            wt_train = weight[train_mask]
            fold_model.fit(
                X_np[train_mask],
                time_np[train_mask],
                event_np[train_mask],
                prov_arr[train_mask],
                sample_weight=wt_train,
            )

            # --- Evaluate on validation fold ---
            n_fold_lam = len(fold_model.lambda_path_)
            X_val = X_np[val_mask]
            event_val = event_np[val_mask]
            prov_val = prov_arr[val_mask]

            # Use integer codes from full-data discretization
            # so alpha indices are consistent with the fold model.
            time_int_val = time_int_full[val_mask]

            # Provider label → index lookup for the fold model.
            label_to_idx = {
                lab: i
                for i, lab in enumerate(fold_model.provider_labels_)
            }

            # Person-period expansion for loss computation.
            pp = person_period_expand(time_int_val, event_val)
            y_pp = pp['y']

            losses = np.full(n_lambda, np.nan)
            for l_idx in range(min(n_fold_lam, n_lambda)):
                coef = fold_model.coef_path_[l_idx]
                alpha_bh = fold_model.baseline_hazard_path_[l_idx]
                gamma = fold_model.gamma_path_[l_idx]

                # Linear predictor: X @ beta + gamma[provider].
                eta_val = X_val @ coef
                for row_idx, pid in enumerate(prov_val):
                    pidx = label_to_idx.get(pid, -1)
                    if pidx >= 0:
                        eta_val[row_idx] += gamma[pidx]
                    # Unseen providers get gamma=0.

                # Predicted hazard on expanded person-periods.
                p_hat = predict_discrete_hazard(
                    alpha_bh, eta_val, time_int_val,
                )
                p_hat = np.clip(p_hat, 1e-5, 1.0 - 1e-5)

                # Binary cross-entropy deviance (matches R
                # loss.Disc.Surv).
                loss = -2.0 * (
                    y_pp * np.log(p_hat)
                    + (1.0 - y_pp) * np.log(1.0 - p_hat)
                )
                losses[l_idx] = float(np.mean(loss))

            fold_losses.append(losses)

        # =============================================================
        # 5. Aggregate CV results and select lambda
        # =============================================================
        loss_matrix = np.array(fold_losses)  # (n_folds, n_lambda)

        # Eliminate saturated lambda values (any fold is NaN).
        valid = np.all(np.isfinite(loss_matrix), axis=0)
        if not np.any(valid):
            raise RuntimeError(
                "All lambda values produced non-finite CV losses. "
                "Consider reducing n_lambda or increasing "
                "lambda_min_ratio."
            )

        cv_mean = np.full(n_lambda, np.nan)
        cv_se = np.full(n_lambda, np.nan)
        cv_mean[valid] = np.mean(
            loss_matrix[:, valid], axis=0,
        )
        cv_se[valid] = (
            np.std(loss_matrix[:, valid], axis=0, ddof=1)
            / np.sqrt(n_folds_actual)
        )

        # lambda_min: minimum mean CV error.
        valid_idx = np.where(valid)[0]
        idx_min = valid_idx[int(np.argmin(cv_mean[valid_idx]))]
        lambda_min = lambda_path[idx_min]

        # lambda_1se: largest lambda (earliest index in the
        # descending path) whose mean CV error is within 1 SE of
        # the minimum.
        threshold = cv_mean[idx_min] + cv_se[idx_min]
        candidates = valid_idx[cv_mean[valid_idx] <= threshold]
        idx_1se = int(candidates[0])
        lambda_1se = lambda_path[idx_1se]

        # =============================================================
        # 6. Store fitted attributes
        # =============================================================
        self.cv_mean_ = cv_mean
        self.cv_se_ = cv_se
        self.lambda_min_ = float(lambda_min)
        self.lambda_1se_ = float(lambda_1se)
        self.lambda_min_idx_ = int(idx_min)
        self.lambda_1se_idx_ = int(idx_1se)
        idx_selected = idx_1se if (self.se_rule == "1se") else idx_min
        self.lambda_ = float(lambda_path[idx_selected])

        # Full-data model and convenience accessors.
        self.model_ = full_model
        self.coef_ = full_model.coef_path_[idx_selected]
        self.baseline_hazard_ = (
            full_model.baseline_hazard_path_[idx_selected]
        )
        self.gamma_ = full_model.gamma_path_[idx_selected]

        # Expose full paths from the full-data model.
        self.coef_path_ = full_model.coef_path_
        self.baseline_hazard_path_ = (
            full_model.baseline_hazard_path_
        )
        self.gamma_path_ = full_model.gamma_path_
        self.lambda_path_ = full_model.lambda_path_
        self.provider_labels_ = full_model.provider_labels_
        self.time_points_ = full_model.time_points_
        self.n_timepoints_ = full_model.n_timepoints_
        self.n_providers_ = full_model.n_providers_

        return self

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def _check_is_fitted(self):
        if not hasattr(self, 'model_'):
            raise NotFittedError(
                f"This {type(self).__name__} is not fitted yet."
            )

    def predict_hazard(
        self, X, provider_id=None, which=None,
    ) -> np.ndarray:
        """Predicted hazard at the selected lambda.

        Returns
        -------
        ndarray, shape (n_new, n_timepoints)
        """
        self._check_is_fitted()
        if which is None:
            which = (
                self.lambda_1se_idx_
                if (self.se_rule == "1se")
                else self.lambda_min_idx_
            )
        return self.model_.predict_hazard(X, provider_id, which=which)

    def predict_survival(
        self, X, provider_id=None, which=None,
    ) -> np.ndarray:
        """Predicted survival at the selected lambda.

        Returns
        -------
        ndarray, shape (n_new, n_timepoints)
        """
        self._check_is_fitted()
        if which is None:
            which = (
                self.lambda_1se_idx_
                if (self.se_rule == "1se")
                else self.lambda_min_idx_
            )
        return self.model_.predict_survival(
            X, provider_id, which=which,
        )

    def predict_provider_effect(self, which=None) -> pd.DataFrame:
        """Provider effects at the selected lambda."""
        self._check_is_fitted()
        if which is None:
            which = (
                self.lambda_1se_idx_
                if (self.se_rule == "1se")
                else self.lambda_min_idx_
            )
        return self.model_.predict_provider_effect(which=which)
