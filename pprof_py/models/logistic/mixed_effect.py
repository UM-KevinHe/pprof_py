"""Logistic Mixed Effect Model: Fixed provider + Random cluster effects.

Implements Newton-Raphson with Gauss-Hermite quadrature for models where:
- Provider/facility effects are FIXED (estimated directly via NR)
- Cluster/hospital effects are RANDOM (integrated out via GH quadrature)
- Covariates have fixed effects

Primary use case: Standardized Readmission Ratio (srr) and similar measures.

Reference: He et al. (2013) "Evaluating Hospital Readmission Rates in
Dialysis Facilities; Adjusting for Hospital Effects", Lifetime Data Analysis.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit as plogis
from numpy.polynomial.hermite import hermgauss
from typing import Optional, List

from ...base import ProviderModel

from ...exceptions import NotFittedError
from ...inference.logistic import LogisticMixedEffectInferenceMixin
from ...measures.logistic import LogisticMixedEffectMeasuresMixin

logger = logging.getLogger(__name__)

#: Floor for a provider's Newton information (it can reach <= 0 once alpha_var > 4).
_INFO_FLOOR = 1e-8


def gauss_hermite_normal(n_nodes: int, sigma: float):
    """Generate GH quadrature nodes/weights for N(0, sigma^2).

    Matches R's statmod::gauss.quad.prob(n, 'normal', sigma=sigma).

    Parameters
    ----------
    n_nodes : int
        Number of quadrature nodes.
    sigma : float
        Standard deviation of the normal distribution.

    Returns
    -------
    nodes : np.ndarray of shape (n_nodes,)
    weights : np.ndarray of shape (n_nodes,)
    """
    nodes_raw, weights_raw = hermgauss(n_nodes)
    nodes = sigma * np.sqrt(2) * nodes_raw
    weights = weights_raw / np.sqrt(np.pi)
    return nodes, weights


class LogisticMixedEffectModel(ProviderModel, LogisticMixedEffectInferenceMixin, LogisticMixedEffectMeasuresMixin):
    """Logistic model with fixed provider effects and random cluster effects.

    **Stage 3** of a three-stage estimation pipeline (He et al., 2013):

    * Stage 1 (``LogisticFixedEffectModel``): estimate β (covariate effects).
    * Stage 2 (``LogisticRandomEffectModel``): estimate σ² (random-effect
      variance), holding β fixed from Stage 1.
    * **Stage 3 (this class):** estimate γ (provider effects), holding
      β and σ fixed from Stages 1–2, integrating the cluster random
      effect out via Gauss-Hermite quadrature.

    Estimates the model:
        logit(P(Y=1)) = gamma_j + alpha_h + X @ beta
    where:
        gamma_j = fixed effect for provider j (estimated via Newton-Raphson)
        alpha_h ~ N(0, sigma^2) = random effect for cluster h (integrated out)
        beta = fixed covariate effects (from Stage 1, held fixed)

    The algorithm iterates:
    1. GH quadrature to compute posterior moments E[alpha|Y], Var[alpha|Y]
    2. Newton-Raphson updates for gamma (provider) given posterior moments
    3. Convergence via the relative change in the objective (default) or the
       largest change in gamma

    Responsibilities are split across mixins so this class stays focused on
    configuration, input handling, fitting, and prediction:

    - `LogisticMixedEffectInferenceMixin` (`pprof_py.inference.logistic`): covariate (beta)
      statistical inference, `summary()`; provider-effect tests, `test()`; confidence intervals.
    - `LogisticMixedEffectMeasuresMixin` (`pprof_py.measures.logistic`): standardized rates/ratios,
      `calculate_standardized_measures()`.

    Parameters
    ----------
    n_nodes : int, default=20
        Number of Gauss-Hermite quadrature nodes.
    max_iter : int, default=10000
        Maximum Newton-Raphson iterations.
    tol : float, default=1e-5
        Convergence tolerance for ``convergence_criterion``.
    bound : float, default=10.0
        Bound for provider effects; see ``bound_mode``.
    bound_mode : {"relative", "absolute"}, default="relative"
        ``"relative"`` clips gamma to ``median(gamma) +/- bound`` at every
        iteration, which is symmetric around the typical provider whatever the
        baseline log-odds. ``"absolute"`` clips to ``[-bound, bound]``, as R's
        ``glmm.fac.hosp`` does; only providers at the bound differ.
    convergence_criterion : {"relative", "max_delta_gamma"}, default="relative"
        ``"relative"``: ``|obj_t - obj_{t-1}| / |obj_t - obj_1|`` for the
        objective below, as in R (a zero denominator counts as converged when
        the numerator is also zero, and otherwise does not stop the fit).
        ``"max_delta_gamma"``: the largest absolute change in gamma, which does
        not depend on the starting value.

    Notes
    -----
    sigma is held at its Stage 2 value. Earlier versions offered
    ``update_sigma=True``, which re-estimated sigma from the posterior moments;
    that update drove sigma toward zero (as it does in R's ``glmm.fac.hosp``)
    and was removed.

    Attributes
    ----------
    gamma_ : np.ndarray
        Estimated provider fixed effects.
    beta_ : np.ndarray
        Covariate fixed effects.  Held fixed from Stage 1
        (``LogisticFixedEffectModel``); this class (Stage 3) does
        not re-estimate β.
    sigma_ : float
        Cluster random effect standard deviation, held fixed from
        Stage 2 (``LogisticRandomEffectModel``).
    alpha_mean_ : np.ndarray
        Posterior mean of cluster effects (observation-level).
    alpha_var_ : np.ndarray
        Posterior variance of cluster effects (observation-level).
    alpha_mean_cluster_ : np.ndarray
        Posterior mean of cluster effects (cluster-level).
    alpha_var_cluster_ : np.ndarray
        Posterior variance of cluster effects (cluster-level).
    xbeta_ : np.ndarray
        Covariate linear predictor (X @ beta).
    fitted_ : np.ndarray
        Fitted probabilities (including all effects).
    provider_ids_ : np.ndarray
        Unique provider identifiers.
    cluster_ids_ : np.ndarray
        Unique cluster identifiers.
    n_providers_ : int
        Number of providers.
    n_clusters_ : int
        Number of clusters.
    iterations_ : int
        Number of iterations run.
    convergence_ : float
        Final value of the convergence criterion (``inf`` if it could not be
        evaluated).
    converged_ : bool
        Whether the criterion fell below ``tol`` within ``max_iter``.
    stage1_model_ : LogisticFixedEffectModel or None
        The Stage 1 model passed to ``fit``; ``summary()`` reports its Wald table.
    """

    def __init__(
        self,
        n_nodes: int = 20,
        max_iter: int = 10000,
        tol: float = 1e-5,
        bound: float = 10.0,
        bound_mode: str = "relative",
        convergence_criterion: str = "relative",
    ):
        """Logistic mixed-effect provider model."""
        self.n_nodes = n_nodes
        self.max_iter = max_iter
        self.tol = tol
        self.bound = bound
        self.bound_mode = bound_mode
        self.convergence_criterion = convergence_criterion

        # Results (populated after fit)
        self.gamma_: Optional[np.ndarray] = None
        self.beta_: Optional[np.ndarray] = None
        self.coefficients_: Optional[dict] = None
        self.sigma_: Optional[float] = None
        self.alpha_mean_: Optional[np.ndarray] = None
        self.alpha_var_: Optional[np.ndarray] = None
        self.alpha_mean_cluster_: Optional[np.ndarray] = None
        self.alpha_var_cluster_: Optional[np.ndarray] = None
        self.xbeta_: Optional[np.ndarray] = None
        self.fitted_: Optional[np.ndarray] = None
        self.provider_ids_: Optional[np.ndarray] = None
        self.cluster_ids_: Optional[np.ndarray] = None
        self.n_providers_: Optional[int] = None
        self.n_clusters_: Optional[int] = None
        self.iterations_: Optional[int] = None
        self.convergence_: Optional[float] = None
        self.converged_: Optional[bool] = None
        self.stage1_model_ = None

        # Internal indices
        self._provider_idx: Optional[np.ndarray] = None
        self._cluster_idx: Optional[np.ndarray] = None
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._obs: Optional[np.ndarray] = None
        self._x_vars: Optional[List[str]] = None

    def fit(
        self,
        data: pd.DataFrame,
        y_var: str,
        x_vars: List[str],
        provider_var: str,
        cluster_var: str,
        gamma_init: np.ndarray,
        beta_init: np.ndarray,
        sigma_init: float,
        obs_var: Optional[str] = None,
        verbose: bool = True,
        stage1_model=None,
    ) -> "LogisticMixedEffectModel":
        """Fit the mixed effect model via Newton-Raphson + GH quadrature.

        Parameters
        ----------
        data : pd.DataFrame
            Dataset with response, covariates, provider and cluster IDs.
        y_var : str
            Response variable column name used for model fitting (NR
            iterations). Typically 'Y_adj' (boundary-adjusted outcome)
            to ensure finite gamma estimates.
        x_vars : list of str
            Covariate column names.
        provider_var : str
            Provider/facility ID column (fixed effect grouping).
        cluster_var : str
            Cluster/hospital ID column (random effect grouping).
        gamma_init : np.ndarray
            Initial provider effects, shape (n_providers,).
        beta_init : np.ndarray
            Covariate effects from Stage 1
            (``LogisticFixedEffectModel``), shape (n_covariates,).
            Held fixed throughout Stage 3 iteration; this class
            estimates only γ.
        sigma_init : float
            Cluster random effect std dev from Stage 2
            (``LogisticRandomEffectModel``), held fixed.
        obs_var : str, optional
            Column name for the actual observed outcome, used for
            computing SRR observed counts and resampling p-values.
            Defaults to ``y_var`` when not specified.  Use this when
            the model is fitted on an adjusted response (Y_adj) but
            the standardized ratio should reflect the true binary
            outcome (e.g., 'readmit30_flag').
        verbose : bool, default=True
            Print iteration progress.
        stage1_model : LogisticFixedEffectModel, optional
            The fitted Stage 1 model that produced ``beta_init``; stored as
            ``stage1_model_`` for ``summary()``.

        Returns
        -------
        self
        """
        if self.bound_mode not in ("relative", "absolute"):
            raise ValueError("bound_mode must be 'relative' or 'absolute'.")
        if self.convergence_criterion not in ("relative", "max_delta_gamma"):
            raise ValueError("convergence_criterion must be 'relative' or 'max_delta_gamma'.")
        self._x_vars = list(x_vars)

        # Extract arrays
        X = data[x_vars].values.astype(float)
        y = data[y_var].values.astype(float)
        self._X = X
        self._y = y

        # Observed outcome for SRR and resampling (may differ from y_var)
        if obs_var is not None:
            self._obs = data[obs_var].values.astype(float)
        else:
            self._obs = y

        # Provider and cluster indices
        prov_cat = data[provider_var].cat if hasattr(data[provider_var], 'cat') else pd.Categorical(data[provider_var])
        clust_cat = data[cluster_var].cat if hasattr(data[cluster_var], 'cat') else pd.Categorical(data[cluster_var])

        if hasattr(data[provider_var], 'cat'):
            self._provider_idx = data[provider_var].cat.codes.values
            self.provider_ids_ = data[provider_var].cat.categories.values
        else:
            cat = pd.Categorical(data[provider_var])
            self._provider_idx = cat.codes
            self.provider_ids_ = cat.categories.values

        if hasattr(data[cluster_var], 'cat'):
            self._cluster_idx = data[cluster_var].cat.codes.values
            self.cluster_ids_ = data[cluster_var].cat.categories.values
        else:
            cat = pd.Categorical(data[cluster_var])
            self._cluster_idx = cat.codes
            self.cluster_ids_ = cat.categories.values

        self.n_providers_ = len(self.provider_ids_)
        self.n_clusters_ = len(self.cluster_ids_)

        prov_idx = self._provider_idx
        clust_idx = self._cluster_idx

        # Initialize
        gamma = gamma_init.copy()
        beta = beta_init.copy()
        sigma = sigma_init
        xbeta = X @ beta
        gamma_obs = gamma[prov_idx]

        nodes, weights = gauss_hermite_normal(self.n_nodes, sigma)

        if verbose:
            logger.info("Fitting mixed effect model (NR + GH quadrature)...")
            logger.info(f"  Providers: {self.n_providers_}, Clusters: {self.n_clusters_}")
            logger.info(f"  observations: {len(y):,}, Covariates: {len(x_vars)}")

        iter_count = 0
        crit = 1.0
        obj = 0.0
        obj_new = 1.0
        obj_init = None
        n_floored = 0               # provider-iterations whose Newton information was floored
        zero_denominator = False    # relative criterion met 0 in its denominator with a nonzero numerator

        while iter_count <= self.max_iter and crit >= self.tol:
            obj = obj_new
            iter_count += 1

            # === GH Quadrature: posterior moments for cluster effects ===
            lkd = np.zeros((self.n_clusters_, self.n_nodes))
            for k in range(self.n_nodes):
                linear = nodes[k] + gamma_obs + xbeta
                q = plogis(-linear)
                log_lik_obs = np.log(y * (1 - q) + (1 - y) * q)
                # Sum by cluster
                np.add.at(lkd[:, k], clust_idx, log_lik_obs)

            # Scale for numerical precision
            lkd_max = lkd.max(axis=1, keepdims=True)
            lkd_scaled = np.exp(lkd - lkd_max)

            # Posterior moments
            denom = lkd_scaled @ weights
            alpha_mean_cluster = (lkd_scaled @ (nodes * weights)) / denom
            alpha_var_cluster = (
                (lkd_scaled @ (nodes**2 * weights)) / denom
                - alpha_mean_cluster**2
            )

            # expand to observation level
            alpha_mean = alpha_mean_cluster[clust_idx]
            alpha_var = alpha_var_cluster[clust_idx]

            # === Newton-Raphson update: provider effects ===
            q = plogis(-(alpha_mean + gamma_obs + xbeta))
            p = 1.0 - q
            pq = p * q

            # Score and information (He et al. 2013, pp.511)
            gamma_score = y - p - 0.5 * alpha_var * pq * (q - p)
            gamma_info = pq + 0.5 * alpha_var * pq * (p**2 + q**2 - 4 * pq)

            # Aggregate by provider
            score_by_prov = np.bincount(prov_idx, weights=gamma_score,
                                        minlength=self.n_providers_)
            info_by_prov = np.bincount(prov_idx, weights=gamma_info,
                                       minlength=self.n_providers_)
            low_info = info_by_prov <= _INFO_FLOOR
            if low_info.any():      # the step would divide by ~0 or reverse the score's sign
                n_floored += int(low_info.sum())
                info_by_prov = np.maximum(info_by_prov, _INFO_FLOOR)
            gamma_prev = gamma
            gamma = gamma + score_by_prov / info_by_prov
            if self.bound_mode == "absolute":
                gamma = np.clip(gamma, -self.bound, self.bound)
            else:
                gamma_median = np.median(gamma)
                gamma = np.clip(gamma, gamma_median - self.bound, gamma_median + self.bound)
            gamma_obs = gamma[prov_idx]

            # === Convergence criterion (offset approach) ===
            q_new = plogis(-(alpha_mean + gamma_obs + xbeta))
            p_new = 1.0 - q_new
            pq_new = p_new * q_new

            obj_new = np.sum(
                (alpha_mean + gamma_obs + xbeta) * y
                + np.log(q_new)
                - 0.5 * alpha_var * pq_new        # second-order term: Var(alpha) * pq / 2
                - (alpha_mean**2 + alpha_var) / (2 * sigma**2)
            )

            if self.convergence_criterion == "max_delta_gamma":
                crit = float(np.max(np.abs(gamma - gamma_prev)))
            elif iter_count == 1:
                obj_init = obj_new
                crit = 1.0
            else:
                numerator, denominator = obj_new - obj, obj_new - obj_init
                if denominator == 0:
                    # 0/0: the objective has not moved since the first iteration (the start was
                    # already the solution); x/0 cannot be scaled, so keep iterating.
                    crit = 0.0 if numerator == 0 else np.inf
                    zero_denominator = zero_denominator or numerator != 0
                else:
                    crit = abs(numerator / denominator)
            if not np.isfinite(obj_new):
                crit = np.inf       # the fit has broken down (e.g. overflow); stop and report non-convergence
                break

            if verbose and (iter_count % 10 == 0 or crit < self.tol):
                logger.info(f"  Iter {iter_count}: crit = {crit:.8e}")

        converged = bool(crit < self.tol)
        if verbose:
            logger.info(f"Stopped after {iter_count} iterations (crit={crit:.2e}, converged={converged}).")
        if n_floored:
            warnings.warn(f"Newton information was <= {_INFO_FLOOR:g} for {n_floored} provider-iteration(s) and was "
                          "floored; those steps follow the score's sign and can reach the bound.", RuntimeWarning,
                          stacklevel=2)
        if zero_denominator:
            warnings.warn("The relative convergence criterion met a zero denominator with a nonzero numerator; "
                          "consider convergence_criterion='max_delta_gamma'.", RuntimeWarning, stacklevel=2)
        if not converged:
            warnings.warn(f"Did not converge within max_iter={self.max_iter} iterations "
                          f"(criterion {crit:.3g} >= tol {self.tol:g}).", RuntimeWarning, stacklevel=2)

        # Store results
        self.gamma_ = gamma
        self.beta_ = beta
        self.coefficients_ = {"beta": self.beta_, "gamma": self.gamma_}
        self.sigma_ = sigma
        self.alpha_mean_ = alpha_mean
        self.alpha_var_ = alpha_var
        self.alpha_mean_cluster_ = alpha_mean_cluster
        self.alpha_var_cluster_ = alpha_var_cluster
        self.xbeta_ = xbeta
        self.fitted_ = plogis(gamma_obs + alpha_mean + xbeta)
        self.iterations_ = iter_count
        self.convergence_ = float(crit)
        self.converged_ = converged
        self.stage1_model_ = stage1_model

        return self

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        """Raise `NotFittedError` if the model has not been fitted yet."""
        if self.gamma_ is None:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call `fit` first."
            )
