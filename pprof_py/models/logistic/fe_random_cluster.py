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
from ...inference.logistic import LogisticFERandomClusterInferenceMixin
from ...measures.logistic import LogisticFERandomClusterMeasuresMixin

logger = logging.getLogger(__name__)

#: Floor for a provider's Newton information (it can reach <= 0 once alpha_var > 4).
_INFO_FLOOR = 1e-8


def _cluster_modes(gamma, xbeta, y, prov_idx, clust_idx, sigma, n_clust, start=None, max_steps=100):
    """Mode and curvature scale of each cluster effect's posterior (the adaptive quadrature's center and scale).

    The log posterior of cluster h's effect ``a``, ``sum_h log f(y | eta + a) - a^2 / (2 sigma^2)``, is concave,
    so a guarded Newton iteration finds its mode; the scale is ``1 / sqrt(-second derivative)`` there.
    """
    base = gamma[prov_idx] + xbeta
    a = np.zeros(n_clust) if start is None else np.array(start, dtype=np.float64)
    for _ in range(max_steps):
        p = plogis(base + a[clust_idx])
        grad = np.bincount(clust_idx, weights=y - p, minlength=n_clust) - a / sigma ** 2
        curv = np.bincount(clust_idx, weights=p * (1.0 - p), minlength=n_clust) + 1.0 / sigma ** 2
        step = np.clip(grad / curv, -1.0, 1.0)
        a = a + step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = plogis(base + a[clust_idx])
    curv = np.bincount(clust_idx, weights=p * (1.0 - p), minlength=n_clust) + 1.0 / sigma ** 2
    return a, 1.0 / np.sqrt(curv)


def _marginal_terms(gamma, xbeta, y, prov_idx, clust_idx, t_nodes, log_wt, sigma, center, scale, n_prov, n_clust,
                    cell_idx=None, n_cells=0):
    """Marginal log-likelihood of the provider effects by adaptive Gauss-Hermite quadrature, its score, the posterior.

    Cluster h's effect is integrated at the nodes ``center_h + sqrt(2) scale_h t_k``, which follow the effect's
    posterior, so a few nodes are accurate at any cluster size. A row contributes ``y log p + (1 - y) log(1 - p)``
    at ``logit p = gamma_i + x beta + a`` (concave in ``gamma`` also for an adjusted, fractional ``y``). For fixed
    centers and scales the returned score and the Hessian built from the cell terms are this objective's exact
    derivatives. Returns the log-likelihood, the score per provider, the posterior node weights and node values
    per cluster, and, when ``cell_idx`` (provider x cluster cells) is given, each cell's score and curvature at
    every node.
    """
    n_nodes = t_nodes.size
    a_nodes = center[:, None] + np.sqrt(2.0) * scale[:, None] * t_nodes[None, :]
    base = gamma[prov_idx] + xbeta
    ll = np.empty((n_clust, n_nodes))
    prob = np.empty((base.size, n_nodes))
    for k in range(n_nodes):
        eta = base + a_nodes[clust_idx, k]
        ll[:, k] = np.bincount(clust_idx, weights=-(y * np.logaddexp(0.0, -eta) + (1.0 - y) * np.logaddexp(0.0, eta)),
                               minlength=n_clust)
        prob[:, k] = plogis(eta)
    log_post = ll + log_wt - a_nodes ** 2 / (2.0 * sigma ** 2)
    peak = log_post.max(axis=1, keepdims=True)
    log_z = np.log(np.exp(log_post - peak).sum(axis=1, keepdims=True)) + peak
    post = np.exp(log_post - log_z)
    loglik = float(np.sum(log_z[:, 0] + np.log(np.sqrt(2.0) * scale) - np.log(np.sqrt(2.0 * np.pi) * sigma)))
    score = np.bincount(prov_idx, weights=y - (prob * post[clust_idx]).sum(axis=1), minlength=n_prov)
    if cell_idx is None:
        return loglik, score, post, a_nodes, None, None
    s_cell = np.stack([np.bincount(cell_idx, weights=y - prob[:, k], minlength=n_cells) for k in range(n_nodes)], axis=1)
    v_cell = np.stack([np.bincount(cell_idx, weights=prob[:, k] * (1.0 - prob[:, k]), minlength=n_cells)
                       for k in range(n_nodes)], axis=1)
    return loglik, score, post, a_nodes, s_cell, v_cell


def _adaptive_rule(n_nodes):
    """Standard Gauss-Hermite nodes and the log weights of the adaptive rule (``log w_k + t_k^2``)."""
    t_nodes, w = hermgauss(n_nodes)
    return t_nodes, np.log(w) + t_nodes ** 2


def _marginal_loglik(gamma, xbeta, y, prov_idx, clust_idx, sigma, n_nodes, n_prov, n_clust):
    """The adaptive-quadrature marginal log-likelihood at ``gamma``, with the nodes centered for it."""
    center, scale = _cluster_modes(gamma, xbeta, y, prov_idx, clust_idx, sigma, n_clust)
    t_nodes, log_wt = _adaptive_rule(n_nodes)
    return _marginal_terms(gamma, xbeta, y, prov_idx, clust_idx, t_nodes, log_wt, sigma, center, scale, n_prov, n_clust)[0]


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


class LogisticFERandomClusterModel(ProviderModel, LogisticFERandomClusterInferenceMixin, LogisticFERandomClusterMeasuresMixin):
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

    - `LogisticFERandomClusterInferenceMixin` (`pprof_py.inference.logistic`): covariate (beta)
      statistical inference, `summary()`; provider-effect tests, `test()`; confidence intervals.
    - `LogisticFERandomClusterMeasuresMixin` (`pprof_py.measures.logistic`): standardized rates/ratios,
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
    convergence_criterion : {"max_delta_gamma", "relative"}, default="max_delta_gamma"
        ``"max_delta_gamma"``: the largest absolute change in gamma, which does
        not depend on the starting value. ``"relative"``: ``|obj_t - obj_{t-1}| /
        |obj_t - obj_1|`` for the objective below, as in R (a zero denominator
        counts as converged when the numerator is also zero, and otherwise does
        not stop the fit); it can stop well short of the fixed point, so use it
        for output comparable with R.
    estimator : {"he2013", "marginal"}, default="he2013"
        ``"he2013"``: the iteration of He et al. (2013) above, as in R's
        ``glmm.fac.hosp``; its fixed point depends on the start and is not the
        maximum likelihood estimate. ``"marginal"``: the maximum of the exact
        Gauss-Hermite marginal likelihood in gamma (beta and sigma fixed), by a
        projected Newton iteration with a line search. That likelihood is
        concave in gamma, so the estimate is unique and does not depend on the
        start. Rows contribute ``y log p + (1 - y) log(1 - p)``, also for the
        adjusted outcome; ``tol`` bounds the largest score and ``max_iter`` the
        Newton steps; the posterior moments of the cluster effects are those at
        the estimate.

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
    loglik_ : float
        The exact Gauss-Hermite marginal log-likelihood at ``gamma_``, under
        either estimator, which compares their fits.
    stage1_ : LogisticFixedEffectModel or None
        The Stage 1 model passed to ``fit``; ``summary()`` reports its Wald table.
    """

    def __init__(
        self,
        n_nodes: int = 20,
        max_iter: int = 10000,
        tol: float = 1e-5,
        bound: float = 10.0,
        bound_mode: str = "relative",
        convergence_criterion: str = "max_delta_gamma",
        estimator: str = "he2013",
    ):
        """Stage 3 of the three-stage logistic model."""
        self.n_nodes = n_nodes
        self.max_iter = max_iter
        self.tol = tol
        self.bound = bound
        self.bound_mode = bound_mode
        self.convergence_criterion = convergence_criterion
        self.estimator = estimator

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
        self.stage1_ = None

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
        *,
        stage1=None,
        stage2=None,
        beta: Optional[np.ndarray] = None,
        sigma: Optional[float] = None,
        gamma_init: Optional[np.ndarray] = None,
        obs_var: Optional[str] = None,
        verbose: bool = True,
    ) -> "LogisticFERandomClusterModel":
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
        stage1 : LogisticFixedEffectModel, optional
            The fitted Stage 1 model. beta is its covariate effects, matched to
            ``x_vars`` by name and held fixed; this class estimates only γ. Stored
            as ``stage1_`` for ``summary()``.
        stage2 : LogisticRandomEffectModel, optional
            The fitted Stage 2 model, with ``provider_var`` as its provider and
            ``cluster_var`` as its one cluster factor. sigma is its cluster SD,
            taken by name and held fixed; the iteration starts from its provider
            effects, matched to this model's providers by ID, plus its intercept.
        beta, sigma, gamma_init : optional
            Explicit values instead of the stages (for example from R): the
            covariate effects in ``x_vars`` order and the cluster SD, both held
            fixed, and the starting provider effects in ``provider_ids_`` order.
            Pass either ``stage1`` and ``stage2`` or all three values.
        obs_var : str, optional
            Column name for the actual observed outcome, used for
            computing SRR observed counts and resampling p-values.
            Defaults to ``y_var`` when not specified.  Use this when
            the model is fitted on an adjusted response (Y_adj) but
            the standardized ratio should reflect the true binary
            outcome (e.g., 'readmit30_flag').
        verbose : bool, default=True
            Print iteration progress.

        Returns
        -------
        self
        """
        if self.bound_mode not in ("relative", "absolute"):
            raise ValueError("bound_mode must be 'relative' or 'absolute'.")
        if self.convergence_criterion not in ("relative", "max_delta_gamma"):
            raise ValueError("convergence_criterion must be 'relative' or 'max_delta_gamma'.")
        if self.estimator not in ("he2013", "marginal"):
            raise ValueError("estimator must be 'he2013' or 'marginal'.")
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

        beta_init, sigma_init, gamma_init = self._stage_values(stage1, stage2, beta, sigma, gamma_init, x_vars,
                                                               provider_var, cluster_var)

        # Initialize
        gamma = gamma_init.copy()
        beta = beta_init.copy()
        sigma = sigma_init
        xbeta = X @ beta
        gamma_obs = gamma[prov_idx]

        nodes, weights = gauss_hermite_normal(self.n_nodes, sigma)
        if self.estimator == "marginal":
            return self._fit_marginal(y, xbeta, prov_idx, clust_idx, gamma, beta, sigma, nodes, weights, stage1, verbose)

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
        self.stage1_ = stage1
        self.loglik_ = _marginal_loglik(gamma, xbeta, y, prov_idx, clust_idx, sigma, self.n_nodes, self.n_providers_,
                                        self.n_clusters_)

        return self

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _fit_marginal(self, y, xbeta, prov_idx, clust_idx, gamma, beta, sigma, nodes, weights, stage1, verbose):
        """Maximize the marginal likelihood in gamma (beta, sigma fixed): projected Newton with a line search.

        The exact marginal likelihood is concave in gamma (its integrand is jointly log-concave), so its
        maximizer is unique. It is computed by adaptive Gauss-Hermite quadrature: each iteration centers and
        scales every cluster's nodes at its posterior, then takes a Newton step, with the Hessian the posterior
        covariance of the per-node cell scores minus their posterior mean curvature (sparse: providers interact
        only through shared clusters), and a backtracking line search at those nodes.
        """
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import spsolve
        n_prov, n_clust = self.n_providers_, self.n_clusters_
        t_nodes, log_wt = _adaptive_rule(self.n_nodes)
        codes, cell_idx = np.unique(np.asarray(prov_idx, np.int64) * n_clust + clust_idx, return_inverse=True)
        cell_prov, cell_clust = codes // n_clust, codes % n_clust
        order = np.argsort(cell_clust, kind="stable")
        edges = np.r_[0, np.cumsum(np.bincount(cell_clust, minlength=n_clust))]
        pair_a = np.concatenate([np.repeat(order[edges[h]:edges[h + 1]], edges[h + 1] - edges[h]) for h in range(n_clust)])
        pair_b = np.concatenate([np.tile(order[edges[h]:edges[h + 1]], edges[h + 1] - edges[h]) for h in range(n_clust)])

        def bounds(g):
            if self.bound_mode == "absolute":
                return -self.bound, self.bound
            center = np.median(g)
            return center - self.bound, center + self.bound

        def terms(g, center, scale, hessian=False):
            return _marginal_terms(g, xbeta, y, prov_idx, clust_idx, t_nodes, log_wt, sigma, center, scale,
                                   n_prov, n_clust, cell_idx if hessian else None, codes.size)

        gamma = np.asarray(gamma, dtype=np.float64)
        gamma = np.clip(gamma, *bounds(gamma))
        iter_count, crit, stalled, center = 0, np.inf, False, None
        while True:
            center, scale = _cluster_modes(gamma, xbeta, y, prov_idx, clust_idx, sigma, n_clust, start=center)
            loglik, score, post, a_nodes, s_cell, v_cell = terms(gamma, center, scale, hessian=True)
            lo, hi = bounds(gamma)
            held = ((gamma <= lo + 1e-12) & (score < 0)) | ((gamma >= hi - 1e-12) & (score > 0))
            free = np.flatnonzero(~held)
            crit = float(np.max(np.abs(score[free]))) if free.size else 0.0
            if verbose:
                logger.info(f"  Newton {iter_count}: log-likelihood {loglik:.10f}, max |score| {crit:.3e}")
            if crit < self.tol or iter_count >= self.max_iter or stalled:
                break
            post_cell = post[cell_clust]
            mean_cell = (post_cell * s_cell).sum(axis=1)
            cov = (post_cell[pair_a] * s_cell[pair_a] * s_cell[pair_b]).sum(axis=1) - mean_cell[pair_a] * mean_cell[pair_b]
            rows = np.r_[cell_prov[pair_a], cell_prov]
            cols = np.r_[cell_prov[pair_b], cell_prov]
            vals = np.r_[-cov, (post_cell * v_cell).sum(axis=1)]          # the negative Hessian
            neg_hess = coo_matrix((vals, (rows, cols)), shape=(n_prov, n_prov)).tocsc()
            step = np.zeros(n_prov)
            step[free] = spsolve(neg_hess[free][:, free], score[free])
            slope, t = float(score[free] @ step[free]), 1.0
            while True:
                proposal = gamma + t * step
                trial = np.clip(proposal, *bounds(proposal))
                if terms(trial, center, scale)[0] >= loglik + 1e-4 * t * slope:
                    break
                t *= 0.5
                if t < 1e-10:
                    stalled = True
                    break
            if not stalled:
                gamma = trial
                iter_count += 1
        converged = bool(crit < self.tol)
        if not converged:
            warnings.warn(f"The marginal-likelihood Newton iteration did not converge within max_iter={self.max_iter} "
                          f"iterations (max |score| {crit:.3g} >= tol {self.tol:g}).", RuntimeWarning, stacklevel=3)
        alpha_mean_cluster = (post * a_nodes).sum(axis=1)
        alpha_var_cluster = (post * a_nodes ** 2).sum(axis=1) - alpha_mean_cluster ** 2
        self.gamma_ = gamma
        self.beta_ = beta
        self.coefficients_ = {"beta": self.beta_, "gamma": self.gamma_}
        self.sigma_ = sigma
        self.alpha_mean_cluster_ = alpha_mean_cluster
        self.alpha_var_cluster_ = alpha_var_cluster
        self.alpha_mean_ = alpha_mean_cluster[clust_idx]
        self.alpha_var_ = alpha_var_cluster[clust_idx]
        self.xbeta_ = xbeta
        self.fitted_ = plogis(gamma[prov_idx] + self.alpha_mean_ + xbeta)
        self.iterations_ = iter_count
        self.convergence_ = crit
        self.converged_ = converged
        self.stage1_ = stage1
        self.loglik_ = loglik
        return self

    def _stage_values(self, stage1, stage2, beta, sigma, gamma_init, x_vars, provider_var, cluster_var):
        """beta, sigma and the starting gamma: from the fitted stages, or given explicitly."""
        given = [name for name, value in (("beta", beta), ("sigma", sigma), ("gamma_init", gamma_init)) if value is not None]
        if stage1 is not None or stage2 is not None:
            if stage1 is None or stage2 is None:
                raise ValueError("Pass both stage1 and stage2, or beta, sigma and gamma_init instead.")
            if given:
                raise ValueError(f"Pass either the fitted stages or {given}, not both.")
            return self._values_from_stages(stage1, stage2, x_vars, provider_var, cluster_var)
        missing = [name for name in ("beta", "sigma", "gamma_init") if name not in given]
        if missing:
            raise ValueError(f"Pass stage1 and stage2, or all of beta, sigma and gamma_init (missing: {missing}).")
        beta = np.asarray(beta, dtype=np.float64).ravel()
        gamma_init = np.asarray(gamma_init, dtype=np.float64).ravel()
        if beta.size != len(x_vars):
            raise ValueError(f"beta has {beta.size} entries for {len(x_vars)} covariates.")
        if gamma_init.size != self.n_providers_:
            raise ValueError(f"gamma_init has {gamma_init.size} entries for {self.n_providers_} providers.")
        return beta, float(sigma), gamma_init

    def _values_from_stages(self, stage1, stage2, x_vars, provider_var, cluster_var):
        """beta from Stage 1 by covariate name; sigma and the start from Stage 2, matched by name and ID."""
        names = list(getattr(stage1, "covariate_names_", None) or [])
        coef = getattr(stage1, "coefficients_", None)
        if not names or not isinstance(coef, dict) or "beta" not in coef:
            raise ValueError("stage1 must be a fitted LogisticFixedEffectModel.")
        if sorted(names) != sorted(x_vars):
            raise ValueError(f"stage1 was fit on covariates {names}, not x_vars {list(x_vars)}.")
        beta = pd.Series(np.asarray(coef["beta"], dtype=np.float64).ravel(), index=names).reindex(list(x_vars)).to_numpy()
        if (getattr(stage2, "sigma_", None) is None or getattr(stage2, "_provider_var", None) != provider_var
                or list(getattr(stage2, "_group_vars", None) or []) != [provider_var, cluster_var]):
            raise ValueError(f"stage2 must be a fitted LogisticRandomEffectModel with provider_var={provider_var!r} "
                             f"and cluster_vars=[{cluster_var!r}].")
        sigma = float(stage2.sigma_[cluster_var])
        fixed = stage2.coefficients_["beta"]
        if "(Intercept)" not in fixed.index:
            raise ValueError("stage2 needs an intercept (include_intercept=True).")
        blups = stage2.get_random_effects(provider_var)
        start = blups.reindex(self.provider_ids_).to_numpy(dtype=np.float64)
        if np.isnan(start).any():     # the stages saw the IDs as different types (for example numbers and text)
            by_text = {str(k): float(v) for k, v in blups.items()}
            start = np.array([by_text.get(str(k), np.nan) for k in self.provider_ids_])
        missing = [p for p, v in zip(self.provider_ids_, start) if np.isnan(v)]
        if missing:
            raise ValueError(f"stage2 has no effect for {len(missing)} provider(s), e.g. {missing[:5]}.")
        return beta, sigma, start + float(fixed["(Intercept)"])

    def _check_is_fitted(self) -> None:
        """Raise `NotFittedError` if the model has not been fitted yet."""
        if self.gamma_ is None:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call `fit` first."
            )
