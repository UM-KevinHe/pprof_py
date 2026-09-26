"""Pure-Python logistic GLMM with random intercepts, designed to mirror lme4::glmer.

Key design choices (for Bernoulli/logit models):

* spherical random effects: b = Lambda(theta) u, u ~ N(0, I)
* PIRLS uses lme4's joint fixed/random-effect equations
* nAGQ=0-style first stage: optimize theta while PIRLS updates BOTH beta and u
* Laplace (nAGQ=1) second stage: optimize theta and beta while PIRLS updates u
* full random-effect Hessian H = I + A' W A, including off-diagonal blocks for
  crossed random intercepts
* Laplace deviance = -2 log p(y | u_hat, beta) + u_hat'u_hat + log|H|

For a simple random-intercept term (1|group), lme4's theta parameter is the
standard deviation itself.  This implementation therefore optimizes sigma
directly and permits sigma=0, matching lme4's boundary parameterization.

This is a statistical/numerical reimplementation of the lme4 formulation, not
a promise of bit-for-bit identity with Eigen/CHOLMOD floating-point results.

The implementation follows the current lme4 glmer workflow more closely than
v2: stage 1 is an nAGQ=0 theta-only optimization (default BOBYQA when nlopt is
available), stage 2 is an nAGQ=1 theta+beta optimization (Nelder-Mead), and
all inner updates solve the joint penalized weighted least-squares system.
For Bernoulli/logit data, lme4s deviance is reproduced up to constants that
do not affect optimization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ...base import ProviderModel
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import splu

from ...inference.logistic import LogisticRandomEffectInferenceMixin
from ...measures.logistic import LogisticRandomEffectMeasuresMixin
from ...plotting.logistic import LogisticRandomEffectPlottingMixin
from ...exceptions import NotFittedError

try:
    import nlopt
except ImportError:  # optional; scipy fallbacks are used if unavailable
    nlopt = None

logger = logging.getLogger(__name__)


Array = np.ndarray


@dataclass
class _FitState:
    beta: Array
    u: Array
    sigma: Array
    pwrss: float
    converged: bool
    iterations: int


class LogisticRandomEffectModel(LogisticRandomEffectInferenceMixin, LogisticRandomEffectMeasuresMixin, LogisticRandomEffectPlottingMixin,
                                ProviderModel):
    """Bernoulli-logit GLMM with random intercepts, lme4-style.

    Model:

        logit(P(Y_i=1 | u)) = offset_i + X_i beta + A_i(theta) u

        u ~ N(0, I)

    For independent random-intercept terms,

        A = Z Lambda(theta),

    where Lambda is diagonal with entries sigma_k repeated over the levels of
    grouping factor k.

    Parameters
    ----------
    max_iter_pirls : int
        Maximum PIRLS iterations for each inner solve.
    tol_pirls : float
        Relative PWRSS convergence tolerance.
    max_iter_outer : int
        Maximum optimizer iterations/evaluations for each outer stage.
    tol_outer : float
        Optimizer tolerance.
    sigma_upper : float
        Upper bound used for sigma parameters.  The actual lme4-style lower
        bound is zero.  The upper bound is purely a numerical safety bound.
    stage2 : bool
        Run the Laplace/nAGQ=1 second stage.  Default True.
    verbose : bool
        Print progress.
    optimizer_stage1 : str
        Optimizer for stage 1.  "bobyqa" (requires nlopt) or "powell".
    optimizer_stage2 : str
        Optimizer for stage 2.  "nelder-mead".
    """

    def __init__(
        self,
        max_iter_pirls: int = 100,
        tol_pirls: float = 1e-8,
        max_iter_outer: int = 200,
        tol_outer: float = 1e-7,
        sigma_upper: float = np.inf,
        stage2: bool = True,
        verbose: bool = True,
        optimizer_stage1: str = "bobyqa",
        optimizer_stage2: str = "nelder-mead",
    ) -> None:
        """Logistic random-intercept provider model (R: ``lme4::glmer``)."""
        if sigma_upper <= 0 and not np.isinf(sigma_upper):
            raise ValueError("sigma_upper must be positive or np.inf")
        self.max_iter_pirls = int(max_iter_pirls)
        self.tol_pirls = float(tol_pirls)
        self.max_iter_outer = int(max_iter_outer)
        self.tol_outer = float(tol_outer)
        self.sigma_upper = float(sigma_upper)
        self.stage2 = bool(stage2)
        self.verbose = bool(verbose)
        self.optimizer_stage1 = str(optimizer_stage1).lower()
        self.optimizer_stage2 = str(optimizer_stage2).lower()

        # Public results
        self.coefficients_: Optional[Dict] = None
        self.variances_: Optional[Dict] = None
        self.fitted_: Optional[Array] = None
        self.residuals_: Optional[Array] = None
        self.aic_: Optional[float] = None
        self.bic_: Optional[float] = None
        self.loglike_: Optional[float] = None
        self.sigma_: Optional[Dict[str, float]] = None
        self.provider_ids_: Optional[Array] = None
        self.provider_sizes_: Optional[Array] = None
        self.provider_indices_: Optional[Array] = None
        self.cluster_ids_: Optional[Dict[str, Array]] = None
        self.cluster_sizes_: Optional[Dict[str, Array]] = None
        self.cluster_indices_: Optional[Dict[str, Array]] = None
        self.xbeta_: Optional[Array] = None
        self.covariate_names_: Optional[List[str]] = None
        self.outcome_: Optional[Array] = None

        # Diagnostics
        self.converged_: bool = False
        self.stage1_result_ = None
        self.stage2_result_ = None
        self.stage1_iterations_: int = 0
        self.stage2_iterations_: int = 0
        self.pirls_iterations_: int = 0
        self.pirls_converged_: bool = False
        self.stage1_objective_: Optional[float] = None
        self.stage2_objective_: Optional[float] = None
        self.pwrss_: Optional[float] = None
        self.ldL2_: Optional[float] = None
        self.ussq_: Optional[float] = None

        # Internal data
        self._X: Optional[Array] = None
        self._offset: Optional[Array] = None
        self._y: Optional[Array] = None
        self._group_vars: Optional[List[str]] = None
        self._group_indices: Optional[List[Array]] = None
        self._n_groups: Optional[List[int]] = None
        self._group_labels: Optional[List[Array]] = None
        self._q_slices: Optional[List[slice]] = None
        self._q: int = 0
        self._p: int = 0
        self._n: int = 0
        self._beta_names: Optional[List[str]] = None

        self._beta: Optional[Array] = None
        self._u: Optional[Array] = None
        self._sigma: Optional[Array] = None
        self._last_u: Optional[Array] = None

    def _check_is_fitted(self) -> None:
        """Raise `NotFittedError` if the model has not been fitted yet."""
        if self.coefficients_ is None:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. Call `fit` first."
            )

    # ------------------------------------------------------------------
    # Public fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y_var: str,
        x_vars: Optional[List[str]] = None,
        provider_var: Optional[str] = None,
        cluster_vars: Optional[Union[str, Sequence[str]]] = None,
        offset_var: Optional[str] = None,
        include_intercept: bool = True,
        verbose: Optional[bool] = None,
        **kwargs,
    ) -> "LogisticRandomEffectModel":
        """Fit the Bernoulli logistic random-intercept model.

        Parameters
        ----------
        X : pd.DataFrame
            Data containing response, covariates, group variables, and
            optionally an offset column.
        y_var : str
            Response variable column name (binary 0/1).
        x_vars : list of str, optional
            Covariate column names to estimate jointly with random effects.
            If None, only the intercept (if include_intercept=True) and
            random effects are estimated.
        provider_var : str
            Column of provider IDs; its random intercepts are the provider effects that
            the measures, tests, and plots report.
        cluster_vars : str or list of str, optional
            Further grouping columns with their own (crossed) random intercepts, e.g.
            the hospital in Stage 2 of the three-stage model.
        offset_var : str, optional
            Column name for a known offset (e.g. X @ beta from a prior
            stage). Added to the linear predictor as-is.
        include_intercept : bool, default True
            Whether to include an intercept in the fixed-effect design.
        verbose : bool, optional
            Override instance-level verbosity.
        """
        if verbose is None:
            verbose = self.verbose

        if provider_var is None:
            raise ValueError("provider_var is required.")
        if isinstance(cluster_vars, str):
            cluster_vars = [cluster_vars]
        group_vars = [provider_var, *(cluster_vars or [])]
        self._provider_var = provider_var

        if not group_vars:
            raise ValueError("At least one grouping variable is required")

        x_vars = list(x_vars or [])
        required = [y_var, *x_vars, *group_vars]
        if offset_var is not None:
            required.append(offset_var)

        missing_cols = [c for c in required if c not in X.columns]
        if missing_cols:
            raise KeyError(f"Missing columns: {missing_cols}")

        work = X.loc[:, required].copy()
        complete = work.notna().all(axis=1)
        dropped = int((~complete).sum())
        work = work.loc[complete].reset_index(drop=True)

        y = work[y_var].to_numpy(dtype=float)
        if np.any(~np.isfinite(y)) or np.any((y < 0) | (y > 1)):
            raise ValueError("y must contain only finite values in [0, 1]")

        self.outcome_ = y.copy()
        self._y = y
        self._n = len(y)
        self._group_vars = group_vars

        # Fixed-effect design.
        if x_vars:
            X_fe = work[x_vars].to_numpy(dtype=float)
        else:
            X_fe = np.empty((self._n, 0), dtype=float)

        if include_intercept:
            X_fe = np.column_stack([np.ones(self._n), X_fe])
            self._beta_names = ["(Intercept)", *x_vars]
            self.covariate_names_ = list(x_vars)
        else:
            self._beta_names = list(x_vars)
            self.covariate_names_ = list(x_vars)

        self._X = X_fe
        self._p = X_fe.shape[1]

        if offset_var is None:
            self._offset = np.zeros(self._n, dtype=float)
        else:
            self._offset = work[offset_var].to_numpy(dtype=float)

        # Group coding: the provider factor first, then the cluster factors.
        sizes_by_var = []
        self._group_indices = []
        self._n_groups = []
        self._group_labels = []

        for gv in group_vars:
            cat = pd.Categorical(work[gv])
            idx = cat.codes.astype(np.int64, copy=False)
            if np.any(idx < 0):
                raise ValueError(f"Invalid missing group values in {gv}")
            labels = np.asarray(cat.categories)
            sizes = np.bincount(idx, minlength=len(labels))
            self._group_indices.append(idx)
            self._n_groups.append(len(labels))
            self._group_labels.append(labels)
            sizes_by_var.append(sizes)
        self.provider_ids_ = self._group_labels[0]
        self.provider_sizes_ = sizes_by_var[0]
        self.provider_indices_ = self._group_indices[0]
        self.cluster_ids_ = {gv: self._group_labels[j] for j, gv in enumerate(group_vars) if j > 0}
        self.cluster_sizes_ = {gv: sizes_by_var[j] for j, gv in enumerate(group_vars) if j > 0}
        self.cluster_indices_ = {gv: self._group_indices[j] for j, gv in enumerate(group_vars) if j > 0}

        self._q_slices = []
        start = 0
        for g in self._n_groups:
            self._q_slices.append(slice(start, start + g))
            start += g
        self._q = start

        # Initial values
        beta0 = np.zeros(self._p, dtype=float)
        if self._p:
            pbar = np.clip(y.mean(), 1e-8, 1 - 1e-8)
            if include_intercept:
                beta0[0] = np.log(pbar / (1 - pbar))
            else:
                beta0[:] = 0.0

        sigma0 = np.ones(len(group_vars), dtype=float)
        u0 = np.zeros(self._q, dtype=float)

        if verbose:
            logger.info("Fitting logistic GLMM (lme4-style PIRLS + Laplace)...")
            logger.info(f"  Observations: {self._n:,} (dropped {dropped:,})")
            logger.info(f"  Fixed effects: {self._p}")
            for gv, g in zip(group_vars, self._n_groups):
                logger.info(f"  RE \''{gv}\'': {g:,} levels")

        # ---------------- Stage 1: nAGQ = 0 ----------------
        stage1_cache = {"beta": beta0.copy(), "u": u0.copy()}

        def stage1_objective(sigma: Array) -> float:  # noqa: D401
            """Stage-1 (nAGQ=0) deviance as a function of *sigma*."""
            sigma = np.asarray(sigma, dtype=float)
            state = self._pirls(
                sigma=sigma,
                beta0=stage1_cache["beta"],
                u0=stage1_cache["u"],
                update_beta=True,
            )
            stage1_cache["beta"] = state.beta.copy()
            stage1_cache["u"] = state.u.copy()
            b = self._u_to_random_effects(sigma, state.u)
            eta = self._eta(state.beta, b)
            mu = expit(eta)
            w = np.maximum(mu * (1.0 - mu), 1e-12)
            H, _ = self._build_H_C(sigma, w)
            ldH = self._logdet_H(H)
            return float(state.pwrss + ldH)

        stage1 = self._optimize_stage1(stage1_objective, sigma0)

        sigma1 = np.asarray(stage1["x"], dtype=float)
        sigma1 = np.maximum(sigma1, 0.0)
        if np.isfinite(self.sigma_upper):
            sigma1 = np.minimum(sigma1, self.sigma_upper)
        state1 = self._pirls(
            sigma=sigma1,
            beta0=stage1_cache["beta"],
            u0=stage1_cache["u"],
            update_beta=True,
        )

        self.stage1_result_ = stage1
        self.stage1_iterations_ = int(stage1.get("nit", 0))
        self.stage1_objective_ = float(stage1.get("fun", np.nan))

        if verbose:
            s1_success = stage1.get('success')
            s1_opt = stage1.get('optimizer')
            logger.info(f"  Stage 1 (nAGQ=0): success={s1_success}, iters={self.stage1_iterations_}, optimizer={s1_opt}")
            logger.info("  Stage 1 sigma: " + ", ".join(f"{s:.8f}" for s in sigma1))

        if not self.stage2:
            sigma_opt = sigma1
            beta_opt = state1.beta
            u_opt = state1.u
            stage2 = None
            laplace_dev = self._laplace_deviance(sigma_opt, beta_opt, u_opt)
        else:
            # ---------------- Stage 2: nAGQ = 1 ----------------
            par0 = np.r_[sigma1, state1.beta]
            stage2_cache = {"u": state1.u.copy()}

            def stage2_objective(par: Array) -> float:  # noqa: D401
                """Stage-2 (nAGQ=1) Laplace deviance over sigma and beta."""
                sigma = np.clip(np.asarray(par[: len(group_vars)], dtype=float), 0.0, self.sigma_upper)
                beta = np.asarray(par[len(group_vars) :], dtype=float)
                state = self._pirls(
                    sigma=sigma,
                    beta0=beta,
                    u0=stage2_cache["u"],
                    update_beta=False,
                )
                stage2_cache["u"] = state.u.copy()
                return self._laplace_deviance(sigma, beta, state.u)

            bounds = [(0.0, self.sigma_upper)] * len(group_vars) + [(None, None)] * self._p
            stage2 = self._optimize_stage2(stage2_objective, par0, bounds)

            sigma_opt = np.maximum(np.asarray(stage2["x"][: len(group_vars)], dtype=float), 0.0)
            if np.isfinite(self.sigma_upper):
                sigma_opt = np.minimum(sigma_opt, self.sigma_upper)
            beta_opt = np.asarray(stage2["x"][len(group_vars) :], dtype=float)
            final_inner = self._pirls(
                sigma=sigma_opt,
                beta0=beta_opt,
                u0=stage2_cache["u"],
                update_beta=False,
            )
            u_opt = final_inner.u
            laplace_dev = self._laplace_deviance(sigma_opt, beta_opt, u_opt)

        self.pwrss_, self.ldL2_, self.ussq_ = self._fit_components(sigma_opt, beta_opt, u_opt)

        self.stage2_result_ = stage2
        self.stage2_iterations_ = int(stage2.get("nit", 0)) if stage2 is not None else 0
        self.stage2_objective_ = float(stage2.get("fun", np.nan)) if stage2 is not None else None
        self.pirls_iterations_ = int(final_inner.iterations if self.stage2 else state1.iterations)
        self.pirls_converged_ = bool(final_inner.converged if self.stage2 else state1.converged)
        self.converged_ = bool(
            stage1.get("success", False)
            and (stage2 is None or stage2.get("success", False))
            and self.pirls_converged_
        )

        # Store final parameters.
        self._sigma = sigma_opt.copy()
        self._beta = beta_opt.copy()
        self._u = u_opt.copy()
        self._last_u = u_opt.copy()

        b_by_group = self._u_to_random_effects(self._sigma, self._u)
        eta = self._eta(beta_opt, b_by_group)
        fitted = expit(eta)

        self.fitted_ = fitted
        self.residuals_ = y - fitted
        self.xbeta_ = self._offset + (self._X @ beta_opt if self._p else 0.0)

        self.sigma_ = {gv: float(s) for gv, s in zip(group_vars, sigma_opt)}
        self.theta_ = sigma_opt.copy()  # lme4 theta analogue for random-intercept terms
        self.coefficients_ = {
            "beta": pd.Series(beta_opt, index=self._beta_names, dtype=float),
            "alpha": {
                gv: pd.Series(b_by_group[k], index=self._group_labels[k], dtype=float)
                for k, gv in enumerate(group_vars)
            },
        }

        vcov = self._fixed_effect_vcov(sigma_opt, beta_opt, u_opt)
        self.variances_ = {
            "beta": vcov,
            "alpha": {gv: float(s) ** 2 for gv, s in zip(group_vars, sigma_opt)},
        }

        self.loglike_ = -0.5 * float(laplace_dev)
        n_params = self._p + len(group_vars)
        self.aic_ = float(laplace_dev + 2 * n_params)
        self.bic_ = float(laplace_dev + np.log(max(self._n, 1)) * n_params)

        if verbose:
            s2_success = stage2.get('success') if stage2 is not None else 'n/a'
            s2_opt = stage2.get('optimizer') if stage2 is not None else 'n/a'
            logger.info(f"  Stage 2 (Laplace): success={s2_success}, "
                        f"iters={self.stage2_iterations_}, optimizer={s2_opt}")
            for gv, s in self.sigma_.items():
                logger.info(f"  sigma({gv}): {s:.8f}")
            logger.info(f"  beta: {beta_opt}")
            logger.info(f"  PIRLS: iters={self.pirls_iterations_}, converged={self.pirls_converged_}")
            logger.info(f"  Laplace deviance: {laplace_dev:.6f}")
            logger.info(f"  logLik: {self.loglike_:.6f}")
            logger.info(f"  AIC: {self.aic_:.6f}, BIC: {self.bic_:.6f}")
            logger.info(f"  Overall converged: {self.converged_}")

        return self

    def _optimize_stage1(self, fun, x0: Array) -> Dict:
        """Stage-1 optimizer matching glmer's default BOBYQA role."""
        if self.optimizer_stage1 == "bobyqa" and nlopt is None:
            import warnings
            warnings.warn(
                "optimizer_stage1='bobyqa' requires the nlopt package, "
                "which is not installed.  Falling back to scipy Powell.  "
                "Install nlopt (pip install pprof_py[random-effect]) or "
                "pass optimizer_stage1='powell' explicitly to silence "
                "this warning.",
                RuntimeWarning,
                stacklevel=3,
            )
            # Update the attribute so it reflects the optimizer actually used
            self.optimizer_stage1 = "powell"
        if self.optimizer_stage1 == "bobyqa" and nlopt is not None:
            opt = nlopt.opt(nlopt.LN_BOBYQA, len(x0))
            lo = np.zeros(len(x0), dtype=float)
            hi = np.full(len(x0), self.sigma_upper if np.isfinite(self.sigma_upper) else 1e6, dtype=float)
            opt.set_lower_bounds(lo)
            opt.set_upper_bounds(hi)
            opt.set_xtol_abs(self.tol_outer)
            opt.set_ftol_abs(self.tol_outer)
            opt.set_maxeval(max(self.max_iter_outer, 100))
            calls = [0]

            def wrapped(x, grad):  # noqa: D401
                """NLopt-compatible wrapper (ignores *grad*)."""
                calls[0] += 1
                return float(fun(np.asarray(x, dtype=float)))

            opt.set_min_objective(wrapped)
            x0c = np.clip(np.asarray(x0, dtype=float), lo, hi)
            x = opt.optimize(x0c)
            f = float(opt.last_optimum_value())
            code = int(opt.last_optimize_result())
            success = code > 0
            return {"x": np.asarray(x), "fun": f, "success": success,
                    "nit": calls[0], "optimizer": "nlopt.BOBYQA", "status": code}

        # Fallback: scipy Powell (bound-constrained, derivative-free)
        ub = self.sigma_upper if np.isfinite(self.sigma_upper) else None
        bounds = [(0.0, ub)] * len(x0)
        r = minimize(fun, np.asarray(x0, dtype=float), method="Powell", bounds=bounds,
                     options={"maxiter": self.max_iter_outer, "xtol": self.tol_outer,
                              "ftol": self.tol_outer})
        return {"x": r.x, "fun": float(r.fun), "success": bool(r.success),
                "nit": int(getattr(r, "nfev", 0)), "optimizer": "scipy.Powell",
                "status": int(getattr(r, "status", 0)),
                "message": str(getattr(r, "message", ""))}

    def _optimize_stage2(self, fun, x0: Array, bounds) -> Dict:
        """Stage-2 optimizer matching glmer's default Nelder-Mead role."""
        r = minimize(fun, np.asarray(x0, dtype=float), method="Nelder-Mead", bounds=bounds,
                     options={"maxiter": self.max_iter_outer, "xatol": self.tol_outer,
                              "fatol": self.tol_outer, "adaptive": True})
        return {"x": r.x, "fun": float(r.fun), "success": bool(r.success),
                "nit": int(getattr(r, "nit", 0)), "optimizer": "scipy.Nelder-Mead",
                "status": int(getattr(r, "status", 0)),
                "message": str(getattr(r, "message", ""))}

    # ------------------------------------------------------------------
    # Core lme4-style computations
    # ------------------------------------------------------------------

    def _eta(self, beta: Array, b_by_group: List[Array]) -> Array:
        eta = self._offset.copy()
        if self._p:
            eta += self._X @ beta
        for k, idx in enumerate(self._group_indices):
            eta += b_by_group[k][idx]
        return eta

    def _u_to_random_effects(self, sigma: Array, u: Array) -> List[Array]:
        result: List[Array] = []
        for k, sl in enumerate(self._q_slices):
            result.append(sigma[k] * u[sl])
        return result

    @staticmethod
    def _binomial_loglik(y: Array, eta: Array) -> float:
        return float(np.sum(y * eta - np.logaddexp(0.0, eta)))

    @staticmethod
    def _pwrss(y: Array, eta: Array, u: Array) -> float:
        loglik = LogisticRandomEffectModel._binomial_loglik(y, eta)
        return float(-2.0 * loglik + np.dot(u, u))

    def _build_H_C(self, sigma: Array, w: Array) -> Tuple[csr_matrix, Array]:
        """Build H = I + A\'WA and C = X\'WA without constructing A (n x q)."""
        q = self._q
        p = self._p

        rows: List[Array] = []
        cols: List[Array] = []
        vals: List[Array] = []

        # Identity + within-factor diagonal contributions.
        diag = np.ones(q, dtype=float)
        for k, idx in enumerate(self._group_indices):
            sl = self._q_slices[k]
            sums = np.bincount(idx, weights=w, minlength=self._n_groups[k])
            diag[sl] += sigma[k] ** 2 * sums

        d = np.arange(q, dtype=np.int64)
        rows.append(d)
        cols.append(d)
        vals.append(diag)

        # Cross-factor blocks (essential for crossed random effects).
        ng = len(self._group_indices)
        for k in range(ng):
            for l in range(k + 1, ng):
                if sigma[k] == 0.0 or sigma[l] == 0.0:
                    continue
                idx_k = self._group_indices[k]
                idx_l = self._group_indices[l]
                r = self._q_slices[k].start + idx_k
                c = self._q_slices[l].start + idx_l
                v = w * sigma[k] * sigma[l]
                rows.extend([r, c])
                cols.extend([c, r])
                vals.extend([v, v])

        H = coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(q, q),
        ).tocsr()

        if p == 0:
            C = np.empty((0, q), dtype=float)
        else:
            C = np.zeros((p, q), dtype=float)
            for k, idx in enumerate(self._group_indices):
                sl = self._q_slices[k]
                for j in range(p):
                    C[j, sl] = sigma[k] * np.bincount(
                        idx, weights=w * self._X[:, j], minlength=self._n_groups[k]
                    )

        return H, C

    def _score_random(self, sigma: Array, residual: Array) -> Array:
        r = np.zeros(self._q, dtype=float)
        for k, idx in enumerate(self._group_indices):
            sl = self._q_slices[k]
            r[sl] = sigma[k] * np.bincount(
                idx, weights=residual, minlength=self._n_groups[k]
            )
        return r

    def _solve_pirls_increment(
        self,
        sigma: Array,
        beta: Array,
        u: Array,
        update_beta: bool,
        eta: Array,
        mu: Array,
        w: Array,
        residual: Array,
    ) -> Tuple[Array, Array]:
        """One lme4-style PIRLS Newton/Fisher-scoring increment."""
        H, C = self._build_H_C(sigma, w)
        r_u = self._score_random(sigma, residual) - u

        # u-only solve: H du = score_u
        if not update_beta or self._p == 0:
            du = splu(H.tocsc()).solve(r_u)
            if self._p == 0:
                db = np.empty(0, dtype=float)
            else:
                db = np.zeros(self._p, dtype=float)
            return du, db

        r_b = self._X.T @ residual
        XWX = self._X.T @ (w[:, None] * self._X)

        lu = splu(H.tocsc())
        Hinv_ru = lu.solve(r_u)
        Hinv_Ct = lu.solve(C.T)

        # Schur complement for beta.
        S = XWX - C @ Hinv_Ct
        rhs_b = r_b - C @ Hinv_ru

        S = 0.5 * (S + S.T)
        try:
            db = np.linalg.solve(S, rhs_b)
        except np.linalg.LinAlgError:
            db = np.linalg.lstsq(S, rhs_b, rcond=None)[0]

        du = Hinv_ru - Hinv_Ct @ db
        return du, db

    def _pirls(
        self,
        sigma: Array,
        beta0: Array,
        u0: Array,
        update_beta: bool,
    ) -> _FitState:
        """Solve conditional mode / PIRLS for fixed sigma."""
        sigma = np.asarray(sigma, dtype=float)
        beta = np.asarray(beta0, dtype=float).copy()
        u = np.asarray(u0, dtype=float).copy()

        if len(sigma) != len(self._group_vars):
            raise ValueError("Incorrect sigma length")

        b = self._u_to_random_effects(sigma, u)
        eta = self._eta(beta, b)
        pdev = self._pwrss(self._y, eta, u)
        converged = False
        iterations = 0

        for iteration in range(1, self.max_iter_pirls + 1):
            mu = expit(eta)
            w = np.maximum(mu * (1.0 - mu), 1e-12)
            residual = self._y - mu

            du, db = self._solve_pirls_increment(
                sigma=sigma,
                beta=beta,
                u=u,
                update_beta=update_beta,
                eta=eta,
                mu=mu,
                w=w,
                residual=residual,
            )

            # Step-halving for monotone PWRSS.
            accepted = False
            step = 1.0
            old_beta = beta.copy()
            old_u = u.copy()
            old_pdev = pdev

            for _ in range(12):
                trial_u = old_u + step * du
                trial_beta = old_beta + step * db if update_beta else old_beta
                trial_b = self._u_to_random_effects(sigma, trial_u)
                trial_eta = self._eta(trial_beta, trial_b)
                trial_pdev = self._pwrss(self._y, trial_eta, trial_u)
                if np.isfinite(trial_pdev) and trial_pdev <= old_pdev + 1e-10:
                    beta = trial_beta
                    u = trial_u
                    eta = trial_eta
                    pdev = trial_pdev
                    accepted = True
                    break
                step *= 0.5

            if not accepted:
                beta = old_beta
                u = old_u
                eta = self._eta(beta, self._u_to_random_effects(sigma, u))
                pdev = old_pdev
                converged = True
                iterations = iteration
                break

            iterations = iteration
            rel = abs((old_pdev - pdev) / max(abs(pdev), 1e-12))
            if rel < self.tol_pirls:
                converged = True
                break

        return _FitState(beta=beta, u=u, sigma=sigma.copy(), pwrss=pdev,
                         converged=converged, iterations=iterations)

    def _logdet_H(self, H: csr_matrix) -> float:
        lu = splu(H.tocsc())
        d = lu.U.diagonal()
        if np.any(~np.isfinite(d)) or np.any(d == 0):
            raise np.linalg.LinAlgError("Conditional random-effect Hessian is singular")
        return float(np.sum(np.log(np.abs(d))))

    def _fit_components(self, sigma: Array, beta: Array, u: Array) -> Tuple[float, float, float]:
        """Return lme4-like PWRSS, ldL2, and ||u||^2 diagnostics."""
        b = self._u_to_random_effects(sigma, u)
        eta = self._eta(beta, b)
        mu = expit(eta)
        w = np.maximum(mu * (1.0 - mu), 1e-12)
        H, _ = self._build_H_C(sigma, w)
        ldL2 = self._logdet_H(H)
        ussq = float(np.dot(u, u))
        pwrss = float(-2.0 * self._binomial_loglik(self._y, eta) + ussq)
        return pwrss, ldL2, ussq

    def _laplace_deviance(self, sigma: Array, beta: Array, u: Array) -> float:
        pwrss, ldL2, _ = self._fit_components(sigma, beta, u)
        return float(pwrss + ldL2)

    def _fixed_effect_vcov(self, sigma: Array, beta: Array, u: Array) -> pd.DataFrame:
        if self._p == 0:
            return pd.DataFrame()
        b = self._u_to_random_effects(sigma, u)
        eta = self._eta(beta, b)
        mu = expit(eta)
        w = np.maximum(mu * (1.0 - mu), 1e-12)
        H, C = self._build_H_C(sigma, w)
        XWX = self._X.T @ (w[:, None] * self._X)
        lu = splu(H.tocsc())
        Hinv_Ct = lu.solve(C.T)
        info = XWX - C @ Hinv_Ct
        info = 0.5 * (info + info.T)
        try:
            vcov = np.linalg.inv(info)
        except np.linalg.LinAlgError:
            vcov = np.linalg.pinv(info)
        return pd.DataFrame(vcov, index=self._beta_names, columns=self._beta_names)

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def get_random_effects(self, var: Optional[str] = None) -> pd.Series:
        """Per-provider random intercepts (BLUPs).

        Parameters
        ----------
        var : str, optional
            Required when multiple grouping variables are present.

        Returns
        -------
        Series
        """
        var = self._provider_var if var is None else var
        self._check_is_fitted()
        re = self.coefficients_["alpha"]
        if var is None:
            if len(re) != 1:
                raise ValueError(f"Specify var; available={list(re)}")
            return list(re.values())[0]
        if var not in re:
            raise ValueError(f"Unknown grouping variable \'{var}\'")
        return re[var]

    def get_sigma(self, var: Optional[str] = None) -> float:
        """Estimated random-effect standard deviation.

        Parameters
        ----------
        var : str, optional
            Required when multiple grouping variables are present.

        Returns
        -------
        float
        """
        var = self._provider_var if var is None else var
        if self.sigma_ is None:
            raise ValueError("Model has not been fitted")
        if var is None:
            if len(self.sigma_) != 1:
                raise ValueError(f"Specify var; available={list(self.sigma_)}")
            return next(iter(self.sigma_.values()))
        if var not in self.sigma_:
            raise ValueError(f"Unknown grouping variable \'{var}\'")
        return self.sigma_[var]

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        X: pd.DataFrame,
        *,
        x_vars: Optional[List[str]] = None,
        re_vars: Optional[Union[str, Sequence[str]]] = None,
        offset_var: Optional[str] = None,
        use_re: bool = False,
        type: str = "response",
    ) -> Array:
        """Predict probabilities or linear predictor for new data.

        Parameters
        ----------
        X : pd.DataFrame
            New data for prediction.
        x_vars : list of str, optional
            Covariate columns. If None, uses covariates from fitting.
        re_vars : str or list of str, optional
            Grouping columns whose random effects ``use_re=True`` adds; defaults
            to every fitted grouping column.
        offset_var : str, optional
            Offset column in ``X``. If None, offset is zero.
        use_re : bool, default False
            If True, include random effects (BLUPs) for known groups.
            Unknown groups receive RE = 0.
        type : str, default "response"
            "response" returns probabilities, "link" returns log-odds.

        Returns
        -------
        np.ndarray
            Predictions (probabilities or linear predictor).
        """
        self._check_is_fitted()
        if type not in ("response", "link"):
            raise ValueError("type must be 'response' or 'link'")

        # Fixed-effect design matrix
        fe = self.coefficients_["beta"]
        beta = fe.to_numpy()

        if x_vars is None:
            x_vars = self.covariate_names_ or []

        # Build design matrix
        if "(Intercept)" in fe.index:
            X_fe = np.column_stack([np.ones(len(X)), X[x_vars].to_numpy(dtype=float)])
        else:
            X_fe = X[x_vars].to_numpy(dtype=float)

        # Linear predictor: offset + X @ beta
        if offset_var is not None:
            eta = X[offset_var].to_numpy(dtype=float) + X_fe @ beta
        else:
            eta = X_fe @ beta

        # Add random effects if requested
        if use_re:
            re_vars = list(self._group_vars) if re_vars is None else ([re_vars] if isinstance(re_vars, str) else list(re_vars))
            unknown = [v for v in re_vars if v not in self._group_vars]
            if unknown:
                raise ValueError(f"Unknown grouping variables {unknown}; fitted: {self._group_vars}")
            for gv in re_vars:
                re = self.coefficients_["alpha"][gv]
                groups_new = X[gv].astype(str).values
                re_vals = np.array([re.get(g, 0.0) for g in groups_new])
                eta += re_vals

        if type == "link":
            return eta
        return expit(eta)

    # ------------------------------------------------------------------
    # Residuals
    # ------------------------------------------------------------------

    def pearson_residuals(self) -> Array:
        """Pearson residuals: (y - mu) / sqrt(mu * (1 - mu)).

        Returns
        -------
        np.ndarray
            Pearson residuals for the training data.
        """
        self._check_is_fitted()
        mu = self.fitted_
        var = np.maximum(mu * (1.0 - mu), 1e-12)
        return (self._y - mu) / np.sqrt(var)

    def deviance_residuals(self) -> Array:
        """Deviance residuals: sign(y - mu) * sqrt(d_i).

        For binomial (n=1):
            d_i = 2 * [y*log(y/mu) + (1-y)*log((1-y)/(1-mu))]

        Returns
        -------
        np.ndarray
            Deviance residuals for the training data.
        """
        self._check_is_fitted()
        y = self._y
        mu = np.clip(self.fitted_, 1e-12, 1.0 - 1e-12)

        # d_i = 2 * [ y*log(y/mu) + (1-y)*log((1-y)/(1-mu)) ]
        term1 = np.where(y > 0, y * np.log(y / mu), 0.0)
        term2 = np.where(y < 1, (1.0 - y) * np.log((1.0 - y) / (1.0 - mu)), 0.0)
        d = 2.0 * (term1 + term2)
        return np.sign(y - mu) * np.sqrt(np.maximum(d, 0.0))
