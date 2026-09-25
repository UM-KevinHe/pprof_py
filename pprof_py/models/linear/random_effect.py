"""Pure-Python linear mixed-effects model with random intercepts, designed to mirror lme4::lmer.

Key design choices (for Gaussian linear mixed-effects models):

* spherical random effects: b = sigma_e * Lambda(theta) u, u ~ N(0, I)
* for a simple random-intercept term, theta is the random-effect SD divided by
  the residual SD, matching lme4's relative covariance parameterization
* inner mixed-model equations solve the joint fixed/random-effect penalized
  least-squares problem
* ML and REML objectives are evaluated exactly for the Gaussian random-intercept
  covariance model, up to floating-point linear-algebra differences
* multiple independent random-intercept terms are supported, including crossed
  grouping factors through the full random-effect Hessian
* zero random-effect variance is allowed as a boundary solution
* random effects are empirical BLUPs (conditional modes)
* fixed-effect covariance is the inverse GLS information matrix

This is a statistical/numerical reimplementation of the relevant lme4
formulation, not a promise of bit-for-bit identity with Eigen/CHOLMOD.
The implementation intentionally mirrors the organization and documentation
style of the companion logistic random-effect model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ...base import ProviderModel
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import splu
from scipy.stats import t

from ...exceptions import NotFittedError
from ...inference.linear import RandomEffectInferenceMixin
from ...measures.linear import RandomEffectMeasuresMixin
from ...plotting.linear import RandomEffectPlottingMixin

logger = logging.getLogger(__name__)

Array = np.ndarray


@dataclass
class _FitState:
    beta: Array
    u: Array
    sigma_e: float
    pwrss: float
    logdet_H: float
    converged: bool
    iterations: int


class LinearRandomEffectModel(
    RandomEffectInferenceMixin,
    RandomEffectMeasuresMixin,
    RandomEffectPlottingMixin,
    ProviderModel,
):
    """Gaussian linear mixed-effects model with random intercepts, lme4-style.

    Model:

        y = offset + X beta + Z b + epsilon
        b = sigma_e * Lambda(theta) u,   u ~ N(0, I)
        epsilon ~ N(0, sigma_e^2 I)

    For independent random-intercept terms, Lambda(theta) is diagonal with
    theta_k repeated over the levels of grouping factor k.

    Parameters
    ----------
    max_iter_inner : int
        Maximum iterations retained for the inner solve.  Gaussian mixed-model
        equations are solved directly, so normally only one linear solve is
        needed; the parameter is kept for API symmetry and diagnostics.
    max_iter_outer : int
        Maximum optimizer iterations/evaluations for variance components.
    tol_outer : float
        Optimizer tolerance.
    theta_upper : float
        Upper bound for relative random-effect SD parameters.
        The lower bound is zero.
    reml : bool, default True
        Use REML.  lme4::lmer defaults to REML=TRUE.
    verbose : bool, default True
        Print progress.
    optimizer : str, default "powell"
        "powell" or "nelder-mead".
    """

    def __init__(
        self,
        max_iter_inner: int = 100,
        max_iter_outer: int = 200,
        tol_outer: float = 1e-8,
        theta_upper: float = np.inf,
        reml: bool = True,
        verbose: bool = True,
        optimizer: str = "powell",
    ) -> None:
        """Linear random-intercept provider model (R: ``lme4::lmer``)."""
        if theta_upper <= 0 and not np.isinf(theta_upper):
            raise ValueError("theta_upper must be positive or np.inf")
        optimizer = str(optimizer).lower()
        if optimizer not in {"powell", "nelder-mead"}:
            raise ValueError("optimizer must be 'powell' or 'nelder-mead'")

        self.max_iter_inner = int(max_iter_inner)
        self.max_iter_outer = int(max_iter_outer)
        self.tol_outer = float(tol_outer)
        self.theta_upper = float(theta_upper)
        self.reml = bool(reml)
        self.verbose = bool(verbose)
        self.optimizer = optimizer

        # Public results
        self.coefficients_: Optional[Dict] = None
        self.variances_: Optional[Dict] = None
        self.fitted_: Optional[Array] = None
        self.residuals_: Optional[Array] = None
        self.aic_: Optional[float] = None
        self.bic_: Optional[float] = None
        self.loglike_: Optional[float] = None
        self.sigma_: Optional[float] = None
        self.random_effect_sd_: Optional[Dict[str, float]] = None
        self.theta_: Optional[Array] = None
        self.groups_: Optional[Union[Dict[str, Array], Array]] = None
        self.group_sizes_: Optional[Union[Dict[str, Array], Array]] = None
        self.group_indices_: Optional[Array] = None
        self.xbeta_: Optional[Array] = None
        self.covariate_names_: Optional[List[str]] = None
        self.outcome_: Optional[Array] = None
        self.residual_variance_: Optional[float] = None

        # Diagnostics
        self.converged_: bool = False
        self.optimizer_result_ = None
        self.outer_iterations_: int = 0
        self.objective_: Optional[float] = None
        self.pwrss_: Optional[float] = None
        self.ldL2_: Optional[float] = None
        self.ussq_: Optional[float] = None
        self.reml_: bool = self.reml
        self.nobs_: int = 0
        self.n_fixed_effects_: int = 0
        self.n_random_effects_: int = 0

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
        self._sigma_e: Optional[float] = None
        self._b: Optional[Array] = None
        self._last_theta: Optional[Array] = None
        self._weights: Optional[Array] = None
        self._alpha_dict: Optional[Dict[str, pd.Series]] = None
        self._var_alpha_dict: Optional[Dict[str, float]] = None
        self._groups_dict: Optional[Dict[str, Array]] = None
        self._group_sizes_dict: Optional[Dict[str, Array]] = None
        self.result = None  # stub for plotting mixin protocol

    def _check_is_fitted(self) -> None:
        """Raise `NotFittedError` if the model has not been fitted yet."""
        if self.coefficients_ is None:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call `fit` first."
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
        weights_var: Optional[str] = None,
        include_intercept: bool = True,
        reml: Optional[bool] = None,
        verbose: Optional[bool] = None,
        **kwargs,
    ) -> "LinearRandomEffectModel":
        """Fit the Gaussian linear random-intercept model.

        Parameters
        ----------
        X : pd.DataFrame
            Data containing response, covariates, grouping variables, and
            optionally an offset and observation weights.
        y_var : str
            Response variable column name.
        x_vars : list of str, optional
            Fixed-effect covariate columns.
        provider_var : str
            Column of provider IDs; its random intercepts are the provider effects that
            the measures, tests, and plots report.
        cluster_vars : str or list of str, optional
            Further grouping columns with their own (crossed) random intercepts, e.g.
            the hospital in Stage 2 of the three-stage model.
        offset_var : str, optional
            Known offset column added to the linear predictor as-is.
        weights_var : str, optional
            Prior observation-weight column.  Weights are treated as inverse
            residual-variance weights, matching lme4's basic prior-weight
            convention for Gaussian models.
        include_intercept : bool, default True
            Whether to include a fixed intercept.
        reml : bool, optional
            Override the instance-level ML/REML setting.
        verbose : bool, optional
            Override instance-level verbosity.
        """
        if verbose is None:
            verbose = self.verbose
        use_reml = self.reml if reml is None else bool(reml)

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
        if weights_var is not None:
            required.append(weights_var)

        missing_cols = [c for c in required if c not in X.columns]
        if missing_cols:
            raise KeyError(f"Missing columns: {missing_cols}")

        work = X.loc[:, required].copy()
        complete = work.notna().all(axis=1)
        dropped = int((~complete).sum())
        work = work.loc[complete].reset_index(drop=True)

        y = work[y_var].to_numpy(dtype=float)
        if np.any(~np.isfinite(y)):
            raise ValueError("y must contain only finite values")

        self.outcome_ = y.copy()
        self._y = y
        self._n = len(y)
        self.nobs_ = self._n
        self._group_vars = group_vars

        # Fixed-effect design.
        if x_vars:
            X_fe = work[x_vars].to_numpy(dtype=float)
            if np.any(~np.isfinite(X_fe)):
                raise ValueError("Fixed-effect covariates must be finite")
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
        self.n_fixed_effects_ = self._p

        if offset_var is None:
            self._offset = np.zeros(self._n, dtype=float)
        else:
            self._offset = work[offset_var].to_numpy(dtype=float)
            if np.any(~np.isfinite(self._offset)):
                raise ValueError("Offset values must be finite")

        if weights_var is None:
            self._weights = np.ones(self._n, dtype=float)
        else:
            w = work[weights_var].to_numpy(dtype=float)
            if np.any(~np.isfinite(w)) or np.any(w <= 0):
                raise ValueError("weights must contain finite positive values")
            self._weights = w

        # Group coding.
        self.groups_ = {}
        self.group_sizes_ = {}
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
            self.groups_[gv] = labels
            self.group_sizes_[gv] = sizes

        self._q_slices = []
        start = 0
        for g in self._n_groups:
            self._q_slices.append(slice(start, start + g))
            start += g
        self._q = start
        self.n_random_effects_ = self._q

        if self._n <= self._p:
            criterion_df = self._n - self._p
            if use_reml:
                raise ValueError(
                    "REML requires n > number of fixed-effect parameters"
                )

        # Initial relative SDs.
        theta0 = np.ones(len(group_vars), dtype=float)
        beta0 = self._weighted_ols_start()

        if verbose:
            logger.info("Fitting linear mixed model (lme4-style ML/REML)...")
            logger.info(f"  Observations: {self._n:,} (dropped {dropped:,})")
            logger.info(f"  Fixed effects: {self._p}")
            for gv, g in zip(group_vars, self._n_groups):
                logger.info(f"  RE '{gv}': {g:,} levels")
            logger.info(f"  Criterion: {'REML' if use_reml else 'ML'}")

        cache = {"beta": beta0.copy(), "u": np.zeros(self._q, dtype=float)}

        def objective(theta: Array) -> float:
            theta = self._clip_theta(theta)
            state = self._solve_inner(theta, cache["beta"], cache["u"])
            cache["beta"] = state.beta.copy()
            cache["u"] = state.u.copy()
            value = self._objective_from_state(theta, state, use_reml)
            if not np.isfinite(value):
                return 1e300
            return float(value)

        theta_opt, opt = self._optimize(objective, theta0)
        theta_opt = self._clip_theta(theta_opt)

        final = self._solve_inner(
            theta_opt, cache["beta"], cache["u"]
        )
        # The mixed-model solution is scale-free for a fixed relative
        # covariance parameter.  Profile the residual scale using the actual
        # REML/ML choice supplied to this fit call.
        scale_df = self._n - self._p if use_reml else self._n
        final.sigma_e = float(
            np.sqrt(max(final.pwrss / scale_df, np.finfo(float).tiny))
        )
        objective_value = self._objective_from_state(
            theta_opt, final, use_reml
        )

        # Store core fit.
        self._beta = final.beta.copy()
        self._u = final.u.copy()
        self._theta = theta_opt.copy()
        self._sigma_e = final.sigma_e
        # The inner Gaussian normal equations are parameterized in terms of
        # the response-scale random contribution b = Lambda(theta) u.
        # The profiled residual scale is applied only to its covariance,
        # not to the BLUP itself.
        self._b = self._theta_expand(theta_opt, final.u)

        self.theta_ = theta_opt.copy()
        self.sigma_ = float(final.sigma_e)
        self.residual_variance_ = float(final.sigma_e**2)
        self.random_effect_sd_ = {
            gv: float(theta_opt[k] * final.sigma_e)
            for k, gv in enumerate(group_vars)
        }
        self._last_theta = theta_opt.copy()

        b_by_group = self._b_by_group(self._b)
        fitted = self._eta(self._beta, b_by_group)
        self.fitted_ = fitted
        self.residuals_ = y - fitted
        self.xbeta_ = self._offset + (self._X @ self._beta if self._p else 0.0)

        self.pwrss_ = float(final.pwrss)
        self.ldL2_ = float(final.logdet_H)
        self.ussq_ = float(np.dot(final.u, final.u))

        # Exact Gaussian deviance / log likelihood.
        logdet_v = self._logdet_V0(theta_opt)
        qform = final.pwrss
        if use_reml:
            criterion_df = self._n - self._p
            sigma2 = max(qform / criterion_df, np.finfo(float).tiny)
            logdet_x = self._logdet_gls_information(theta_opt)
            dev = (
                criterion_df * (np.log(2.0 * np.pi * sigma2) + 1.0)
                + logdet_v
                + logdet_x
                - float(np.sum(np.log(self._weights)))
            )
            # ML logLik is not equal to the REML criterion.  Store the REML
            # criterion as the fitting objective and provide ML logLik only
            # when fitting by ML.
            self.loglike_ = -0.5 * dev
            n_cov_params = len(group_vars) + 1
            n_params = self._p + n_cov_params
            self.aic_ = float(dev + 2.0 * n_params)
            self.bic_ = float(
                dev + np.log(max(self._n, 1)) * n_params
            )
        else:
            sigma2 = max(qform / self._n, np.finfo(float).tiny)
            dev = (
                self._n * (np.log(2.0 * np.pi * sigma2) + 1.0)
                + logdet_v
                - float(np.sum(np.log(self._weights)))
            )
            self.loglike_ = -0.5 * dev
            n_cov_params = len(group_vars) + 1
            n_params = self._p + n_cov_params
            self.aic_ = float(dev + 2.0 * n_params)
            self.bic_ = float(dev + np.log(max(self._n, 1)) * n_params)

        # ---- Multi-group internal storage (always available) ----
        self._alpha_dict = {
            gv: pd.Series(
                b_by_group[k],
                index=self._group_labels[k],
                dtype=float,
            )
            for k, gv in enumerate(group_vars)
        }
        self._var_alpha_dict = {
            gv: float(theta_opt[k] ** 2 * final.sigma_e**2)
            for k, gv in enumerate(group_vars)
        }
        vcov = self._fixed_effect_vcov(theta_opt, final.beta)

        # ---- Mixin-compatibility attributes ----
        # The inference, measures, and plotting mixins expect flat
        # (single-grouping-variable) attributes.  When there is exactly
        # one grouping variable, expose the flat form directly.
        # Multi-group models retain the dict form and can be accessed
        # via get_random_effects() / _alpha_dict.
        self._groups_dict = dict(self.groups_)
        self._group_sizes_dict = dict(self.group_sizes_)

        if len(group_vars) == 1:
            gv0 = group_vars[0]
            self.groups_ = self._groups_dict[gv0]
            self.group_sizes_ = self._group_sizes_dict[gv0]
            self.group_indices_ = self._group_indices[0]
            self.coefficients_ = {
                "beta": pd.Series(
                    self._beta, index=self._beta_names, dtype=float
                ),
                "alpha": self._alpha_dict[gv0],
            }
            self.variances_ = {
                "beta": vcov,
                "sigma": float(final.sigma_e**2),
                "alpha": pd.DataFrame(
                    [[self._var_alpha_dict[gv0]]],
                    index=[gv0],
                    columns=[gv0],
                ),
            }
        else:
            self.group_indices_ = None
            self.coefficients_ = {
                "beta": pd.Series(
                    self._beta, index=self._beta_names, dtype=float
                ),
                "alpha": self._alpha_dict,
            }
            self.variances_ = {
                "beta": vcov,
                "sigma": float(final.sigma_e**2),
                "alpha": self._var_alpha_dict,
            }

        self.optimizer_result_ = opt
        self.outer_iterations_ = int(
            getattr(opt, "nfev", getattr(opt, "nit", 0))
        )
        self.objective_ = float(objective_value)
        self.converged_ = bool(getattr(opt, "success", False))

        if verbose:
            logger.info(
                f"  Outer optimizer: success={self.converged_}, "
                f"iters={self.outer_iterations_}, optimizer={self.optimizer}"
            )
            for gv, sd in self.random_effect_sd_.items():
                logger.info(f"  SD({gv}): {sd:.8f}")
            logger.info(f"  Residual SD: {final.sigma_e:.8f}")
            logger.info(f"  beta: {self._beta}")
            logger.info(f"  Criterion: {objective_value:.6f}")
            logger.info(f"  Overall converged: {self.converged_}")

        return self

    # ------------------------------------------------------------------
    # Variance-component optimization
    # ------------------------------------------------------------------

    def _clip_theta(self, theta: Array) -> Array:
        theta = np.asarray(theta, dtype=float)
        theta = np.maximum(theta, 0.0)
        if np.any(~np.isfinite(theta)):
            raise ValueError("theta must be finite")
        if np.isfinite(self.theta_upper):
            theta = np.minimum(theta, self.theta_upper)
        return theta

    def _optimize(self, fun, x0: Array):
        ub = self.theta_upper if np.isfinite(self.theta_upper) else None
        bounds = [(0.0, ub)] * len(x0)

        if self.optimizer == "powell":
            result = minimize(
                fun,
                np.asarray(x0, dtype=float),
                method="Powell",
                bounds=bounds,
                options={
                    "maxiter": self.max_iter_outer,
                    "xtol": self.tol_outer,
                    "ftol": self.tol_outer,
                },
            )
        else:
            result = minimize(
                fun,
                np.asarray(x0, dtype=float),
                method="Nelder-Mead",
                bounds=bounds,
                options={
                    "maxiter": self.max_iter_outer,
                    "xatol": self.tol_outer,
                    "fatol": self.tol_outer,
                    "adaptive": True,
                },
            )
        return self._clip_theta(result.x), result

    # ------------------------------------------------------------------
    # Core Gaussian mixed-model calculations
    # ------------------------------------------------------------------

    def _weighted_ols_start(self) -> Array:
        y_adj = self._y - self._offset
        w = self._weights
        if self._p == 0:
            return np.empty(0, dtype=float)

        xtwx = self._X.T @ (w[:, None] * self._X)
        xtwy = self._X.T @ (w * y_adj)
        try:
            return np.linalg.solve(xtwx, xtwy)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(xtwx, xtwy, rcond=None)[0]

    def _group_expand(self, theta: Array) -> Array:
        out = np.zeros(self._q, dtype=float)
        for k, sl in enumerate(self._q_slices):
            out[sl] = theta[k]
        return out

    def _theta_expand(self, theta: Array, u: Array) -> Array:
        return self._group_expand(theta) * u

    def _b_by_group(self, b: Array) -> List[Array]:
        return [b[sl] for sl in self._q_slices]

    def _eta(self, beta: Array, b_by_group: List[Array]) -> Array:
        eta = self._offset.copy()
        if self._p:
            eta += self._X @ beta
        for k, idx in enumerate(self._group_indices):
            eta += b_by_group[k][idx]
        return eta

    def _build_H_C(self, theta: Array) -> Tuple[csr_matrix, Array, Array]:
        """Build H = I + A'WA and C = X'WA without constructing A.

        A is the random-effect design after scaling by the relative covariance
        parameters.  W contains prior observation weights.
        """
        q = self._q
        p = self._p
        w = self._weights
        rows: List[Array] = []
        cols: List[Array] = []
        vals: List[Array] = []

        diag = np.ones(q, dtype=float)
        for k, idx in enumerate(self._group_indices):
            sl = self._q_slices[k]
            sums = np.bincount(
                idx, weights=w, minlength=self._n_groups[k]
            )
            diag[sl] += theta[k] ** 2 * sums

        d = np.arange(q, dtype=np.int64)
        rows.append(d)
        cols.append(d)
        vals.append(diag)

        ng = len(self._group_indices)
        for k in range(ng):
            for l in range(k + 1, ng):
                if theta[k] == 0.0 or theta[l] == 0.0:
                    continue
                idx_k = self._group_indices[k]
                idx_l = self._group_indices[l]
                r = self._q_slices[k].start + idx_k
                c = self._q_slices[l].start + idx_l
                v = w * theta[k] * theta[l]
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
                if theta[k] == 0.0:
                    continue
                for j in range(p):
                    C[j, sl] = theta[k] * np.bincount(
                        idx,
                        weights=w * self._X[:, j],
                        minlength=self._n_groups[k],
                    )

        # d = A'W(y-offset).  The fixed-effect equations use the adjusted y.
        y_adj = self._y - self._offset
        dvec = np.zeros(q, dtype=float)
        for k, idx in enumerate(self._group_indices):
            sl = self._q_slices[k]
            dvec[sl] = theta[k] * np.bincount(
                idx,
                weights=w * y_adj,
                minlength=self._n_groups[k],
            )

        return H, C, dvec

    def _solve_inner(
        self,
        theta: Array,
        beta0: Array,
        u0: Array,
    ) -> _FitState:
        theta = self._clip_theta(theta)
        y_adj = self._y - self._offset
        w = self._weights

        H, C, dvec = self._build_H_C(theta)
        xwx = self._X.T @ (w[:, None] * self._X)
        xwy = self._X.T @ (w * y_adj)

        if self._p == 0:
            if self._q:
                lu = splu(H.tocsc())
                u = lu.solve(dvec)
            else:
                u = np.empty(0, dtype=float)
            beta = np.empty(0, dtype=float)
        else:
            lu = splu(H.tocsc()) if self._q else None
            if self._q:
                h_inv_d = lu.solve(dvec)
                h_inv_ct = lu.solve(C.T)
                schur = xwx - C @ h_inv_ct
                rhs = xwy - C @ h_inv_d
            else:
                schur = xwx
                rhs = xwy

            schur = 0.5 * (schur + schur.T)
            try:
                beta = np.linalg.solve(schur, rhs)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(schur, rhs, rcond=None)[0]

            if self._q:
                u = h_inv_d - h_inv_ct @ beta
            else:
                u = np.empty(0, dtype=float)

        # u is the spherical random effect.  At the scale-free stage,
        # b / sigma_e = Lambda(theta) u; the fitted random contribution is
        # therefore Lambda(theta) u.  The actual BLUP scale is restored after
        # profiling sigma_e in fit().
        b = self._theta_expand(theta, u)

        eta = y_adj * 0.0 + (self._X @ beta if self._p else 0.0)
        for k, idx in enumerate(self._group_indices):
            eta += b[self._q_slices[k]][idx]

        resid = y_adj - eta
        rss = float(np.sum(w * resid * resid))
        ussq = float(np.dot(u, u))
        pwrss = rss + ussq

        logdet_h = self._logdet_H(H)

        # Exact Gaussian scale parameter under ML or REML is obtained after
        # restoring the residual scale.  For the profile objective, the ratio
        # parameter theta makes beta and u scale-free.
        denom = self._n if not self.reml else max(self._n - self._p, 1)
        sigma_e = np.sqrt(max(pwrss / denom, np.finfo(float).tiny))

        return _FitState(
            beta=beta,
            u=u,
            sigma_e=float(sigma_e),
            pwrss=float(pwrss),
            logdet_H=float(logdet_h),
            converged=True,
            iterations=1,
        )
    def _logdet_H(self, H: csr_matrix) -> float:
        if H.shape[0] == 0:
            return 0.0
        lu = splu(H.tocsc())
        d = lu.U.diagonal()
        if np.any(~np.isfinite(d)) or np.any(d == 0):
            raise np.linalg.LinAlgError("Random-effect Hessian is singular")
        return float(np.sum(np.log(np.abs(d))))

    def _logdet_V0(self, theta: Array) -> float:
        H, _, _ = self._build_H_C(theta)
        return self._logdet_H(H)

    def _logdet_gls_information(self, theta: Array) -> float:
        if self._p == 0:
            return 0.0
        H, C, _ = self._build_H_C(theta)
        xwx = self._X.T @ (self._weights[:, None] * self._X)
        if self._q:
            lu = splu(H.tocsc())
            h_inv_ct = lu.solve(C.T)
            info = xwx - C @ h_inv_ct
        else:
            info = xwx
        info = 0.5 * (info + info.T)
        sign, logdet = np.linalg.slogdet(info)
        if sign <= 0 or not np.isfinite(logdet):
            raise np.linalg.LinAlgError(
                "GLS fixed-effect information matrix is not positive definite"
            )
        return float(logdet)

    def _fixed_effect_vcov(
        self,
        theta: Array,
        beta: Array,
    ) -> pd.DataFrame:
        if self._p == 0:
            return pd.DataFrame()

        H, C, _ = self._build_H_C(theta)
        xwx = self._X.T @ (self._weights[:, None] * self._X)
        if self._q:
            lu = splu(H.tocsc())
            h_inv_ct = lu.solve(C.T)
            info = xwx - C @ h_inv_ct
        else:
            info = xwx

        info = 0.5 * (info + info.T)
        try:
            inv_info = np.linalg.inv(info)
        except np.linalg.LinAlgError:
            inv_info = np.linalg.pinv(info)

        return pd.DataFrame(
            inv_info * self.residual_variance_,
            index=self._beta_names,
            columns=self._beta_names,
        )

    def _objective_from_state(
        self,
        theta: Array,
        state: _FitState,
        reml: bool,
    ) -> float:
        qform = state.pwrss
        if reml:
            if self._n <= self._p:
                return 1e300
            sigma2 = max(
                qform / (self._n - self._p),
                np.finfo(float).tiny,
            )
            logdet_x = self._logdet_gls_information(theta)
            return float(
                (self._n - self._p)
                * (np.log(2.0 * np.pi * sigma2) + 1.0)
                + self._logdet_V0(theta)
                + logdet_x
                - float(np.sum(np.log(self._weights)))
            )

        sigma2 = max(qform / self._n, np.finfo(float).tiny)
        return float(
            self._n * (np.log(2.0 * np.pi * sigma2) + 1.0)
            + self._logdet_V0(theta)
            - float(np.sum(np.log(self._weights)))
        )

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def get_random_effects(
        self,
        var: Optional[str] = None,
    ) -> pd.Series:
        """Return BLUPs for the given grouping variable."""
        var = self._provider_var if var is None else var
        if self._alpha_dict is None:
            raise ValueError("Model has not been fitted")
        re = self._alpha_dict
        if var is None:
            if len(re) != 1:
                raise ValueError(
                    f"Specify var; available={list(re)}"
                )
            return list(re.values())[0]
        if var not in re:
            raise ValueError(f"Unknown grouping variable '{var}'")
        return re[var]

    def get_sigma(self) -> float:
        """Profiled residual standard deviation.

        Returns
        -------
        float
        """
        if self.sigma_ is None:
            raise ValueError("Model has not been fitted")
        return float(self.sigma_)

    def get_random_effect_sd(
        self,
        var: Optional[str] = None,
    ) -> Union[float, Dict[str, float]]:
        """Estimated random-effect standard deviation.

        Parameters
        ----------
        var : str, optional
            Required when multiple grouping variables are present.

        Returns
        -------
        float or dict
        """
        var = self._provider_var if var is None else var
        if self.random_effect_sd_ is None:
            raise ValueError("Model has not been fitted")
        if var is None:
            if len(self.random_effect_sd_) != 1:
                return dict(self.random_effect_sd_)
            return next(iter(self.random_effect_sd_.values()))
        if var not in self.random_effect_sd_:
            raise ValueError(f"Unknown grouping variable '{var}'")
        return float(self.random_effect_sd_[var])

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
    ) -> Array:
        """Predict responses for new data.

        By default only fixed effects are used.  When ``use_re=True``, BLUPs
        for known grouping levels are added and unknown levels receive zero,
        matching lme4's conditional prediction convention.
        """
        if self.coefficients_ is None:
            raise ValueError("Model has not been fitted")

        fe = self.coefficients_["beta"]
        beta = fe.to_numpy()

        if x_vars is None:
            x_vars = self.covariate_names_ or []

        if "(Intercept)" in fe.index:
            X_fe = np.column_stack(
                [np.ones(len(X)), X[x_vars].to_numpy(dtype=float)]
            )
        else:
            X_fe = X[x_vars].to_numpy(dtype=float)

        if offset_var is not None:
            eta = X[offset_var].to_numpy(dtype=float) + X_fe @ beta
        else:
            eta = X_fe @ beta

        if use_re:
            re_vars = list(self._group_vars) if re_vars is None else ([re_vars] if isinstance(re_vars, str) else list(re_vars))
            unknown = [v for v in re_vars if v not in self._group_vars]
            if unknown:
                raise ValueError(f"Unknown grouping variables {unknown}; fitted: {self._group_vars}")

            for gv in re_vars:
                re = self._alpha_dict[gv]
                groups_new = X[gv].astype(str).values
                # pandas Index/Series labels may not be strings, so use
                # explicit string comparison through a small dictionary.
                mapping = {str(k): float(v) for k, v in re.items()}
                eta += np.array(
                    [mapping.get(g, 0.0) for g in groups_new],
                    dtype=float,
                )

        return eta

    def standard_errors(self) -> pd.Series:
        """Return fixed-effect standard errors."""
        if self.variances_ is None:
            raise ValueError("Model has not been fitted")
        vcov = self.variances_["beta"]
        return pd.Series(
            np.sqrt(np.maximum(np.diag(vcov), 0.0)),
            index=vcov.index,
            dtype=float,
        )

    def fixed_effects_table(self) -> pd.DataFrame:
        """Return fixed-effect estimates, SEs, t statistics, and p-values.

        P-values use a residual-df Student-t reference distribution.  This is
        a convenience inference table rather than a claim of exact lme4
        denominator-df or small-sample inference.
        """
        if self.coefficients_ is None:
            raise ValueError("Model has not been fitted")
        beta = self.coefficients_["beta"]
        se = self.standard_errors()
        stat = beta / se.replace(0.0, np.nan)
        df = max(self._n - self._p, 1)
        p_value = 2.0 * t.sf(np.abs(stat), df=df)
        return pd.DataFrame(
            {
                "Estimate": beta,
                "Std. Error": se,
                "t value": stat,
                "df": float(df),
                "Pr(>|t|)": p_value,
            }
        )

    def conf_int(
        self,
        level: float = 0.95,
    ) -> pd.DataFrame:
        """Return Wald confidence intervals for fixed effects."""
        if not 0.0 < level < 1.0:
            raise ValueError("level must be between 0 and 1")
        if self.coefficients_ is None:
            raise ValueError("Model has not been fitted")
        df = max(self._n - self._p, 1)
        zcrit = t.ppf(0.5 + level / 2.0, df=df)
        beta = self.coefficients_["beta"]
        se = self.standard_errors()
        return pd.DataFrame(
            {
                "lower": beta - zcrit * se,
                "upper": beta + zcrit * se,
            }
        )

    def residual_standard_error(self) -> float:
        """Return the profiled residual standard deviation."""
        return self.get_sigma()

    def fitted_values(self) -> Array:
        """Fitted (predicted) values from the training data.

        Returns
        -------
        ndarray, shape (n_train,)
        """
        if self.fitted_ is None:
            raise ValueError("Model has not been fitted")
        return self.fitted_.copy()
