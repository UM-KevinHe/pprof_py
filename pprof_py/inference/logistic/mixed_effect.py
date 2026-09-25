"""Covariate inference, provider-effect tests (exact, plug-in Poisson-binomial,
or resampling, with optional empirical-null calibration) and confidence
intervals for ``LogisticMixedEffectModel``.  Mixed
into the model class so that ``models/logistic/mixed_effect.py`` stays
focused on configuration, fitting, and prediction.

Stage 3 holds beta fixed at its Stage 1 estimate, so covariate inference
belongs to Stage 1: ``summary()`` returns the Stage 1 model's Wald table, as
R's ``summary.glmm.covar`` does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import warnings
from typing import List, Optional, Union
from scipy.special import expit as plogis
from ..effect_tests import (EXACT_P_FLOOR, clustered_poibin_tails, effect_test, integrated_poibin_tails, invert_decreasing, normalize_alternative, poibin_tails, reference_effect, resample_tails, z_from_tails)


class LogisticMixedEffectInferenceMixin:
    """Covariate (beta) inference for `LogisticMixedEffectModel`, taken from Stage 1."""

    def summary(self, stage1_model=None, covariates=None, level: float = 0.95, null: float = 0.0,
                alternative: str = "two_sided") -> pd.DataFrame:
        """Wald inference for the covariate effects, from the Stage 1 model.

        Beta is estimated in Stage 1 (``LogisticFixedEffectModel``) and held
        fixed here, so its standard errors are Stage 1's, whose Wald variance
        accounts for the estimated provider effects. (Earlier versions computed
        an information matrix from the Stage 3 fit at fixed beta and gamma,
        which understated the standard errors.)

        Parameters
        ----------
        stage1_model : LogisticFixedEffectModel, optional
            The fitted Stage 1 model whose beta was passed as ``beta_init``.
            Defaults to the one given to ``fit(stage1_model=...)``.
        covariates, level, null, alternative
            Passed to the Stage 1 model's ``summary(test_method="wald")``.

        Returns
        -------
        pandas.DataFrame
            The Stage 1 Wald table.

        Raises
        ------
        ValueError
            If no Stage 1 model is available, or its beta differs from the beta
            this model was fit with.
        """
        self._check_is_fitted()
        stage1 = stage1_model if stage1_model is not None else getattr(self, "stage1_model_", None)
        if stage1 is None:
            raise ValueError("summary() requires the Stage 1 model for covariate inference. "
                             "Pass it via stage1_model= at fit() or summary() time.")
        coefs = getattr(stage1, "coefficients_", None)
        if not isinstance(coefs, dict) or "beta" not in coefs:
            raise ValueError("stage1_model must be a fitted LogisticFixedEffectModel (it has no coefficients_['beta']).")
        beta1 = np.asarray(coefs["beta"], dtype=np.float64).ravel()
        if beta1.shape != np.shape(self.beta_) or not np.allclose(beta1, self.beta_, rtol=1e-10, atol=1e-12):
            raise ValueError("stage1_model's beta differs from the beta this model was fit with (beta_init); "
                             "pass the Stage 1 model whose beta was used.")
        return stage1.summary(covariates=covariates, level=level, null=null, alternative=alternative,
                              test_method="wald")

    def test(
        self,
        providers=None,
        *,
        test_method: str = "exact",
        reference="median",
        null_model=None,
        alternative: str = "two_sided",
        level: float = 0.95,
        critical: Optional[float] = None,
        n_resample: int = 10000,
        seed=None,
    ) -> pd.DataFrame:
        """Test each provider's Stage 3 effect against the reference effect gamma_0.

        Parameters
        ----------
        providers : array-like, optional
            Report only these providers; gamma_0 and any empirical null use all.
        test_method : {"exact", "poibin_exact", "resampling"}, default "exact"
            The provider's event count is compared with its distribution when its
            effect is gamma_0 and the cluster effects follow their posterior.

            ``"exact"``: each cluster's effect is drawn once and shared by all of
            the provider's patients in that cluster (He et al. 2013, Section 3.3,
            step (ii)). The count is then a convolution over clusters of
            Gauss-Hermite mixtures of Poisson-binomial distributions, computed
            exactly.
            ``"poibin_exact"``: exact Poisson-binomial test with the cluster
            effects at their posterior means (their uncertainty is ignored).
            ``"resampling"``: Monte Carlo with a separate draw of the cluster
            effect for every patient, as in R's ``summary.glmm.fac`` (which,
            unlike this method, passes the posterior variance to ``rnorm`` as the
            SD). With independent draws the count is Poisson-binomial with
            posterior-integrated probabilities, so ``"poibin_exact"`` approximates
            this null closely. Providers whose simulated tail falls to the
            resolution floor ``0.5 / n_resample`` get the exact tails of this same
            null instead, with a warning: a capped z-statistic can make the most
            extreme providers impossible to flag under an empirical null.
        reference : "median", "mean", or float
            The reference effect gamma_0: the median of the estimated effects,
            their size-weighted mean, or a value on the effect scale.
        null_model : NullModel or callable, optional
            Null for the z-statistics: :class:`~pprof_py.inference.TheoreticalNull`
            by default, or an instance such as ``FixedNull(sd=...)``, or a callable
            that receives the z-statistics, such as ``EmpiricalNull.fitter(...)``.
            R's ``summary.glmm.fac`` calibrates within quartiles of a facility-size
            variable with ``MASS::rlm`` defaults, i.e.
            ``EmpiricalNull.fitter(size=sizes, n_groups=4, grouping="quantile",
            estimator=HUBER_RLM)``; R sets missing sizes to 0, while the quantile
            grouping requires the caller to handle them.
        alternative, level, critical
            As in :func:`~pprof_py.inference.provider_test`.
        n_resample, seed : int, optional
            Monte Carlo draws and seed for ``"resampling"``.

        Returns
        -------
        pandas.DataFrame
            Indexed by provider with columns
            :data:`~pprof_py.inference.PROVIDER_TEST_COLUMNS`: ``flag`` is +1
            above gamma_0, -1 below, 0 not significant, NA not tested. For
            ``"exact"`` and ``"poibin_exact"``, ``ci_lower`` and ``ci_upper``
            invert the (calibrated) test, so a limit excludes gamma_0 exactly
            when the provider is flagged; they are NaN for ``"resampling"``.
        """
        self._check_is_fitted()
        alt = normalize_alternative(alternative)
        if test_method not in ("exact", "poibin_exact", "resampling"):
            raise ValueError(f"test_method={test_method!r} is not supported; "
                             "use 'exact', 'poibin_exact', or 'resampling'.")
        idx = np.asarray(self._provider_idx).ravel()
        counts = np.bincount(idx, minlength=self.n_providers_)
        g0 = reference_effect(self.gamma_, counts, reference)
        order = np.argsort(idx, kind="stable")
        edges = np.r_[0, np.cumsum(counts)]
        rows_of = [order[edges[j]:edges[j + 1]] for j in range(self.n_providers_)]
        obs = np.array([self._obs[rows].sum() for rows in rows_of])
        two = alt == "two_sided"

        def z_of(tails, floor):
            t = np.atleast_2d(tails)
            return z_from_tails(t[:, 0] if two else t[:, 2], t[:, 1] if two else t[:, 3], alt, floor)

        limits = None
        if test_method == "resampling":
            rng = np.random.default_rng(seed)
            tails = np.empty((self.n_providers_, 4))
            for j, rows in enumerate(rows_of):
                tails[j] = resample_tails(obs[j], self.xbeta_[rows], self.alpha_mean_[rows], self.alpha_var_[rows],
                                          g0, n_resample, rng)
            floor = 0.5 / n_resample
            z = z_of(tails, floor)
            up, lo = (tails[:, 0], tails[:, 1]) if two else (tails[:, 2], tails[:, 3])
            at_floor = (np.minimum(up, lo) if two else (up if alt == "greater" else lo)) <= floor
            for j in np.flatnonzero(at_floor):
                rows = rows_of[j]
                z[j] = z_of(integrated_poibin_tails(obs[j], g0 + self.alpha_mean_[rows] + self.xbeta_[rows],
                                                    self.alpha_var_[rows], self._POSTERIOR_NODES), EXACT_P_FLOOR)[0]
            if at_floor.any():
                warnings.warn(f"test_method='resampling': {int(at_floor.sum())} of {self.n_providers_} providers had "
                              f"simulated tails at the Monte Carlo floor (0.5/n_resample = {floor:.2g}); their "
                              "z-statistics use the exact tails of the same per-patient null instead.", stacklevel=2)
        else:
            tails_at = self._exact_tails_function(test_method, rows_of, obs)
            z = z_of(np.array([tails_at(j, g0) for j in range(self.n_providers_)]), EXACT_P_FLOOR)
            wanted = (np.ones(self.n_providers_, dtype=bool) if providers is None
                      else np.isin(self.provider_ids_, np.atleast_1d(providers)))

            def limits(z_lower, z_upper):
                lower = np.full(self.n_providers_, np.nan)
                upper = np.full(self.n_providers_, np.nan)
                for j in np.flatnonzero(wanted):
                    def zfun(g, j=j):
                        return float(z_of(tails_at(j, g), EXACT_P_FLOOR)[0])
                    lower[j] = invert_decreasing(zfun, self.gamma_[j], z_lower[j])
                    upper[j] = invert_decreasing(zfun, self.gamma_[j], z_upper[j])
                return lower, upper
        return effect_test(self.provider_ids_, self.gamma_, z, g0, null_model=null_model, alternative=alt,
                           level=level, critical=critical, providers=providers, test_method=test_method,
                           limits=limits)

    def _exact_tails_function(self, test_method, rows_of, obs):
        """``tails_at(j, g)``: exact tails of provider j's count with its effect at ``g``."""
        if test_method == "exact":
            clusters = np.asarray(self._cluster_idx).ravel()
            mean_c, var_c = self.alpha_mean_cluster_, self.alpha_var_cluster_

            def tails_at(j, g):
                rows = rows_of[j]
                return clustered_poibin_tails(obs[j], g + self.xbeta_[rows], clusters[rows], mean_c, var_c,
                                              self._POSTERIOR_NODES)
        else:
            def tails_at(j, g):
                rows = rows_of[j]
                return poibin_tails(obs[j], plogis(g + self.alpha_mean_[rows] + self.xbeta_[rows]))
        return tails_at

    def calculate_confidence_intervals(
        self,
        providers: Optional[Union[List, np.ndarray]] = None,
        level: float = 0.95,
        option: str = "SM",
        stdz: Union[str, List[str]] = "indirect",
        reference: Union[str, float] = "median",
        measure: Union[str, List[str]] = ("rate", "ratio"),
        alternative: str = "two_sided",
        test_method: str = "exact",
        null_model=None,
    ) -> dict:
        """Confidence intervals for provider effects (``option="gamma"``) or standardized measures (``"SM"``).

        Mirrors ``LogisticFixedEffectModel.calculate_confidence_intervals``. The
        provider-effect limits are those of :meth:`test` (the inverted exact
        test, calibrated by ``null_model``); standardized-measure limits map them
        through the measure, which increases with the provider effect: the
        predicted count at a limit divided by the expected count (indirect), or
        the provider's standard-population prediction at a limit (direct).

        Parameters
        ----------
        providers : list or np.ndarray, optional
            Report only these providers.
        level : float, default=0.95
        option : {"gamma", "SM"}, default="SM"
        stdz : "indirect", "direct", or both, default="indirect"
        reference : "median", "mean", or float, default="median"
            Reference effect: gamma_0 of :meth:`test` and the standardization norm.
        measure : "rate", "ratio", or both, default=("rate", "ratio")
        alternative : {"two_sided", "greater", "less"}, default="two_sided"
        test_method : {"exact", "poibin_exact"}, default="exact"
        null_model : NullModel or callable, optional
            As in :meth:`test`.

        Returns
        -------
        dict
            ``option="gamma"``: ``{"gamma_ci": DataFrame[provider_id, gamma,
            gamma_lower, gamma_upper]}``. ``option="SM"``: any of
            ``"indirect_ratio"``, ``"indirect_rate"``, ``"direct_ratio"``,
            ``"direct_rate"``, each the :meth:`calculate_standardized_measures`
            table with ``ci_ratio_lower``/``ci_ratio_upper`` or
            ``ci_rate_lower``/``ci_rate_upper``.
        """
        self._check_is_fitted()
        if option not in ("gamma", "SM"):
            raise ValueError("option must be 'gamma' or 'SM'.")
        if test_method not in ("exact", "poibin_exact"):
            raise ValueError("Intervals need a deterministic test: test_method must be 'exact' or 'poibin_exact'.")
        ids = self.provider_ids_
        res = self.test(test_method=test_method, reference=reference, null_model=null_model, alternative=alternative,
                        level=level)
        lower = res["ci_lower"].reindex(ids).to_numpy(dtype=np.float64)
        upper = res["ci_upper"].reindex(ids).to_numpy(dtype=np.float64)
        keep = np.ones(ids.size, dtype=bool) if providers is None else np.isin(ids, np.atleast_1d(providers))
        meta = {"confidence_level": f"{level * 100}%", "model": type(self).__name__, "test_method": test_method}
        if option == "gamma":
            out = pd.DataFrame({"provider_id": ids, "gamma": self.gamma_, "gamma_lower": lower,
                                "gamma_upper": upper})[keep].reset_index(drop=True)
            out.attrs.update(meta, description="Provider effect")
            return {"gamma_ci": out}
        stdz_list = [stdz] if isinstance(stdz, str) else list(stdz)
        measures = [measure] if isinstance(measure, str) else list(measure)
        if not measures or not set(measures) <= {"ratio", "rate"}:
            raise ValueError("measure must be 'ratio', 'rate', or both.")
        sm = self.calculate_standardized_measures(stdz=stdz_list, reference=reference)
        idx = np.asarray(self._provider_idx).ravel()
        n_obs = len(self._obs)
        population_rate = float(np.sum(self._obs)) / n_obs * 100.0
        with np.errstate(invalid="ignore"):
            bounds = {}
            if "indirect" in sm:
                expected = sm["indirect"]["expected"].to_numpy(dtype=np.float64)
                own = lambda g: np.bincount(idx, weights=plogis(g[idx] + self.alpha_mean_ + self.xbeta_),
                                            minlength=self.n_providers_)
                bounds["indirect"] = tuple(np.where(expected > 0, own(g) / expected, np.nan) for g in (lower, upper))
            if "direct" in sm:
                total = float(np.sum(self._obs))
                everyone = lambda g: np.array([np.nan if np.isnan(gj) else plogis(gj + self.alpha_mean_ + self.xbeta_).sum()
                                               for gj in g])
                bounds["direct"] = tuple(everyone(g) / total if total > 0 else np.full(ids.size, np.nan)
                                         for g in (lower, upper))
        results = {}
        for kind, (rl, ru) in bounds.items():
            base = sm[kind]
            label = kind.capitalize()
            if "ratio" in measures:
                df = base.assign(ci_ratio_lower=rl, ci_ratio_upper=ru)[keep].reset_index(drop=True)
                df.attrs.update(meta, description=f"{label} Standardized Ratio")
                results[f"{kind}_ratio"] = df
            if "rate" in measures:
                df = base.assign(ci_rate_lower=np.clip(rl * population_rate, 0.0, 100.0),
                                 ci_rate_upper=np.clip(ru * population_rate, 0.0, 100.0))[keep].reset_index(drop=True)
                df.attrs.update(meta, description=f"{label} Standardized Rate")
                results[f"{kind}_rate"] = df
        return results
