"""Provider-effect hypothesis testing for logistic fixed-effect models.

Contains ``_ProviderTestMethods``, a mixin fragment providing
:meth:`test` (gamma-level tests) and :meth:`test_standardized`
(standardized-measure tests with empirical-null calibration).
"""
from __future__ import annotations

import logging
from typing import Optional

import warnings

import numpy as np
import pandas as pd

from ....utils.numerical import sigmoid
from ....inference.decision import provider_test, resolve_null_model
from ....inference._recording import record
from ....inference.count_tests import MonteCarlo, PlugIn, count_test, rows_by_provider
from ....inference.effect_tests import bootstrap_tails, effect_test, normalize_alternative, reference_effect
from ....inference.standardized import _reference_gamma, standardized_measure
from ....inference.zstat import z_statistic

logger = logging.getLogger(__name__)

PATIENT_CLUSTERED = ("robust_wald", "robust_score")


def _patient_clusters(provider_indices: np.ndarray, obs_ids: np.ndarray):
    """(provider, patient) clusters: a patient treated at two providers forms two clusters.

    Returns the cluster of each row, the provider of each cluster and the number of rows in each cluster.
    """
    codes, labels = pd.factorize(np.asarray(obs_ids, dtype=object).ravel())
    if (codes < 0).any():
        raise ValueError("Patient IDs (obs_id_var) must not be missing for the patient-clustered tests.")
    k = max(len(labels), 1)
    key = np.asarray(provider_indices, dtype=np.int64).ravel() * k + codes
    keys, cluster = np.unique(key, return_inverse=True)
    cluster = cluster.ravel()
    return cluster, keys // k, np.bincount(cluster)


class _ProviderTestMethods:
    """Mixin fragment: provider-effect hypothesis tests."""

    def test(
        self,
        providers=None,
        *,
        test_method: str = "poibin_exact",
        reference="median",
        null_model=None,
        alternative: str = "two_sided",
        level: float = 0.95,
        critical: Optional[float] = None,
        interval: str = "inversion",
        n_resample: int = 10000,
        seed=None,
    ) -> pd.DataFrame:
        """Test each provider's effect against the reference effect gamma_0.

        Parameters
        ----------
        providers : array-like, optional
            Report only these providers; gamma_0 and any empirical null use all.
        test_method : {"poibin_exact", "score", "wald", "bootstrap_exact", "robust_wald", "robust_score"}
            ``"poibin_exact"``: exact Poisson-binomial test of the provider's
            event count at gamma_0 (two-sided mid-p). ``"score"``:
            ``(O - E) / sqrt(Var_0(O))`` at gamma_0. ``"wald"``:
            ``(gamma_j - gamma_0) / SE(gamma_j)`` with a normal reference (as
            R pprof) and the variance at the average case mix,
            ``variances_["gamma_case_mix"]`` (R uses ``variances_["gamma"]``,
            which depends on the covariates' origin; C34); unreliable for
            providers without a finite estimate (see
            :func:`~pprof_py.inference.at_bound`). ``"bootstrap_exact"``: the
            exact test by simulation. Binomial models are weighted by trials.

            ``"robust_wald"`` and ``"robust_score"`` are R pprofR's patient-clustered
            tests, with beta treated as known; they need the model fitted with
            ``obs_id_var``. A cluster is a (provider, patient) pair, so a patient
            treated at two providers forms two clusters. With row residuals
            ``r = Y - N p`` summed within each patient, ``V_j`` is the sum over
            provider j's patients of the squared sums. ``"robust_wald"`` is
            ``(gamma_j - gamma_0) / SE`` with ``SE = sqrt(V_j) / I_j`` at the
            fitted ``p`` (``I_j`` the provider's information): the
            ``variance="robust_fixed_beta"`` SE of :meth:`test_standardized`.
            ``"robust_score"`` is ``(O - E) / sqrt(V_j)``, with ``p`` and the
            residuals at gamma_0 as in ``"score"``. A statistic that is not finite
            is reported as not tested.
        reference : "median", "mean", or float
            The reference effect gamma_0: the median of the estimated effects,
            their size-weighted mean, or a value on the effect scale.
        null_model : NullModel or callable, optional
            Null for the z-statistics: :class:`~pprof_py.inference.TheoreticalNull`
            by default, or an instance such as ``FixedNull(sd=...)``, or a callable
            that receives the z-statistics, such as ``EmpiricalNull.fitter(...)``.
        alternative, level, critical, interval
            As in :func:`~pprof_py.inference.provider_test`.
            Intervals are available for the Wald tests (``"wald"`` and
            ``"robust_wald"``) and, by inverting the exact test, for
            ``"poibin_exact"``.
        n_resample, seed : int, optional
            Monte Carlo draws and seed for ``"bootstrap_exact"``.

        Returns
        -------
        pandas.DataFrame
            Indexed by provider with columns
            :data:`~pprof_py.inference.PROVIDER_TEST_COLUMNS`: ``flag`` is +1
            above gamma_0, -1 below, 0 not significant, NA not tested. For the
            patient-clustered tests, ``attrs`` also record the variance
            (``"patient-clustered, beta fixed"``), the number of clusters and the
            largest cluster's number of rows.
        """
        if self.coefficients_ is None or self.variances_ is None:
            raise ValueError("The model must be fitted before testing.")
        alt = normalize_alternative(alternative)
        gamma = np.asarray(self.coefficients_["gamma"], dtype=np.float64).ravel()
        g0 = reference_effect(gamma, self.provider_sizes_, reference)
        m = gamma.size
        idx = np.asarray(self.provider_indices_).ravel()
        xb = np.asarray(self.xbeta_, dtype=np.float64).ravel()
        y = np.asarray(self.outcome_, dtype=np.float64).ravel()
        trials = None if self.N_ is None else np.asarray(self.N_, dtype=np.float64).ravel()
        se = None
        limits = None
        clusters = None
        if test_method in PATIENT_CLUSTERED:
            if self.obs_ids_ is None or getattr(self, "robust_variances_", None) is None:
                raise ValueError(f"test_method='{test_method}' needs patient IDs: fit the model with obs_id_var.")
            clusters = _patient_clusters(idx, self.obs_ids_)
        if test_method in ("wald", "robust_wald"):
            if test_method == "wald":
                se = np.sqrt(np.asarray(self.variances_["gamma_case_mix"], dtype=np.float64).ravel())   # C34
            else:   # beta fixed, as R pprofR's robust_wald; the same SE as test_standardized(variance="robust_fixed_beta")
                se = np.sqrt(np.asarray(self.robust_variances_["gamma_fixed_beta"], dtype=np.float64).ravel())
            z = (gamma - g0) / se
            events = np.bincount(idx, weights=y, minlength=m)
            size = np.bincount(idx, weights=np.ones_like(y) if trials is None else trials, minlength=m)
            unbounded = (events == 0) | (events == size)       # no finite estimate: the effect sits at the bound
            if providers is not None:
                unbounded &= np.isin(self.provider_ids_, np.atleast_1d(providers))
            if unbounded.any():
                warnings.warn(f"{int(unbounded.sum())} provider(s) have no events or only events, so their effects have "
                              "no finite estimate and are held at the bound; their Wald statistics and intervals are "
                              "unreliable (test_method='poibin_exact' handles them).", UserWarning, stacklevel=2)
        elif test_method == "score":
            w = np.ones(xb.size) if trials is None else trials
            p0 = np.clip(sigmoid(g0 + xb), 1e-10, 1 - 1e-10)
            expected = np.bincount(idx, weights=w * p0, minlength=m)
            var0 = np.bincount(idx, weights=w * p0 * (1 - p0), minlength=m)
            observed = np.bincount(idx, weights=y, minlength=m)
            with np.errstate(divide="ignore", invalid="ignore"):
                z = np.where(var0 >= 1e-14, (observed - expected) / np.sqrt(var0), 0.0)
            record("score", observed=observed, expected=expected, var0=var0)
        elif test_method == "robust_score":
            cluster, cluster_provider, _ = clusters
            w = np.ones(xb.size) if trials is None else trials
            p0 = np.clip(sigmoid(g0 + xb), 1e-10, 1 - 1e-10)           # as in "score"
            residual = y - w * p0
            u = np.bincount(idx, weights=residual, minlength=m)          # O - E
            v = np.bincount(cluster_provider, weights=np.bincount(cluster, weights=residual) ** 2, minlength=m)
            with np.errstate(divide="ignore", invalid="ignore"):
                z = np.where(v > 0, u / np.sqrt(v), np.nan)
        elif test_method in ("poibin_exact", "bootstrap_exact"):
            rows_of = rows_by_provider(idx, m)
            obs = np.array([y[rows].sum() for rows in rows_of])
            n_of = [None if trials is None else trials[rows] for rows in rows_of]
            if test_method == "poibin_exact":
                nulls = [PlugIn(prob=lambda g, rows=rows: sigmoid(g + xb[rows]), trials=n_j) for rows, n_j in zip(rows_of, n_of)]
            else:
                rng = np.random.default_rng(seed)
                nulls = [MonteCarlo(simulate=lambda o, g, rows=rows, n_j=n_j: bootstrap_tails(o, sigmoid(g + xb[rows]), n_j,
                                                                                               n_resample, rng),
                                    n_resample=n_resample)
                         for rows, n_j in zip(rows_of, n_of)]
            wanted = None if providers is None else np.isin(self.provider_ids_, np.atleast_1d(providers))
            z, limits = count_test(obs, nulls, g0, alternative=alt, start=gamma, wanted=wanted)
        else:
            raise ValueError("test_method must be 'poibin_exact', 'score', 'wald', 'bootstrap_exact', 'robust_wald' "
                             "or 'robust_score'.")
        result = effect_test(self.provider_ids_, gamma, z, g0, se=se, null_model=null_model, alternative=alt,
                             level=level, critical=critical, interval=interval, providers=providers,
                             test_method=test_method, limits=limits)
        sizes = dict(zip(self.provider_ids_, self.provider_sizes_))
        result.attrs["provider_size"] = {g: sizes[g] for g in result.index}
        if clusters is not None:
            result.attrs.update({"variance": "patient-clustered, beta fixed", "clusters": int(clusters[2].size),
                                 "largest_cluster_rows": int(clusters[2].max())})
        return result

    def funnel_limits(self, providers=None, *, test_method: str = "score", reference="median", null_model=None,
                      alternative: str = "two_sided", level: float = 0.95, critical: Optional[float] = None,
                      levels=None):
        """Funnel coordinates and control limits that agree with :meth:`test` by construction.

        Parameters
        ----------
        providers, reference, null_model, alternative, level, critical
            As in :meth:`test`.
        test_method : {"score", "poibin_exact", "wald"}, default "score"
            The test whose flags the limits reproduce. ``"score"`` (the funnel default) gives smooth limit curves
            in the precision ``E^2/V0``; ``"poibin_exact"`` (the default of :meth:`test`) gives exact limits per
            provider at half-integer counts; ``"wald"`` gives limits for the effect in ``1/SE^2``.
            ``"bootstrap_exact"`` is a Monte Carlo test and has no funnel limits; the patient-clustered tests
            have none yet, since each provider's variance depends on its own patients.
        levels : sequence of float, optional
            Levels of the limit curves (default: ``level`` only). Flags exist only at ``level``.

        Returns
        -------
        FunnelLimits
            See :func:`pprof_py.inference.funnel_limits`.
        """
        if test_method == "bootstrap_exact":
            raise ValueError("test_method='bootstrap_exact' is a Monte Carlo test and has no funnel limits; use "
                             "'score', 'poibin_exact' or 'wald'.")
        if test_method in PATIENT_CLUSTERED:
            raise ValueError(f"test_method='{test_method}' has no funnel limits yet: each provider's patient-clustered "
                             "variance depends on its own patients. Use 'score', 'poibin_exact' or 'wald'.")
        from ....inference.funnel import build_funnel_limits
        return build_funnel_limits(
            self, lambda: self.test(providers, test_method=test_method, reference=reference, null_model=null_model,
                                    alternative=alternative, level=level, critical=critical),
            order=self.provider_ids_, levels=levels)

    def _provider_event_counts(self):
        """Provider ids, events, trials and finite-estimate status (see ``degenerate_providers``)."""
        from ....inference.standardized import at_bound
        m = np.asarray(self.coefficients_["gamma"]).size
        idx = np.asarray(self.provider_indices_).ravel()
        y = np.asarray(self.outcome_, dtype=np.float64).ravel()
        trials = np.ones(idx.size) if self.N_ is None else np.asarray(self.N_, dtype=np.float64).ravel()
        return (pd.Index(self.provider_ids_, name="provider_id"), np.bincount(idx, weights=y, minlength=m),
                np.bincount(idx, weights=trials, minlength=m), ~at_bound(self))

    def test_standardized(
        self,
        measure: str = "direct_rate",
        *,
        providers=None,
        null_value="reference",
        transform="auto",
        null_model=None,
        population=None,
        reference="median",
        variance: str = "model",
        indirect_variance: str = "null",
        alternative: str = "two_sided",
        level: float = 0.95,
        critical: Optional[float] = None,
        interval: str = "inversion",
        bounds="auto",
    ) -> pd.DataFrame:
        """Test providers on a standardized measure (or on the provider effect).

        A convenience composition of the layers in :mod:`pprof_py.inference`:
        :func:`~pprof_py.inference.standardized_measure`, then
        :func:`~pprof_py.inference.z_statistic`, then a null model, then
        :func:`~pprof_py.inference.provider_test`. Call those directly for any
        step this method does not expose.

        Parameters
        ----------
        measure : {"direct_rate", "direct_ratio", "indirect_rate", "indirect_ratio", "gamma"}
        providers : array-like, optional
            Report only these providers. Every step (including any empirical
            null and any ``"mean"``/``"median"`` null value) uses all providers.
        null_value : "reference", "mean", "median", float, or callable
            Value under the null, on the measure's own scale. ``"reference"``
            (default) is the measure at the reference effect gamma_0, so the
            test agrees with a test of gamma_j = gamma_0.
        transform : "auto", "identity", "logit", "log", or Transform
            Working scale for the test and intervals.
        null_model : NullModel or callable, optional
            A null model (default :class:`~pprof_py.inference.TheoreticalNull`;
            for example ``FixedNull(sd=1.81)``), or a callable that receives the
            z-statistics and returns one, such as
            ``EmpiricalNull.fitter(size=sizes, n_groups=4)``.
        population, reference, variance, indirect_variance
            Passed to :func:`~pprof_py.inference.standardized_measure`.
        bounds : "auto", (float, float), or None
            Limits to clip intervals to. ``"auto"`` clips identity-scale rates
            to [0, 1] and identity-scale ratios to [0, inf) and leaves other
            scales alone (their inverse transforms already respect the range).
        alternative, level, critical, interval
            Passed to :func:`~pprof_py.inference.provider_test`.

        Returns
        -------
        pandas.DataFrame
            Indexed by provider with columns
            :data:`~pprof_py.inference.PROVIDER_TEST_COLUMNS`; ``flag`` is +1
            above the null value, -1 below, 0 not significant, NA not tested.
        """
        m = standardized_measure(self, measure, population=population, reference=reference,
                                 variance=variance, indirect_variance=indirect_variance)
        z = z_statistic(m, null_value=null_value, transform=transform)
        null = resolve_null_model(null_model, z)
        if isinstance(bounds, str):
            if bounds != "auto":
                raise ValueError("bounds must be 'auto', a (lower, upper) pair, or None.")
            bounds = None
            if z.transform.name == "identity" and measure.endswith("_rate"):
                bounds = (0.0, 1.0)
            elif z.transform.name == "identity" and measure.endswith("_ratio"):
                bounds = (0.0, np.inf)
        result = provider_test(z, null, alternative=alternative, level=level, critical=critical,
                               interval=interval, bounds=bounds)
        indirect = measure in ("indirect_rate", "indirect_ratio")
        # (O/E - 1) with the variance of O at gamma_0 on the identity scale is the score statistic
        score = indirect and indirect_variance == "null" and z.transform.name == "identity"
        result.attrs["test_method"] = "score" if score else "wald"
        result.attrs["variance"] = indirect_variance if indirect else variance
        result.attrs["reference"] = _reference_gamma(np.asarray(self.coefficients_["gamma"], dtype=np.float64).ravel(),
                                                     np.asarray(self.provider_sizes_, dtype=np.float64).ravel(),
                                                     reference)
        if providers is not None:
            result = result.loc[result.index.isin(np.atleast_1d(providers))]
        return result
