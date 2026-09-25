"""Provider-effect hypothesis testing for logistic fixed-effect models.

Contains ``_ProviderTestMethods``, a mixin fragment providing
:meth:`test` (gamma-level tests) and :meth:`test_standardized`
(standardized-measure tests with empirical-null calibration).
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import norm, t
from fast_poibin import PoiBin

from ....utils.numerical import sigmoid
from ....inference.decision import provider_test, resolve_null_model
from ....inference.effect_tests import (EXACT_P_FLOOR, bootstrap_tails, effect_test, normalize_alternative,
                                        poibin_tails, reference_effect, z_from_tails)
from ....inference.empirical_null.models import NullModel, TheoreticalNull
from ....inference.standardized import standardized_measure
from ....inference.zstat import z_statistic

logger = logging.getLogger(__name__)


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
        test_method : {"poibin_exact", "score", "wald", "bootstrap_exact"}
            ``"poibin_exact"``: exact Poisson-binomial test of the provider's
            event count at gamma_0 (two-sided mid-p). ``"score"``:
            ``(O - E) / sqrt(Var_0(O))`` at gamma_0. ``"wald"``:
            ``(gamma_j - gamma_0) / SE(gamma_j)`` with a normal reference (as
            R pprof); unreliable for providers at the numerical bound (see
            :func:`~pprof_py.inference.at_bound`). ``"bootstrap_exact"``: the
            exact test by simulation. Binomial models are weighted by trials.
        reference : "median", "mean", or float
            The reference effect gamma_0: the median of the estimated effects,
            their size-weighted mean, or a value on the effect scale.
        null_model : NullModel or callable, optional
            Null for the z-statistics: :class:`~pprof_py.inference.TheoreticalNull`
            by default, or an instance such as ``FixedNull(sd=...)``, or a callable
            that receives the z-statistics, such as ``EmpiricalNull.fitter(...)``.
        alternative, level, critical, interval
            As in :func:`~pprof_py.inference.provider_test`.
            Intervals are available for the Wald test.
        n_resample, seed : int, optional
            Monte Carlo draws and seed for ``"bootstrap_exact"``.

        Returns
        -------
        pandas.DataFrame
            Indexed by provider with columns
            :data:`~pprof_py.inference.PROVIDER_TEST_COLUMNS`: ``flag`` is +1
            above gamma_0, -1 below, 0 not significant, NA not tested.
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
        if test_method == "wald":
            se = np.sqrt(np.asarray(self.variances_["gamma"], dtype=np.float64).ravel())
            z = (gamma - g0) / se
        elif test_method == "score":
            w = np.ones(xb.size) if trials is None else trials
            p0 = np.clip(sigmoid(g0 + xb), 1e-10, 1 - 1e-10)
            expected = np.bincount(idx, weights=w * p0, minlength=m)
            var0 = np.bincount(idx, weights=w * p0 * (1 - p0), minlength=m)
            observed = np.bincount(idx, weights=y, minlength=m)
            with np.errstate(divide="ignore", invalid="ignore"):
                z = np.where(var0 >= 1e-14, (observed - expected) / np.sqrt(var0), 0.0)
        elif test_method in ("poibin_exact", "bootstrap_exact"):
            p0 = sigmoid(g0 + xb)
            order = np.argsort(idx, kind="stable")
            edges = np.r_[0, np.cumsum(np.bincount(idx, minlength=m))]
            rng = np.random.default_rng(seed) if test_method == "bootstrap_exact" else None
            tails = np.empty((m, 4))
            for j in range(m):
                rows = order[edges[j]:edges[j + 1]]
                n_j = None if trials is None else trials[rows]
                if test_method == "poibin_exact":
                    tails[j] = poibin_tails(y[rows].sum(), p0[rows], n_j)
                else:
                    tails[j] = bootstrap_tails(y[rows].sum(), p0[rows], n_j, n_resample, rng)
            two = alt == "two_sided"
            z = z_from_tails(tails[:, 0] if two else tails[:, 2], tails[:, 1] if two else tails[:, 3], alt,
                             EXACT_P_FLOOR if test_method == "poibin_exact" else 0.5 / n_resample)
        else:
            raise ValueError("test_method must be 'poibin_exact', 'score', 'wald', or 'bootstrap_exact'.")
        result = effect_test(self.provider_ids_, gamma, z, g0, se=se, null_model=null_model, alternative=alt,
                             level=level, critical=critical, interval=interval, providers=providers,
                             test_method=test_method)
        sizes = dict(zip(self.provider_ids_, self.provider_sizes_))
        result.attrs["provider_size"] = {g: sizes[g] for g in result.index}
        return result

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
        if providers is not None:
            result = result.loc[result.index.isin(np.atleast_1d(providers))]
        return result
