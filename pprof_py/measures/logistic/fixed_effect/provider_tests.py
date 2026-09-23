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
from ....inference.decision import provider_test
from ....inference.empirical_null.models import NullModel, TheoreticalNull
from ....inference.standardized import standardized_measure
from ....inference.zstat import z_statistic

logger = logging.getLogger(__name__)


class _ProviderTestMethods:
    """Mixin fragment: provider-effect hypothesis tests."""

    def test(
        self,
        providers: Optional[Union[list, np.ndarray]] = None,
        level: float = 0.95,
        test_method: str = "poibin_exact",
        score_modified: bool = True,
        null: Union[str, float] = "median",
        n_bootstrap: int = 10000,
        alternative: str = "two_sided",
        random_state=None,
    ) -> pd.DataFrame:
        """Conduct hypothesis tests on provider effects.

        Supported test methods:
        - "poibin_exact"  (exact test using Poisson-binomial DP approach)
        - "bootstrap_exact" (exact test via bootstrap resampling)
        - "score"         (score test; can be "modified" or standard)
        - "wald"          (wald test; disclaim for outlying providers)

        Parameters
        ----------
        providers : list or np.ndarray, optional
            Subset of provider IDs to test. If None, all providers are included.
        level : float, default=0.95
            Confidence level => alpha = 1 - level is significance.
        test_method : {"poibin_exact","bootstrap_exact","score","wald"}, default="poibin_exact"
            Which testing approach to use.
        score_modified : bool, default=True
            If True, uses a simpler "modified" score approach that does not re-fit 
            restricted models for each provider. If False, you would do the standard 
            approach (placeholder or partial).
        null : {"median"} or float, default="median"
            The null hypothesis value for gamma. If "median", uses median(gamma_hat).
            If numeric, that numeric is used instead.
        n_bootstrap : int, default=10000
            Resample size for "bootstrap_exact" approach.
        alternative : {"two_sided","greater","less"}, default="two_sided"
            Direction of test. "two_sided" is default.
        random_state : None, int, numpy.random.Generator or RandomState
            Seed for ``test_method="bootstrap_exact"``; ignored otherwise.
            ``None`` (default) keeps the legacy behaviour of drawing from
            NumPy's global RNG, reproducible only via ``np.random.seed``.
            Pass an int or ``Generator`` for a self-contained stream.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns ["flag", "p_value", "stat", "std_error"] (if applicable)
            indexed by provider ID. Also has an attribute "provider_size".

        Raises
        ------
        ValueError
            If the model is not fitted or if arguments are invalid.
        """

        if self.coefficients_ is None or self.variances_ is None:
            raise ValueError("The model must be fitted with valid coefficients_ and variances_.")

        alpha = 1.0 - level
        gamma_vals = self.coefficients_["gamma"].flatten()
        se_gamma = np.sqrt(self.variances_["gamma"].flatten())
        n_obs = self.outcome_.size if self.outcome_ is not None else 0
        p = len(self.coefficients_["beta"])
        m = len(gamma_vals)
        df = n_obs - (m + p)

        if null == "median":
            gamma_null = np.median(gamma_vals)
        elif isinstance(null, (int, float)):
            gamma_null = float(null)
        else:
            raise ValueError("Argument 'null' must be 'median' or a numeric value.")

        full_index = np.arange(m)
        if providers is not None:
            mask = np.isin(self.groups_, providers)
            indices = full_index[mask]
        else:
            indices = full_index

        tested_groups = self.groups_[indices]
        size_dict = dict(zip(self.groups_, self.group_sizes_))

        if test_method == "wald":
            flags, pvals, stats, se = self._compute_wald_gamma(indices, gamma_null, alpha, alternative, gamma_vals, se_gamma, df)
        elif test_method == "score":
            flags, pvals, stats, se = self._compute_score_gamma(indices, gamma_null, alpha, alternative, score_modified)
        elif test_method == "poibin_exact":
            flags, pvals, stats, se = self._compute_poibin_gamma(indices, gamma_null, alpha, alternative)
        elif test_method == "bootstrap_exact":
            flags, pvals, stats, se = self._compute_bootstrap_gamma(indices, gamma_null, alpha, alternative, n_bootstrap, random_state)
        else:
            raise ValueError("test_method must be one of {'poibin_exact','bootstrap_exact','score','wald'}.")

        df_res = pd.DataFrame({
            "flag": pd.Categorical(flags, categories=[-1, 0, 1]),
            "p_value": pvals,
            "stat": stats,
            "std_error": se
        }, index=tested_groups)
        df_res.attrs["provider_size"] = {gid: size_dict[gid] for gid in tested_groups}
        return df_res

    def _compute_wald_gamma(
        self, 
        indices: np.ndarray,
        gamma_null: float, 
        alpha: float, 
        alternative: str, 
        gamma_vals: np.ndarray, 
        se_gamma: np.ndarray, 
        df: int
    ) -> Tuple[List[int], List[float], List[float], List[float]]:
        """Compute Wald test for gamma coefficients.

        Parameters
        ----------
        indices : np.ndarray
            Indices of tested groups.
        gamma_null : float
            Null hypothesis value.
        alpha : float
            Significance level.
        alternative : str
            Hypothesis type ("two_sided", "greater", or "less").
        gamma_vals : np.ndarray
            Gamma coefficient values.
        se_gamma : np.ndarray
            Standard errors for gamma.
        df : int
            Degrees of freedom.

        Returns
        -------
        Tuple[List[int], List[float], List[float], List[float]]
            Flags, p-values, test statistics, and standard errors.
        """
        tested_gamma = gamma_vals[indices]
        tested_se = se_gamma[indices]
        wald_stat = (tested_gamma - gamma_null) / tested_se
        prob = t.sf(wald_stat, df=df) if df > 0 else norm.sf(wald_stat)

        flags, pvals, stats, se = [], [], [], []
        for i, st in enumerate(wald_stat):
            pr = prob[i]
            if alternative == "two_sided":
                f_ = 1 if pr < alpha / 2 else -1 if pr > 1 - alpha / 2 else 0
                p_val = 2 * min(pr, 1 - pr)
            elif alternative == "greater":
                f_ = 1 if pr < alpha else 0
                p_val = pr
            elif alternative == "less":
                f_ = -1 if (1 - pr) < alpha else 0
                p_val = 1 - pr
            else:
                raise ValueError("Argument 'alternative' must be 'two_sided','greater','less'.")
            flags.append(f_)
            pvals.append(round(p_val, 7))
            stats.append(st)
            se.append(tested_se[i])
        return flags, pvals, stats, se

    def _compute_score_gamma(
        self, 
        indices: np.ndarray, 
        gamma_null: float, 
        alpha: float, 
        alternative: str, 
        score_modified: bool
    ) -> Tuple[List[int], List[float], List[float], List[float]]:
        """Compute Score test for gamma coefficients.

        Parameters
        ----------
        indices : np.ndarray
            Indices of tested groups.
        gamma_null : float
            Null hypothesis value.
        alpha : float
            Significance level.
        alternative : str
            Hypothesis type ("two_sided", "greater", or "less").
        score_modified : bool
            Use modified score approach if True.

        Returns
        -------
        Tuple[List[int], List[float], List[float], List[float]]
            Flags, p-values, test statistics, and standard errors.
        """
        if not score_modified:
            raise NotImplementedError("Standard (unmodified) score test not implemented. Use score_modified=True.")

        pvec = 1.0 / (1.0 + np.exp(-(gamma_null + self.xbeta_)))
        pvec = np.clip(pvec, 1e-10, 1 - 1e-10)

        flags, pvals, stats, se = [], [], [], []
        for g_ind in indices:
            mask_g = (self.group_indices_ == g_ind)
            obs_count = np.sum(self.outcome_[mask_g])
            sum_p = np.sum(pvec[mask_g])
            sum_var = np.sum(pvec[mask_g] * (1 - pvec[mask_g]))
            zscore = (obs_count - sum_p) / np.sqrt(sum_var) if sum_var >= 1e-14 else 0.0

            if alternative == "two_sided":
                p_one_side = norm.sf(abs(zscore))
                p_val = 2 * p_one_side
                f_ = 1 if p_one_side < alpha / 2 and zscore > 0 else -1 if p_one_side < alpha / 2 else 0
            elif alternative == "greater":
                p_val = norm.sf(zscore)
                f_ = 1 if p_val < alpha else 0
            elif alternative == "less":
                p_val = norm.cdf(zscore)
                f_ = -1 if p_val < alpha else 0
            else:
                raise ValueError("Argument 'alternative' must be 'two_sided','greater','less'.")
            flags.append(f_)
            pvals.append(round(p_val, 7))
            stats.append(zscore)
            se.append(np.nan)
        return flags, pvals, stats, se

    def _compute_poibin_gamma(
        self, 
        indices: np.ndarray, 
        gamma_null: float, 
        alpha: float, 
        alternative: str
    ) -> Tuple[List[int], List[float], List[float], List[float]]:
        """Compute Poisson-Binomial exact test for gamma coefficients.

        Parameters
        ----------
        indices : np.ndarray
            Indices of tested groups.
        gamma_null : float
            Null hypothesis value.
        alpha : float
            Significance level.
        alternative : str
            Hypothesis type ("two_sided", "greater", or "less").

        Returns
        -------
        Tuple[List[int], List[float], List[float], List[float]]
            Flags, p-values, test statistics, and standard errors.
        """
        flags, pvals, stats, se = [], [], [], []
        for g_ind in indices:
            mask_g = (self.group_indices_ == g_ind)
            x_mat = self.xbeta_[mask_g]
            pvec = 1.0 / (1.0 + np.exp(-(gamma_null + x_mat)))
            pvec = np.clip(pvec, 1e-10, 1 - 1e-10)
            obs = int(np.sum(self.outcome_[mask_g]))  # Convert obs to integer for indexing

            pb = PoiBin(pvec)
            cdf_obs = pb.cdf[obs] 
            pmf_obs = pb.pmf[obs]          
            cdf_obs_minus_1 = pb.cdf[obs - 1] if obs > 0 else 0.0  # Handle edge case

            if alternative == "two_sided":
                pr = 1.0 - cdf_obs + 0.5 * pmf_obs
                zscore = norm.isf(pr)
                f_ = 1 if pr < alpha / 2 else -1 if pr > 1 - alpha / 2 else 0
                p_val = 2 * min(pr, 1 - pr)
            elif alternative == "greater":
                pr = 1.0 - cdf_obs_minus_1 if obs > 0 else 1.0
                zscore = norm.isf(pr)
                p_val = pr
                f_ = 1 if pr < alpha else 0
            elif alternative == "less":
                pr = cdf_obs
                zscore = norm.ppf(pr)
                p_val = pr
                f_ = -1 if pr < alpha else 0
            else:
                raise ValueError("Argument 'alternative' must be 'two_sided', 'greater', or 'less'.")
            flags.append(f_)
            pvals.append(round(p_val, 7))
            stats.append(zscore)
            se.append(np.nan)
        return flags, pvals, stats, se

    def _compute_bootstrap_gamma(
        self, 
        indices: np.ndarray, 
        gamma_null: float, 
        alpha: float, 
        alternative: str, 
        n_bootstrap: int,
        random_state=None,
    ) -> Tuple[List[int], List[float], List[float], List[float]]:
        """Compute Bootstrap exact test for gamma coefficients.

        Parameters
        ----------
        indices : np.ndarray
            Indices of tested groups.
        gamma_null : float
            Null hypothesis value.
        alpha : float
            Significance level.
        alternative : str
            Hypothesis type ("two_sided", "greater", or "less").
        n_bootstrap : int
            Number of bootstrap resamples.
        random_state : None, int, numpy.random.Generator or RandomState
            See :meth:`test`.

        Returns
        -------
        Tuple[List[int], List[float], List[float], List[float]]
            Flags, p-values, test statistics, and standard errors.
        """
        # REV-016: build the uniform source ONCE, outside the provider loop,
        # so the stream continues across providers (per-provider seeding
        # would hand every provider identical draws).  random_state=None
        # deliberately keeps the legacy global RNG: switching it to
        # default_rng(None) would silently break callers who currently get
        # reproducibility from np.random.seed().
        if random_state is None:
            _uniform = np.random.rand
        elif isinstance(random_state, np.random.RandomState):
            _uniform = random_state.rand
        else:
            _uniform = np.random.default_rng(random_state).random

        flags, pvals, stats, se = [], [], [], []
        for g_ind in indices:
            mask_g = (self.group_indices_ == g_ind)
            x_mat = self.xbeta_[mask_g]
            pvec = 1.0 / (1.0 + np.exp(-(gamma_null + x_mat)))
            pvec = np.clip(pvec, 1e-10, 1 - 1e-10)
            obs = np.sum(self.outcome_[mask_g])

            draws = np.empty(n_bootstrap, dtype=np.int_)
            group_size = pvec.size
            for i_bs in range(n_bootstrap):
                r = _uniform(group_size)
                draws[i_bs] = np.sum(r < pvec)

            if alternative == "two_sided":
                bigger = np.sum(draws > obs)
                equal = np.sum(draws == obs)
                pr = (bigger + 0.5 * equal) / n_bootstrap
                zscore = norm.isf(pr)
                f_ = 1 if pr < alpha / 2 else -1 if pr > 1 - alpha / 2 else 0
                p_val = 2 * min(pr, 1 - pr)
            elif alternative == "greater":
                pr = np.sum(draws >= obs) / n_bootstrap
                zscore = norm.isf(pr)
                p_val = pr
                f_ = 1 if pr < alpha else 0
            elif alternative == "less":
                pr = np.sum(draws <= obs) / n_bootstrap
                zscore = norm.ppf(pr)
                p_val = pr
                f_ = -1 if pr < alpha else 0
            else:
                raise ValueError("Argument 'alternative' must be 'two_sided','greater','less'.")
            flags.append(f_)
            pvals.append(round(p_val, 7))
            stats.append(zscore)
            se.append(np.nan)
        return flags, pvals, stats, se

    # -------------------------------------------------------------------------
    # STANDARDIZED MEASURE SE AND EMPIRICAL NULL TESTING
    # -------------------------------------------------------------------------

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
        if null_model is None:
            null = TheoreticalNull()
        elif isinstance(null_model, NullModel):
            null = null_model
        elif callable(null_model):
            null = null_model(z)
        else:
            raise TypeError("null_model must be a NullModel or a callable returning one.")
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
