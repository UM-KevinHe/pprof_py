"""Provider-level confidence intervals for logistic fixed-effect models.

Contains ``_ConfidenceIntervalMethods``, a mixin fragment providing
:meth:`calculate_confidence_intervals` and supporting helpers.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import norm, t
from scipy.optimize import root_scalar
from fast_poibin import PoiBin

from ....utils.numerical import sigmoid

logger = logging.getLogger(__name__)


class _ConfidenceIntervalMethods:
    """Mixin fragment: confidence-interval computation."""

    # -------------------------------------------------------------------------
    # BASIC (T-BASED) CI BOUNDS FOR GAMMA
    # -------------------------------------------------------------------------
    def _compute_ci_bounds(self, gamma: np.ndarray, se: np.ndarray, df: int, level: float, alternative: str) -> Tuple[np.ndarray, np.ndarray]:
        """Compute two-sided or one-sided confidence interval bounds for gamma using a t-distribution.

        Parameters
        ----------
        gamma : np.ndarray
            Point estimates of provider effects.
        se : np.ndarray
            Standard errors for gamma.
        df : int
            Degrees of freedom for the t-distribution.
        level : float
            Confidence level (e.g., 0.95 for 95% CI).
        alternative : str
            Hypothesis type: 'two_sided', 'greater', or 'less'.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Lower and upper bounds for the confidence intervals.

        Raises
        ------
        ValueError
            If 'alternative' is not one of the allowed values.
        """
        alpha = 1.0 - level
        if alternative == "two_sided":
            crit_value = t.ppf(1.0 - alpha / 2.0, df)
            lower = gamma - crit_value * se
            upper = gamma + crit_value * se
        elif alternative == "greater":
            crit_value = t.ppf(1.0 - alpha, df)
            lower = gamma - crit_value * se
            upper = np.full_like(gamma, np.inf)
        elif alternative == "less":
            crit_value = t.ppf(1.0 - alpha, df)
            lower = np.full_like(gamma, -np.inf)
            upper = gamma + crit_value * se
        else:
            raise ValueError("Argument 'alternative' must be 'two_sided', 'greater', or 'less'.")
        return lower, upper

    def _search_root(self, func, bracket: Tuple[float, float], max_attempts: int = 3) -> Optional[float]:
        """Attempt to find a root of 'func' within an initial bracket, expanding if necessary.

        Parameters
        ----------
        func : callable
            Function for which to find the root.
        bracket : Tuple[float, float]
            Initial [left, right] bracket for the root.
        max_attempts : int, default=3
            Number of bracket expansions to attempt.

        Returns
        -------
        Optional[float]
            The root if found, otherwise None.
        """
        left, right = bracket
        for i in range(max_attempts):
            try:
                sol = root_scalar(func, bracket=[left, right], method='bisect')
                if sol.converged:
                    return sol.root
            except ValueError:
                pass
            expand_amt = 5.0 * (i + 1)
            left -= expand_amt
            right += expand_amt
        logger.warning(f"Root-finding failed after {max_attempts} attempts with bracket {bracket}")
        return None

    def _get_no_all_events(self, group_idx: int) -> Tuple[bool, bool]:
        """Check if a provider has no events or all events.

        Parameters
        ----------
        group_idx : int
            Index of the provider group.

        Returns
        -------
        Tuple[bool, bool]
            (no_events, all_events) indicating if the provider has no events or all events.
        """
        sum_y = np.sum(self.outcome_[self.provider_indices_ == group_idx])
        gsize = self.provider_sizes_[group_idx]
        return sum_y == 0, sum_y == gsize

    def _score_ci_for_one_group(self, group_idx: int, alpha: float, alternative: str, gamma_guess: float) -> Tuple[float, float]:
        """Compute score-based confidence interval for a single provider's gamma.

        Parameters
        ----------
        group_idx : int
            Provider group index.
        alpha : float
            Significance level.
        alternative : str
            Hypothesis type: 'two_sided', 'greater', or 'less'.
        gamma_guess : float
            Initial guess for gamma.

        Returns
        -------
        Tuple[float, float]
            Lower and upper bounds of the confidence interval.
        """
        # Check for no events or all events
        no_events, all_events = self._get_no_all_events(group_idx)
        observed = np.sum(self.outcome_[self.provider_indices_ == group_idx])
        xbeta_group = self.xbeta_[self.provider_indices_ == group_idx]

        qnorm_half = norm.ppf(1.0 - alpha / 2.0)
        qnorm_1side = norm.ppf(1.0 - alpha)

        def upper_func(gamma):
            probs = 1.0 / (1.0 + np.exp(-(gamma + xbeta_group)))
            score = (observed - probs.sum()) / np.sqrt((probs * (1.0 - probs)).sum())
            return score + (qnorm_half if alternative == "two_sided" else qnorm_1side if alternative == "less" else 0)

        def lower_func(gamma):
            probs = 1.0 / (1.0 + np.exp(-(gamma + xbeta_group)))
            score = (observed - probs.sum()) / np.sqrt((probs * (1.0 - probs)).sum())
            return score - (qnorm_half if alternative == "two_sided" else qnorm_1side if alternative == "greater" else 0)

        if no_events:
            def no_events_func(gamma):
                probs = 1.0 / (1.0 + np.exp(-(gamma + xbeta_group)))
                return qnorm_1side - probs.sum() / np.sqrt((probs * (1.0 - probs)).sum())
            upper_bound = self._search_root(no_events_func, (gamma_guess - 5.0, gamma_guess + 5.0))
            return (-np.inf, upper_bound if upper_bound is not None else np.inf)

        if all_events:
            def all_events_func(gamma):
                probs = 1.0 / (1.0 + np.exp(-(gamma + xbeta_group)))
                return ((1.0 - probs).sum() / np.sqrt((probs * (1.0 - probs)).sum())) - qnorm_1side
            lower_bound = self._search_root(all_events_func, (gamma_guess - 5.0, gamma_guess + 5.0))
            return (lower_bound if lower_bound is not None else -np.inf, np.inf)

        lower_bound, upper_bound = -np.inf, np.inf
        if alternative in ["two_sided", "less"]:
            upper_bound = self._search_root(upper_func, (gamma_guess, gamma_guess + 5.0)) or np.inf
        if alternative in ["two_sided", "greater"]:
            lower_bound = self._search_root(lower_func, (gamma_guess - 5.0, gamma_guess)) or -np.inf
        return (lower_bound, upper_bound)

    def _exact_ci_for_one_group(self, group_idx: int, alpha: float, alternative: str, gamma_guess: float) -> (float, float):
        """Compute 'exact' CI for a single provider's gamma using the fast_poibin package.

        Parameters
        ----------
        group_idx : int
            Provider index in [0, M-1].
        alpha : float
            1 - confidence level.
        alternative : {'two_sided','greater','less'}
            Specifies the alternative hypothesis.
        gamma_guess : float
            An initial guess for gamma.

        Returns
        -------
        (float, float)
            (lower_bound, upper_bound) for this provider's gamma.

        Notes
        -----
        - Uses the fast_poibin.PoiBin class for computing PMF and CDF.
        - For the "mid-p" approach when alternative == "two_sided", we compute:
            lower tail: P(X <= obs-1) + 0.5 * P(X=obs) - alpha/2
            upper tail: P(X >= obs) + 0.5 * P(X=obs) - alpha/2
                      = (1 - P(X <= obs)) + P(X=obs) + 0.5 * P(X=obs) - alpha/2
                      = 1 - P(X <= obs-1) - 0.5 * P(X=obs) - alpha/2
        """
        # Check for no events or all events
        no_events, all_events = self._get_no_all_events(group_idx)
        observed = int(np.sum(self.outcome_[self.provider_indices_ == group_idx]))
        xbeta_group = self.xbeta_[self.provider_indices_ == group_idx]
        n_trials = len(xbeta_group)

        if n_trials == 0:
            logger.warning(f"Group {group_idx} has no observations. Returning (-inf, inf).")
            return (-np.inf, np.inf)

        if no_events:
            if alternative not in ["two_sided", "less"]:
                return (-np.inf, np.inf)

            def upper_func(gamma):
                pvec = 1.0 / (1.0 + np.exp(-(gamma + xbeta_group)))
                if np.any(np.isnan(pvec)) or np.any(np.isinf(pvec)):
                    return np.nan
                pb = PoiBin(pvec)
                alpha_level = alpha if alternative == "less" else alpha / 2.0
                return 0.5 * pb.pmf[0] - alpha_level if len(pb.pmf) > 0 else np.nan

            upper_bound = self._search_root(upper_func, (gamma_guess - 10.0, gamma_guess + 10.0))
            return (-np.inf, upper_bound if upper_bound is not None else np.inf)

        if all_events:
            if alternative not in ["two_sided", "greater"]:
                return (-np.inf, np.inf)

            def lower_func(gamma):
                pvec = 1.0 / (1.0 + np.exp(-(gamma + xbeta_group)))
                if np.any(np.isnan(pvec)) or np.any(np.isinf(pvec)):
                    return np.nan
                pb = PoiBin(pvec)
                alpha_level = alpha if alternative == "greater" else alpha / 2.0
                pmf_n = pb.pmf[n_trials] if len(pb.pmf) > n_trials else np.nan
                return pmf_n - alpha_level if alternative == "greater" else 0.5 * pmf_n - alpha_level

            lower_bound = self._search_root(lower_func, (gamma_guess - 10.0, gamma_guess + 10.0))
            return (lower_bound if lower_bound is not None else -np.inf, np.inf)

        def upper_func(gamma):
            pvec = 1.0 / (1.0 + np.exp(-(gamma + xbeta_group)))
            if np.any(np.isnan(pvec)) or np.any(np.isinf(pvec)):
                return np.nan
            pb = PoiBin(pvec)
            if len(pb.pmf) <= observed or len(pb.cdf) <= observed:
                logger.warning(f"fast_poibin arrays too short for obs={observed}.")
                return np.nan
            pmf_obs = pb.pmf[observed]
            cdf_minus_1 = pb.cdf[observed - 1] if observed > 0 else 0.0
            if alternative == "two_sided":
                return cdf_minus_1 + 0.5 * pmf_obs - alpha / 2.0
            elif alternative == "less":
                return cdf_minus_1 - alpha
            return 1.0

        def lower_func(gamma):
            pvec = 1.0 / (1.0 + np.exp(-(gamma + xbeta_group)))
            if np.any(np.isnan(pvec)) or np.any(np.isinf(pvec)):
                return np.nan
            pb = PoiBin(pvec)
            if len(pb.pmf) <= observed or len(pb.cdf) <= observed:
                logger.warning(f"fast_poibin arrays too short for obs={observed}.")
                return np.nan
            pmf_obs = pb.pmf[observed]
            cdf_obs = pb.cdf[observed]
            if alternative == "two_sided":
                return (1.0 - cdf_obs) + 0.5 * pmf_obs - alpha / 2.0
            elif alternative == "greater":
                cdf_minus_1 = pb.cdf[observed - 1] if observed > 0 else 0.0
                return (1.0 - cdf_minus_1) - alpha
            return 1.0

        lower_bound, upper_bound = -np.inf, np.inf
        search_interval_half_width = 5.0
        if alternative in ["two_sided", "less"]:
            upper_bound = self._search_root(upper_func, (gamma_guess, gamma_guess + search_interval_half_width)) or np.inf
        if alternative in ["two_sided", "greater"]:
            lower_bound = self._search_root(lower_func, (gamma_guess - search_interval_half_width, gamma_guess)) or -np.inf

        if lower_bound > upper_bound:
            logger.warning(f"Lower bound {lower_bound} > Upper bound {upper_bound} for group {group_idx}. Resetting.")
            return (-np.inf, np.inf)
        return (lower_bound, upper_bound)
    
    def _validate_ci_arguments(self, option: str, stdz: Union[str, list], alternative: str) -> None:
        """Validate arguments for confidence interval calculations.

        Parameters
        ----------
        option : {'gamma', 'SM'}
            Whether to calculate intervals for provider effects or standardized measures.
        stdz : str or list
            Standardization method(s).
        alternative : str
            Hypothesis type.

        Raises
        ------
        ValueError 
            If arguments are inconsistent.
        """
        if option not in {"gamma", "SM"}:
            raise ValueError("Argument 'option' must be one of {'gamma','SM'}.")

        if isinstance(stdz, str):
            stdz = [stdz]

        if option == "gamma" and alternative not in {"two_sided", "greater", "less"}:
            raise ValueError("option='gamma' requires 'alternative' in {'two_sided','greater','less'}.")

        if option == "SM" and not any(m in stdz for m in ["indirect", "direct"]):
            raise ValueError("If option='SM', 'stdz' must include at least one of {'indirect','direct'}.")

    def _compute_gamma_intervals(
        self,
        group_ids: np.ndarray,
        level: float,
        alternative: str,
        test_method: str
    ) -> pd.DataFrame:
        """Compute intervals for all providers' gamma (option='gamma'), 
        delegating 'wald' to _compute_ci_bounds, 
        and 'score'/'exact' to specialized logic.

        Returns
        -------
        pd.DataFrame with columns ["provider_id","gamma","gamma_lower","gamma_upper"].
        """
        gamma_vals = self.coefficients_["gamma"].flatten()
        se_gamma = np.sqrt(self.variances_["gamma"].flatten())
        n_obs = len(self.outcome_)
        p = len(self.coefficients_["beta"])
        m = len(gamma_vals)
        df = n_obs - (m + p)
        alpha = 1.0 - level

        records = []
        for i, gid in enumerate(self.provider_ids_):
            if gid not in group_ids:
                continue

            gamma_est = gamma_vals[i]
            if test_method == "wald":
                lower, upper = -np.inf, np.inf
                if alternative in {"two_sided", "less", "greater"}:
                    # Reuse _compute_ci_bounds for just *one* provider if we wish
                    g_lower, g_upper = self._compute_ci_bounds(
                        gamma=np.array([gamma_est]),
                        se=np.array([se_gamma[i]]),
                        df=df,
                        level=level,
                        alternative=alternative
                    )
                    lower, upper = g_lower[0], g_upper[0]
                else:
                    raise ValueError("alternative must be one of {'two_sided','less','greater'}")

            elif test_method == "score":
                lower, upper = self._score_ci_for_one_group(
                    group_idx=i, alpha=alpha, alternative=alternative, gamma_guess=gamma_est
                )
            elif test_method == "exact":
                lower, upper = self._exact_ci_for_one_group(
                    group_idx=i, alpha=alpha, alternative=alternative, gamma_guess=gamma_est
                )
            else:
                raise ValueError("test_method must be wald, score, exact")

            records.append({
                "provider_id": gid,
                "gamma": gamma_est,
                "gamma_lower": lower,
                "gamma_upper": upper
            })

        df_res = pd.DataFrame(records)
        return df_res

    def _compute_sm_intervals(
        self,
        group_ids: np.ndarray,
        level: float,
        stdz: Union[str, list],
        measure: Union[str, list],
        alternative: str,
        test_method: str,
        reference: Union[str,float]
    ) -> dict:
        """Factor out the logic for "option='SM'" to a dedicated helper,
        returning intervals for indirect/direct ratio/rate. 
        Based on your prior code or the approach from your R function.
        """
        alpha = 1.0 - level
        gamma_vals = self.coefficients_["gamma"].flatten()
        n_obs = len(self.outcome_)

        # force measure to list
        if isinstance(measure, str):
            measure = [measure]
        want_ratio = "ratio" in measure
        want_rate  = "rate"  in measure

        # get the raw indirect/direct values
        sm_data = self.calculate_standardized_measures(
            providers=None, stdz=stdz, reference=reference
        )

        # build gamma CI maps
        df_gamma_ci = self._compute_gamma_intervals(
            group_ids=self.provider_ids_,
            level=level,
            alternative=alternative,
            test_method=test_method
        )
        gamma_lower_map = df_gamma_ci.set_index("provider_id")["gamma_lower"].to_dict()
        gamma_upper_map = df_gamma_ci.set_index("provider_id")["gamma_upper"].to_dict()

        population_rate = np.mean(self.outcome_) * 100.0
        results = {}

        # ---- INDIRECT ----
        if "indirect" in sm_data:
            base = sm_data["indirect"].copy().set_index("provider_id")

            # always compute ratio CIs if either ratio OR rate was requested
            if want_ratio or want_rate:
                expected_arr = base["expected"]
                rl, ru = [], []
                for gid in base.index:
                    if gid not in group_ids:
                        rl.append(np.nan); ru.append(np.nan); continue
                    gl = gamma_lower_map[gid]; gu = gamma_upper_map[gid]
                    expv = expected_arr[gid]
                    low_sum = self._sum_logistic_for_provider(gid, gl)
                    up_sum  = self._sum_logistic_for_provider(gid, gu)
                    rl.append(low_sum/expv if expv>0 else np.nan)
                    ru.append(up_sum /expv if expv>0 else np.nan)
                base["ci_ratio_lower"] = rl
                base["ci_ratio_upper"] = ru

            # only export the ratio table if they asked for it
            if want_ratio:
                df_ratio = base.copy()
                df_ratio.attrs.update({
                    "confidence_level": f"{level*100}%",
                    "description": "Indirect Standardized Ratio",
                    "model": type(self).__name__
                })
                results["indirect_ratio"] = df_ratio.copy()

            # now compute rate, reusing ratio bounds
            if want_rate:
                # make a fresh copy so we don't clobber df_ratio
                df_rate = base.copy()
                # ci_rate_lower = ci_ratio_lower * pop_rate
                df_rate["ci_rate_lower"] = np.clip(
                    df_rate["ci_ratio_lower"] * population_rate, 0, 100
                )
                df_rate["ci_rate_upper"] = np.clip(
                    df_rate["ci_ratio_upper"] * population_rate, 0, 100
                )
                df_rate.attrs.update({
                    "confidence_level": f"{level*100}%",
                    "description": "Indirect Standardized Rate",
                    "model": type(self).__name__,
                    "population_rate": population_rate
                })
                results["indirect_rate"] = df_rate.copy()

        # ---- DIRECT ----
        if "direct" in sm_data:
            base = sm_data["direct"].copy().set_index("provider_id")

            if want_ratio or want_rate:
                observed_arr = base["observed"]
                rl, ru = [], []
                for gid in base.index:
                    if gid not in group_ids:
                        rl.append(np.nan); ru.append(np.nan); continue
                    gl = gamma_lower_map[gid]; gu = gamma_upper_map[gid]
                    obs  = observed_arr[gid]
                    low_sum = self._sum_logistic_overall(gl)
                    up_sum  = self._sum_logistic_overall(gu)
                    rl.append(low_sum/obs if obs>0 else np.nan)
                    ru.append(up_sum /obs if obs>0 else np.nan)
                base["ci_ratio_lower"] = rl
                base["ci_ratio_upper"] = ru

            if want_ratio:
                df_ratio = base.copy()
                df_ratio.attrs.update({
                    "confidence_level": f"{level*100}%",
                    "description": "Direct Standardized Ratio",
                    "model": type(self).__name__
                })
                results["direct_ratio"] = df_ratio.copy()

            if want_rate:
                df_rate = base.copy()
                df_rate["ci_rate_lower"] = np.clip(
                    df_rate["ci_ratio_lower"] * population_rate, 0, 100
                )
                df_rate["ci_rate_upper"] = np.clip(
                    df_rate["ci_ratio_upper"] * population_rate, 0, 100
                )
                df_rate.attrs.update({
                    "confidence_level": f"{level*100}%",
                    "description": "Direct Standardized Rate",
                    "model": type(self).__name__,
                    "population_rate": population_rate
                })
                results["direct_rate"] = df_rate.copy()

        # finally, turn provider_id back into a column
        for k, df in results.items():
            if df.index.name == "provider_id":
                results[k] = df.reset_index()

        return results
    
    def _sum_logistic_for_provider(self, gid, gamma_val):
        """Sum logistic(gamma_val + xbeta_) for the observations belonging to provider gid."""
        idx = np.where(self.provider_ids_ == gid)[0]
        if len(idx) == 0: return np.nan
        group_idx = idx[0]
        mask = (self.provider_indices_ == group_idx)
        xbeta_group = self.xbeta_[mask]
        pvals = 1.0/(1.0 + np.exp(-(gamma_val + xbeta_group)))
        return pvals.sum()

    def _sum_logistic_overall(self, gamma_val):
        """Sum logistic(gamma_val + xbeta_) over the entire dataset (for direct approach)."""
        pvals = 1.0/(1.0 + np.exp(-(gamma_val + self.xbeta_)))
        return pvals.sum()

    # -------------------------------------------------------------------------
    # PUBLIC METHOD TO COMPUTE CONFIDENCE INTERVALS
    # -------------------------------------------------------------------------
    def calculate_confidence_intervals(
        self,
        providers: Optional[Union[list, np.ndarray]] = None,
        level: float = 0.95,
        option: str = "SM",
        stdz: Union[str, list] = "indirect",
        reference: Union[str, float] = "median",
        measure: Union[str, list] = ("rate", "ratio"),
        alternative: str = "two_sided",
        test_method: str = "exact"
    ) -> dict:
        """Compute confidence intervals for either provider effects (option="gamma") or
        standardized measures (option="SM") in a logistic fixed-effects model.

        When option="gamma", the method calculates two-sided confidence intervals
        for each provider's fixed effect (γ) using a specified test method 
        ("wald", "score", or "exact"). The user must supply alternative="two_sided" 
        if option="gamma", and the method ignores any stdz or measure arguments.

        When option="SM", the method produces confidence intervals for indirect or direct
        standardized measures (ratio and/or rate) at the specified confidence level.
        It calculates provider-specific γ intervals first, then transforms lower/upper
        γ bounds into standardized measure intervals. The user must supply stdz ∈ 
        {"indirect","direct"} and measure ∈ {"ratio","rate"}.

        Parameters
        ----------
        providers : list or np.ndarray, optional
            Subset of providers to include. If None, all providers are included.
        level : float, default=0.95
            Confidence level → alpha = 1 - level significance threshold.
        option : {'gamma','SM'}, default='SM'
            Type of intervals to compute:
            - 'gamma': intervals for each provider effect (γ).
            - 'SM': intervals for standardized measures (indirect/direct ratio/rate).
        stdz : {'indirect','direct'} or list, default='indirect'
            Standardization method(s) to use if option="SM". Ignored if option="gamma".
        reference : {'median','mean'} or float, default='median'
            Baseline norm for direct standardization. Ignored if option="gamma".
        measure : {'rate','ratio'} or list of these, default=('rate','ratio')
            Which standardized measures to produce intervals for if option="SM".
            Ignored if option="gamma".
        alternative : {'two_sided','greater','less'}, default='two_sided'
            Hypothesis direction. Must be "two_sided" if option="gamma".
            If option="SM", relevant only for how γ intervals are computed in "score" or "exact".
        test_method : {'wald','score','exact'}, default='wald'
            Method for computing γ intervals. 
            - 'wald' uses a t-based approach (via self._compute_ci_bounds).
            - 'score' uses partial derivative root-finding. 
            - 'exact' uses Poisson-binomial root-finding logic.

        Returns
        -------
        dict
            If option="gamma": 
                { "gamma_ci": DataFrame with columns [provider_id, gamma, gamma_lower, gamma_upper] }
            If option="SM": 
                Possibly includes keys:
                - "indirect_ratio"
                - "indirect_rate"
                - "direct_ratio"
                - "direct_rate"
            depending on stdz and measure selected.

        Notes
        -----
        - By default, if option="gamma", the method enforces alternative="two_sided".
        - For option="SM", standardization intervals are derived from summing logistic(γ_lower + Xβ)
        or logistic(γ_upper + Xβ), dividing by "expected," and (for rates) multiplying by the 
        population event rate. 
        - DataFrame .attrs usage:
        Storing metadata (e.g., "confidence_level", "description") in DataFrame.attrs is a
        convenient way to keep additional information attached without adding columns. This
        is optional. Some developers prefer returning a separate dictionary for metadata.
        """

        if self.coefficients_ is None or self.variances_ is None:
            raise ValueError("Model must be fitted, with valid coefficients_ and variances_.")

        # Enforce that if option="gamma", user must supply two_sided, 
        # and ignore stdz, measure, null
        if option == "gamma":
            # Force the intervals to be two-sided
            if alternative != "two_sided":
                raise ValueError(
                    "For option='gamma', only two-sided intervals are supported."
                )
            # If the user provided stz or measure, that doesn't make sense here
            if stdz is not None and (isinstance(stdz, str) or isinstance(stdz, list)):
                # It's simpler just to ignore them or raise an error:
                # We'll raise an error so user doesn't get confused:
                if (isinstance(stdz, list) and len(stdz) > 0) or (isinstance(stdz, str) and stdz != "indirect"):
                    raise ValueError("For option='gamma', stdz is not applicable.")
            if measure is not None and (
                (isinstance(measure, list) and len(measure) > 0) or
                (isinstance(measure, str))
            ):
                raise ValueError("For option='gamma', 'measure' is not applicable.")
            
            # Subset or gather final group IDs
            final_groups = self.provider_ids_ if providers is None else [g for g in self.provider_ids_ if g in providers]
            # Compute gamma intervals
            gamma_ci_df = self._compute_gamma_intervals(
                group_ids=np.array(final_groups),
                level=level,
                alternative=alternative,
                test_method=test_method
            )
            return {"gamma_ci": gamma_ci_df}

        elif option == "SM":
            # For standardized measures, we rely on a helper approach
            # that handles indirect/direct ratio/rate intervals:
            final_groups = self.provider_ids_ if providers is None else [g for g in self.provider_ids_ if g in providers]
            return self._compute_sm_intervals(
                group_ids=np.array(final_groups),
                level=level,
                stdz=stdz,
                measure=measure,
                alternative=alternative,
                test_method=test_method,
                reference=reference
            )

        else:
            raise ValueError("option must be either 'gamma' or 'SM'.")
