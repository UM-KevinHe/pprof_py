"""Covariate-level statistical inference for `LinearRandomEffectModel`,
mixed into the model class so that `models/linear/random_effect.py` can
stay focused on configuration, fitting, and prediction (see AGENTS.md
Sections 9 and 15).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Union

import numpy as np
import pandas as pd
from scipy.stats import t


class _LinearREInferenceHost(Protocol):
    """Attribute contract that `RandomEffectInferenceMixin` expects from its
    host class (`LinearRandomEffectModel`).
    """
    coefficients_: Optional[Dict[str, Any]]
    variances_: Optional[Dict[str, Any]]
    fitted_: Optional[np.ndarray]

    def _check_is_fitted(self) -> None: ...


class RandomEffectInferenceMixin:
    """Fixed-effect (covariate) statistical inference for
    `LinearRandomEffectModel`."""

    def summary(
        self, covariates: Optional[Union[list, np.ndarray]] = None, 
        level: float = 0.95,
        null: float = 0, 
        alternative: str = "two_sided"
    ) -> pd.DataFrame:
        """Provide summary statistics for the fixed effects in the model.

        Parameters:
        ----------
        - covariates: Optional[Union[list, np.ndarray]]
            Subset of covariates for which summary statistics are provided. Defaults to all.
        - level: float, default=0.95
            Confidence level for the intervals.
        - null: float, default=0
            Null hypothesis value for the parameter estimates.
        - alternative: str, default="two_sided"
            The alternative hypothesis ("two_sided", "greater", or "less").

        Returns:
        -------
        - summary_df: pd.DataFrame
            A DataFrame with columns:
              - estimate
              - std_error
              - stat
              - p_value
              - ci_lower
              - ci_upper
        """
        self._check_is_fitted()

        fe_estimates = self.coefficients_["beta"]
        se_fe = np.sqrt(np.diag(self.variances_["beta"]))
        stat = (fe_estimates - null) / se_fe

        n = len(self.fitted_)
        p = len(fe_estimates)
        m = len(self.coefficients_["alpha"])
        df = n - p - m  # consistent with fixed_effect model
        
        if alternative == "two_sided":
            p_value = 2 * (1 - t.cdf(np.abs(stat), df=df))
            crit_value = t.ppf(1 - (1 - level) / 2, df=df)
            ci_lower = fe_estimates - crit_value * se_fe
            ci_upper = fe_estimates + crit_value * se_fe
        elif alternative == "greater":
            p_value = 1 - t.cdf(stat, df=df)
            crit_value = t.ppf(1 - (1 - level), df=df)
            ci_lower = fe_estimates - crit_value * se_fe
            ci_upper = np.inf
        elif alternative == "less":
            p_value = t.cdf(stat, df=df)
            crit_value = t.ppf(1 - (1 - level), df=df)
            ci_lower = -np.inf
            ci_upper = fe_estimates + crit_value * se_fe
        else:
            raise ValueError("argument 'alternative' should be 'two_sided', 'greater', or 'less'.")
        
        p_value = np.round(p_value, 7)
        summary_df = pd.DataFrame({
            "estimate": fe_estimates,
            "std_error": se_fe,
            "stat": stat,
            "p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        }, index=fe_estimates.index)
        
        if covariates is not None:
            if isinstance(covariates, (list, np.ndarray)):
                summary_df = summary_df.loc[covariates]
            else:
                raise ValueError("argument 'covariates' should be a list or array of covariate names or indices.")
        return summary_df
