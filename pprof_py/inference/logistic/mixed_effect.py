"""Covariate-effect inference for `LogisticMixedEffectModel`, mixed into
the model class so that `models/logistic/mixed_effect.py` can stay
focused on configuration, fitting, and prediction (see AGENTS.md
Sections 9 and 15).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit as plogis
from scipy.stats import norm


class MixedEffectInferenceMixin:
    """Covariate (beta) statistical inference for
    `LogisticMixedEffectModel`."""

    def summary(self, alpha: float = 0.05) -> pd.DataFrame:
        """Covariate effect inference (asymptotic normal approximation).

        Uses the adjusted Fisher information that accounts for
        posterior variance of cluster random effects.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for confidence intervals.

        Returns
        -------
        pd.DataFrame with columns: covariate, beta, se, z_stat,
            p_value, ci_lower, ci_upper, odds_ratio, odds_ratio_lower,
            odds_ratio_upper
        """
        self._check_is_fitted()

        X = self._X
        gamma_obs = self.gamma_[self._provider_idx]
        linear_pred = gamma_obs + self.alpha_mean_ + self.xbeta_
        p = plogis(linear_pred)
        q = 1 - p
        pq = p * q

        # Adjusted weights (He et al. 2013)
        w = pq + 0.5 * self.alpha_var_ * pq * (p**2 + q**2 - 4 * pq)

        # Information matrix
        info_beta = X.T @ (w[:, None] * X)
        var_beta = np.linalg.inv(info_beta)
        se_beta = np.sqrt(np.diag(var_beta))

        # Statistics
        z_stat = self.beta_ / se_beta
        p_values = 2 * norm.sf(np.abs(z_stat))

        z_crit = norm.ppf(1 - alpha / 2)
        ci_lower = self.beta_ - z_crit * se_beta
        ci_upper = self.beta_ + z_crit * se_beta

        return pd.DataFrame({
            'covariate': self._x_vars,
            'beta': self.beta_,
            'se': se_beta,
            'z_stat': z_stat,
            'p_value': p_values,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'odds_ratio': np.exp(self.beta_),
            'odds_ratio_lower': np.exp(ci_lower),
            'odds_ratio_upper': np.exp(ci_upper),
        })
