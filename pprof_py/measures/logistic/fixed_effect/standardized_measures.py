"""Standardized rate/ratio computation for logistic fixed-effect models.

Contains ``_StandardizedMeasureMethods``, a mixin fragment providing
:meth:`calculate_standardized_measures`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ....utils.numerical import sigmoid


class _StandardizedMeasureMethods:
    """Mixin fragment: standardized-measure computation."""

    def calculate_standardized_measures(
        self,
        providers=None,
        stdz="indirect",
        reference="median",
        include_extreme_obs: bool = False,
        extreme_obs_total_n: Optional[float] = None,
    ) -> dict:
        """Calculate direct/indirect standardized ratios and rates for a fixed-effects logistic model.

        Supports both Bernoulli (N=1) and binomial (N>1) models. When the model
        was fitted with n_var (binomial N), computations are automatically
        N-weighted to produce correct standardized rates.

        Parameters
        ----------
        providers : list or np.ndarray, optional
            Subset of provider/group IDs for whom to calculate measures. If None,
            all providers are included.
        stdz : str or list of str, default="indirect"
            Standardization method(s). Must include at least one of {"indirect", "direct"}.
        reference : {'median', 'mean'} or float, default="median"
            Defines the population norm if "direct" standardization is requested.
            - 'median': uses median(gamma)
            - 'mean': uses weighted mean(gamma, weights=provider_sizes_)
            - float: uses a user-specified numeric reference level.
        include_extreme_obs : bool, default=False
            If True, include extreme provider observations (with xbeta=0) in the
            population sum for direct standardization. This is required for AOH
            and recommended for PPPW/SFR when extreme patients should be in the
            denominator. Requires extreme_obs_total_n to be specified.
        extreme_obs_total_n : float, optional
            Total binomial N (sum of N_i) across all extreme provider observations.
            For Bernoulli models, this equals the number of extreme observations.
            Only used when include_extreme_obs=True. Extreme obs are assumed to
            have xbeta=0 (R convention).

        Returns
        -------
        dict
            A dictionary with keys for each requested standardization type:
            - 'indirect' -> DataFrame with columns ["provider_id", "indirect_ratio",
            "indirect_rate", "observed", "expected"].
            - 'direct' -> DataFrame with columns ["provider_id", "direct_ratio",
            "direct_rate", "observed", "expected", "n_pop"].

        Raises
        ------
        ValueError
            If the model is not deemed fitted or if arguments are invalid.

        Notes
        -----
        When self.N_ is not all ones (binomial model):
        - Direct rate: sum(N_i * sigmoid(gamma_k + xbeta_i)) / Ntot
        - Indirect expected: sum(N_i * sigmoid(gamma_null + xbeta_i)) per group
        - This matches R's AOH_piehat / SR_output for binomial models.

        When include_extreme_obs=True:
        - Extreme obs contribute sigmoid(gamma_k) * extreme_obs_total_n to direct preds
        - Ntot is increased by extreme_obs_total_n
        - This matches R's convention where extreme patients have xbeta=0
        """
        if self.coefficients_ is None or self.fitted_ is None or self.outcome_ is None:
            raise ValueError("The model must be fitted with stored outcomes, coefficients, and fitted probabilities.")

        if isinstance(stdz, str):
            stdz = [stdz]
        if not any(m in stdz for m in ["indirect", "direct"]):
            raise ValueError("Argument 'stdz' must include 'indirect' and/or 'direct'.")

        gamma = self.coefficients_["gamma"].flatten()
        n_samples = len(self.outcome_)

        # Binomial N per observation (defaults to 1 for Bernoulli)
        N_obs = self.N_ if self.N_ is not None else np.ones(n_samples)
        Ntot_model = np.sum(N_obs)  # total N from model observations

        # Total denominator (includes extreme obs if requested)
        extreme_n = float(extreme_obs_total_n) if (include_extreme_obs and extreme_obs_total_n) else 0.0
        Ntot = Ntot_model + extreme_n

        # Determine gamma_null if needed
        if reference == "median":
            gamma_null = np.median(gamma)
        elif reference == "mean":
            gamma_null = np.average(gamma, weights=self.provider_sizes_)
        elif isinstance(reference, (int, float)):
            gamma_null = float(reference)
        else:
            raise ValueError("Invalid 'null' argument for standardization baseline.")

        selected_groups = (
            self.provider_ids_ if providers is None
            else self.provider_ids_[np.isin(self.provider_ids_, providers)]
        )
        results = {}

        # Indirect standardization (N-weighted for binomial)
        if "indirect" in stdz:
            expected_prob = 1.0 / (1.0 + np.exp(-(gamma_null + self.xbeta_)))
            # N-weighted expected: sum(N_i * p_null_i) per group
            expected_by_group = np.array([
                np.sum(N_obs[self.provider_indices_ == i] * expected_prob[self.provider_indices_ == i])
                for i in range(len(self.provider_ids_))
            ])
            observed_by_group = np.array([
                np.sum(self.outcome_[self.provider_indices_ == i])
                for i in range(len(self.provider_ids_))
            ])
            indirect_ratio = observed_by_group / expected_by_group
            population_rate = observed_by_group.sum() / Ntot_model * 100.0
            indirect_rate = np.clip(indirect_ratio * population_rate, 0.0, 100.0)

            indirect_df = pd.DataFrame({
                "provider_id": self.provider_ids_,
                "indirect_ratio": indirect_ratio,
                "indirect_rate": indirect_rate,
                "observed": observed_by_group,
                "expected": expected_by_group
            })

            if providers is not None:
                indirect_df = indirect_df[indirect_df["provider_id"].isin(selected_groups)].reset_index(drop=True)
            results["indirect"] = indirect_df

        # Direct standardization (N-weighted for binomial, with optional extreme obs)
        if "direct" in stdz:
            obs_total = np.sum(self.outcome_)
            population_rate = obs_total / Ntot * 100.0

            direct_preds = np.empty(len(gamma))
            for j, g_val in enumerate(gamma):
                # Model obs contribution: sum(N_i * sigmoid(gamma_k + xbeta_i))
                p_temp = 1.0 / (1.0 + np.exp(-(g_val + self.xbeta_)))
                pred_model = np.sum(N_obs * p_temp)

                # Extreme obs contribution: sigmoid(gamma_k) * sum(N_extreme)
                # (all extreme obs have xbeta=0, so sigmoid only depends on gamma_k)
                if extreme_n > 0:
                    p_extreme = 1.0 / (1.0 + np.exp(-g_val))
                    pred_extreme = extreme_n * p_extreme
                else:
                    pred_extreme = 0.0

                direct_preds[j] = pred_model + pred_extreme

            ds_ratio = direct_preds / obs_total
            ds_rate = np.clip(direct_preds / Ntot * 100.0, 0.0, 100.0)

            direct_df = pd.DataFrame({
                "provider_id": self.provider_ids_,
                "direct_ratio": ds_ratio,
                "direct_rate": ds_rate,
                "observed": np.full(len(self.provider_ids_), obs_total),
                "expected": direct_preds,
                "n_pop": Ntot,
            })

            if providers is not None:
                direct_df = direct_df[direct_df["provider_id"].isin(selected_groups)].reset_index(drop=True)
            results["direct"] = direct_df

        return results
