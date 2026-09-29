"""Standardized measures for `LinearFixedEffectModel`: direct and indirect
standardized differences. Provider-effect tests and confidence intervals
are in `pprof_py.inference.linear.fixed_effect`. Mixed into the model class so that
`models/linear/fixed_effect.py` can stay focused on configuration,
fitting, and prediction.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

import numpy as np
import pandas as pd


class _LinearFEMeasuresHost(Protocol):
    """Attribute contract that `LinearFixedEffectMeasuresMixin` expects from its
    host class (`LinearFixedEffectModel`).
    """
    coefficients_: Optional[Dict[str, Any]]
    variances_: Optional[Dict[str, Any]]
    fitted_: Optional[np.ndarray]
    provider_ids_: Optional[np.ndarray]
    provider_indices_: Optional[np.ndarray]
    provider_sizes_: Optional[np.ndarray]
    outcome_: Optional[np.ndarray]
    xbeta_: Optional[np.ndarray]

    def _check_is_fitted(self) -> None: ...
    def calculate_standardized_measures(self, **kwargs: Any) -> dict: ...


class LinearFixedEffectMeasuresMixin:
    """Standardized measures, provider-level confidence intervals, and
    provider-effect hypothesis tests for `LinearFixedEffectModel`."""

    def calculate_standardized_measures(
        self, providers=None, stdz="indirect", reference="median"
    ) -> dict:
        """Calculate direct/indirect standardized differences for a fixed effects linear model.

        Parameters
        ----------
        providers : Optional[list or np.ndarray]
            The specific groups or provider identifiers for which the differences should be calculated.
            If None, calculates for all groups.
        stdz : Union[str, list], default="indirect"
            Methods for standardization; can be "indirect", "direct", or both.
        reference : Union[str, float], default="median"
            Baseline norm used for standardization; can be "median", "mean", or a specific numeric value.

        Returns
        -------
        dict
            A dictionary containing DataFrames of standardized differences and observed/expected outcomes
            grouped by method. The keys will be "indirect" and/or "direct" based on the selected methods.
        """

        if self.coefficients_ is None or self.fitted_ is None:
            raise ValueError("The model must be fitted before calculating standardized differences.")

        if self.outcome_ is None:
            raise ValueError("Original outcomes were not stored during fitting.")

        if isinstance(stdz, str):
            stdz = [stdz]

        if not any(method in stdz for method in ["indirect", "direct"]):
            raise ValueError("Argument 'stdz' must include 'indirect' and/or 'direct'.")
        
        # Extract model components
        gamma = self.coefficients_["gamma"].flatten()  # shape (m,)
        n_samples = len(self.outcome_)
        group_sizes = self.provider_sizes_
        
        # Determine the null value for gamma
        if reference == "median":
            gamma_null = np.median(gamma)
        elif reference == "mean":
            gamma_null = np.average(gamma, weights=group_sizes)
        elif isinstance(reference, (int, float)):
            gamma_null = reference
        else:
            raise ValueError("Invalid 'null' argument provided. Must be 'median', 'mean', or a numeric value.")
        
        # If providers are specified, select those groups; otherwise, use all groups.
        if providers is not None:
            mask = np.isin(self.provider_ids_, providers)
            selected_groups = self.provider_ids_[mask]
        else:
            selected_groups = self.provider_ids_
        
        results = {}
        
        # Indirect Standardization
        if "indirect" in stdz:
            n_groups = len(self.provider_ids_)
            # For each observation, expected outcome = gamma_null + linear predictor.
            expected = gamma_null + self.xbeta_.flatten()
            # Sum expected and observed by group.
            expected_by_group = np.bincount(self.provider_indices_, weights=expected, minlength=n_groups)
            observed_by_group = np.bincount(self.provider_indices_, weights=self.outcome_, minlength=n_groups)
            # Standardized difference is the (Obs - Exp) divided by the group size.
            indirect_diff = (observed_by_group - expected_by_group) / group_sizes
            
            indirect_df = pd.DataFrame({
                "provider_id": self.provider_ids_,
                "indirect_difference": indirect_diff,
                "observed": observed_by_group,
                "expected": expected_by_group
            })
            if providers is not None:
                indirect_df = indirect_df[indirect_df['provider_id'].isin(selected_groups)].reset_index(drop=True)
            results["indirect"] = indirect_df
        
        # Direct Standardization
        if "direct" in stdz:
            # Overall observed is the sum over all observations of (gamma_null + xbeta).
            obs_direct_total = (gamma_null + self.xbeta_.flatten()).sum()
            # For each group, compute expected sum using the group-specific gamma:
            # sum(gamma_val + xbeta) == gamma_val * n_samples + xbeta.sum().
            xbeta_sum = self.xbeta_.sum()
            exp_direct_by_group = gamma * n_samples + xbeta_sum
            # Standardized difference is (expected_by_group - overall observed) divided by total sample size.
            direct_diff = (exp_direct_by_group - obs_direct_total) / n_samples
            
            direct_df = pd.DataFrame({
                "provider_id": self.provider_ids_,
                "direct_difference": direct_diff,
                "observed": np.full(len(self.provider_ids_), obs_direct_total),
                "expected": exp_direct_by_group
            })
            if providers is not None:
                direct_df = direct_df[direct_df['provider_id'].isin(selected_groups)].reset_index(drop=True)
            results["direct"] = direct_df
            
        return results
