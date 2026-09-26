"""Standardized measures for `LinearRandomEffectModel`: direct and indirect
standardized differences. Provider-effect tests and confidence intervals
are in `pprof_py.inference.linear.random_effect`. Mixed into the model class so that
`models/linear/random_effect.py` can stay focused on configuration,
fitting, and prediction.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Union

import numpy as np
import pandas as pd


class _LinearREMeasuresHost(Protocol):
    """Attribute contract that `LinearRandomEffectMeasuresMixin` expects from its
    host class (`LinearRandomEffectModel`).
    """
    def _require_one_factor(self) -> None: ...
    coefficients_: Optional[Dict[str, Any]]
    variances_: Optional[Dict[str, Any]]
    fitted_: Optional[np.ndarray]
    provider_ids_: Optional[np.ndarray]
    provider_indices_: Optional[np.ndarray]
    provider_sizes_: Optional[np.ndarray]
    sigma_: Optional[float]
    xbeta_: Optional[np.ndarray]

    def _check_is_fitted(self) -> None: ...
    def calculate_standardized_measures(self, **kwargs: Any) -> dict: ...


class LinearRandomEffectMeasuresMixin:
    """Standardized measures, provider-level confidence intervals, and
    provider-effect hypothesis tests for `LinearRandomEffectModel`."""

    def calculate_standardized_measures(
        self, providers: Optional[Union[list, np.ndarray]] = None,
        stdz: Union[str, list] = "indirect",
        reference: str = "median"
    ) -> dict:
        """Calculate direct/indirect standardized differences for the random effect model.

        Parameters
        ----------
        providers : Optional[Union[list, np.ndarray]]
            Specifies a subset of providers (groups) for which the measures are calculated.
            Defaults to all providers.
        stdz : Union[str, list], default="indirect"
            Standardization method(s); can be "indirect", "direct", or both.
        reference : Union[str, float], default="median"
            Baseline norm used for standardization; can be "median", "mean", or a specific numeric value.

        Returns
        -------
        dict
            A dictionary containing DataFrames of standardized differences and observed/expected outcomes
            grouped by method. The keys will be "indirect" and/or "direct" based on the selected methods.
        """
        self._require_one_factor()

        self._check_is_fitted()
        if isinstance(stdz, str):
            stdz = [stdz]

        if not any(method in stdz for method in ["indirect", "direct"]):
            raise ValueError("Argument 'stdz' must include 'indirect' and/or 'direct'.")

        # Extract model components
        random_effects = self.coefficients_["alpha"]
        group_names = self.provider_ids_
        group_indices = self.provider_indices_
        group_sizes = self.provider_sizes_
        total_samples = len(self.fitted_)

        # Determine the null value for random effects
        if reference == "median":
            re_null = np.median(random_effects)
        elif reference == "mean":
            re_null = np.average(random_effects, weights=group_sizes)
        elif isinstance(reference, (int, float)):
            re_null = reference
        else:
            raise ValueError("Invalid 'null' argument provided. Must be 'median', 'mean', or a numeric value.")

        # If providers are specified, select those groups; otherwise, use all groups
        if providers is not None:
            mask = np.isin(group_names, providers)
            selected_groups = group_names[mask]
        else:
            selected_groups = group_names

        results = {}

        # Indirect Standardization
        if "indirect" in stdz:
            n_groups = len(group_names)
            # Compute expected outcomes by group by excluding random effects
            expected_by_group = np.bincount(group_indices, weights=self.xbeta_, minlength=n_groups)
            # Compute observed outcomes using the full fitted values including random effects
            observed_by_group = np.bincount(group_indices, weights=self.fitted_, minlength=n_groups)
            # Indirect standardized difference
            indirect_diff = (observed_by_group - expected_by_group) / group_sizes

            indirect_df = pd.DataFrame({
                "provider_id": group_names,
                "indirect_difference": indirect_diff,
                "observed": observed_by_group,
                "expected": expected_by_group
            })

            if providers is not None:
                indirect_df = indirect_df[indirect_df['provider_id'].isin(selected_groups)].reset_index(drop=True)

            results["indirect"] = indirect_df

        # Direct Standardization
        if "direct" in stdz:
            # Overall observed is the total sum of the fitted values
            total_observed = self.fitted_.sum()
            # Calculate direct expected outcomes using group-specific random effects:
            # sum(xbeta + re_val) == xbeta.sum() + re_val * n_samples.
            xbeta_sum = self.xbeta_.sum()
            expected_direct_by_group = xbeta_sum + random_effects * total_samples
            # Direct standardized difference
            direct_diff = (expected_direct_by_group - total_observed) / total_samples

            direct_df = pd.DataFrame({
                "provider_id": group_names,
                "direct_difference": direct_diff,
                "observed": np.full(len(group_names), total_observed),
                "expected": expected_direct_by_group
            })

            if providers is not None:
                direct_df = direct_df[direct_df['provider_id'].isin(selected_groups)].reset_index(drop=True)

            results["direct"] = direct_df

        return results
