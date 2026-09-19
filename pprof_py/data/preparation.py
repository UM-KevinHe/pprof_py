"""Data preparation for provider-profiling models.

Provides ``DataPrep``, the main entry point for converting a raw
DataFrame into the arrays and index structures that model classes
consume: provider sorting and renaming, covariate extraction,
train/test splitting, and diagnostic checks (missingness, variation,
collinearity).
"""
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple

from .validation import (
    check_missingness as _check_missingness,
    check_variation as _check_variation,
    check_correlation as _check_correlation,
    check_vif as _check_vif,
)


@dataclass
class DataPrepOptions:
    """Screening/threshold knobs for `DataPrep`.

    Parameters
    ----------
    cutoff : int, default=10
        Minimum number of records per provider.
    screen_providers : bool, default=False
        Whether to screen and filter small providers.
    log_event_providers : bool, default=False
        Whether to log event statistics.
    threshold_cor : float, default=0.9
        Threshold for pairwise correlation.
    threshold_vif : int, default=10
        Threshold for Variance Inflation Factor.
    binary_response : bool, default=False
        Whether to enforce a binary response (for logistic models).
    """
    cutoff: int = 10
    screen_providers: bool = False
    log_event_providers: bool = False
    threshold_cor: float = 0.9
    threshold_vif: int = 10
    binary_response: bool = False


class DataPrep:
    """Prepare and validate data for regression models.

    This class performs data checks, optionally screens providers based on record counts, 
    and logs event statistics. It ensures data suitability for modeling.

    Attributes:
    -----------
    data (pd.DataFrame): The input data, sorted by provider.
    Y_char (str): The name of the response variable.
    X_char (List[str]): The names of the covariate variables.
    prov_char (str): The name of the provider variable.
    check (bool): Whether to perform data checks.
    options (DataPrepOptions): Screening/threshold knobs (cutoff, screen_providers,
        log_event_providers, threshold_cor, threshold_vif, binary_response).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        Y_char: str,
        X_char: List[str],
        prov_char: str,
        options: DataPrepOptions = None,
        check: bool = True,
        logging=logging
    ) -> None:
        """Initialize the DataPrep object.

        Parameters:
            data (pd.DataFrame): The input data.
            Y_char (str): The response variable name.
            X_char (List[str]): The covariate variable names.
            prov_char (str): The provider variable name.
            options (DataPrepOptions, optional): Screening/threshold knobs. Defaults
                to `DataPrepOptions()` if not provided.
            check (bool, optional): Whether to perform data checks (default is True).
            logging: The logging module to use.

        Raises:
            AssertionError: If specified columns are not in the data.
            ValueError: If data dimensions do not match or response is not binary (if enforced).
        """
        if options is None:
            options = DataPrepOptions()

        # Input validation
        assert Y_char in data.columns, f"Response variable '{Y_char}' NOT found!"
        missing_X_char = [char for char in X_char if char not in data.columns]
        assert not missing_X_char, f"Covariate(s) '{', '.join(missing_X_char)}' NOT found!"
        assert prov_char in data.columns, f"Provider '{prov_char}' NOT found!"

        # Check dimensions
        if len(data[Y_char]) != len(data[prov_char]) or any(len(data[prov_char]) != len(data[x]) for x in X_char):
            raise ValueError("Dimensions of the input data do not match!")

        # Optional: Enforce binary response if binary_response=True
        if options.binary_response and not set(data[Y_char].unique()).issubset({0, 1}):
            raise ValueError("Response variable must be binary (0 or 1) when binary_response=True.")

        self.data = data.sort_values(by=prov_char)
        self.Y_char = Y_char
        self.X_char = X_char
        self.prov_char = prov_char
        self.cutoff = options.cutoff
        self.check = check
        self.screen_providers = options.screen_providers
        self.log_event_providers = options.log_event_providers
        self.threshold_cor = options.threshold_cor
        self.threshold_vif = options.threshold_vif
        self.logging = logging

    def check_missingness(self) -> None:
        """Check for missing values in the data.

        Delegates to :func:`~pprof_py.data.validation.check_missingness`.

        Raises:
            ValueError: If missing values are found in any of the specified columns.
        """
        columns = self.X_char + [self.Y_char, self.prov_char]
        _check_missingness(self.data, columns)

    def check_variation(self) -> None:
        """Check for zero variance in the covariates.

        Delegates to :func:`~pprof_py.data.validation.check_variation`.

        Raises:
            ValueError: If any covariate has zero variance.
        """
        _check_variation(self.data, self.X_char)

    def check_correlation(self) -> None:
        """Check for pairwise correlation among covariates.

        Delegates to :func:`~pprof_py.data.validation.check_correlation`.
        Logs a warning if any pair exceeds ``self.threshold_cor``.
        """
        _check_correlation(self.data, self.X_char, threshold=self.threshold_cor)

    def check_vif(self) -> None:
        """Check the Variance Inflation Factor (VIF) of the covariates.

        Delegates to :func:`~pprof_py.data.validation.check_vif`.
        Logs a warning if any covariate has a VIF above ``self.threshold_vif``.
        """
        _check_vif(self.data, self.X_char, threshold=self.threshold_vif)

    def provider_screening(self) -> None:
        """Screen providers based on the number of records.

        Adds 'prov_size' and 'included' columns to the data.
        """
        self.data["prov_size"] = self.data.groupby(self.prov_char)[self.prov_char].transform("count")
        self.data["included"] = np.where(self.data["prov_size"] >= self.cutoff, 1, 0)

    def filter_small_providers(self) -> None:
        """Filter out providers with fewer than 'cutoff' records.

        Drops 'prov_size' and 'included' columns after filtering.
        """
        total_prov = self.data[self.prov_char].nunique()
        n_prov_small = self.data[self.data["included"] == 0][self.prov_char].nunique()
        if n_prov_small > 0:
            self.logging.warning(f"{n_prov_small} out of {total_prov} providers are small and will be filtered out.")
        self.data = self.data[self.data["included"] == 1].drop(columns=["included", "prov_size"])

    def log_no_all_event_providers(self) -> None:
        """Log statistics about providers with no or all events.

        Does not filter the data but provides information for further analysis.
        """
        self.n_prov = self.data[self.prov_char].nunique()
        prov_no_event = self.data.groupby(self.prov_char).filter(lambda x: x[self.Y_char].sum() == 0)[self.prov_char].unique()
        prov_all_event = self.data.groupby(self.prov_char).filter(lambda x: x[self.Y_char].sum() == len(x))[self.prov_char].unique()
        self.logging.info(f"{len(prov_no_event)} out of {self.n_prov} providers have no events.")
        self.logging.info(f"{len(prov_all_event)} out of {self.n_prov} providers have all events.")
        event_rate = self.data[self.Y_char].mean() * 100
        self.logging.info(f"After screening, {round(event_rate, 2)}% of records have events (Y == 1).")

    def data_prep(self) -> pd.DataFrame:
        """Prepare the data for regression modeling.

        Performs checks (if enabled), optionally screens providers, and logs event statistics.

        Returns:
            pd.DataFrame: The prepared data.
        """
        if self.check:
            self.check_missingness()
            self.check_variation()
            self.check_correlation()
        if self.screen_providers:
            self.provider_screening()
            self.filter_small_providers()
        if self.log_event_providers:
            self.log_no_all_event_providers()
        return self.data