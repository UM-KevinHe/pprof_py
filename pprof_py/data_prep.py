import logging
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor


class DataPrep:
    """Prepare and validate data for regression models.

    This class performs data checks, optionally screens providers based on record counts, 
    and logs event statistics. It ensures data suitability for modeling.

    Attributes:
        data (pd.DataFrame): The input data, sorted by provider.
        Y_char (str): The name of the response variable.
        X_char (List[str]): The names of the covariate variables.
        prov_char (str): The name of the provider variable.
        cutoff (int): Minimum number of records per provider.
        check (bool): Whether to perform data checks.
        screen_providers (bool): Whether to screen and filter small providers.
        log_event_providers (bool): Whether to log event statistics.
        threshold_cor (float): Threshold for pairwise correlation.
        threshold_vif (int): Threshold for Variance Inflation Factor.
        binary_response (bool): Whether to enforce a binary response (for logistic models).
    """

    def __init__(
        self, 
        data: pd.DataFrame, 
        Y_char: str,
        X_char: List[str], 
        prov_char: str,
        cutoff: int = 10,
        check: bool = True, 
        screen_providers: bool = False,
        log_event_providers: bool = False,
        threshold_cor: float = 0.9, 
        threshold_vif: int = 10,
        binary_response: bool = False,  # Optional flag for binary response check
        logging=logging
    ) -> None:
        """Initialize the DataPrep object.

        Parameters:
            data (pd.DataFrame): The input data.
            Y_char (str): The response variable name.
            X_char (List[str]): The covariate variable names.
            prov_char (str): The provider variable name.
            cutoff (int, optional): Minimum records per provider (default is 10).
            check (bool, optional): Whether to perform data checks (default is True).
            screen_providers (bool, optional): Whether to filter small providers (default is False).
            log_event_providers (bool, optional): Whether to log event statistics (default is False).
            threshold_cor (float, optional): Correlation threshold (default is 0.9).
            threshold_vif (int, optional): VIF threshold (default is 10).
            binary_response (bool, optional): Whether to enforce binary response (default is False).
            logging: The logging module to use.

        Raises:
            AssertionError: If specified columns are not in the data.
            ValueError: If data dimensions do not match or response is not binary (if enforced).
        """
        # Input validation
        assert Y_char in data.columns, f"Response variable '{Y_char}' NOT found!"
        missing_X_char = [char for char in X_char if char not in data.columns]
        assert not missing_X_char, f"Covariate(s) '{', '.join(missing_X_char)}' NOT found!"
        assert prov_char in data.columns, f"Provider '{prov_char}' NOT found!"

        # Check dimensions
        if len(data[Y_char]) != len(data[prov_char]) or any(len(data[prov_char]) != len(data[x]) for x in X_char):
            raise ValueError("Dimensions of the input data do not match!")

        # Optional: Enforce binary response if binary_response=True
        if binary_response and not set(data[Y_char].unique()).issubset({0, 1}):
            raise ValueError("Response variable must be binary (0 or 1) when binary_response=True.")

        self.data = data.sort_values(by=prov_char)
        self.Y_char = Y_char
        self.X_char = X_char
        self.prov_char = prov_char
        self.cutoff = cutoff
        self.check = check
        self.screen_providers = screen_providers
        self.log_event_providers = log_event_providers
        self.threshold_cor = threshold_cor
        self.threshold_vif = threshold_vif
        self.logging = logging

    def check_missingness(self) -> None:
        """Check for missing values in the data.

        Raises:
            ValueError: If missing values are found in any of the specified columns.
        """
        self.logging.info("Checking missingness of variables ...")
        columns_to_check = self.X_char + [self.Y_char, self.prov_char]
        missing_data = self.data[columns_to_check].isnull().sum()
        if missing_data.any():
            self.logging.error(f"Missing values found in columns: {', '.join(missing_data.index[missing_data > 0])}")
            raise ValueError("Missing values found in the data.")
        self.logging.info("Missing values NOT found. Checking missingness completed!")

    def check_variation(self) -> None:
        """Check for zero variance in the covariates.

        Raises:
            ValueError: If any covariate has zero variance.
        """
        self.logging.info("Checking variation in covariates ...")
        selector = VarianceThreshold()
        selector.fit(self.data[self.X_char])
        zero_var_cols = [col for col, var in zip(self.X_char, selector.variances_) if var == 0]
        if zero_var_cols:
            self.logging.error(f"Covariates with zero variance: {', '.join(zero_var_cols)}")
            raise ValueError("Covariates with zero variance found.")
        self.logging.info("Checking variation in covariates completed!")

    def check_correlation(self) -> None:
        """Check for pairwise correlation among covariates.

        Logs a warning if any pair of covariates has a correlation above the threshold.
        """
        self.logging.info("Checking pairwise correlation among covariates ...")
        corr_matrix = self.data[self.X_char].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr = [(col1, col2) for col1 in upper_tri.columns for col2 in upper_tri.index
                     if upper_tri[col1][col2] > self.threshold_cor]
        if high_corr:
            self.logging.warning(f"Highly correlated covariates found: {high_corr}")
        else:
            self.logging.info("No highly correlated covariates found.")

    def check_vif(self) -> None:
        """Check the Variance Inflation Factor (VIF) of the covariates.

        Logs a warning if any covariate has a VIF above the threshold.
        """
        self.logging.info("Checking VIF of covariates ...")
        vif_data = pd.DataFrame()
        vif_data["variables"] = self.X_char
        vif_data["VIF"] = [variance_inflation_factor(self.data[self.X_char].values, i)
                          for i in range(len(self.X_char))]
        high_vif = vif_data[vif_data["VIF"] > self.threshold_vif]
        if not high_vif.empty:
            self.logging.warning(f"High VIF found in variables: {', '.join(high_vif['variables'])}")
        else:
            self.logging.info("No high VIF found.")

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
            # self.check_vif()
        if self.screen_providers:
            self.provider_screening()
            self.filter_small_providers()
        if self.log_event_providers:
            self.log_no_all_event_providers()
        return self.data