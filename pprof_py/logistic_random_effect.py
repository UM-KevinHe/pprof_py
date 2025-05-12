import pandas as pd
import numpy as np
from scipy.stats import norm
from typing import Optional, Union, List, Dict, Tuple, Literal, Any
import logging
import warnings
import matplotlib.pyplot as plt

from rpy2.robjects import r

# --- Pymer4 Import ---
try:
    from pymer4.models import Lmer
except ImportError:
    raise ImportError("The 'pymer4' package is required for LogisticRandomEffectModel. Please install it (pip install pymer4) and ensure R and lme4 are configured.")


from .base_model import BaseModel
from .mixins import SummaryMixin, PlotMixin, TestMixin


class LogisticRandomEffectModel(BaseModel, SummaryMixin, PlotMixin, TestMixin):
    """
    Logistic Random Effect Model using pymer4.models.Lmer.

    Estimates a generalized linear mixed model (GLMM) for binary outcomes using
    R's lme4 package via pymer4. It incorporates provider-specific random
    intercepts to account for clustering.

    Parameters
    ----------
    (No initialization parameters specific to this model beyond BaseModel)

    Attributes
    ----------
    model : pymer4.models.Lmer or None
        The fitted pymer4 Lmer model object.
    coefficients_ : dict
        Contains 'fixed_effect' (fixed effects estimates as pd.Series) and
        'random_effect' (predicted random effects/BLUPs as pd.Series/DataFrame).
    variances_ : dict
        Contains 'fe_var_cov' (fixed effects variance-covariance matrix as pd.DataFrame)
        and 're_var' (estimated variance(s) of the random effects from model.ranef_var).
    fitted_ : pd.Series or None
        Fitted probabilities (response scale) for the training data.
    residuals_ : pd.Series or None
        Response residuals (y - fitted_prob) for the training data.
    pearson_residuals_ : pd.Series or None
        Pearson residuals (from pymer4 model if available).
    # sigma_ attribute from BaseModel is generally not applicable/meaningful for logistic models.
    aic_ : float or None
        Akaike Information Criterion.
    bic_ : float or None
        Bayesian Information Criterion.
    loglike_ : float or None
        Log-likelihood of the fitted model.
    groups_ : np.ndarray or None
        Unique group identifiers present in the fitted data.
    group_sizes_ : np.ndarray or None
        Number of observations per group.
    xbeta_ : np.ndarray or None
        Linear predictor component from fixed effects only ($X\hat{\beta}$) for the fitted data.
    covariate_names_ : List[str] or None
        Names of the fixed effect covariates (excluding intercept).
    # Attributes related to statsmodels results are removed or replaced
    # by pymer4 equivalents where applicable.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Generate synthetic data (replace with actual data)
    >>> np.random.seed(123)
    >>> n_groups = 20; n_obs_per_group = 50; N = n_groups * n_obs_per_group
    >>> groups = np.repeat([f'G{i+1}' for i in range(n_groups)], n_obs_per_group) # Use string IDs
    >>> x1 = np.random.randn(N); x2 = np.random.binomial(1, 0.4, N)
    >>> true_beta = np.array([0.5, -1.0]); true_intercept = -0.5; true_re_sd = 0.8
    >>> true_re_dict = {f'G{i+1}': np.random.normal(0, true_re_sd) for i in range(n_groups)}
    >>> true_re_obs = np.array([true_re_dict[g] for g in groups])
    >>> lin_pred = true_intercept + x1 * true_beta[0] + x2 * true_beta[1] + true_re_obs
    >>> prob = 1 / (1 + np.exp(-lin_pred)); y = np.random.binomial(1, prob)
    >>> data = pd.DataFrame({'Y': y, 'X1': x1, 'X2': x2, 'GroupID': groups})
    >>>
    >>> # Fit the model using pymer4
    >>> logit_re_model = LogisticRandomEffectModel()
    >>> logit_re_model.fit(X=data, y_var='Y', x_vars=['X1', 'X2'], group_var='GroupID')
    >>>
    >>> # Get summary (fixed effects)
    >>> print(logit_re_model.summary())
    >>>
    >>> # Get standardized differences (based on BLUPs)
    >>> # sm = logit_re_model.calculate_standardized_measures(null='median')
    >>> # print(sm['indirect'].head())
    >>>
    >>> # Plot provider effects (BLUPs)
    >>> # logit_re_model.plot_provider_effects() # Requires matplotlib
    """

    def __init__(self) -> None:
        """Initialize the LogisticRandomEffectModel with placeholders."""
        super().__init__() # Initialize BaseModel attributes
        self.model: Optional[Lmer] = None # Type hint for pymer4 model
        # Remove attributes specific to statsmodels results
        self.result = None # No separate result object needed like statsmodels
        self.sigma_: Optional[float] = None
        self.pearson_residuals_: Optional[pd.Series] = None
        self.deviance_residuals_: Optional[pd.Series] = None # Lmer might not provide this easily
        self.loglike_: Optional[float] = None
        self.endog_: Optional[pd.Series] = None # Store original endog if needed
        self.exog_fe_: Optional[pd.DataFrame] = None # Store design matrix if needed
        self.exog_re_: Optional[pd.DataFrame] = None # Store RE design matrix if needed


    def fit(
        self, 
        X: Optional[pd.DataFrame] = None, 
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        groups: Optional[Union[pd.Series, np.ndarray]] = None,
        x_vars: Optional[List[str]] = None,
        y_var: Optional[str] = None, 
        group_var: Optional[str] = None,
        **kwargs
    ) -> "LogisticRandomEffectModel":
        """
        Fit the logistic random effect model using pymer4 Lmer with family='binomial'.

        Requires either a DataFrame `X` with `y_var`, `x_vars`, `group_var` specified,
        or separate arrays/Series for `X`, `y`, and `groups`.

        Parameters
        ----------
        X : pd.DataFrame or array-like, optional
            Dataset containing covariates and potentially outcome and group ID, or a design matrix.
        y : pd.Series or array-like, optional
            Binary response variable (0 or 1). Required if not in `X` DataFrame via `y_var`.
        groups : pd.Series or array-like, optional
            Group identifiers for random effects. Required if not in `X` DataFrame via `group_var`.
        x_vars : list of str, optional
            Predictor column names in `X` DataFrame. Required if `X` is DataFrame.
        y_var : str, optional
            Response column name in `X` DataFrame. Required if `X` is DataFrame.
        group_var : str, optional
            Group identifier column name in `X` DataFrame. Required if `X` is DataFrame.
        **kwargs : dict
            Additional keyword arguments passed to `pymer4.models.Lmer`. Common arguments
            might include `factors` to specify categorical variables if not auto-detected.

        Returns
        -------
        self : LogisticRandomEffectModel
            The fitted model instance.

        Raises
        ------
        ValueError
            If input data is insufficient or response variable is not binary.
        ImportError
            If pymer4 or its dependencies (R, lme4) are not available.
        """
        # _validate_and_convert_inputs is expected to set self.covariate_names_ if X is array
        # and to return numpy arrays for X_processed, y_processed, groups_processed.
        X_processed, y_processed, groups_processed = self._validate_and_convert_inputs(
            X, y, groups, x_vars, y_var, group_var, use_dataprep=True
        )
        # self.covariate_names_ should be set by _validate_and_convert_inputs
        if self.covariate_names_ is None:
            raise ValueError("Covariate names were not properly set during input validation.")

        # Ensure binary response
        unique_y_values = np.unique(y_processed)
        if not np.all(np.isin(unique_y_values, [0, 1])):
            raise ValueError(f"Response variable must be binary (0 or 1). Found unique values: {unique_y_values}")

        # --- Prepare DataFrame for Lmer ---
        # Use safe column names for formula construction if original names might be problematic
        # For this example, we assume provided x_vars, y_var, group_var are R-compatible
        # If not, sanitization would be needed here.
        current_y_var = y_var if y_var else 'outcome_LRE'
        current_x_vars = self.covariate_names_
        current_group_var = group_var if group_var else 'group_LRE'

        df_for_lmer = pd.DataFrame(X_processed, columns=current_x_vars)
        df_for_lmer[current_y_var] = y_processed
        df_for_lmer[current_group_var] = groups_processed

        if (not pd.api.types.is_string_dtype(df_for_lmer[current_group_var]) and
            not pd.api.types.is_categorical_dtype(df_for_lmer[current_group_var])):
            df_for_lmer[current_group_var] = df_for_lmer[current_group_var].astype(str)

        # --- Fit the GLMM using pymer4 ---
        fixed_part = " + ".join(current_x_vars)
        formula = f"{current_y_var} ~ {fixed_part} + (1 | {current_group_var})"

        print(f"Fitting Lmer model with formula: {formula}")
        try:
            self.model = Lmer(formula, data=df_for_lmer, family='binomial')
            # Pass summarize=False by default, let summary method handle it
            self.model.fit(summarize=kwargs.pop('summarize', False), **kwargs)
        except Exception as e:
            print(f"Error during pymer4 Lmer fitting: {e}")
            if hasattr(self.model, 'logfile') and self.model.logfile:
                print("--- R Logfile ---")
                print(self.model.logfile)
                print("-----------------")
            raise

        if not self.model.fitted: # Check if fitting actually succeeded
            log_content = self.model.logfile if hasattr(self.model, 'logfile') else "No logfile available."
            raise RuntimeError(f"pymer4 model fitting failed. Check R/lme4 installation and model specification. Log: {log_content}")
        
        self._store_results(y_processed, groups_processed, current_x_vars, current_y_var, current_group_var)

        # --- Calculate fixed-effect linear predictor (X*beta) ---
        if self.model and hasattr(self.model, 'design_matrix') and self.model.design_matrix is not None and \
           self.coefficients_ and 'fixed_effect' in self.coefficients_:

            design_matrix_fe = self.model.design_matrix
            fe_params = self.coefficients_['fixed_effect'] # Should have '(Intercept)'

            # Check alignment (more robust check)
            if set(design_matrix_fe.columns) == set(fe_params.index):
                # Ensure order matches for dot product
                ordered_design_matrix = design_matrix_fe[fe_params.index]
                self.exog_fe_ = ordered_design_matrix # Store the aligned design matrix
                self.xbeta_ = np.dot(ordered_design_matrix.values, fe_params.values)
            else:
                # This warning should ideally not trigger if naming is fixed
                warnings.warn(
                    "Mismatch between design matrix columns and fixed effect names even after attempting consistency. "
                    f"Design Matrix cols: {list(design_matrix_fe.columns)}, "
                    f"FE params index: {list(fe_params.index)}. "
                    "Attempting reconstruction for xbeta.", UserWarning
                )
                # Fallback reconstruction (less ideal)
                temp_df_for_design = df_for_lmer[current_x_vars]
                fe_param_names = fe_params.index.tolist()

                if '(Intercept)' in fe_param_names:
                    # Use add_constant matching pymer4/lme4 behavior (prepend=True)
                    design_matrix_fe_recon = sm.add_constant(temp_df_for_design, has_constant='add', prepend=True)
                    # Ensure column name is '(Intercept)' to match R default
                    design_matrix_fe_recon = design_matrix_fe_recon.rename(columns={'const': '(Intercept)'})
                else:
                    design_matrix_fe_recon = temp_df_for_design

                # Ensure columns and order match fe_params exactly
                try:
                    design_matrix_fe_recon = design_matrix_fe_recon[fe_param_names]
                    self.exog_fe_ = design_matrix_fe_recon
                    self.xbeta_ = np.dot(self.exog_fe_.values, fe_params.values)
                except KeyError as e:
                     warnings.warn(f"Failed to reconstruct design matrix for xbeta calculation due to missing columns: {e}. xbeta_ set to zeros.", UserWarning)
                     self.xbeta_ = np.zeros(len(y_processed))

        else:
            self.xbeta_ = np.zeros(len(y_processed))
            warnings.warn("Could not calculate xbeta_ due to missing model components (design_matrix or fixed_effects).", UserWarning)

        print("Model fitting complete.")
        return self
    
    def _store_results(
        self, 
        y_orig: np.ndarray, 
        groups_orig: np.ndarray,
        x_vars: List[str], 
        y_var: str, 
        group_var: str
    ) -> None:
        """
        Store fitted pymer4 Lmer model results in class attributes.
        
        Orchestrates the extraction of model components and stores them in
        standardized class attributes.
        
        Parameters
        ----------
        y_orig : np.ndarray
            Original response variable array.
        groups_orig : np.ndarray
            Original group identifiers array.
        x_vars : List[str]
            Names of fixed effect predictors.
        y_var : str
            Name of response variable in model data.
        group_var : str
            Name of grouping variable in model data.
        """
        if not self._validate_model():
            return
        
        # Store original data
        self._store_original_data(y_orig, y_var)
        
        # Extract model components
        # Store original data reference (outcome_ is y, endog_ is y from model data)
        self._store_original_data(y_orig, y_var)

        # Extract model components (using consistent naming)
        self._extract_fixed_effects() # Keeps (Intercept)
        self._extract_random_effects(group_var)
        self._extract_variance_components()

        # Calculate fitted values (probabilities, including RE) - THIS IS THE KEY PART
        self._calculate_fitted_values() # Tries fits, predict, R predict

        # Calculate residuals based on the full fitted values
        self._calculate_residuals()

        # Extract AIC, BIC, LogLik
        self._extract_fit_statistics()

        # Process group info
        self._process_group_information(groups_orig)

        # Note: xbeta_ (fixed effects only) is calculated *after* _store_results in the fit method.
        # _calculate_linear_predictor is removed as its logic is now split/handled elsewhere.


    def _validate_model(self) -> bool:
        """Validate that the model has been properly fitted"""
        if self.model is None:
            logging.warning("Model is None. Results not stored.")
            return False
        # Check for coefs attribute as a proxy for successful fitting summary
        if not hasattr(self.model, 'coefs') or self.model.coefs is None:
            logging.warning("Model fitting may have failed or model attributes (like coefs) are missing.")
            # Check logfile if fit failed
            if hasattr(self.model, 'logfile') and self.model.logfile:
                 print("--- R Logfile Snippet ---")
                 print(self.model.logfile[-500:]) # Print last part of log
                 print("------------------------")
            return False
        return True
    
    def _store_original_data(self, y_orig: np.ndarray, y_var: str) -> None:
        """Store original outcome and endogenous variable from model data"""
        self.outcome_ = pd.Series(y_orig) # Store original outcome array/series passed to fit

        # Get endogenous variable from the DataFrame used by pymer4 model if available
        if hasattr(self.model, 'data') and y_var in self.model.data:
            self.endog_ = self.model.data[y_var].copy() # Use the y-variable from the model's data
        else:
            # Fallback to the originally passed y_orig if model.data is not accessible
            self.endog_ = self.outcome_.copy()
    
    def _extract_fixed_effects(self) -> None:
        """Extract fixed effects coefficients, keeping '(Intercept)' name."""
        if not hasattr(self.model, 'coefs') or self.model.coefs is None:
             self.coefficients_ = {"fixed_effect": pd.Series(dtype=float)}
             self.covariate_names_ = []
             logging.warning("Could not extract fixed effects: model.coefs missing.")
             return

        fe_params = self.model.coefs.copy()

        # Ensure it's a Series of estimates
        if isinstance(fe_params, pd.DataFrame) and 'Estimate' in fe_params.columns:
            fe_params = fe_params['Estimate']
        elif not isinstance(fe_params, pd.Series):
             logging.warning(f"Unexpected format for model.coefs: {type(fe_params)}. Cannot extract fixed effects.")
             self.coefficients_ = {"fixed_effect": pd.Series(dtype=float)}
             self.covariate_names_ = []
             return

        self.coefficients_ = {"fixed_effect": fe_params}
        # Covariate names are non-intercept terms
        self.covariate_names_ = list(fe_params.index.difference(['(Intercept)']))

    
    def _extract_random_effects(self, group_var: str) -> None:
        """Extract random effects (BLUPs) from the model"""
        extractors = [
            # Method 1: Direct ranef DataFrame with (Intercept) column
            lambda m: m.ranef['(Intercept)'].copy() if (
                hasattr(m, 'ranef') and 
                isinstance(m.ranef, pd.DataFrame) and 
                '(Intercept)' in m.ranef.columns
            ) else None,
            
            # Method 2: Direct ranef DataFrame first column
            lambda m: m.ranef.iloc[:, 0].copy() if (
                hasattr(m, 'ranef') and 
                isinstance(m.ranef, pd.DataFrame) and 
                len(m.ranef.columns) == 1
            ) else None,
            
            # Method 3: Ranef dictionary format
            lambda m: m.ranef[group_var]['(Intercept)'].copy() if (
                hasattr(m, 'ranef') and 
                isinstance(m.ranef, dict) and
                group_var in m.ranef and
                isinstance(m.ranef[group_var], pd.DataFrame) and
                '(Intercept)' in m.ranef[group_var].columns
            ) else None,
            
            # Method 4: Ranef table format
            lambda m: pd.Series(
                m.ranef_table[m.ranef_table['Effect'] == '(Intercept)']['Value'].values,
                index=m.ranef_table[m.ranef_table['Effect'] == '(Intercept)']['Group'].values
            ) if (
                hasattr(m, 'ranef_table') and 
                isinstance(m.ranef_table, pd.DataFrame) and
                'Effect' in m.ranef_table and
                'Value' in m.ranef_table and
                'Group' in m.ranef_table and
                not m.ranef_table[m.ranef_table['Effect'] == '(Intercept)'].empty
            ) else None
        ]
        
        # Try each extractor until one works
        re_blups = None
        for extractor in extractors:
            try:
                re_blups = extractor(self.model)
                if re_blups is not None:
                    break
            except Exception as e:
                continue
        
        # Last resort fallback
        if re_blups is None:
            logging.warning("Could not extract random effects. Using empty Series.")
            re_blups = pd.Series(dtype=float)
        
        # Ensure index name and Series name
        if isinstance(re_blups, pd.Series):
            re_blups.index.name = 'group_id'
            re_blups.name = "random_effect"
            
        self.coefficients_["random_effect"] = re_blups
    
    def _extract_variance_components(self) -> None:
        """Extract variance-covariance matrices for fixed and random effects"""
        # Fixed effects variance-covariance matrix
        if hasattr(self.model, 'vcov') and self.model.vcov is not None:
            fe_vcv = self.model.vcov.copy()

        else:
            # Fallback to diagonal matrix from SE values
            if hasattr(self.model, 'coefs') and 'SE' in self.model.coefs.columns:
                se_vals = self.model.coefs['SE'].values
                fe_vcv_diag = np.diag(se_vals**2)
                fe_vcv = pd.DataFrame(
                    fe_vcv_diag, 
                    index=self.coefficients_["fixed_effect"].index, 
                    columns=self.coefficients_["fixed_effect"].index
                )
            else:
                logging.warning("Cannot find variance-covariance matrix. Using placeholder.")
                fe_vcv = pd.DataFrame(
                    np.nan, 
                    index=self.coefficients_["fixed_effect"].index, 
                    columns=self.coefficients_["fixed_effect"].index
                )
        
        self.variances_ = {"fe_var_cov": fe_vcv}
        
        # Random effects variance
        extractors = [
            # Method 1: Direct ranef_var attribute
            lambda m: m.ranef_var['Var'].iloc[0] if (
                hasattr(m, 'ranef_var') and 
                isinstance(m.ranef_var, pd.DataFrame) and
                'Var' in m.ranef_var.columns and 
                not m.ranef_var.empty
            ) else np.nan,
            
            # Method 2: Variance components via varcor
            lambda m: m.varcor['sdcor'].iloc[0]**2 if (
                hasattr(m, 'varcor') and
                isinstance(m.varcor, pd.DataFrame) and
                'sdcor' in m.varcor.columns and
                not m.varcor.empty
            ) else np.nan,
            
            # Method 3: Via summary random effects
            lambda m: m.summary['Random effects']['Std.Dev.'].iloc[0]**2 if (
                hasattr(m, 'summary') and
                isinstance(m.summary, dict) and
                'Random effects' in m.summary and
                isinstance(m.summary['Random effects'], pd.DataFrame) and
                'Std.Dev.' in m.summary['Random effects'].columns
            ) else np.nan
        ]
        
        # Try each extractor
        re_var_est = np.nan
        for extractor in extractors:
            try:
                value = extractor(self.model)
                if pd.notna(value):
                    re_var_est = value
                    break
            except Exception:
                continue
                
        self.variances_["re_var"] = float(re_var_est)
    
    def _calculate_fitted_values(self) -> None:
        """
        Calculate fitted values (probabilities) including random effects.
        Prioritizes direct model outputs, uses predict as fallback.
        """
        fitted_vals = None
        n_obs = len(self.outcome_) if self.outcome_ is not None else None

        # --- Method 1: Use model.fits (most direct from pymer4) ---
        if hasattr(self.model, 'fits') and self.model.fits is not None:
            try:
                fits_array = np.asarray(self.model.fits)
                if n_obs is None: n_obs = len(fits_array) # Get length if not known
                if len(fits_array) == n_obs:
                    fitted_vals = pd.Series(fits_array, index=self.outcome_.index[:n_obs])
                    logging.debug("Using model.fits for fitted values.")
                else:
                    logging.warning(f"Length mismatch: model.fits ({len(fits_array)}) vs outcome ({n_obs}). Cannot use model.fits.")
            except Exception as e:
                logging.warning(f"Error processing model.fits: {e}")

        # --- Method 2: Use pymer4's predict method (corrected call) ---
        if fitted_vals is None and hasattr(self.model, 'predict') and hasattr(self.model, 'data') and self.model.data is not None:
            logging.debug("Attempting fitted values using model.predict().")
            try:
                # Predict on training data, use rfx, disable verification warning
                predictions = self.model.predict(
                    data=self.model.data,
                    pred_type='response', # Get probabilities
                    use_rfx=True,         # Include random effects
                    verify_predictions=False # Suppress warning for predicting on training data
                )
                pred_array = np.asarray(predictions)
                if n_obs is None: n_obs = len(pred_array)
                if len(pred_array) == n_obs:
                     fitted_vals = pd.Series(pred_array, index=self.outcome_.index[:n_obs])
                     logging.debug("Using model.predict(verify_predictions=False) for fitted values.")
                else:
                     logging.warning(f"Length mismatch: model.predict ({len(pred_array)}) vs outcome ({n_obs}). Cannot use model.predict.")

            except Exception as e:
                # Log the actual error from predict
                logging.warning(f"Error using self.model.predict(data=self.model.data, verify_predictions=False): {e}")


        # --- Method 3: Use R's predict function directly (robust fallback) ---
        if fitted_vals is None and hasattr(self.model, '_R_model') and hasattr(self.model, 'r'):
            logging.debug("Attempting fitted values using R predict(type='response').")
            try:
                # Get probabilities directly from R's predict.glmerMod
                # Not specifying newdata uses the training data
                r_predictions = np.array(self.model.r.predict(self.model._R_model, type="response"))
                if n_obs is None: n_obs = len(r_predictions)
                if len(r_predictions) == n_obs:
                    fitted_vals = pd.Series(r_predictions, index=self.outcome_.index[:n_obs])
                    logging.debug("Using R predict(type='response') for fitted values.")
                else:
                    logging.warning(f"Length mismatch: R predict ({len(r_predictions)}) vs outcome ({n_obs}). Cannot use R predict.")
            except Exception as e:
                logging.warning(f"Could not calculate fitted values from R predict(type='response'): {e}")
                # Try R predict with type='link' as a last resort calculation
                if fitted_vals is None:
                     logging.debug("Attempting fitted values using R predict(type='link').")
                     try:
                         full_linear_predictor = np.array(self.model.r.predict(self.model._R_model, type="link"))
                         if n_obs is None: n_obs = len(full_linear_predictor)
                         if len(full_linear_predictor) == n_obs:
                              fitted_vals = pd.Series(
                                   1 / (1 + np.exp(-np.clip(full_linear_predictor, -500, 500))),
                                   index=self.outcome_.index[:n_obs]
                              )
                              logging.debug("Using R predict(type='link') and inverse logit for fitted values.")
                         else:
                              logging.warning(f"Length mismatch: R predict(link) ({len(full_linear_predictor)}) vs outcome ({n_obs}).")
                     except Exception as e_link:
                          logging.warning(f"Could not calculate fitted values from R predict(type='link'): {e_link}")


        # --- Final Assignment ---
        if fitted_vals is not None:
             self.fitted_ = fitted_vals
        else:
             # Only log warning if all attempts failed
             logging.warning(
                 "Could not calculate full fitted values (including random effects) using any method. "
                 "Check model fitting success and pymer4/R logs. self.fitted_ will be None."
             )
             self.fitted_ = None
        
    def _calculate_residuals(self) -> None:
        """Calculate various residuals based on full fitted probabilities."""
        if self.fitted_ is None or self.outcome_ is None:
            self.residuals_ = None
            self.pearson_residuals_ = None
            self.deviance_residuals_ = None # Still likely unavailable
            logging.debug("Skipping residual calculation as fitted_ or outcome_ is None.")
            return

        # Ensure alignment (use intersection of indices)
        common_index = self.outcome_.index.intersection(self.fitted_.index)
        if len(common_index) < len(self.outcome_):
             logging.warning(f"Index mismatch between outcome ({len(self.outcome_)}) and fitted ({len(self.fitted_)})."
                              "Calculating residuals on {len(common_index)} common observations.")

        outcome_aligned = self.outcome_.loc[common_index]
        fitted_aligned = self.fitted_.loc[common_index]

        # Response residuals (raw difference)
        self.residuals_ = outcome_aligned - fitted_aligned
        self.residuals_.name = "response_residuals"

        # Pearson residuals (raw / sqrt(variance))
        # Add small epsilon for numerical stability if fitted values are exactly 0 or 1
        fitted_var = fitted_aligned * (1 - fitted_aligned)
        epsilon = 1e-10
        self.pearson_residuals_ = self.residuals_ / np.sqrt(np.maximum(fitted_var, epsilon))
        self.pearson_residuals_.name = "pearson_residuals"

        # Deviance residuals for logistic are more complex, usually obtained from R directly
        # Formula: sign(y - p) * sqrt(-2 * [y*log(p) + (1-y)*log(1-p)])
        # Requires careful handling of p=0 or p=1 cases.
        # If pymer4 doesn't provide them easily, leave as None.
        self.deviance_residuals_ = None
        # try:
        #     # Attempt to get from R if possible (conceptual)
        #     if hasattr(self.model, '_R_model') and hasattr(self.model, 'r'):
        #          dev_res = np.array(self.model.r.residuals(self.model._R_model, type="deviance"))
        #          if len(dev_res) == len(common_index):
        #               self.deviance_residuals_ = pd.Series(dev_res, index=common_index, name="deviance_residuals")
        #          else:
        #               logging.warning("Could not get deviance residuals from R (length mismatch).")
        # except Exception as e:
        #      logging.warning(f"Could not get deviance residuals from R: {e}")


    def _extract_fit_statistics(self) -> None:
        """Extract model fit statistics (LogLik, AIC, BIC)."""
        # Use getattr for safe access
        self.loglike_ = getattr(self.model, 'logLike', None)
        self.aic_ = getattr(self.model, 'AIC', None)
        self.bic_ = getattr(self.model, 'BIC', None)

        # Convert to float if they are extracted but not already floats
        if self.loglike_ is not None: self.loglike_ = float(self.loglike_)
        if self.aic_ is not None: self.aic_ = float(self.aic_)
        if self.bic_ is not None: self.bic_ = float(self.bic_)

    def _process_group_information(self, groups_orig: np.ndarray) -> None:
        """Process grouping information (unique groups, sizes, indices)."""
        # Ensure groups_orig is treated consistently (e.g., as strings if they were converted)
        if pd.api.types.is_numeric_dtype(groups_orig) and not pd.api.types.is_integer_dtype(groups_orig):
             # If original groups were float/numeric and converted to string for R factor:
             groups_processed = pd.Series(groups_orig).astype(str).values
        else:
             groups_processed = np.asarray(groups_orig) # Use as is (int or string)

        # Get unique group identifiers and the index mapping for each observation
        self.groups_, self.group_indices_ = np.unique(groups_processed, return_inverse=True)
        # Count number of observations per unique group
        self.group_sizes_ = np.bincount(self.group_indices_, minlength=len(self.groups_))

    def predict(self, X: pd.DataFrame, x_vars: Optional[List[str]] = None) -> np.ndarray:
        """
        Predict probabilities for new data using only the fixed effects part of the model.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing new data for prediction. Must include columns specified
            in `x_vars` (or the covariates used during fitting if `x_vars` is None).
        x_vars : list of str, optional
            Predictor column names in X. If None, uses covariates from the fitted model.

        Returns
        -------
        np.ndarray
            Predicted probabilities based on fixed effects.
        """
        if self.model is None or not hasattr(self.model, 'coefs'):
            raise ValueError("Model must be fitted before prediction.")
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame for prediction.")

        fixed_effects = self.coefficients_['fixed_effect']
        
        # Check if we have 'Intercept' in the model
        has_intercept = 'Intercept' in fixed_effects.index

        if x_vars is None:
            # Use stored covariate names (excluding intercept)
            if self.covariate_names_ is None:
                raise ValueError("Cannot determine covariates for prediction. Fit model or provide x_vars.")
            pred_vars = self.covariate_names_
        else:
            pred_vars = x_vars

        missing_cols = [col for col in pred_vars if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in prediction data: {missing_cols}")

        X_pred = X[pred_vars].copy()

        # Prepare design matrix, adding intercept if needed
        if has_intercept:
            # Add an intercept column
            X_pred_design = sm.add_constant(X_pred, has_constant='add')
            # Rename 'const' to 'Intercept' to match our standardized name
            X_pred_design = X_pred_design.rename(columns={'const': 'Intercept'})
            
            # Make sure all coefficient names are in the design matrix
            missing_cols = [col for col in fixed_effects.index if col not in X_pred_design.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in design matrix: {missing_cols}")
            
            # Ensure column order matches coefficients
            X_pred_design = X_pred_design[fixed_effects.index]
        else:
            # Ensure column order matches if no intercept
            X_pred_design = X_pred[fixed_effects.index]

        # Calculate linear predictor using fixed effects
        try:
            linear_pred = np.dot(X_pred_design.values, fixed_effects.values)
        except Exception as e:
            raise ValueError(f"Error calculating predictions: {e}. Check that design matrix and "
                            f"coefficients are compatible.")

        # Convert to probability using the logistic function (sigmoid)
        probabilities = 1 / (1 + np.exp(-np.clip(linear_pred, -500, 500)))
        return probabilities

    def summary(
        self, 
        covariates: Optional[Union[List[str], np.ndarray]] = None,
        level: float = 0.95, 
        null: float = 0, 
        alternative: str = "two_sided"
    ) -> pd.DataFrame:
        """
        Provide summary statistics for the fixed effects coefficients (beta).

        Extracts results from the fitted pymer4 model summary. Uses Z-tests.

        Parameters
        ----------
        covariates : list or array, optional
            Subset of fixed effect covariate names to summarize (e.g., ['X1', 'Intercept']).
            Defaults to all fixed effects.
        level : float, default=0.95
            Confidence level for the intervals.
        null : float, default=0
            Null hypothesis value (ignored by pymer4 summary, test is vs 0).
        alternative : str, default="two_sided"
            Alternative hypothesis type (ignored by pymer4 summary, p-value is two-sided).

        Returns
        -------
        pd.DataFrame
            Summary table with columns: 'estimate', 'std_error', 'statistic' (Z-score),
            'p_value', 'ci_lower', 'ci_upper'. Indexed by fixed effect name.
        """
        if self.model is None or not hasattr(self.model, 'coefs'):
            raise ValueError("Model must be fitted before summarizing.")
        if null != 0:
             warnings.warn("`null` parameter is ignored for summary; pymer4 tests against 0.")
        if alternative != "two_sided":
             warnings.warn("`alternative` parameter is ignored for summary; pymer4 provides two-sided p-values.")

        # pymer4 model.coefs DataFrame usually contains: Estimate, SE, T-stat (or Z-stat), P-val, DF
        # For GLMM, it's typically Z-stat and P-values based on Normal approx.
        summary_df_raw = self.model.coefs.copy()

        # Rename columns for consistency
        rename_map = {
            'Estimate': 'estimate', 'SE': 'std_error',
            'Z-stat': 'statistic', # Check if it's 'T-stat' or 'Z-stat' in your pymer4 version
            'P-val': 'p_value'
        }
        # Handle potential T-stat column name
        if 'T-stat' in summary_df_raw.columns and 'Z-stat' not in summary_df_raw.columns:
            rename_map['T-stat'] = 'statistic'
            warnings.warn("Using T-stat from pymer4 summary as statistic.")

        summary_df = summary_df_raw.rename(columns=rename_map)

        # Calculate CIs using Normal approximation (consistent with Z-stat/P-val)
        alpha = 1 - level
        crit_value = norm.ppf(1 - alpha / 2)
        summary_df['ci_lower'] = summary_df['estimate'] - crit_value * summary_df['std_error']
        summary_df['ci_upper'] = summary_df['estimate'] + crit_value * summary_df['std_error']

        # Select and order columns
        final_cols = ["estimate", "std_error", "statistic", "p_value", "ci_lower", "ci_upper"]
        missing_final_cols = [c for c in final_cols if c not in summary_df.columns]
        if missing_final_cols:
             raise RuntimeError(f"Could not find expected columns in pymer4 summary: {missing_final_cols}")

        summary_df = summary_df[final_cols]

        # Format p-value
        summary_df['p_value'] = summary_df['p_value'].apply(lambda x: f"{min(x, 1.0):.4g}")

        if covariates is not None:
            try:
                summary_df = summary_df.loc[covariates]
            except KeyError as e:
                raise ValueError(f"One or more specified covariates not found in fixed effects: {e}")

        return summary_df.round(4)

    def calculate_standardized_measures(
        self, 
        group_ids: Optional[Union[List, np.ndarray]] = None,
        stdz: Union[str, List[str]] = "indirect",
        null: Union[str, float] = "median",
        measure: Union[str, List[str]] = ["rate", "ratio"]
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate standardized ratios and rates based on predicted random effects (BLUPs).
        This implementation follows the R function SM_output.logis_re logic.

        Parameters
        ----------
        group_ids : list or array, optional
            Subset of group IDs. Defaults to all groups in the model.
        stdz : str or list, default="indirect"
            Standardization method(s): "indirect", "direct".
        null : str or float, default="median"
            Baseline for random effects comparison:
            - 'median': uses median BLUP
            - 'mean': uses weighted mean of BLUPs (weighted by group sizes)
            - float: uses a user-specified numeric reference level (typically 0.0)
        measure : str or list, default=["rate", "ratio"]
            Measures to calculate: "rate", "ratio", or both.

        Returns
        -------
        dict
            Dictionary with DataFrames for 'indirect' and/or 'direct' measures.
            Each DataFrame contains columns:
            - 'group_id'
            - '{stdz}_difference': Raw difference between group RE and null RE
            - '{stdz}_ratio': Ratio of observed to expected (if "ratio" in measure)
            - '{stdz}_rate': Standardized rate (if "rate" in measure)
            - 'observed': For indirect - sum of observed outcomes; for direct - total observed
            - 'expected': For indirect - sum of expected probs; for direct - total expected under group's RE
        """
        if self.coefficients_ is None or self.fitted_ is None or self.xbeta_ is None or self.outcome_ is None:
            raise ValueError("Model must be fitted with coefficients, fitted values, xbeta, and outcome.")
        if self.groups_ is None or self.group_indices_ is None or self.group_sizes_ is None:
            raise ValueError("Group information (groups_, group_indices_, group_sizes_) missing.")

        # Ensure stdz and measure are lists
        if isinstance(stdz, str): 
            stdz = [stdz]
        if isinstance(measure, str):
            measure = [measure]
            
        # Validate parameters
        if not any(method in stdz for method in ["indirect", "direct"]):
            raise ValueError("Argument 'stdz' must include 'indirect' and/or 'direct'.")
        if not any(m in measure for m in ["rate", "ratio"]):
            raise ValueError("Argument 'measure' must include 'rate' and/or 'ratio'.")

        random_effects = self.coefficients_["random_effect"]  # Series indexed by group_id
        group_names = self.groups_  # Array of unique group IDs in fit order
        n_samples_total = len(self.outcome_)
        
        # Calculate population rate (percentage)
        total_observed = self.outcome_.sum()
        population_rate = (total_observed / n_samples_total) * 100.0

        # Determine null value for random effects
        if null == "median": 
            re_null = np.median(random_effects.dropna())
        elif null == "mean":
            valid_re = random_effects.dropna()
            weights = pd.Series(self.group_sizes_, index=self.groups_).reindex(valid_re.index)
            re_null = np.average(valid_re, weights=weights.fillna(0))
        elif isinstance(null, (int, float)): 
            re_null = float(null)
        else: 
            raise ValueError("Invalid 'null' argument. Use 'median', 'mean', or float.")

        # Map group ID to its index (0..m-1) and BLUP
        group_to_idx_map = {gid: i for i, gid in enumerate(group_names)}
        group_to_re_map = random_effects.to_dict()

        # Select groups to process
        if group_ids is None: 
            current_groups_ids = group_names
        else: 
            current_groups_ids = group_names[np.isin(group_names, np.asarray(group_ids))]

        results = {}
        sigmoid = lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))

        if "indirect" in stdz:
            # Get the full fitted values (including random effects)
            # These should be the fitted probabilities from the model
            # Make sure self.fitted_ contains the actual model predicted probabilities
            full_fitted_values = self.fitted_
            
            # Get expected probabilities without random effects (null model)
            exp_prob_null = sigmoid(self.xbeta_)
            
            # Create dataframes to store group-level calculations
            group_data = {}
            
            # First pass: collect observed and expected counts by group
            for gid in group_names:
                idx = group_to_idx_map.get(gid)
                if idx is None or self.group_sizes_[idx] == 0:
                    continue
                
                mask = (self.group_indices_ == idx)
                
                # FIXED: Use fitted values including random effects as "observed"
                # This matches R's approach using fit$fitted
                obs_sum = full_fitted_values[mask].sum()  # Sum of fitted probabilities (with RE)
                exp_sum = exp_prob_null[mask].sum()  # Sum of expected probabilities (null model)
                
                group_data[gid] = {
                    "observed": obs_sum,
                    "expected": exp_sum,
                }
            
            # Second pass: calculate ratios and rates for requested groups
            indirect_data = []
            for gid in current_groups_ids:
                if gid not in group_data:
                    continue
                    
                obs_sum = group_data[gid]["observed"]
                exp_sum = group_data[gid]["expected"]
                
                # Calculate standardized ratio and rate
                ratio = obs_sum / exp_sum if exp_sum > 0 else np.nan
                rate = np.clip(ratio * population_rate, 0.0, 100.0) if not np.isnan(ratio) else np.nan
                
                row = {
                    "group_id": gid,
                    "indirect_difference": group_to_re_map.get(gid, np.nan) - re_null,
                    "observed": obs_sum,
                    "expected": exp_sum
                }
                
                if "ratio" in measure:
                    row["indirect_ratio"] = ratio
                
                if "rate" in measure:
                    row["indirect_rate"] = rate
                    
                indirect_data.append(row)
            
            if indirect_data:
                indirect_df = pd.DataFrame(indirect_data)
                results["indirect"] = indirect_df

        if "direct" in stdz:
            # Direct standardization calculation (this part remains the same)
            direct_data = []
            
            for gid in current_groups_ids:
                re_k = group_to_re_map.get(gid)
                if re_k is None or np.isnan(re_k):
                    continue
                
                # Calculate expected probabilities for all observations using this group's RE
                exp_prob_k = sigmoid(self.xbeta_ + re_k)
                total_expected_k = exp_prob_k.sum()
                
                # Calculate standardized ratio and rate
                ratio = total_expected_k / total_observed if total_observed > 0 else np.nan
                rate = np.clip(ratio * population_rate, 0.0, 100.0) if not np.isnan(ratio) else np.nan
                
                row = {
                    "group_id": gid,
                    "direct_difference": re_k - re_null,
                    "observed": total_observed,
                    "expected": total_expected_k
                }
                
                if "ratio" in measure:
                    row["direct_ratio"] = ratio
                
                if "rate" in measure:
                    row["direct_rate"] = rate
                    
                direct_data.append(row)
            
            if direct_data:
                direct_df = pd.DataFrame(direct_data)
                results["direct"] = direct_df

        return results
    
    def _get_blup_post_se(self) -> pd.Series:
        """
        Extract posterior standard errors for BLUPs from the underlying lme4 model.
        
        Returns
        -------
        pd.Series
            Standard errors for the random effect BLUPs, indexed by group IDs.
        """
        if not hasattr(self.model, 'model_obj'):
            raise ValueError("Model must be fitted with pymer4's Lmer.")

        # Access the underlying R model
        r_model = self.model.model_obj

        # Extract random effects with posterior variances
        ranef_result = r['ranef'](r_model, condVar=True)

        # Assuming one random effect term (e.g., random intercept)
        re_df = ranef_result[0]  # First random effect term
        post_var = r['attr'](re_df, 'postVar')  # Posterior variances

        # Convert to numpy array and squeeze to 1D (for random intercept)
        post_var_array = np.array(post_var).squeeze()

        # Get group IDs from the random effects DataFrame
        group_ids = re_df.rownames if hasattr(re_df, 'rownames') else range(len(post_var_array))

        # Compute standard errors
        se_blup = np.sqrt(post_var_array)

        return pd.Series(se_blup, index=group_ids, name='blup_se')

    def calculate_confidence_intervals(
        self, 
        group_ids: Optional[Union[List, np.ndarray]] = None,
        level: float = 0.95, 
        option: str = "alpha",
        stdz: Union[str, List[str]] = "indirect",
        null: Union[str, float] = 0.0,
        measure: Union[str, List[str]] = ["ratio", "rate"],
        alternative: str = "two_sided"
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate CIs for random effects (BLUPs) or standardized measures.

        For `option='alpha'`, it provides approximate CIs for the random effect BLUPs
        on the log-odds scale.

        For `option='SM'`, calculation of robust CIs for standardized ratios/rates
        from logistic random effects models is complex and not directly implemented here.
        This method will raise a NotImplementedError for SM CIs, suggesting alternative
        approaches or consultation.

        Parameters
        ----------
        group_ids : list or array, optional
            Subset of group IDs.
        level : float, default=0.95
            Confidence level.
        option : str, default="SM"
            "alpha" for random effects (BLUPs), "SM" for standardized measures.
        stdz : str or list, default="indirect"
            Standardization method(s) if `option='SM'`.
        null : str or float, default=0.0
            Baseline for SM calculation or null for BLUP comparison.
        measure : str or list, default=["rate", "ratio"]
            Measures for SM if `option='SM'`.
        alternative : str, default="two_sided"
            Interval type. Must be 'two_sided' if `option='alpha'`.

        Returns
        -------
        dict
            If `option='alpha'`, returns `{'alpha_ci': DataFrame}` with BLUP CIs.
            Raises `NotImplementedError` if `option='SM'`.

        Raises
        ------
        ValueError
            If model not fitted or arguments invalid.
        NotImplementedError
            If `option='SM'` is chosen.
        """
        if self.coefficients_ is None or self.variances_ is None or self.group_sizes_ is None:
            raise ValueError("Model must be fitted with coefficients, variances, and group sizes.")
        if option == "alpha" and alternative != "two_sided":
            raise ValueError("Option 'alpha' (random effects) only supports two-sided CIs.")
        if option not in {"alpha", "SM"}:
            raise ValueError("Option must be 'alpha' or 'SM'.")

        if option == "SM":
            raise NotImplementedError(
                "Calculating robust confidence intervals for standardized ratios/rates "
                "from logistic random effect models is statistically complex and beyond "
                "simple transformation of BLUP confidence intervals. This typically requires "
                "methods like the Delta method on the log/logit scale, or extensive "
                "bootstrapping of the entire standardization process. "
                "Please consult with a statistician or consider these advanced methods. "
                "You can obtain CIs for the random effects (BLUPs) on the log-odds scale "
                "using option='alpha'."
            )

        # --- option == "alpha" ---
        random_effects = self.coefficients_["random_effect"]
        se_blup = self._get_blup_post_se()  # Use posterior SEs

        # Compute CI bounds using Normal distribution
        z_value = norm.ppf(1 - (1 - level) / 2)  # Two-sided critical value
        lower_re_blup = random_effects.values - z_value * se_blup.values
        upper_re_blup = random_effects.values + z_value * se_blup.values

        df_re_ci_full = pd.DataFrame({
            "group_id": random_effects.index,
            "alpha": random_effects.values,  # BLUP
            "lower": lower_re_blup,  # CI lower bound
            "upper": upper_re_blup  # CI upper bound
        }).set_index("group_id")

        if group_ids is not None:
            df_re_ci = df_re_ci_full.loc[df_re_ci_full.index.isin(np.asarray(group_ids))].copy()
            if df_re_ci.empty: return {"alpha_ci": pd.DataFrame()}  # Return empty DF
        else:
            df_re_ci = df_re_ci_full.copy()

        return {"alpha_ci": df_re_ci.reset_index()}

    def test(
        self, 
        group_ids: Optional[Union[List, np.ndarray]] = None,
        level: float = 0.95, null: Union[str, float] = 0.0, # Changed default null to 0 for RE
        alternative: str = "two_sided"
    ) -> pd.DataFrame:
        """
        Test random effects (BLUPs) for significance against a null value using Z-tests.

        Parameters
        ----------
        group_ids : list or array, optional
            Subset of group IDs to test. Defaults to all groups.
        level : float, default=0.95
            Confidence level (alpha = 1 - level).
        null : str or float, default=0.0
            Null hypothesis value for the random effect ('median' BLUP, 'mean' BLUP, or float).
        alternative : str, default="two_sided"
            Alternative hypothesis type ('two_sided', 'greater', 'less').

        Returns
        -------
        pd.DataFrame
            Test results indexed by group ID, with columns 'flag', 'p_value', 'stat' (Z-score), 'std_error'.
        """
        if self.coefficients_ is None or self.variances_ is None or self.group_sizes_ is None:
            raise ValueError("Model must be fitted.")

        alpha = 1 - level
        random_effects = self.coefficients_["random_effect"]  # Series indexed by group_id
        se_blup = self._get_blup_post_se()  # Use posterior SEs

        # Determine null value
        if null == "median": re_null = np.median(random_effects.dropna())
        elif null == "mean":
            valid_re = random_effects.dropna()
            weights = pd.Series(self.group_sizes_, index=self.groups_).reindex(valid_re.index)
            re_null = np.average(valid_re, weights=weights.fillna(0))
        elif isinstance(null, (int, float)): re_null = float(null)
        else: raise ValueError("Invalid 'null' argument.")

        # Compute Z-statistic, handle NaN SE
        z_score = pd.Series(np.nan, index=random_effects.index)
        valid_se = se_blup.notna() & (se_blup > 1e-10)
        z_score[valid_se] = (random_effects[valid_se] - re_null) / se_blup[valid_se]
        zero_se = se_blup.notna() & (se_blup <= 1e-10)
        z_score[zero_se] = np.sign(random_effects[zero_se] - re_null) * np.inf

        # Calculate p-values using Normal distribution
        p = pd.Series(norm.sf(z_score), index=z_score.index)
        
        # Compute p_value and flag based on alternative hypothesis
        if alternative == "two_sided":
            p_value = 2 * pd.Series(np.minimum(p, 1 - p), index=p.index)
            flag = np.where(p < alpha / 2, 1, np.where(p > 1 - alpha / 2, -1, 0))
        elif alternative == "greater":
            p_value = p
            flag = np.where(p_value < alpha, 1, 0)
        elif alternative == "less":
            p_value = pd.Series(norm.cdf(z_score), index=z_score.index)
            flag = np.where(p_value < alpha, -1, 0)
        else:
            raise ValueError("Argument 'alternative' should be one of 'two_sided', 'greater', or 'less'")
        
        # Handle edge cases for p_value
        p_value = p_value.fillna(1.0)
        p_value[~np.isfinite(z_score)] = 0.0
        
        # Convert flag to pandas Series for consistency
        flag = pd.Series(flag, index=z_score.index)
        
        # Create the result DataFrame
        result = pd.DataFrame({
            "flag": pd.Categorical(flag.astype(int), categories=[-1, 0, 1]),
            "p_value": np.round(p_value, 7),
            "stat": z_score.fillna(0),
            "std_error": se_blup
        }, index=random_effects.index)
        result.index.name = "group_id"
        
        # Filter by group_ids if provided
        if group_ids is not None:
            if not isinstance(group_ids, (list, np.ndarray, pd.Series)):
                raise ValueError("group_ids must be a list, array, or Series")
            result = result.loc[result.index.isin(group_ids)]
        
        return result

    # --- Plotting Methods ---
    def plot_funnel(self, **kwargs) -> None:
        """
        Funnel plot for Logistic Random Effect Model.

        Standard funnel plots with control limits based on expected variance
        are complex to define rigorously for random effect models, especially
        logistic ones, due to shrinkage and the nature of random effect variance.
        Consider using plot_provider_effects (caterpillar plot) instead,
        or plotting BLUPs vs group size/precision without control limits.

        Raises
        ------
        NotImplementedError
            This method is not implemented due to theoretical complexities.
        """
        raise NotImplementedError(
            "Standard funnel plots with control limits are complex for logistic random effect models. "
            "Consider plot_provider_effects or a custom scatter plot of BLUPs vs precision."
            )
        
    def plot_provider_effects(
        self, 
        group_ids=None, 
        level: float = 0.95,          
        use_flags: bool = True, 
        null: Union[str, float] = 0.0,
        **plot_kwargs
    ) -> None:
        """
        Plots predicted random effects (BLUPs) with approximate confidence intervals.

        Uses the `plot_caterpillar` helper function for visualization.
        Confidence intervals for BLUPs are approximate and based on asymptotic normality.

        Parameters
        ----------
        group_ids : list or array, optional
            Subset of provider IDs to plot. Defaults to all.
        level : float, default=0.95
            Confidence level for intervals.
        use_flags : bool, default=True
            Color points based on significance flags from `self.test()`.
        null : str or float, default=0.0
            Null value for flagging random effects (typically 0 for BLUPs).
        **plot_kwargs
            Additional arguments passed to `plot_caterpillar` (e.g., `orientation`,
            `title`, `figsize`). See `plot_caterpillar` docstring.
        """
        if self.coefficients_ is None or self.variances_ is None:
            raise ValueError("Model must be fitted first.")

        # Get CIs for BLUPs (option='alpha')
        ci_results = self.calculate_confidence_intervals(
            group_ids=group_ids, level=level, option='alpha', alternative='two_sided'
        )
        if 'alpha_ci' not in ci_results or ci_results['alpha_ci'].empty:
            print("Warning: No random effect CI data found. Cannot plot.")
            return

        df_plot = ci_results['alpha_ci'] # Columns: group_id, alpha, lower, upper

        flag_col_name = None
        if use_flags:
            flag_col_name = 'flag'
            try:
                test_df = self.test(group_ids=df_plot['group_id'].unique().tolist(), 
                                    level=level, 
                                    null=null, 
                                    alternative='two_sided')
                # if not test_df.index.name == 'group_id': 
                #     test_df = test_df.set_index('group_id')
                df_plot = df_plot.merge(test_df[['flag']], left_on='group_id', right_index=True, how='left')
                df_plot[flag_col_name] = df_plot[flag_col_name].fillna(0).astype(int)
            except Exception as e:
                warnings.warn(f"Could not generate flags. Plotting without flags. Error: {e}")
                flag_col_name = None

        # Set defaults for plot_caterpillar
        plot_kwargs.setdefault('plot_title', 'Provider Random Effects')
        plot_kwargs.setdefault('refline_value', 0.0) # RE are typically centered around 0

        # Default orientation and labels for plot_caterpillar
        orientation = plot_kwargs.pop('orientation', 'vertical')
        if orientation == 'vertical':
            plot_kwargs.setdefault('xlab', 'Predicted Random Effect (Log-Odds Scale)')
            plot_kwargs.setdefault('ylab', 'Provider')
        else: # horizontal
            plot_kwargs.setdefault('xlab', 'Provider')
            plot_kwargs.setdefault('ylab', 'Predicted Random Effect (Log-Odds Scale)')
        plot_kwargs['orientation'] = orientation # Add it back

        # Call plot_caterpillar, it creates its own figure/axes
        plot_caterpillar(
            df=df_plot, 
            estimate_col='alpha', 
            ci_lower_col='lower', 
            ci_upper_col='upper',
            group_col='group_id', 
            flag_col=flag_col_name, 
            **plot_kwargs
        )

    def plot_standardized_measures(self, **kwargs) -> None:
        """
        Plot standardized measures for Logistic Random Effect Model.

        This plot would typically display standardized ratios or rates with their
        confidence intervals. However, as robust confidence intervals for these
        standardized measures (derived from logistic random effect models) are
        not directly implemented in `calculate_confidence_intervals` due to
        statistical complexity, this plotting method is also not implemented.

        To visualize provider differences, consider using `plot_provider_effects`
        to show the random effects (BLUPs) on the log-odds scale with their
        approximate confidence intervals.

        Raises
        ------
        NotImplementedError
            This method is not implemented because robust CIs for standardized
            ratios/rates from this model are not provided by the corresponding
            CI calculation method.
        """
        raise NotImplementedError(
            "Plotting standardized measures with robust CIs is not implemented for "
            "LogisticRandomEffectModel because the underlying CI calculation for these "
            "measures is complex. Please use `plot_provider_effects` to visualize "
            "random effects (BLUPs) on the log-odds scale."
        )

    def plot_coefficient_forest(
        self,
        orientation: Literal["vertical", "horizontal"] = "vertical",
        refline_value: Optional[float] = 0.0,
        level: float = 0.95,
        point_color: str = "#34495E",
        point_alpha: float = 0.8,
        edge_color: Optional[str] = None,
        edge_linewidth: float = 0,
        point_size: float = 0.05,
        error_color: str = "#95A5A6",
        capsize: float = 5,
        errorbar_size: float = 0.5,
        errorbar_alpha: float = 0.5,
        line_color: str = "red",
        line_style: str = "--",
        line_size: float = 0.8,
        font_size: float = 12,
        tick_label_size: float = 10,
        add_grid: bool = True,
        grid_style: str = ":",
        grid_alpha: float = 0.6,
        remove_top_right_spines: bool = True,
        figure_size: Tuple[float, float] = (10, 6),
        plot_title: str = "Forest Plot of Fixed Effect Coefficients",
        xlab: str = "Coefficient Estimate (Log-Odds)",
        ylab: str = "Covariate",
        save_path: Optional[str] = None,
        dpi: int = 300
    ) -> None:
        """
        Create a forest plot of fixed effect coefficients with confidence intervals.

        Plots each covariate's coefficient estimate and its confidence interval
        in a vertical or horizontal layout, with a reference line at a specified value.

        Parameters
        ----------
        orientation : {'vertical','horizontal'}, default 'vertical'
            'vertical': covariate names on the y-axis, estimates on the x-axis.
            'horizontal': covariate names on the x-axis, estimates on the y-axis.
        refline_value : float or None, default 0.0
            Draws a reference line at this value (vertical or horizontal). None disables it.
        level : float, default 0.95
            Confidence level for the intervals (e.g., 0.95 for 95% CI).
        point_color : str, default "#34495E"
            Color of the coefficient marker.
        point_alpha : float, default 0.8
            Opacity of the coefficient marker.
        edge_color : str or None, default None
            Edge color of the marker. None for no edge.
        edge_linewidth : float, default 0
            Width of the marker edge.
        point_size : float, default 0.05
            Scale factor for marker size.
        error_color : str, default "#95A5A6"
            Color of the error bars.
        capsize : float, default 5
            Cap size for error bars.
        errorbar_size : float, default 0.5
            Thickness of the error bar lines.
        errorbar_alpha : float, default 0.5
            Opacity of the error bars.
        line_color : str, default "red"
            Color of the reference line.
        line_style : str, default "--"
            Line style of the reference line.
        line_size : float, default 0.8
            Thickness of the reference line.
        font_size : float, default 12
            Font size for labels and title.
        tick_label_size : float, default 10
            Font size for tick labels.
        add_grid : bool, default True
            Whether to draw a light grid.
        grid_style : str, default ":"
            Line style for the grid.
        grid_alpha : float, default 0.6
            Opacity of the grid.
        remove_top_right_spines : bool, default True
            Hide the top and right spines.
        figure_size : tuple, default (10, 6)
            Size of the figure in inches.
        plot_title : str, default "Forest Plot of Fixed Effect Coefficients"
            Title of the plot.
        xlab : str, default "Coefficient Estimate (Log-Odds)"
            Label for the x-axis (or y-axis if horizontal).
        ylab : str, default "Covariate"
            Label for the y-axis (or x-axis if horizontal).
        save_path : str or None, default None
            File path to save the figure. If None, the plot is shown.
        dpi : int, default 300
            Resolution (dots per inch) for saving.

        Raises
        ------
        ValueError
            If the model is not fitted or if an invalid orientation is provided.
        """
        # Preconditions
        if self.coefficients_ is None or self.variances_ is None:
            raise ValueError("Model must be fitted before plotting coefficients.")
        if orientation not in ("vertical", "horizontal"):
            raise ValueError("orientation must be 'vertical' or 'horizontal'")

        # Use summary method to get CIs based on Normal distribution (Z-scores)
        summary_df = self.summary(level=level, alternative="two_sided") # Get two-sided CIs

        # Prepare DataFrame for plot_caterpillar
        coef_df = summary_df[['estimate', 'ci_lower', 'ci_upper']].copy() # Use renamed columns
        coef_df['covariate'] = coef_df.index

        # Positions for plotting
        n = len(coef_df)
        positions = np.arange(n)

        # Prepare coordinates and errors based on orientation
        if orientation == "vertical":
            x_vals, y_vals = coef_df["estimate"], positions
            xerr = [
                coef_df["estimate"] - coef_df["ci_lower"],
                coef_df["ci_upper"] - coef_df["estimate"]
            ]
        else:
            x_vals, y_vals = positions, coef_df["estimate"]
            yerr = [
                coef_df["estimate"] - coef_df["ci_lower"],
                coef_df["ci_upper"] - coef_df["estimate"]
            ]

        # Set up the plot
        fig, ax = plt.subplots(figsize=figure_size)

        # Draw error bars and points
        if orientation == "vertical":
            ax.errorbar(
                x_vals, y_vals,
                xerr=xerr,
                fmt="o",
                color=point_color,
                ecolor=error_color,
                capsize=capsize,
                markersize=point_size * 30,
                alpha=point_alpha,
                linewidth=errorbar_size,
                markeredgecolor=edge_color if edge_color else 'none',
                markeredgewidth=edge_linewidth
            )
        else:
            ax.errorbar(
                x_vals, y_vals,
                yerr=yerr,
                fmt="o",
                color=point_color,
                ecolor=error_color,
                capsize=capsize,
                markersize=point_size * 30,
                alpha=point_alpha,
                linewidth=errorbar_size,
                markeredgecolor=edge_color if edge_color else 'none',
                markeredgewidth=edge_linewidth
            )

        # Add reference line
        if refline_value is not None:
            if orientation == "vertical":
                ax.axvline(refline_value, color=line_color, linestyle=line_style, linewidth=line_size)
            else:
                ax.axhline(refline_value, color=line_color, linestyle=line_style, linewidth=line_size)

        # Set labels, ticks, and grid
        if orientation == "vertical":
            ax.set_xlabel(xlab, fontsize=font_size)
            ax.set_ylabel(ylab, fontsize=font_size)
            ax.set_yticks(positions)
            ax.set_yticklabels(coef_df["covariate"], fontsize=tick_label_size)
            ax.tick_params(axis="x", labelsize=tick_label_size)
            if add_grid:
                ax.grid(True, axis="x", linestyle=grid_style, alpha=grid_alpha, color="lightgrey")
        else:
            ax.set_xlabel(ylab, fontsize=font_size)
            ax.set_ylabel(xlab, fontsize=font_size)
            ax.set_xticks(positions)
            ax.set_xticklabels(coef_df["covariate"], rotation=45, ha="right", fontsize=tick_label_size)
            ax.tick_params(axis="y", labelsize=tick_label_size)
            if add_grid:
                ax.grid(True, axis="y", linestyle=grid_style, alpha=grid_alpha, color="lightgrey")

        # Adjust spines
        if remove_top_right_spines:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        if orientation == "vertical":
            ax.spines["left"].set_linewidth(0.8)
            ax.spines["bottom"].set_linewidth(0.8)
        else:
            ax.spines["bottom"].set_linewidth(0.8)
            ax.spines["left"].set_linewidth(0.8)

        # Set title and layout
        ax.set_title(plot_title, fontsize=font_size + 2, pad=15)
        plt.tight_layout()

        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    def plot_residuals(
        self,
        residual_type: Literal['response', 'pearson', 'deviance'] = "pearson",
        figsize: tuple = (8, 5),
        point_color: str = "#1F78B4",
        point_alpha: float = 0.6,
        edge_color: Optional[str] = "grey",
        edge_linewidth: float = 0.5,
        point_size: float = 30,
        line_color: str = "red",
        line_style: str = "--",
        line_width: float = 1.5,
        xlabel: str = "Fitted Probabilities",
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        font_size: float = 12,
        tick_label_size: float = 10,
        add_grid: bool = True,
        grid_style: str = ':',
        grid_alpha: float = 0.6,
        remove_top_right_spines: bool = True
    ) -> None:
        """
        Plots residuals versus fitted probabilities for the logistic GLMM.

        Parameters are as described previously. This method creates its own plot.
        """
        if self.fitted_ is None:
             raise ValueError("Model must be fitted before plotting residuals.")

        residuals_to_plot = None
        if residual_type == "response":
            residuals_to_plot = self.residuals_
            default_ylabel = "Response Residuals (y - p_hat)"
            default_title = "Response Residuals vs. Fitted Probabilities"
        elif residual_type == "pearson":
            residuals_to_plot = self.pearson_residuals_
            default_ylabel = "Pearson Residuals"
            default_title = "Pearson Residuals vs. Fitted Probabilities"
        elif residual_type == "deviance":
            residuals_to_plot = self.deviance_residuals_
            default_ylabel = "Deviance Residuals"
            default_title = "Deviance Residuals vs. Fitted Probabilities"
        else:
            raise ValueError("residual_type must be 'response', 'pearson', or 'deviance'.")

        if residuals_to_plot is None:
            raise ValueError(f"'{residual_type}' residuals are not available. Ensure they were calculated.")

        plot_ylabel = ylabel if ylabel is not None else default_ylabel
        plot_title = title if title is not None else default_title

        fig, ax = plt.subplots(figsize=figsize) # Create figure/axes internally

        fitted_vals = self.fitted_.values if isinstance(self.fitted_, pd.Series) else self.fitted_
        resid_vals = residuals_to_plot.values if isinstance(residuals_to_plot, pd.Series) else residuals_to_plot

        ax.scatter(
            fitted_vals, resid_vals, color=point_color, alpha=point_alpha,
            edgecolor=edge_color, linewidth=edge_linewidth if edge_color else 0, s=point_size
        )
        ax.axhline(0, color=line_color, linestyle=line_style, linewidth=line_width)
        ax.set_xlabel(xlabel, fontsize=font_size); ax.set_ylabel(plot_ylabel, fontsize=font_size)
        ax.set_title(plot_title, fontsize=font_size + 2, pad=15)
        ax.tick_params(axis="both", labelsize=tick_label_size)
        if add_grid: ax.grid(True, linestyle=grid_style, alpha=grid_alpha, color='lightgrey')
        if remove_top_right_spines:
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.8); ax.spines['bottom'].set_linewidth(0.8)
        ax.text(0.99, 0.01, "Note: Patterns expected due to binary outcome & variance dependency",
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes, color='gray', fontsize=font_size-3)

        plt.tight_layout()
        plt.show()
        # No return value

    def plot_qq(self, *args, **kwargs) -> None:
        """
        Create a Q-Q plot of the deviance residuals for logistic regression.

        Raises:
        -------
        NotImplementedError
            This method is not implemented.
        """
        raise NotImplementedError(
            "plot_qq is not implemented for LogisticFixedEffectModel. "
            "Q-Q plots are less meaningful in logistic regression"
            "Residuals from logistic regression are not expected to be normally distributed"
        )