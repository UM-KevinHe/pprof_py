coefficients_lfe = lfe_model_logistic.coefficients_
beta_coeffs_lfe_df = coefficients_lfe['beta']  # Covariate effects (as Series)
gamma_coeffs_lfe_df = coefficients_lfe['gamma'] # Provider fixed effects (as Series)

print("\nEstimated Beta Coefficients (Log-Odds Ratios for Covariates):\n", beta_coeffs_lfe_df)
print("\nEstimated Gamma Coefficients (Log-Odds Adjustments for Providers, first 5):\n", gamma_coeffs_lfe_df.head())

# Interpretation:
# For Beta: A one-unit increase in 'age_patient' is associated with a change of [beta_for_age] in the log-odds of 'readmitted_30day', holding other factors and provider constant.
# For Gamma: Clinic_X has an adjusted log-odds of 'readmitted_30day' that is [gamma_for_Clinic_X] units different from the reference when covariates are at their baseline.