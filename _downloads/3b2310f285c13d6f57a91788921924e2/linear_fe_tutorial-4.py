coefficients = lfe_model.coefficients_
beta_coeffs = coefficients['beta']  # Covariate effects
gamma_coeffs = coefficients['gamma'] # Provider fixed effects

print("\nEstimated Beta Coefficients (Covariates):\n", beta_coeffs)
print("\nEstimated Gamma Coefficients (Providers, first 5):\n", gamma_coeffs[:5])