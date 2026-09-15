variances_lfe = lfe_model_logistic.variances_
var_beta_lfe_df = variances_lfe['beta']    # Variance-covariance matrix for beta
var_gamma_lfe_df = variances_lfe['gamma']  # Variances for gamma (diagonal)

print("\nVariance-Covariance Matrix for Beta:\n", var_beta_lfe_df)
print("\nVariances for Gamma (Providers, first 5):\n", var_gamma_lfe_df.head())