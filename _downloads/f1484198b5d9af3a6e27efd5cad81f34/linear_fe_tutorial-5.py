variances = lfe_model.variances_
var_beta = variances['beta']    # Variance-covariance matrix for beta
var_gamma = variances['gamma']  # Variances for gamma
sigma_hat = lfe_model.sigma_       # Estimated residual standard deviation

print("\nVariance-Covariance Matrix for Beta:\n", var_beta)
print("\nVariances for Gamma (Providers, first 5):\n", var_gamma[:5])
print(f"\nEstimated Sigma (Residual Standard Deviation): {sigma_hat:.4f}")