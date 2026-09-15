variances_re = lre_model.variances_
fe_var_cov_df = variances_re['beta']    # Variance-covariance matrix for fixed effects
re_var_matrix = variances_re['alpha']   # Variance of random effects (1x1 matrix for random intercept)

print("\nVariance-Covariance Matrix for Fixed Effects:\n", fe_var_cov_df)
print("\nVariance of Random Effects (Group Var):\n", re_var_matrix)
# For a random intercept model, re_var is typically the variance of the provider effects (sigma_u^2)