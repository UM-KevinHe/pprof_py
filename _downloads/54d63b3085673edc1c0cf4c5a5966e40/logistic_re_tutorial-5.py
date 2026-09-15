if lre_model_logistic.coefficients_ is not None:
    variances_lre = lre_model_logistic.variances_
    fe_var_cov_lre_df = variances_lre['beta']  # VCV for fixed effects
     re_var_lre = variances_lre['alpha']          # Variance of random effects {group_var: sigma_u^2}

     if fe_var_cov_lre_df is not None:
         print("\nVariance-Covariance Matrix for Fixed Effects:\n", fe_var_cov_lre_df.head())
     else:
         print("\nFixed effects VCV not available.")

     if re_var_lre is not None:
         sigma_u2 = list(re_var_lre.values())[0]  # Get sigma_u^2 for first (only) group
         print(f"\nEstimated Variance of Random Effects (sigma_u^2): {sigma_u2:.4f}")
    else:
        print("\nRandom effects variance not available.")
else:
    print("Model not fitted successfully. Cannot display variances.")