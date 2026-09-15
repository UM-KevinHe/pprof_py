if hasattr(lre_model_logistic, 'model') and lre_model_logistic.model is not None and lre_model_logistic.model.fitted:
    variances_lre = lre_model_logistic.variances_
    fe_var_cov_lre_df = variances_lre.get('fe_var_cov')  # VCV for fixed effects
    re_var_lre = variances_lre.get('re_var')      # Variance of random effects (sigma_u^2)

    if fe_var_cov_lre_df is not None:
        print("\nVariance-Covariance Matrix for Fixed Effects:\n", fe_var_cov_lre_df.head())
    else:
        print("\nFixed effects VCV not available.")

    if re_var_lre is not None:
        print(f"\nEstimated Variance of Random Effects (sigma_u^2): {re_var_lre:.4f}")
    else:
        print("\nRandom effects variance not available.")
else:
    print("Model not fitted successfully. Cannot display variances.")