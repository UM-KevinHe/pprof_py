if lre_model_logistic.coefficients_ is not None:
    coefficients_lre = lre_model_logistic.coefficients_
    fixed_effects_lre_series = coefficients_lre['beta']    # Fixed effects (as Series)
     random_effects_lre_series = lre_model_logistic.get_random_effects()  # BLUPs (as Series)

     if fixed_effects_lre_series is not None:
         print("\nEstimated Fixed Effects (Log-Odds Ratios for Covariates):\n", fixed_effects_lre_series)
    else:
        print("\nFixed effects not available.")

    if random_effects_lre_series is not None:
        print("\nPredicted Random Effects (BLUPs for Providers, first 5):\n", random_effects_lre_series.head())
    else:
        print("\nRandom effects not available.")
else:
    print("Model not fitted successfully. Cannot display coefficients.")