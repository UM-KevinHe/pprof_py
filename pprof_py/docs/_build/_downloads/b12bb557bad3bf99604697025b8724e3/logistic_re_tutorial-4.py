if hasattr(lre_model_logistic, 'model') and lre_model_logistic.model is not None and lre_model_logistic.model.fitted:
    coefficients_lre = lre_model_logistic.coefficients_
    fixed_effects_lre_df = coefficients_lre.get('fixed_effect')    # Fixed effects (as Series)
    random_effects_lre_series = coefficients_lre.get('random_effect') # Random effects (as Series)

    if fixed_effects_lre_df is not None:
        print("\nEstimated Fixed Effects (Log-Odds Ratios for Covariates):\n", fixed_effects_lre_df)
    else:
        print("\nFixed effects not available.")

    if random_effects_lre_series is not None:
        print("\nPredicted Random Effects (BLUPs for Providers, first 5):\n", random_effects_lre_series.head())
    else:
        print("\nRandom effects not available.")
else:
    print("Model not fitted successfully. Cannot display coefficients.")