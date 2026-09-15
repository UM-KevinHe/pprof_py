coefficients_re = lre_model.coefficients_
fixed_effects_df = coefficients_re['fixed_effect']    # Fixed effects (as Series)
random_effects_series = coefficients_re['random_effect'] # Random effects (as Series)

print("\nEstimated Fixed Effects (Betas):\n", fixed_effects_df)
print("\nEstimated Random Effects (Providers, first 5):\n", random_effects_series.head())