coefficients_re = lre_model.coefficients_
fixed_effects_df = coefficients_re['beta']    # Fixed effects (as Series)
random_effects_series = coefficients_re['alpha'] # Random effects (as Series)

print("\nEstimated Fixed Effects (Betas):\n", fixed_effects_df)
print("\nEstimated Random Effects (Providers, first 5):\n", random_effects_series.head())