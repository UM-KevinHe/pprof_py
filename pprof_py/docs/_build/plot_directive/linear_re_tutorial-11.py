random_effect_ci_results = lre_model.calculate_confidence_intervals(
    option='alpha', # 'alpha' for random effects (u_i)
    level=0.95,
    alternative='two_sided' # 'alpha' option only supports 'two_sided'
)
random_effect_ci_df = random_effect_ci_results['alpha_ci']
print("\nConfidence Intervals for Random Effects (alpha/u_i, first 5):\n", random_effect_ci_df.head())