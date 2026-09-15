gamma_ci_results = lfe_model.calculate_confidence_intervals(
    option='gamma',
    level=0.95,
    alternative='two_sided'
)
gamma_ci_df = gamma_ci_results['gamma_ci']
print("\nConfidence Intervals for Gamma (first 5):\n", gamma_ci_df.head())