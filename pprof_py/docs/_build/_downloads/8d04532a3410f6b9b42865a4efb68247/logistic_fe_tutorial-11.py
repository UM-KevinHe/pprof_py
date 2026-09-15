gamma_ci_lfe_results = lfe_model_logistic.calculate_confidence_intervals(
    option='gamma',
    level=0.95,
    test_method='exact', # 'wald', 'score', or 'exact' (Poisson-Binomial based)
    alternative='two_sided' # Must be 'two_sided' for option='gamma'
)
gamma_ci_lfe_df = gamma_ci_lfe_results['gamma_ci']
print("\nConfidence Intervals for Gamma (Exact method, first 5):\n", gamma_ci_lfe_df.head())