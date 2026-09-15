# Test provider effects against the median gamma using Poisson-Binomial exact test
provider_test_lfe_results_df = lfe_model_logistic.test(
    null='median',          # Null hypothesis value for gamma
    level=0.95,             # Corresponds to alpha = 0.05 for flagging
    test_method='poibin_exact', # or 'score', 'wald', 'bootstrap_exact'
    alternative='two_sided'
)

print("\nProvider Test Results (vs median gamma, poibin_exact, first 5):\n", provider_test_lfe_results_df.head())
# Key columns: 'flag' (-1, 0, 1), 'p_value', 'stat'.

print("\nProviders flagged as significantly different (example):\n", provider_test_lfe_results_df[provider_test_lfe_results_df['flag'] != 0].head())