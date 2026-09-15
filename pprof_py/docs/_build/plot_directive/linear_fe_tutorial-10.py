# Test provider effects against the median gamma
provider_test_results_df = lfe_model.test(
    null='median',
    level=0.95,
    alternative='two_sided'
)

print("\nProvider Test Results (vs median gamma, first 5):\n", provider_test_results_df.head())
print("\nProviders flagged as significantly different (example):\n", provider_test_results_df[provider_test_results_df['flag'] != 0].head())