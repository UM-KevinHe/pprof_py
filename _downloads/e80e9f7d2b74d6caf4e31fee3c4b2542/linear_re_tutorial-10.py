# Test provider random effects against a null of 0 (or median/mean of random effects)
provider_test_re_results_df = lre_model.test(
    null=0,                 # Null hypothesis value for random effects
    level=0.95,             # Corresponds to alpha = 0.05 for flagging
    alternative='two_sided'
)

print("\nProvider Random Effect Test Results (vs null=0, first 5):\n", provider_test_re_results_df.head())
# Key columns: 'flag', 'p_value', 'stat', 'std_error' for random effects.

print("\nProviders flagged as significantly different (example):\n", provider_test_re_results_df[provider_test_re_results_df['flag'] != 0].head())