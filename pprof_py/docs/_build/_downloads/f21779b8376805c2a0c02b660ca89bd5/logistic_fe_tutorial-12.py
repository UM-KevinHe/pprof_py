sm_ci_lfe_results = lfe_model_logistic.calculate_confidence_intervals(
    option='SM',
    stdz='indirect',
    measure=['ratio', 'rate'], # Can be 'ratio', 'rate', or both
    null='median',
    level=0.95,
    test_method='exact', # Method used for underlying gamma CIs
    alternative='two_sided'
)
indirect_ratio_ci_lfe_df = sm_ci_lfe_results.get('indirect_ratio')
indirect_rate_ci_lfe_df = sm_ci_lfe_results.get('indirect_rate')

if indirect_ratio_ci_lfe_df is not None:
    print("\nCIs for Indirect Standardized Ratio (first 5):\n", indirect_ratio_ci_lfe_df.head())
if indirect_rate_ci_lfe_df is not None:
    print("\nCIs for Indirect Standardized Rate (first 5):\n", indirect_rate_ci_lfe_df.head())