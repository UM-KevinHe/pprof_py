sm_ci_results = lfe_model.calculate_confidence_intervals(
    option='SM',
    stdz='indirect',
    null='median',
    level=0.95,
    alternative='two_sided'
)
indirect_sm_ci_df = sm_ci_results['indirect_ci']
print("\nCIs for Indirect Standardized Difference (first 5):\n", indirect_sm_ci_df.head())