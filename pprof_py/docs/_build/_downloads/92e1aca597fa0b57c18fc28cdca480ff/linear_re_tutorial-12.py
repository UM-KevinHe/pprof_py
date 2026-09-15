sm_ci_re_results = lre_model.calculate_confidence_intervals(
    option='SM',             # Standardized Measure
    stdz='indirect',
    null='median',           # Baseline for the difference
    level=0.95,
    alternative='two_sided'
)
indirect_sm_re_ci_df = sm_ci_re_results['indirect_ci']
print("\nCIs for Indirect Standardized Difference (first 5):\n", indirect_sm_re_ci_df.head())