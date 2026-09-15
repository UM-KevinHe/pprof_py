# Calculate indirect standardized differences against the median provider performance
sm_results = lfe_model.calculate_standardized_measures(
    stdz='indirect',
    null='median'
)

indirect_diff_df = sm_results['indirect']
print("\nIndirect Standardized Differences (vs median gamma, first 5):\n", indirect_diff_df.head())