# Calculate indirect standardized ratios and rates against the median provider performance
sm_results_lfe = lfe_model_logistic.calculate_standardized_measures(
    stdz='indirect',  # or 'direct', or ['indirect', 'direct']
    null='median'     # Baseline: 'median' or 'mean' of estimated gammas, or a specific float value for gamma_null
)

indirect_sm_lfe_df = sm_results_lfe['indirect']
print("\nIndirect Standardized Measures (vs median gamma, first 5):\n", indirect_sm_lfe_df.head())
# Key columns: 'indirect_ratio' (O/E), 'indirect_rate' (Adjusted Rate), 'observed', 'expected'.
# An indirect_ratio > 1 means more events observed than expected at baseline.