if lre_model_logistic.coefficients_ is not None:
    try:
        sm_results_lre_log = lre_model_logistic.calculate_standardized_measures(
            stdz='indirect',
            null='median'  # Baseline for random effects: 'median', 'mean', or 0.0
        )
        indirect_sm_lre_log_df = sm_results_lre_log.get('indirect')
        if indirect_sm_lre_log_df is not None:
            print("\nIndirect Standardized Measures (vs median BLUP, first 5):\n", indirect_sm_lre_log_df.head())
        else:
            print("\nIndirect standardized measures not calculated.")
    except Exception as e:
        print(f"Could not calculate standardized measures: {e}")
else:
    print("Model not fitted successfully. Cannot calculate standardized measures.")