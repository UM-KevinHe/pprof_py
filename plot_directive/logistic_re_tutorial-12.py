if lre_model_logistic.coefficients_ is not None:
    try:
        sm_ci_results = lre_model_logistic.calculate_confidence_intervals(
            option='SM',
            stdz='indirect',
            null='median',
            measure=['ratio', 'rate'],
            level=0.95
        )
        if 'indirect_ratio' in sm_ci_results:
            print("\nCIs for Indirect Standardized Ratio (first 5):")
            print(sm_ci_results['indirect_ratio'].head())
        if 'indirect_rate' in sm_ci_results:
            print("\nCIs for Indirect Standardized Rate (first 5):")
            print(sm_ci_results['indirect_rate'].head())
    except Exception as e:
        print(f"Could not calculate CIs for standardized measures: {e}")
else:
    print("Model not fitted. Skipping CIs for Standardized Measures.")