if hasattr(lre_model_logistic, 'model') and lre_model_logistic.model is not None and lre_model_logistic.model.fitted:
    try:
        re_ci_results = lre_model_logistic.calculate_confidence_intervals(
            option='alpha', # 'alpha' for random effects (BLUPs)
            level=0.95
        )
        re_ci_df = re_ci_results.get('alpha_ci')
        if re_ci_df is not None:
            print("\nApproximate CIs for Random Effects (BLUPs, first 5):\n", re_ci_df.head())
        else:
            print("\nRandom effect CIs not available.")
    except Exception as e:
        print(f"Could not calculate CIs for random effects: {e}")
else:
    print("Model not fitted successfully. Cannot calculate CIs for random effects.")