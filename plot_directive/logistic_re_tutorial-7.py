if lre_model_logistic.coefficients_ is not None:
    try:
        fixed_effects_summary_lre_df = lre_model_logistic.summary()
        print("\nSummary of Fixed Effects (Betas):\n", fixed_effects_summary_lre_df)
    except Exception as e:
        print(f"Could not generate fixed effects summary: {e}")
else:
    print("Model not fitted successfully. Cannot display fixed effects summary.")