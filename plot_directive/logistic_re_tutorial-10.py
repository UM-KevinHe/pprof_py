if lre_model_logistic.coefficients_ is not None:
    try:
        provider_test_lre_log_df = lre_model_logistic.test(
            null=0.0, # Test BLUPs against 0 (or 'median', 'mean')
            level=0.95,
            alternative='two_sided'
        )
        print("\nProvider Random Effect Test Results (vs null=0, first 5):\n", provider_test_lre_log_df.head())
        print("\nProviders flagged as significantly different (example):\n",
              provider_test_lre_log_df[provider_test_lre_log_df['flag'] != 0].head())
    except Exception as e:
        print(f"Could not perform hypothesis tests for providers: {e}")
else:
    print("Model not fitted successfully. Cannot perform provider tests.")