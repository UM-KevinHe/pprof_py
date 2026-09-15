covariate_summary_lfe_df = lfe_model_logistic.summary(test_method='wald') # or 'lr', 'score'
print("\nSummary of Covariate Effects (Betas - Wald test):\n", covariate_summary_lfe_df)