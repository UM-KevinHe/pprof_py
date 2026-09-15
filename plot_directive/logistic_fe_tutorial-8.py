# Predict on the training data
data_lfe['predicted_prob_readmission'] = lfe_model_logistic.predict(
    X=data_lfe,
    x_vars=covariate_vars_lfe,
    group_var=group_var_lfe
)
print("\nData with predicted probabilities (selected columns, first 5 rows):\n", data_lfe[[group_var_lfe, outcome_var_lfe, 'predicted_prob_readmission']].head())