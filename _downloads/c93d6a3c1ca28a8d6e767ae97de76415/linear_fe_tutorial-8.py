# Predict on the training data
data['predicted_los'] = lfe_model.predict(
    X=data,
    x_vars=covariate_vars,
    group_var=group_var
)
print("\nData with predictions (selected columns, first 5 rows):\n", data[[group_var, outcome_var, 'predicted_los']].head())