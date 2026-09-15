# Predict using only fixed effects
data_re['predicted_cost_fixed_only'] = lre_model.predict(
    data_re,
    x_vars=covariate_vars_re
)
print("\nPredictions (fixed effects only, first 5 rows):")
print(data_re[[group_var_re, outcome_var_re, 'predicted_cost_fixed_only']].head())

# Predictions including random effects (BLUPs)
data_re['predicted_cost_with_re'] = lre_model.predict(
    data_re,
    x_vars=covariate_vars_re,
    group_var=group_var_re,
    use_re=True
)
print("\nWith BLUPs (first 5 rows):")
print(data_re[[group_var_re, outcome_var_re, 'predicted_cost_fixed_only', 'predicted_cost_with_re']].head())