# Predict using only fixed effects
data_re['predicted_cost_fixed_only'] = lre_model.predict(
    X=data_re,
    x_vars=covariate_vars_re
)
print("\nData with predictions (fixed effects only, selected columns, first 5 rows):\n", data_re[[group_var_re, outcome_var_re, 'predicted_cost_fixed_only']].head())

# For predictions including random effects (Best Linear Unbiased Predictors - BLUPs),
# you would typically access the full statsmodels result object:
# fitted_values_with_re = lre_model.result.fittedvalues
# print("\nFitted values including random effects (first 5):\n", fitted_values_with_re.head())