# Initialize the model
lre_model_logistic = LogisticRandomEffectModel()

# Fit the model
# The model internally constructs a formula like 'y_var ~ x_var1 + (1 | group_var)'
lre_model_logistic.fit(
     X=data_lre,
     y_var=outcome_var_lre,
     x_vars=covariate_vars_lre,
     group_var=group_var_lre
 )
 print("Model fitting complete.")