# Initialize the model
# You can specify algorithm='Serbin' (default) or 'Ban'
# Other parameters control data preparation steps if use_dataprep=True (default)
lfe_model_logistic = LogisticFixedEffectModel(
    algorithm='Serbin',
    screen_providers=True, # Example: ensure providers meet minimum size
    cutoff=10              # Example: minimum 10 patients per provider
)

# Fit the model
lfe_model_logistic.fit(
    X=data_lfe,
    y_var=outcome_var_lfe,
    x_vars=covariate_vars_lfe,
    group_var=group_var_lfe,
    max_iter=1000, # Max iterations for the optimization algorithm
    tol=1e-6       # Tolerance for convergence
)
# The "Model fitting complete." message comes from the .fit() method.