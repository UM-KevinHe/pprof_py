# Initialize the model
lre_model = LinearRandomEffectModel(verbose=False)

# Fit the model
# group_var specifies the random intercept grouping variable.
lre_model.fit(
    data_re,                   # Your DataFrame (positional)
    y_var=outcome_var_re,      # Name of the outcome column
    x_vars=covariate_vars_re,  # List of covariate column names (fixed effects)
    group_var=group_var_re,    # Name of the provider ID column (for random effects)
    reml=True                  # Use REML for estimation (common for variance components)
)
print(f"Model converged: {lre_model.converged_}")