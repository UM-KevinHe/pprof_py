# Initialize the model
# gamma_var_option:
#  'complete': Considers uncertainty in covariate effects when calculating variance of provider effects. More accurate but slower.
#  'simplified': A faster approximation for provider effect variance (sigma^2 / n_i).
lfe_model = LinearFixedEffectModel(gamma_var_option='complete')

# Fit the model
lfe_model.fit(
    X=data,                 # Your DataFrame
    y_var=outcome_var,      # Name of the outcome column
    x_vars=covariate_vars,  # List of covariate column names
    group_var=group_var     # Name of the provider ID column
)

print("\nModel fitting complete.")