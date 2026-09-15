# Simulate data for demonstration
np.random.seed(123)
n_providers = 100
n_patients_per_provider = np.random.randint(25, 80, n_providers)
n_total_patients = np.sum(n_patients_per_provider)

provider_ids_list = []
for i, count in enumerate(n_patients_per_provider):
    provider_ids_list.extend([f"Hospital_{i+1}"] * count)

data_re = pd.DataFrame({
    'patient_id': range(n_total_patients),
    'provider_id': provider_ids_list,
    'age': np.random.normal(55, 12, n_total_patients),
    'comorbidity_index': np.random.gamma(2, 1, n_total_patients),
    'emergency_admission': np.random.choice([0, 1], n_total_patients, p=[0.6, 0.4])
})

# Simulate true random provider effects (mean zero) and outcome (e.g., cost)
true_random_effects_map = {f"Hospital_{i+1}": np.random.normal(0, 150) for i in range(n_providers)} # Provider-specific deviations
data_re['true_random_effect'] = data_re['provider_id'].map(true_random_effects_map)

data_re['treatment_cost'] = (
    1000  # Base cost
    + 10 * data_re['age']
    + 50 * data_re['comorbidity_index']
    + 300 * data_re['emergency_admission']
    + data_re['true_random_effect']  # Add provider random effect
    + np.random.normal(0, 200, n_total_patients) # Random error
)
data_re['treatment_cost'] = np.maximum(100, data_re['treatment_cost']) # Ensure positive cost

# Define variable names for the model
outcome_var_re = 'treatment_cost'
covariate_vars_re = ['age', 'comorbidity_index', 'emergency_admission']
group_var_re = 'provider_id'

print("Sample data for Random Effect Model (first 5 rows):")
print(data_re.head())
print(f"\nOutcome: {outcome_var_re}, Covariates: {covariate_vars_re}, Group: {group_var_re}")