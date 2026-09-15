# Simulate data for demonstration
np.random.seed(42)
n_providers = 100
n_patients_per_provider = np.random.randint(30, 100, n_providers) # Varying provider sizes
n_total_patients = np.sum(n_patients_per_provider)

provider_ids = []
for i, count in enumerate(n_patients_per_provider):
    provider_ids.extend([f"Provider_{i+1}"] * count)

data = pd.DataFrame({
    'patient_id': range(n_total_patients),
    'provider_id': provider_ids,
    'age': np.random.normal(60, 10, n_total_patients),
    'severity_score': np.random.rand(n_total_patients) * 5,
    'is_urgent': np.random.choice([0, 1], n_total_patients, p=[0.7, 0.3])
})

# Simulate true provider effects and outcome (e.g., length of stay in days)
provider_effect_map = {f"Provider_{i+1}": np.random.normal(0, 2) for i in range(n_providers)}
data['true_provider_effect'] = data['provider_id'].map(provider_effect_map)

data['length_of_stay'] = (
    5  # Base length of stay
    + 0.05 * data['age']
    + 0.5 * data['severity_score']
    + 1.5 * data['is_urgent']
    + data['true_provider_effect']
    + np.random.normal(0, 1.5, n_total_patients) # Random error
)
data['length_of_stay'] = np.maximum(1, data['length_of_stay']) # Ensure positive LOS

# Define variable names for the model
outcome_var = 'length_of_stay'
covariate_vars = ['age', 'severity_score', 'is_urgent']
group_var = 'provider_id'

print("Sample data (first 5 rows):")
print(data.head())
print(f"\nOutcome: {outcome_var}, Covariates: {covariate_vars}, Group: {group_var}")