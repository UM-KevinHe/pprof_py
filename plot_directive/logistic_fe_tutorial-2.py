# Simulate data for demonstration
np.random.seed(420)
n_providers_logistic = 100
n_patients_per_provider_logistic = np.random.randint(50, 150, n_providers_logistic)
n_total_patients_logistic = np.sum(n_patients_per_provider_logistic)

provider_ids_logistic = []
for i, count in enumerate(n_patients_per_provider_logistic):
    provider_ids_logistic.extend([f"Clinic_{i+1}"] * count)

data_lfe = pd.DataFrame({
    'patient_id': range(n_total_patients_logistic),
    'provider_id': provider_ids_logistic,
    'age_patient': np.random.normal(65, 8, n_total_patients_logistic),
    'chronic_conditions': np.random.randint(0, 4, n_total_patients_logistic),
    'prior_admission': np.random.choice([0, 1], n_total_patients_logistic, p=[0.8, 0.2])
})

# Simulate true provider effects (log-odds adjustments) and binary outcome (e.g., 30-day readmission)
provider_log_odds_effect_map = {f"Clinic_{i+1}": np.random.normal(0, 0.5) for i in range(n_providers_logistic)}
data_lfe['true_provider_log_odds_effect'] = data_lfe['provider_id'].map(provider_log_odds_effect_map)

# Log-odds calculation
log_odds = (
    -2.5  # Base log-odds
    + 0.02 * (data_lfe['age_patient'] - 65) # Centering age
    + 0.3 * data_lfe['chronic_conditions']
    + 0.6 * data_lfe['prior_admission']
    + data_lfe['true_provider_log_odds_effect']
)

# Convert log-odds to probability
probabilities = 1 / (1 + np.exp(-log_odds))

# Simulate binary outcome
data_lfe['readmitted_30day'] = (np.random.rand(n_total_patients_logistic) < probabilities).astype(int)

# Define variable names for the model
outcome_var_lfe = 'readmitted_30day'
covariate_vars_lfe = ['age_patient', 'chronic_conditions', 'prior_admission']
group_var_lfe = 'provider_id'

print("Sample data for Logistic Fixed Effect Model (first 5 rows):")
print(data_lfe.head())
print(f"\nOutcome: {outcome_var_lfe}, Covariates: {covariate_vars_lfe}, Group: {group_var_lfe}")
print(f"\nOverall event rate: {data_lfe[outcome_var_lfe].mean():.3f}")