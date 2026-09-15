# Simulate data for demonstration
np.random.seed(789)
n_providers_log_re = 25
n_patients_per_provider_log_re = np.random.randint(40, 120, n_providers_log_re)
n_total_patients_log_re = np.sum(n_patients_per_provider_log_re)

provider_ids_log_re = []
for i, count in enumerate(n_patients_per_provider_log_re):
    provider_ids_log_re.extend([f"Hospital_RE_{i+1}"] * count) # String IDs for provider groups

data_lre = pd.DataFrame({
    'patient_id': range(n_total_patients_log_re),
    'provider_id': provider_ids_log_re,
    'age_yrs': np.random.normal(60, 10, n_total_patients_log_re),
    'severity_score': np.random.gamma(2.5, 1.2, n_total_patients_log_re),
    'urgent_case': np.random.choice([0, 1], n_total_patients_log_re, p=[0.7, 0.3])
})

# Simulate true random provider effects (log-odds adjustments) and binary outcome
true_re_sd_log = 0.6
provider_log_odds_re_map = {
    f"Hospital_RE_{i+1}": np.random.normal(0, true_re_sd_log) for i in range(n_providers_log_re)
}
data_lre['true_provider_re_log_odds'] = data_lre['provider_id'].map(provider_log_odds_re_map)

# Log-odds calculation
log_odds_re = (
    -2.0  # Base log-odds (Intercept)
    + 0.03 * (data_lre['age_yrs'] - 60)
    + 0.25 * data_lre['severity_score']
    + 0.5 * data_lre['urgent_case']
    + data_lre['true_provider_re_log_odds'] # Add random effect
)

probabilities_re = 1 / (1 + np.exp(-log_odds_re))
data_lre['complication'] = (np.random.rand(n_total_patients_log_re) < probabilities_re).astype(int)

outcome_var_lre = 'complication'
covariate_vars_lre = ['age_yrs', 'severity_score', 'urgent_case']
group_var_lre = 'provider_id'

print("Sample data for Logistic Random Effect Model (first 5 rows):")
print(data_lre.head())
print(f"\nOutcome: {outcome_var_lre}, Covariates: {covariate_vars_lre}, Group: {group_var_lre}")
print(f"\nOverall event rate: {data_lre[outcome_var_lre].mean():.3f}")