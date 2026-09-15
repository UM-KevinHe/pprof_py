lfe_model_logistic.plot_provider_effects(
    level=0.95,
    test_method='exact', # Method for CI calculation: 'wald', 'score', or 'exact'
    use_flags=True,
    null='median',
    title="Provider Effects: Adjusted Log-Odds (Gamma)",
    figure_size=(8, 6)
)