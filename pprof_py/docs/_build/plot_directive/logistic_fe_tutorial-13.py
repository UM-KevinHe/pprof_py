lfe_model_logistic.plot_provider_effects(
    level=0.95,
    test_method='poibin_exact', # Method for CI calculation
    use_flags=True,
    null='median',
    plot_title="Provider Effects: Adjusted Log-Odds (Gamma)",
    figure_size=(8, 6)
)