lfe_model_logistic.plot_standardized_measures(
    stdz='indirect',
    measure='ratio',
    level=0.95,
    test_method='poibin_exact', # Method for underlying gamma CIs
    use_flags=True,
    null='median',
    plot_title="Provider Standardized Ratios (Indirect O/E)",
    figure_size=(8, 6)
)