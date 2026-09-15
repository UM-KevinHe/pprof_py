lfe_model_logistic.plot_funnel(
    test_method='poibin_exact', # 'score' or 'poibin_exact' for control limits
    null='median',
    target=1.0,          # Target O/E ratio
    alpha=[0.05, 0.01],  # For 95% and 99% control limits
    plot_title="Funnel Plot: Indirect Standardized Ratios (O/E)",
    ylab="Indirect Standardized Ratio (O/E)"
)