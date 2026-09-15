if lre_model_logistic.coefficients_ is not None:
    lre_model_logistic.plot_funnel(
        test_method='wald',
        null='median',
        target=1.0,
        alpha=[0.05, 0.01],
        plot_title="Funnel Plot: Indirect Standardized Ratio (O/E)",
    )