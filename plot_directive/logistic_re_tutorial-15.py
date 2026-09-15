if lre_model_logistic.coefficients_ is not None:
    lre_model_logistic.plot_standardized_measures(
        stdz='indirect',
        measure='ratio',
        level=0.95,
        use_flags=True,
        null='median',
        plot_title="Indirect Standardized Ratio (O/E) with CIs",
    )