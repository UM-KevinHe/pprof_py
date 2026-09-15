if lre_model_logistic.coefficients_ is not None:
    lre_model_logistic.plot_provider_effects(
        level=0.95,
        use_flags=True,
        null='median',
        plot_title="Provider Random Effects (BLUPs, Log-Odds Scale)",
    )