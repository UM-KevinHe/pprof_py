if lre_model_logistic.coefficients_ is not None:
    lre_model_logistic.plot_coefficient_forest(
        plot_title="Forest Plot of Covariate Coefficients (Log-Odds)"
    )