if hasattr(lre_model_logistic, 'model') and lre_model_logistic.model is not None and lre_model_logistic.model.fitted:
    aic_lre_log = lre_model_logistic.aic_
    bic_lre_log = lre_model_logistic.bic_
    loglik_lre_log = lre_model_logistic.loglike_

    print(f"\nAIC: {aic_lre_log:.2f}" if aic_lre_log is not None else "\nAIC: Not available")
    print(f"BIC: {bic_lre_log:.2f}" if bic_lre_log is not None else "BIC: Not available")
    print(f"Log-Likelihood: {loglik_lre_log:.2f}" if loglik_lre_log is not None else "Log-Likelihood: Not available")
else:
    print("Model not fitted successfully. Cannot display fit statistics.")