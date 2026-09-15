if hasattr(lre_model_logistic, 'model') and lre_model_logistic.model is not None and lre_model_logistic.model.fitted:
    try:
        lre_model_logistic.plot_residuals(
            residual_type="pearson", # or "response"
            title="Pearson Residuals vs. Fitted Probabilities (LRE)"
        )
        plt.show() # Ensure plot is displayed
    except Exception as e:
        print(f"Could not plot residuals: {e}")
else:
    print("Model not fitted successfully. Cannot plot residuals.")