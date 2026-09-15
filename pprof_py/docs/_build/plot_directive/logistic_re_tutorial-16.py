if hasattr(lre_model_logistic, 'model') and lre_model_logistic.model is not None and lre_model_logistic.model.fitted:
    try:
        lre_model_logistic.plot_coefficient_forest(
            level=0.95,
            plot_title="Forest Plot of Fixed Effect Coefficients (Log-Odds)"
        )
        plt.show() # Ensure plot is displayed
    except Exception as e:
        print(f"Could not plot coefficient forest: {e}")
else:
    print("Model not fitted successfully. Cannot plot coefficient forest.")