if hasattr(lre_model_logistic, 'model') and lre_model_logistic.model is not None and lre_model_logistic.model.fitted:
    try:
        lre_model_logistic.plot_standardized_measures()
    except NotImplementedError as e:
        print(f"\nNote on Plotting Standardized Measures: {e}")
    except Exception as e:
        print(f"An error occurred trying to plot standardized measures: {e}")
else:
    print("Model not fitted. Skipping plot of Standardized Measures.")