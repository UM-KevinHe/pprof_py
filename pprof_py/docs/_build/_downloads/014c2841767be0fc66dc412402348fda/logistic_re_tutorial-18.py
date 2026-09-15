if hasattr(lre_model_logistic, 'model') and lre_model_logistic.model is not None and lre_model_logistic.model.fitted:
    try:
        lre_model_logistic.plot_qq()
    except NotImplementedError as e:
        print(f"\nNote on Q-Q Plot: {e}")
    except Exception as e:
        print(f"An error occurred trying to generate Q-Q plot: {e}")
else:
    print("Model not fitted. Skipping Q-Q plot.")