if hasattr(lre_model_logistic, 'model') and lre_model_logistic.model is not None and lre_model_logistic.model.fitted:
    try:
        lre_model_logistic.calculate_confidence_intervals(option='SM')
    except NotImplementedError as e:
        print(f"\nNote on CIs for Standardized Measures: {e}")
    except Exception as e:
        print(f"An error occurred trying to call CI for SM: {e}")
else:
    print("Model not fitted. Skipping CIs for Standardized Measures.")