if hasattr(lre_model_logistic, 'model') and lre_model_logistic.model is not None and lre_model_logistic.model.fitted:
    try:
        lre_model_logistic.plot_provider_effects(
            level=0.95,
            use_flags=True,
            null=0.0, # Baseline for flagging BLUPs
            plot_title="Provider Random Effects (BLUPs on Log-Odds Scale)",
            figure_size=(10, 8)
        )
        plt.show() # Ensure plot is displayed
    except Exception as e:
        print(f"Could not plot provider effects: {e}")
else:
    print("Model not fitted successfully. Cannot plot provider effects.")