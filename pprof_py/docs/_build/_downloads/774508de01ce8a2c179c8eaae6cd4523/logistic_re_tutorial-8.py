if hasattr(lre_model_logistic, 'model') and lre_model_logistic.model is not None and lre_model_logistic.model.fitted:
    # Predict probabilities using only fixed effects for the training data (as an example)
    data_lre['pred_prob_fixed_only'] = lre_model_logistic.predict(
        X=data_lre, # pymer4 predict needs the dataframe
        x_vars=covariate_vars_lre # x_vars are implicitly used from fitted model if None
    )
    print("\nData with predictions (fixed effects only, first 5 rows):\n",
          data_lre[[group_var_lre, outcome_var_lre, 'pred_prob_fixed_only']].head())

    # Fitted values including random effects (BLUPs) are stored in self.fitted_
    if lre_model_logistic.fitted_ is not None:
        data_lre['fitted_prob_with_re'] = lre_model_logistic.fitted_
        print("\nData with fitted probabilities (including RE, first 5 rows):\n",
              data_lre[[group_var_lre, outcome_var_lre, 'fitted_prob_with_re']].head())
    else:
        print("\nFitted probabilities (including RE) not available.")
else:
    print("Model not fitted successfully. Cannot make predictions.")