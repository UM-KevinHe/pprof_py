aic_lfe = lfe_model_logistic.aic_
bic_lfe = lfe_model_logistic.bic_
auc_lfe = lfe_model_logistic.auc_

print(f"\nAIC: {aic_lfe:.2f}")
print(f"BIC: {bic_lfe:.2f}")
print(f"AUC: {auc_lfe:.4f}")