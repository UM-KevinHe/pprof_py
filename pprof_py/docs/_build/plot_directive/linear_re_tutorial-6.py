aic_re = lre_model.aic_
bic_re = lre_model.bic_
sigma_re = lre_model.sigma_ # Estimated residual standard deviation (sigma_e)

print(f"\nAIC: {aic_re:.2f}")
print(f"BIC: {bic_re:.2f}")
print(f"Estimated Sigma (Residual Std Dev): {sigma_re:.4f}")