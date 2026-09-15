aic_re = lre_model.aic_
bic_re = lre_model.bic_
sigma_re = lre_model.sigma_                # Residual standard deviation (sigma_e)
re_sd = lre_model.random_effect_sd_        # Dict: group_var -> sigma_u
loglik = lre_model.loglike_                # Log-likelihood at convergence

print(f"\nAIC: {aic_re:.2f}")
print(f"BIC: {bic_re:.2f}")
print(f"Estimated Sigma (Residual Std Dev): {sigma_re:.4f}")
print(f"Random-effect SD per group variable: {re_sd}")
print(f"Log-likelihood: {loglik:.2f}")