#!/usr/bin/env Rscript
# Fits coxph(..., cluster=) with strata AND left truncation together --
# the one combination statsmodels.PHReg could not confirm during
# development (returned NaN; see docs/R_COMPATIBILITY.md's robust-variance
# section) -- and writes results for tests/test_phase4_r_comparison.py.

suppressMessages(library(survival))

args <- commandArgs(trailingOnly = TRUE)
base <- if (length(args) >= 1) args[[1]] else "."
datadir <- file.path(base, "data")
resultsdir <- file.path(base, "results")
dir.create(resultsdir, showWarnings = FALSE, recursive = TRUE)

d <- read.csv(file.path(datadir, "robust_strata_truncation.csv"))

cat("== robust + strata + left truncation + clustering ==\n")
fit <- coxph(Surv(start, stop, event) ~ x1 + x2 + strata(strata), data = d, cluster = cluster)
print(summary(fit))

s <- summary(fit)
coefs <- as.data.frame(s$coefficients)
out <- data.frame(
  term = rownames(coefs),
  coef = coefs[["coef"]],
  se_robust = coefs[, grep("robust se", colnames(coefs), fixed = TRUE)]
)
write.csv(out, file.path(resultsdir, "robust_strata_truncation_coefficients.csv"), row.names = FALSE)
write.csv(data.frame(se_naive = sqrt(diag(fit$naive.var))),
          file.path(resultsdir, "robust_strata_truncation_naive_se.csv"), row.names = FALSE)

cat("\nAll Phase 4 robust-variance R reference results written to", resultsdir, "\n")
