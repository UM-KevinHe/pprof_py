#!/usr/bin/env Rscript
# Runs R's own tmerge() on the Phase 4 time-dependent-covariate dataset
# and fits coxph() on the result, writing both the merged dataset itself
# (for a direct row-for-row comparison against data/timedep.py::tmerge)
# and the fitted coefficients to r_reference/results/*.csv.

suppressMessages(library(survival))

args <- commandArgs(trailingOnly = TRUE)
base <- if (length(args) >= 1) args[[1]] else "."
datadir <- file.path(base, "data")
resultsdir <- file.path(base, "results")
dir.create(resultsdir, showWarnings = FALSE, recursive = TRUE)

skeleton <- read.csv(file.path(datadir, "timedep_skeleton.csv"))
updates <- read.csv(file.path(datadir, "timedep_updates.csv"))

cat("== tmerge: skeleton + tdc ==\n")
test1 <- tmerge(skeleton, skeleton, id = id, death = event(stop, death))
test2 <- tmerge(test1, updates, id = id, treated = tdc(time))

write.csv(
  test2[, c("id", "tstart", "tstop", "treated", "death")],
  file.path(resultsdir, "timedep_merged.csv"),
  row.names = FALSE
)

fit <- coxph(Surv(tstart, tstop, death) ~ treated, data = test2)
print(summary(fit))

s <- summary(fit)
coefs <- as.data.frame(s$coefficients)
ci <- as.data.frame(s$conf.int)
out <- data.frame(
  term = rownames(coefs),
  coef = coefs[["coef"]],
  se_coef = coefs[, grep("^se\\(coef\\)$", colnames(coefs))],
  lower_95 = ci[, 3],
  upper_95 = ci[, 4]
)
write.csv(out, file.path(resultsdir, "timedep_coefficients.csv"), row.names = FALSE)
meta <- data.frame(loglik_beta = fit$loglik[2], loglik_null = fit$loglik[1], n = fit$n, nevent = fit$nevent)
write.csv(meta, file.path(resultsdir, "timedep_meta.csv"), row.names = FALSE)

cat("\nAll Phase 4 time-dependent-covariate R reference results written to", resultsdir, "\n")
