#!/usr/bin/env Rscript
# Fits cause-specific and Fine-Gray competing-risks models with R's own
# survival::finegray() + coxph(), on the Phase 4 synthetic datasets, and
# writes results to r_reference/results/*.csv for
# tests/test_phase4_r_comparison.py to compare against.
#
# Writes out finegray()'s own TRANSFORMED dataset (fgstart/fgstop/fgstatus/
# fgwt), not just the final fitted coefficients, so
# algorithms/survival/finegray.py::finegray_transform can be checked directly
# against R's own transform on a dataset larger than the hand-worked
# examples in tests/test_finegray_transform.py -- those already validate
# exactly against R's OWN bundled test cases, but this is R actually
# invoked on fresh, larger data, which is a different (complementary)
# kind of confidence.

suppressMessages(library(survival))

args <- commandArgs(trailingOnly = TRUE)
base <- if (length(args) >= 1) args[[1]] else "."
datadir <- file.path(base, "data")
resultsdir <- file.path(base, "results")
dir.create(resultsdir, showWarnings = FALSE, recursive = TRUE)

write_coef_table <- function(fit, name) {
  s <- summary(fit)
  coefs <- as.data.frame(s$coefficients)
  ci <- as.data.frame(s$conf.int)
  out <- data.frame(
    term = rownames(coefs),
    coef = coefs[["coef"]],
    se_coef = coefs[, grep("^se\\(coef\\)$", colnames(coefs))],
    z = coefs[[grep("^z$", colnames(coefs), value = TRUE)[1]]],
    p = coefs[, grep("^Pr", colnames(coefs))],
    lower_95 = ci[, 3],
    upper_95 = ci[, 4]
  )
  write.csv(out, file.path(resultsdir, paste0(name, "_coefficients.csv")), row.names = FALSE)
  meta <- data.frame(loglik_beta = fit$loglik[2], loglik_null = fit$loglik[1], n = fit$n, nevent = fit$nevent)
  write.csv(meta, file.path(resultsdir, paste0(name, "_meta.csv")), row.names = FALSE)
}

run_one <- function(csvname, prefix, has_truncation) {
  d <- read.csv(file.path(datadir, csvname))
  d$event_f <- factor(d$event, 0:2, c("censor", "cause1", "cause2"))

  cat("\n==", prefix, ": cause-specific (cause 1) ==\n")
  if (has_truncation) {
    cs1 <- coxph(Surv(start, stop, event == 1) ~ x1 + x2, data = d)
    cs2 <- coxph(Surv(start, stop, event == 2) ~ x1 + x2, data = d)
  } else {
    cs1 <- coxph(Surv(stop, event == 1) ~ x1 + x2, data = d)
    cs2 <- coxph(Surv(stop, event == 2) ~ x1 + x2, data = d)
  }
  print(summary(cs1))
  write_coef_table(cs1, paste0(prefix, "_cause1"))
  write_coef_table(cs2, paste0(prefix, "_cause2"))

  cat("\n==", prefix, ": finegray() transform + weighted coxph (cause 1) ==\n")
  if (has_truncation) {
    fg_data <- finegray(Surv(start, stop, event_f) ~ ., id = id, data = d)
  } else {
    fg_data <- finegray(Surv(stop, event_f) ~ ., id = id, data = d)
  }
  # the transform itself, for a direct comparison against
  # algorithms/survival/finegray.py::finegray_transform's own output
  write.csv(
    fg_data[, c("id", "fgstart", "fgstop", "fgstatus", "fgwt")],
    file.path(resultsdir, paste0(prefix, "_finegray_transform.csv")),
    row.names = FALSE
  )

  fg_fit <- coxph(Surv(fgstart, fgstop, fgstatus) ~ x1 + x2, data = fg_data, weight = fgwt, cluster = id)
  print(summary(fg_fit))
  write_coef_table(fg_fit, paste0(prefix, "_finegray_fit"))
}

run_one("competing_risks_simple.csv", "cr_simple", has_truncation = FALSE)
run_one("competing_risks_truncated.csv", "cr_truncated", has_truncation = TRUE)

cat("\nAll Phase 4 competing-risks R reference results written to", resultsdir, "\n")
