#!/usr/bin/env Rscript
# Reference results for CoxPHSelector, using R's own step() on coxph
# objects (the actual generic-AIC/BIC selection procedure this package
# targets -- see pprof_py/selection/selector.py's module docstring for how
# this was verified, not assumed, including nobs.coxph's n_events
# convention for BIC).
suppressMessages(library(survival))

args <- commandArgs(trailingOnly = TRUE)
base <- if (length(args) >= 1) args[[1]] else "."
datadir <- file.path(base, "data")
resultsdir <- file.path(base, "results")
dir.create(resultsdir, showWarnings = FALSE, recursive = TRUE)

write_history <- function(steps, name) {
  write.csv(do.call(rbind, lapply(steps, as.data.frame)),
            file.path(resultsdir, paste0(name, ".csv")), row.names = FALSE)
}

# step() prints its trace to stdout; to get a machine-readable trajectory
# we instead drive the same search manually one AIC/BIC evaluation at a
# time, calling extractAIC ourselves -- this is a more direct, exactly
# reproducible reference than parsing step()'s trace text, and uses
# nothing but extractAIC()/coxph(), which step() itself is built on.
greedy_search <- function(d, surv_formula_lhs, all_vars, forced_vars, direction, k, weights_col = NULL) {
  fit_one <- function(vars) {
    rhs <- if (length(vars) == 0) "1" else paste(vars, collapse = " + ")
    f <- as.formula(paste(surv_formula_lhs, "~", rhs))
    if (is.null(weights_col)) coxph(f, data = d, ties = "breslow")
    else coxph(f, data = d, ties = "breslow", weights = d[[weights_col]])
  }
  candidates <- setdiff(all_vars, forced_vars)
  current <- if (direction == "backward") all_vars else forced_vars
  current_model <- fit_one(current)
  current_aic <- extractAIC(current_model, k = k)[2]

  rows <- list()
  rows[[1]] <- list(step = 0, action = "start", variable = NA, n_variables = length(current), aic = current_aic)

  step_i <- 0
  repeat {
    step_i <- step_i + 1
    moves <- list()
    if (direction %in% c("forward", "both")) {
      for (v in setdiff(candidates, current)) {
        m <- fit_one(c(current, v))
        moves[[length(moves) + 1]] <- list(action = "add", var = v, vars = c(current, v),
                                            aic = extractAIC(m, k = k)[2], model = m)
      }
    }
    if (direction %in% c("backward", "both")) {
      for (v in setdiff(current, forced_vars)) {
        m <- fit_one(setdiff(current, v))
        moves[[length(moves) + 1]] <- list(action = "remove", var = v, vars = setdiff(current, v),
                                            aic = extractAIC(m, k = k)[2], model = m)
      }
    }
    if (length(moves) == 0) break
    aics <- sapply(moves, function(m) m$aic)
    best <- moves[[which.min(aics)]]
    if (best$aic >= current_aic) break
    current <- best$vars
    current_model <- best$model
    current_aic <- best$aic
    rows[[length(rows) + 1]] <- list(step = step_i, action = best$action, variable = best$var,
                                      n_variables = length(current), aic = current_aic)
  }
  list(rows = rows, final_vars = current, final_model = current_model)
}

cat("== forward, AIC ==\n")
d <- read.csv(file.path(datadir, "selector_test_data.csv"))
res <- greedy_search(d, "Surv(stop, event)", c("x1","x2","x3","x4","x5"), c(), "forward", k = 2)
write_history(res$rows, "forward_aic")
cat("selected:", paste(setdiff(names(coef(res$final_model)), character(0)), collapse=","), "\n")
write.csv(as.data.frame(summary(res$final_model)$coefficients), file.path(resultsdir, "forward_aic_final_coef.csv"))

cat("\n== backward, AIC ==\n")
res <- greedy_search(d, "Surv(stop, event)", c("x1","x2","x3","x4","x5"), c(), "backward", k = 2)
write_history(res$rows, "backward_aic")

cat("\n== both, AIC ==\n")
res <- greedy_search(d, "Surv(stop, event)", c("x1","x2","x3","x4","x5"), c(), "both", k = 2)
write_history(res$rows, "both_aic")

cat("\n== backward, BIC (k=log(nevent)) ==\n")
full_fit <- coxph(Surv(stop, event) ~ x1+x2+x3+x4+x5, data = d, ties = "breslow")
k_bic <- log(nobs(full_fit))
write.csv(data.frame(k_bic = k_bic, nevent = nobs(full_fit)), file.path(resultsdir, "bic_k.csv"), row.names = FALSE)
res <- greedy_search(d, "Surv(stop, event)", c("x1","x2","x3","x4","x5"), c(), "backward", k = k_bic)
write_history(res$rows, "backward_bic")

cat("\n== forward, AIC, with x4 forced ==\n")
res <- greedy_search(d, "Surv(stop, event)", c("x1","x2","x3","x4","x5"), c("x4"), "forward", k = 2)
write_history(res$rows, "forward_aic_forced")

cat("\n== strata + offset + weights, forward AIC ==\n")
d2 <- read.csv(file.path(datadir, "selector_strata_data.csv"))
fit_one2 <- function(vars) {
  rhs <- paste(c(vars, "strata(provider)", "offset(off1)"), collapse = " + ")
  coxph(as.formula(paste("Surv(stop, event) ~", rhs)), data = d2, weights = wt, ties = "breslow")
}
candidates2 <- c("x1", "x2", "x3")
current2 <- character(0)
current_aic2 <- extractAIC(fit_one2(current2), k = 2)[2]
rows2 <- list(list(step = 0, action = "start", variable = NA, n_variables = 0, aic = current_aic2))
step_i <- 0
repeat {
  step_i <- step_i + 1
  remaining <- setdiff(candidates2, current2)
  if (length(remaining) == 0) break
  aics <- sapply(remaining, function(v) extractAIC(fit_one2(c(current2, v)), k = 2)[2])
  best_i <- which.min(aics)
  if (aics[best_i] >= current_aic2) break
  current2 <- c(current2, remaining[best_i])
  current_aic2 <- aics[best_i]
  rows2[[length(rows2) + 1]] <- list(step = step_i, action = "add", variable = remaining[best_i],
                                      n_variables = length(current2), aic = current_aic2)
}
write_history(rows2, "forward_aic_strata_offset_weights")

cat("\nAll selection reference results written to", resultsdir, "\n")
