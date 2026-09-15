#!/usr/bin/env Rscript
# Fits glmnet(family="cox") on the shared synthetic datasets in
# r_reference/data/ and writes results to r_reference/results/*.csv for
# the Python PenalizedCoxPH/PenalizedCoxPHCV test suite to compare
# against -- the actual ground truth, not a hand-derived expectation,
# following exactly the same philosophy as run_all.R and run_selection.R.
#
# glmnet's Cox family only implements Breslow ties (confirmed by reading
# R/coxnet.deviance.R directly -- see docs/R_COMPATIBILITY.md, Phase 3
# section), so every fit here uses Breslow; Efron-tie penalized fits are
# validated separately, by self-consistency against this package's own
# (R-validated) unpenalized CoxPH(ties="efron") -- see
# tests/test_penalized_self_consistency.py.

suppressMessages(library(survival))
suppressMessages(library(glmnet))

args <- commandArgs(trailingOnly = TRUE)
base <- if (length(args) >= 1) args[[1]] else "."
datadir <- file.path(base, "data")
resultsdir <- file.path(base, "results")
dir.create(resultsdir, showWarnings = FALSE, recursive = TRUE)

write_path <- function(fit, name) {
  # coef(fit) is p x nlambda (sparse); transpose to nlambda x p, matching
  # this package's PenalizedCoxPH.coef_path_ row-per-lambda convention.
  coef_mat <- as.matrix(t(as.matrix(coef(fit))))
  write.csv(coef_mat, file.path(resultsdir, paste0(name, "_coef_path.csv")), row.names = FALSE)
  write.csv(data.frame(lambda = fit$lambda, dev_ratio = fit$dev.ratio, df = fit$df),
            file.path(resultsdir, paste0(name, "_path_meta.csv")), row.names = FALSE)
}

fit_and_write <- function(X, y, name, alpha, standardize = TRUE, penalty.factor = NULL,
                           weights = NULL, offset = NULL) {
  pf <- if (is.null(penalty.factor)) rep(1, ncol(X)) else penalty.factor
  fit <- glmnet(X, y, family = "cox", alpha = alpha, standardize = standardize,
                penalty.factor = pf, weights = weights, offset = offset,
                thresh = 1e-12, maxit = 1e6)
  write_path(fit, name)
  invisible(fit)
}

# ---------------------------------------------------------- penalized_wide
d <- read.csv(file.path(datadir, "penalized_wide.csv"))
xcols <- grep("^x", names(d), value = TRUE)
X <- as.matrix(d[, xcols])
y <- Surv(d$stop, d$event)

cat("== penalized_wide: lasso (alpha=1) ==\n")
fit_and_write(X, y, "penalized_wide_lasso", alpha = 1.0)

cat("== penalized_wide: ridge (alpha=0) ==\n")
fit_and_write(X, y, "penalized_wide_ridge", alpha = 0.0)

cat("== penalized_wide: elastic net (alpha=0.5) ==\n")
fit_and_write(X, y, "penalized_wide_enet", alpha = 0.5)

cat("== penalized_wide: lasso, standardize=FALSE ==\n")
fit_and_write(X, y, "penalized_wide_lasso_nostd", alpha = 1.0, standardize = FALSE)

cat("== penalized_wide: lasso, first 3 vars unpenalized ==\n")
pf <- rep(1, ncol(X)); pf[1:3] <- 0
fit_and_write(X, y, "penalized_wide_lasso_unpen", alpha = 1.0, penalty.factor = pf)

cat("== penalized_wide: cv.glmnet (alpha=1, explicit foldid) ==\n")
foldid <- read.csv(file.path(datadir, "penalized_wide_foldid.csv"))$fold_id
cvfit <- cv.glmnet(X, y, family = "cox", alpha = 1.0, foldid = foldid,
                    thresh = 1e-12, maxit = 1e6)
write.csv(
  data.frame(lambda = cvfit$lambda, cvm = cvfit$cvm, cvsd = cvfit$cvsd,
             cvup = cvfit$cvup, cvlo = cvfit$cvlo, nzero = cvfit$nzero),
  file.path(resultsdir, "penalized_wide_cv.csv"), row.names = FALSE
)
write.csv(data.frame(lambda_min = cvfit$lambda.min, lambda_1se = cvfit$lambda.1se),
          file.path(resultsdir, "penalized_wide_cv_selected.csv"), row.names = FALSE)
coef_min <- as.matrix(coef(cvfit, s = "lambda.min"))
write.csv(data.frame(term = rownames(coef_min), coef = coef_min[, 1]),
          file.path(resultsdir, "penalized_wide_cv_coef_min.csv"), row.names = FALSE)

# ------------------------------------------------------------------ strata
cat("\n== strata ==\n")
d <- read.csv(file.path(datadir, "strata.csv"))
X <- as.matrix(d[, c("x1", "x2")])
y <- stratifySurv(Surv(d$stop, d$event), d$provider)
fit_and_write(X, y, "penalized_strata", alpha = 1.0)

# ------------------------------------------------------------------ offset
cat("\n== offset ==\n")
d <- read.csv(file.path(datadir, "offset.csv"))
X <- as.matrix(d[, c("x1", "x2")])
y <- Surv(d$stop, d$event)
fit_and_write(X, y, "penalized_offset", alpha = 1.0, offset = d$log_exposure)

# ----------------------------------------------------------------- weights
cat("\n== weights ==\n")
d <- read.csv(file.path(datadir, "weights.csv"))
X <- as.matrix(d[, c("x1", "x2")])
y <- Surv(d$stop, d$event)
fit_and_write(X, y, "penalized_weights", alpha = 1.0, weights = d$weight)

# ----------------------------------------------------------- left_truncation
cat("\n== left_truncation ==\n")
d <- read.csv(file.path(datadir, "left_truncation.csv"))
X <- as.matrix(d[, c("x1", "x2")])
y <- Surv(d$start, d$stop, d$event)
fit_and_write(X, y, "penalized_left_truncation", alpha = 1.0)

# ------------------------------------------------------------------ combined
cat("\n== combined (strata + offset + weights + left truncation) ==\n")
d <- read.csv(file.path(datadir, "combined.csv"))
X <- as.matrix(d[, c("x1", "x2", "x3")])
y <- stratifySurv(Surv(d$start, d$stop, d$event), d$provider)
fit_and_write(X, y, "penalized_combined", alpha = 1.0, weights = d$weight, offset = d$offset1)

# --------------------------------------------------------- basic (has ties)
cat("\n== basic (Breslow ties, heavy tie structure: 27 unique times / 500 rows) ==\n")
d <- read.csv(file.path(datadir, "basic.csv"))
X <- as.matrix(d[, c("x1", "x2", "x3")])
y <- Surv(d$time, d$event)
fit_and_write(X, y, "penalized_basic_ties", alpha = 1.0)

cat("\nDone.\n")
