#!/usr/bin/env Rscript
# Reference values for pprofR's patient-clustered provider tests (robust_wald, stabilized_wald,
# robust_score, stabilized_score), for tests/logistic/test_pprofr_clustered_reference.py.
#
#   Rscript run_pprofr_clustered.R <r_reference directory> <pprofR checkout>
#
# pprofR is github.com/UM-KevinHe/pprof, branch Yubo. The checkout must be at PPROFR_COMMIT; the
# script exports that commit with git archive (so local changes are ignored) and installs it into
# a private library. Rcpp, RcppArmadillo and poibin must already be installed.
#
# Cohorts: the simulated data sets of pprofR's tests/testthat/test-tests.R (sim_fe() from
# helper-sim.R, same seeds), plus two stress cohorts simulated here: s1_spanning, patients treated
# at more than one provider with rows interleaved across providers, and s2_long, binomial rows and
# patients with up to 50 rows. Every fit uses tol = 1e-12 and backtrack = FALSE: with pprofR's default
# step-halving search the fit can stop with score equations off by up to 3e-7 (the log-likelihood
# differences it compares fall below double precision near the maximum), whereas plain Newton steps
# reach the maximum to rounding error. The script checks that every fit solves its score equations.
# test_fe() gives the four tests and the plain score test; clustered_ref(), taken from test-tests.R itself, gives the variance
# ingredients at the null and at the fit. The script stops unless every statistic matches its
# definition to TOL_CHECK (absolute plus relative).
#
# Doubles are written with 15 significant digits when that reads back as the same double, and with
# 17 otherwise (17 always round-trips). The cohort table holds MD5 sums of the IEEE bytes of the
# covariates, gamma-hat and beta-hat, so a reader can confirm it got R's exact doubles, and of
# pprofR's row order (the input row numbers, in the order pprofR keeps the rows).
#
# Outputs (every file has a `cohort` column):
#   data/pprofr_clustered_inputs.csv        the rows in the order given to fe_data_prep()
#   results/pprofr_clustered_providers.csv  per provider: gamma, U/V/Vstab/I, test outputs, intervals
#   results/pprofr_clustered_rhat.csv       the pooled matrix R-hat (s <= t) at the null and the fit
#   results/pprofr_clustered_cohorts.csv    per cohort: sizes, beta, gamma0, fit, self-check maxima
#   results/pprofr_clustered_provenance.csv R, platform and package versions; MD5 of every file

PPROFR_COMMIT <- "f23c2889b59108f9506da3f91327c3f239945a63"
TOL_FIT <- 1e-12
BACKTRACK <- FALSE
TOL_CHECK <- 1e-10
TESTS <- c("robust_wald", "stabilized_wald", "robust_score", "stabilized_score", "score")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: Rscript run_pprofr_clustered.R <r_reference directory> <pprofR checkout>")
base <- normalizePath(args[[1]], mustWork = TRUE)
checkout <- normalizePath(args[[2]], mustWork = TRUE)
script <- normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)))
invisible(Sys.setlocale("LC_COLLATE", "C"))

# ---- pprofR at the pinned commit ------------------------------------------------------------------
head <- system2("git", c("-C", shQuote(checkout), "rev-parse", "HEAD"), stdout = TRUE)
if (!identical(head, PPROFR_COMMIT)) stop("the pprofR checkout is at ", head, ", not ", PPROFR_COMMIT)
work <- file.path(tempdir(), "pprofR")
lib <- file.path(tempdir(), "lib")
dir.create(work)
dir.create(lib)
if (system(sprintf("git -C %s archive %s | tar -x -C %s", shQuote(checkout), PPROFR_COMMIT, shQuote(work))) != 0)
  stop("git archive of the pprofR checkout failed")
if (system2(file.path(R.home("bin"), "R"), c("CMD", "INSTALL", paste0("--library=", shQuote(lib)), shQuote(work)),
            stdout = FALSE, stderr = FALSE) != 0)
  stop("installing pprofR failed")
suppressPackageStartupMessages(library(pprofR, lib.loc = lib))

# sim_fe() and prep_fe() from helper-sim.R; clustered_ref() from test-tests.R, used as written.
tests_dir <- file.path(work, "tests", "testthat")
sys.source(file.path(tests_dir, "helper-sim.R"), envir = globalenv())
exprs <- parse(file.path(tests_dir, "test-tests.R"), keep.source = FALSE)
is_ref <- vapply(exprs, function(e) is.call(e) && identical(e[[1]], as.name("<-")) &&
                   identical(e[[2]], as.name("clustered_ref")), logical(1))
if (sum(is_ref) != 1) stop("test-tests.R does not define clustered_ref() exactly once")
eval(exprs[[which(is_ref)]], globalenv())

# ---- stress cohorts -------------------------------------------------------------------------------
# Covariates and outcome as in sim_fe(); `extra` adds patient-level terms so a patient's rows are
# correlated.
outcome <- function(prov, gam, extra, N) {
  n <- length(prov)
  Z <- data.frame(x1 = round(rnorm(n), 3), x2 = rbinom(n, 1, 0.4), x3 = round(runif(n, -1, 1), 3))
  eta <- gam[prov] + 0.5 * Z$x1 - 0.4 * Z$x2 + 0.3 * Z$x3 + extra
  list(Z = Z, Y = stats::rbinom(n, N, stats::plogis(eta)))
}

as_cohort <- function(d, N, y) {
  list(N = N, Y = data.frame(Y = y$Y), Z = y$Z, ProvID = data.frame(prov = d$prov),
       PatientID = data.frame(pat = d$pat), time = d$time)
}

# s1_spanning: 640 patients with 2-6 visits each; about 30% also visit one or two other providers
# (each visit goes there with probability 0.4). Rows are ordered by visit time, so patients and
# providers interleave and some patients return to a provider after visiting another.
sim_spanning <- function(seed, m = 30, n_patients = 640) {
  set.seed(seed)
  provs <- sprintf("Q%02d", seq_len(m))
  gam <- stats::setNames(rnorm(m, -0.3, 0.5), provs)
  pats <- sprintf("S%04d", seq_len(n_patients))
  u <- stats::setNames(rnorm(n_patients, 0, 0.8), pats)
  home <- sample(provs, n_patients, replace = TRUE)
  d <- do.call(rbind, lapply(seq_len(n_patients), function(k) {
    n_k <- sample(2:6, 1)
    at <- rep(home[k], n_k)
    if (runif(1) < 0.3) {
      others <- sample(setdiff(provs, home[k]), sample(1:2, 1))
      away <- runif(n_k) < 0.4
      at[away] <- others[sample.int(length(others), sum(away), replace = TRUE)]
    }
    data.frame(prov = at, pat = pats[k], time = cumsum(sample(1:30, n_k, replace = TRUE)))
  }))
  d <- d[order(d$time, d$pat), ]
  as_cohort(d, rep(1, nrow(d)), outcome(d$prov, gam, u[d$pat], 1))
}

# s2_long: binomial rows (2-12 trials); most patients have 1-6 rows, 15% have 10-49 and two have 50.
# A patient effect plus an AR(1) term (correlation 0.7 between consecutive rows) gives R-hat
# off-diagonal entries that decay with the distance between rows.
sim_long <- function(seed, m = 20) {
  set.seed(seed)
  provs <- sprintf("L%02d", seq_len(m))
  gam <- stats::setNames(rnorm(m, -0.3, 0.5), provs)
  home <- rep(provs, sample(12:20, m, replace = TRUE))
  K <- length(home)
  n_rows <- ifelse(runif(K) < 0.15, sample(10:49, K, replace = TRUE), 1 + rpois(K, 1.5))
  n_rows[sample.int(K, 2)] <- 50
  pats <- sprintf("T%04d", seq_len(K))
  u <- rnorm(K, 0, 0.6)
  d <- do.call(rbind, lapply(seq_len(K), function(k) {
    w <- numeric(n_rows[k])
    w[1] <- rnorm(1)
    for (t in seq_len(n_rows[k])[-1]) w[t] <- 0.7 * w[t - 1] + sqrt(1 - 0.49) * rnorm(1)
    data.frame(prov = home[k], pat = pats[k], time = cumsum(sample(1:20, n_rows[k], replace = TRUE)),
               extra = u[k] + 0.5 * w)
  }))
  d <- d[order(d$time, d$pat), ]
  N <- sample(2:12, nrow(d), replace = TRUE)
  as_cohort(d, N, outcome(d$prov, gam, d$extra, N))
}

COHORTS <- list(
  list(cohort = "t31_binary", source = "sim_fe", seed = 31, binomial = FALSE),
  list(cohort = "t31_binomial", source = "sim_fe", seed = 31, binomial = TRUE),
  list(cohort = "t13", source = "sim_fe", seed = 13, binomial = FALSE),
  list(cohort = "t14", source = "sim_fe", seed = 14, binomial = FALSE),
  list(cohort = "t16", source = "sim_fe", seed = 16, binomial = FALSE),
  list(cohort = "s1_spanning", source = "sim_spanning", seed = 20261021, binomial = FALSE),
  list(cohort = "s2_long", source = "sim_long", seed = 20261022, binomial = TRUE))

simulate <- function(cfg) {
  if (cfg$source == "sim_fe") {
    d <- sim_fe(seed = cfg$seed, binomial = cfg$binomial)
    d$time <- stats::ave(seq_along(d$PatientID$pat), d$PatientID$pat, FUN = seq_along)
    return(d)
  }
  get(cfg$source)(cfg$seed)
}

# ---- helpers --------------------------------------------------------------------------------------
# The pooled matrix behind clustered_ref()'s "stable" (same steps), kept for output; its meat is
# checked against clustered_ref() below.
pooled <- function(fit, p) {
  din <- fit$data_include
  r <- din$Y - din$N * p
  v <- din$N * p * (1 - p)
  key <- paste(din$prov, din$pat)
  e <- ifelse(v > 0, r / sqrt(v), 0)
  a <- sqrt(v)
  by_pat <- split(seq_along(r), factor(key, levels = unique(key)))
  dim <- max(lengths(by_pat))
  S <- matrix(0, dim, dim)
  m <- numeric(dim)
  for (rows in by_pat) {
    s <- length(rows)
    S[1:s, 1:s] <- S[1:s, 1:s] + tcrossprod(e[rows])
    m[1:s] <- m[1:s] + 1
  }
  R <- S / outer(seq_len(dim), seq_len(dim), function(k, l) m[pmax(k, l)])
  W <- vapply(by_pat, function(rows) {
    s <- length(rows)
    drop(t(a[rows]) %*% R[1:s, 1:s, drop = FALSE] %*% a[rows])
  }, numeric(1))
  list(R = R, m = m, meat = tapply(W, as.character(din$prov)[!duplicated(key)], sum))
}

ieee_md5 <- function(x) {
  f <- tempfile()
  writeBin(as.double(x), f, size = 8, endian = "little")
  unname(tools::md5sum(f))
}

num_str <- function(x) {
  s <- sprintf("%.15g", x)
  long <- !is.na(x) & suppressWarnings(as.numeric(s)) != x
  s[long] <- sprintf("%.17g", x[long])
  s
}

int_md5 <- function(x) {
  f <- tempfile()
  writeBin(as.integer(x), f, size = 4, endian = "little")
  unname(tools::md5sum(f))
}

write_table <- function(df, path) {
  for (j in seq_along(df)) if (is.double(df[[j]])) df[[j]] <- num_str(df[[j]])
  utils::write.csv(df, path, row.names = FALSE, quote = FALSE, na = "NA")
}

flag_int <- function(f) as.integer(as.character(f))

# ---- one cohort -----------------------------------------------------------------------------------
run_cohort <- function(cfg) {
  d <- simulate(cfg)
  n <- nrow(d$Y)
  fit <- suppressMessages(logis_fe(prep_fe(d, binomial = cfg$binomial), tol = TOL_FIT, backtrack = BACKTRACK))
  din <- fit$data_include
  order_r <- as.integer(rownames(din))
  if (!identical(sort(order_r), seq_len(n))) stop(cfg$cohort, ": pprofR left rows out of the fit")
  gamma <- fit$df.prov$gamma_est
  if (!identical(unname(as.numeric(fit$gamma)), gamma)) stop(cfg$cohort, ": fit$gamma differs from gamma_est")
  provs <- rownames(fit$gamma)
  g0 <- stats::median(gamma)
  p_fit <- fit$pred
  p_null <- as.numeric(stats::plogis(g0 + fit$linear_pred))
  at <- list(null = clustered_ref(fit, p_null), fit = clustered_ref(fit, p_fit))
  ref <- function(basis, what) as.numeric(at[[basis]][[what]][provs])
  out <- lapply(stats::setNames(TESTS, TESTS), function(tst) test_fe(fit, test = tst))
  for (tst in TESTS) if (!identical(rownames(out[[tst]]), provs)) stop(cfg$cohort, ": ", tst, " rows out of order")
  ci <- confint(fit, option = "gamma", test = "robust_wald")
  if (!identical(as.numeric(ci$gamma), gamma)) stop(cfg$cohort, ": confint() rows out of order")
  pool <- list(null = pooled(fit, p_null), fit = pooled(fit, p_fit))

  checks <- list()
  check <- function(what, got, want) {
    got <- as.numeric(got)
    want <- as.numeric(want)
    err <- abs(got - want)
    if (anyNA(err) || any(err > TOL_CHECK + TOL_CHECK * abs(want)))
      stop(sprintf("%s: %s differs from its definition (max error %.3g)", cfg$cohort, what, max(err)))
    checks[[paste0("chk_", what)]] <<- max(err / (1 + abs(want)))
  }
  resid <- din$Y - din$N * p_fit
  check("score_equations", c(tapply(resid, din$prov, sum), crossprod(as.matrix(din[, c("x1", "x2", "x3")]), resid)), 0)
  se_rw <- sqrt(ref("fit", "robust")) / ref("fit", "info")
  se_sw <- sqrt(ref("fit", "stable")) / ref("fit", "info")
  check("robust_wald_se", out$robust_wald$se.gamma, se_rw)
  check("robust_wald_stat", out$robust_wald$stat, (gamma - g0) / se_rw)
  check("stabilized_wald_se", out$stabilized_wald$se.gamma, se_sw)
  check("stabilized_wald_stat", out$stabilized_wald$stat, (gamma - g0) / se_sw)
  check("robust_score_stat", out$robust_score$stat, ref("null", "U") / sqrt(ref("null", "robust")))
  check("stabilized_score_stat", out$stabilized_score$stat, ref("null", "U") / sqrt(ref("null", "stable")))
  check("score_stat", out$score$stat, ref("null", "U") / sqrt(ref("null", "info")))
  q <- stats::qnorm(0.975)
  check("robust_wald_ci", c(ci$gamma.lower, ci$gamma.upper),
        c(gamma - q * out$robust_wald$se.gamma, gamma + q * out$robust_wald$se.gamma))
  check("rhat_null_meat", pool$null$meat[provs], ref("null", "stable"))
  check("rhat_fit_meat", pool$fit$meat[provs], ref("fit", "stable"))

  # sizes and stress features, from the data as given
  key <- paste(din$prov, din$pat)
  pat_in <- d$PatientID$pat
  prov_in <- d$ProvID$prov
  per_prov_patients <- tapply(din$pat, din$prov, function(x) length(unique(x)))[provs]
  n_events <- tapply(din$Y, din$prov, sum)[provs]
  n_trials <- tapply(din$N, din$prov, sum)[provs]
  if (any(per_prov_patients < 10) || any(n_events == 0) || any(n_events == n_trials))
    stop(cfg$cohort, ": a provider has fewer than 10 patients, or no or only events")
  spanning <- tapply(prov_in, pat_in, function(x) length(unique(x)) > 1)
  interleaved <- tapply(prov_in, pat_in, function(x) anyDuplicated(rle(x)$values) > 0)

  inputs <- data.frame(cohort = cfg$cohort, row_id = seq_len(n), provider = prov_in, patient = pat_in,
                       time = d$time, N = d$N, Y = d$Y$Y, d$Z)
  providers <- data.frame(
    cohort = cfg$cohort, provider = provs, n_rows = as.integer(table(din$prov)[provs]),
    n_patients = as.integer(per_prov_patients), gamma = gamma,
    U_null = ref("null", "U"), V_null = ref("null", "robust"), Vstab_null = ref("null", "stable"),
    I_null = ref("null", "info"), U_fit = ref("fit", "U"), V_fit = ref("fit", "robust"),
    Vstab_fit = ref("fit", "stable"), I_fit = ref("fit", "info"))
  for (tst in TESTS) {
    res <- out[[tst]]
    if ("se.gamma" %in% names(res)) providers[[paste0(tst, "_se")]] <- res$se.gamma
    providers[[paste0(tst, "_stat")]] <- res$stat
    providers[[paste0(tst, "_p")]] <- res$p
    providers[[paste0(tst, "_flag")]] <- flag_int(res$flag)
  }
  providers$robust_wald_ci_lower <- ci$gamma.lower
  providers$robust_wald_ci_upper <- ci$gamma.upper
  rhat <- do.call(rbind, lapply(names(pool), function(basis) {
    R <- pool[[basis]]$R
    st <- which(upper.tri(R, diag = TRUE), arr.ind = TRUE)
    st <- st[order(st[, 1], st[, 2]), , drop = FALSE]
    data.frame(cohort = cfg$cohort, basis = basis, s = st[, 1], t = st[, 2], R = R[st],
               m = as.integer(pool[[basis]]$m[pmax(st[, 1], st[, 2])]))
  }))
  beta <- as.numeric(fit$beta)
  cohort <- data.frame(
    cohort = cfg$cohort, source = cfg$source, seed = cfg$seed, binomial = cfg$binomial,
    n_rows = n, n_providers = length(provs), n_patients = length(unique(pat_in)),
    n_clusters = length(unique(key)), max_cluster_rows = max(table(key)),
    min_patients_per_provider = min(per_prov_patients), n_spanning_patients = sum(spanning),
    n_interleaved_patients = sum(interleaved), gamma0 = g0, beta_x1 = beta[1], beta_x2 = beta[2],
    beta_x3 = beta[3], fit_tol = TOL_FIT, fit_iter = as.integer(fit$control$iter),
    fit_algorithm = fit$control$algorithm, checks,
    md5_x1 = ieee_md5(d$Z$x1), md5_x3 = ieee_md5(d$Z$x3), md5_gamma = ieee_md5(gamma),
    md5_beta = ieee_md5(beta), md5_r_order = int_md5(order_r))
  cat(sprintf("%-13s %5d rows %3d providers %4d clusters (max %2d rows) %3d spanning, %3d interleaved; worst check %.2g\n",
              cfg$cohort, n, length(provs), length(unique(key)), max(table(key)), sum(spanning), sum(interleaved),
              max(unlist(checks))))
  list(inputs = inputs, providers = providers, rhat = rhat, cohort = cohort)
}

# ---- run and write --------------------------------------------------------------------------------
results <- lapply(COHORTS, run_cohort)
stack <- function(part) do.call(rbind, lapply(results, `[[`, part))
files <- c(inputs = "data/pprofr_clustered_inputs.csv", providers = "results/pprofr_clustered_providers.csv", rhat = "results/pprofr_clustered_rhat.csv",
           cohorts = "results/pprofr_clustered_cohorts.csv")
parts <- c(inputs = "inputs", providers = "providers", rhat = "rhat", cohorts = "cohort")
for (f in names(files)) write_table(stack(parts[[f]]), file.path(base, files[[f]]))

si <- utils::sessionInfo()
pkg <- function(p) as.character(utils::packageVersion(p))
desc <- function(p, field) {
  value <- utils::packageDescription(p, fields = field)
  if (is.na(value)) "" else gsub("[\r\n,]+", " ", value)
}
md5 <- tools::md5sum(file.path(base, files))
provenance <- data.frame(key = c(
  "generator", "generator_md5", "generated_utc", "R", "platform", "os", "blas", "lapack", "rng", "collate",
  "pprofR_source", "pprofR_commit", "pprofR_version", "Rcpp", "RcppArmadillo", "poibin",
  "poibin_date_publication", "fit_tol", "fit_backtrack", "check_tol", paste0("md5:", files)),
  value = c(
    "r_reference/run_pprofr_clustered.R", unname(tools::md5sum(script)),
    format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"), R.version.string, R.version$platform, si$running,
    si$BLAS, si$LAPACK, paste(RNGkind(), collapse = " "), Sys.getlocale("LC_COLLATE"),
    "github.com/UM-KevinHe/pprof branch Yubo", PPROFR_COMMIT, pkg("pprofR"), pkg("Rcpp"), pkg("RcppArmadillo"),
    pkg("poibin"), desc("poibin", "Date/Publication"), num_str(TOL_FIT), as.character(BACKTRACK), num_str(TOL_CHECK),
    unname(md5)))
provenance$value <- gsub(",", ";", provenance$value)
utils::write.csv(provenance, file.path(base, "results/pprofr_clustered_provenance.csv"), row.names = FALSE,
                 quote = FALSE)
cat("wrote", length(files) + 1, "files under", base, "\n")
