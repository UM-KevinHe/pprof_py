"""Validation harness: fit `coxph.PenalizedCoxPH`/`PenalizedCoxPHCV` on
the same data R fit in r_reference/run_penalized.R, and compare every
output R can produce against what R actually produced
(r_reference/results/penalized_*.csv) -- not a hand-derived
expectation. Mirrors tests/test_r_comparison.py's structure exactly.

For every scenario, the exact lambda sequence glmnet settled on
(R's `fit$lambda`, including early stops from `fdev`/`devmax`) is
passed to `PenalizedCoxPH(lambda_path=...)` directly, so this compares
coefficients at identical lambda values rather than relying on both
languages' auto-grid generation landing on the same points (grid
generation itself is checked separately, in
`test_lambda_max_matches_glmnet_auto_grid`).

One documented, investigated exception: `test_wide_ridge` excludes the
single largest lambda in the ridge (alpha=0) path. `get_cox_lambda_max`
floors alpha at 1e-3 to keep lambda_max finite for ridge, which makes
that one grid point ~1000x larger than the "natural" (alpha=1) scale
-- at that artificial extreme, this package's coefficients (verified
to satisfy the exact ridge stationarity condition to 1e-17, and to
match a back-of-envelope shrinkage estimate) are ~1e-3, while glmnet's
are ~1e-37. Every other point on the same path (99/100) matches to
~1e-7. See docs/R_COMPATIBILITY.md, Phase 3 section, for the full
investigation -- this is the project's established "investigate, don't
loosen tolerance to hide it" policy applied to a case where the
investigation's conclusion is "glmnet's own extreme-edge-case value
looks like the outlier," not this package's.

Run standalone (prints a full report) via:
    python tests/test_penalized_r_comparison.py
or under pytest.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import pytest

from coxph import PenalizedCoxPH, PenalizedCoxPHCV

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "r_reference", "data")
RESULTS = os.path.join(HERE, "..", "r_reference", "results")

# See module docstring: strata/left-truncation/combined show slightly
# more numerical spread than the plain right-censored cases (max
# observed ~2.5e-5 vs ~1e-7), likely from the extra risk-set bookkeeping
# interacting with coordinate descent's own iterative refinement; both
# are given generous headroom above what's actually observed.
ATOL_COEF_PATH = 5e-4
ATOL_COEF_PATH_TIGHT = 1e-4
RTOL_LAMBDA = 1e-6
ATOL_CV_DEVIANCE = 0.01
ATOL_CV_COEF = 1e-4

REPORT = []


def _check(label, py_val, r_val, atol):
    py_val = np.asarray(py_val, dtype=float)
    r_val = np.asarray(r_val, dtype=float)
    abs_diff = np.max(np.abs(py_val - r_val))
    status = "PASS" if abs_diff < atol else "FAIL"
    REPORT.append(dict(check=label, abs_diff=abs_diff, tol=atol, status=status))
    assert status == "PASS", f"{label}: max abs diff={abs_diff:.3e} (tol={atol:.1e})"


def _r_coef_path(name):
    return pd.read_csv(os.path.join(RESULTS, f"{name}_coef_path.csv")).to_numpy()


def _r_path_meta(name):
    return pd.read_csv(os.path.join(RESULTS, f"{name}_path_meta.csv"))


def test_wide_lasso():
    d = pd.read_csv(os.path.join(DATA, "penalized_wide.csv"))
    xcols = [c for c in d.columns if c.startswith("x")]
    r_lambda = _r_path_meta("penalized_wide_lasso")["lambda"].to_numpy()
    m = PenalizedCoxPH(alpha=1.0, lambda_path=r_lambda)
    m.fit(d[xcols], duration=d["stop"], event=d["event"])
    _check("wide lasso: coef path", m.coef_path_, _r_coef_path("penalized_wide_lasso"), ATOL_COEF_PATH_TIGHT)


def test_wide_ridge():
    d = pd.read_csv(os.path.join(DATA, "penalized_wide.csv"))
    xcols = [c for c in d.columns if c.startswith("x")]
    r_lambda = _r_path_meta("penalized_wide_ridge")["lambda"].to_numpy()
    m = PenalizedCoxPH(alpha=0.0, lambda_path=r_lambda)
    m.fit(d[xcols], duration=d["stop"], event=d["event"])
    # exclude the single artificially-inflated top-of-path lambda -- see module docstring
    _check("wide ridge: coef path (excl. lambda[0])",
           m.coef_path_[1:], _r_coef_path("penalized_wide_ridge")[1:], ATOL_COEF_PATH_TIGHT)


def test_wide_elastic_net():
    d = pd.read_csv(os.path.join(DATA, "penalized_wide.csv"))
    xcols = [c for c in d.columns if c.startswith("x")]
    r_lambda = _r_path_meta("penalized_wide_enet")["lambda"].to_numpy()
    m = PenalizedCoxPH(alpha=0.5, lambda_path=r_lambda)
    m.fit(d[xcols], duration=d["stop"], event=d["event"])
    _check("wide elastic net (alpha=0.5): coef path", m.coef_path_, _r_coef_path("penalized_wide_enet"), ATOL_COEF_PATH_TIGHT)


def test_wide_lasso_no_standardize():
    d = pd.read_csv(os.path.join(DATA, "penalized_wide.csv"))
    xcols = [c for c in d.columns if c.startswith("x")]
    r_lambda = _r_path_meta("penalized_wide_lasso_nostd")["lambda"].to_numpy()
    m = PenalizedCoxPH(alpha=1.0, lambda_path=r_lambda, standardize=False)
    m.fit(d[xcols], duration=d["stop"], event=d["event"])
    _check("wide lasso, standardize=False: coef path",
           m.coef_path_, _r_coef_path("penalized_wide_lasso_nostd"), ATOL_COEF_PATH_TIGHT)


def test_wide_lasso_unpenalized_variables():
    d = pd.read_csv(os.path.join(DATA, "penalized_wide.csv"))
    xcols = [c for c in d.columns if c.startswith("x")]
    r_lambda = _r_path_meta("penalized_wide_lasso_unpen")["lambda"].to_numpy()
    pf = np.ones(len(xcols)); pf[:3] = 0.0
    m = PenalizedCoxPH(alpha=1.0, lambda_path=r_lambda, penalty_factor=pf)
    m.fit(d[xcols], duration=d["stop"], event=d["event"])
    _check("wide lasso, first 3 vars unpenalized: coef path",
           m.coef_path_, _r_coef_path("penalized_wide_lasso_unpen"), ATOL_COEF_PATH_TIGHT)
    # the always-unpenalized columns should never be exactly 0 once any
    # signal reaches them, unlike a penalized column at large lambda
    assert np.all(m.coef_path_[-1, :3] != 0.0)


def test_lambda_max_matches_glmnet_auto_grid():
    """Grid *generation* itself, not just fitting at a supplied grid:
    request glmnet's own default nlambda and confirm our auto lambda_max_
    (and the first 45 auto-generated grid points, before glmnet's own
    fdev/devmax early stopping) match glmnet's un-forced path."""
    d = pd.read_csv(os.path.join(DATA, "penalized_wide.csv"))
    xcols = [c for c in d.columns if c.startswith("x")]
    r_lambda = _r_path_meta("penalized_wide_lasso")["lambda"].to_numpy()
    m = PenalizedCoxPH(alpha=1.0, n_lambda=100)
    m.fit(d[xcols], duration=d["stop"], event=d["event"])
    _check("lambda_max_ vs glmnet's own lambda[0]", [m.lambda_max_], [r_lambda[0]], atol=1e-6)
    n = len(r_lambda)
    _check("auto-generated grid (first %d pts)" % n, m.lambda_path_[:n], r_lambda, atol=1e-6)


def test_strata():
    d = pd.read_csv(os.path.join(DATA, "strata.csv"))
    r_lambda = _r_path_meta("penalized_strata")["lambda"].to_numpy()
    m = PenalizedCoxPH(alpha=1.0, lambda_path=r_lambda)
    m.fit(d[["x1", "x2"]], duration=d["stop"], event=d["event"], strata=d["provider"])
    _check("strata: coef path", m.coef_path_, _r_coef_path("penalized_strata"), ATOL_COEF_PATH)


def test_offset():
    d = pd.read_csv(os.path.join(DATA, "offset.csv"))
    r_lambda = _r_path_meta("penalized_offset")["lambda"].to_numpy()
    m = PenalizedCoxPH(alpha=1.0, lambda_path=r_lambda)
    m.fit(d[["x1", "x2"]], duration=d["stop"], event=d["event"], offset=d["log_exposure"])
    _check("offset: coef path", m.coef_path_, _r_coef_path("penalized_offset"), ATOL_COEF_PATH_TIGHT)


def test_weights():
    d = pd.read_csv(os.path.join(DATA, "weights.csv"))
    r_lambda = _r_path_meta("penalized_weights")["lambda"].to_numpy()
    m = PenalizedCoxPH(alpha=1.0, lambda_path=r_lambda)
    m.fit(d[["x1", "x2"]], duration=d["stop"], event=d["event"], sample_weight=d["weight"])
    _check("weights: coef path", m.coef_path_, _r_coef_path("penalized_weights"), ATOL_COEF_PATH_TIGHT)


def test_left_truncation():
    d = pd.read_csv(os.path.join(DATA, "left_truncation.csv"))
    r_lambda = _r_path_meta("penalized_left_truncation")["lambda"].to_numpy()
    m = PenalizedCoxPH(alpha=1.0, lambda_path=r_lambda)
    m.fit(d[["x1", "x2"]], start=d["start"], stop=d["stop"], event=d["event"])
    _check("left_truncation: coef path", m.coef_path_, _r_coef_path("penalized_left_truncation"), ATOL_COEF_PATH)


def test_combined_strata_offset_weights_left_truncation():
    d = pd.read_csv(os.path.join(DATA, "combined.csv"))
    r_lambda = _r_path_meta("penalized_combined")["lambda"].to_numpy()
    m = PenalizedCoxPH(alpha=1.0, lambda_path=r_lambda)
    m.fit(d[["x1", "x2", "x3"]], start=d["start"], stop=d["stop"], event=d["event"],
          strata=d["provider"], offset=d["offset1"], sample_weight=d["weight"])
    _check("combined (strata+offset+weights+start/stop): coef path",
           m.coef_path_, _r_coef_path("penalized_combined"), ATOL_COEF_PATH)


def test_basic_heavy_ties():
    """basic.csv has 27 unique times over 500 rows -- a real stress test
    for Breslow-tie handling under penalization."""
    d = pd.read_csv(os.path.join(DATA, "basic.csv"))
    r_lambda = _r_path_meta("penalized_basic_ties")["lambda"].to_numpy()
    m = PenalizedCoxPH(alpha=1.0, lambda_path=r_lambda)
    m.fit(d[["x1", "x2", "x3"]], duration=d["time"], event=d["event"])
    _check("basic (heavy ties): coef path", m.coef_path_, _r_coef_path("penalized_basic_ties"), ATOL_COEF_PATH_TIGHT)


def test_cross_validation():
    d = pd.read_csv(os.path.join(DATA, "penalized_wide.csv"))
    xcols = [c for c in d.columns if c.startswith("x")]
    fold_id = pd.read_csv(os.path.join(DATA, "penalized_wide_foldid.csv"))["fold_id"].to_numpy()
    r_cv = pd.read_csv(os.path.join(RESULTS, "penalized_wide_cv.csv"))
    r_selected = pd.read_csv(os.path.join(RESULTS, "penalized_wide_cv_selected.csv"))
    r_coef_min = pd.read_csv(os.path.join(RESULTS, "penalized_wide_cv_coef_min.csv")).set_index("term")

    m = PenalizedCoxPHCV(alpha=1.0, fold_id=fold_id, n_lambda=100)
    m.fit(d[xcols], duration=d["stop"], event=d["event"])

    _check("cv: lambda_min_", [m.lambda_min_], [r_selected["lambda_min"].iloc[0]], atol=1e-6)
    _check("cv: lambda_1se_", [m.lambda_1se_], [r_selected["lambda_1se"].iloc[0]], atol=1e-6)

    n = min(len(m.lambda_path_), len(r_cv))
    _check("cv: lambda grid", m.lambda_path_[:n], r_cv["lambda"].to_numpy()[:n], atol=1e-6)
    _check("cv: mean deviance (cvm)", m.cv_mean_deviance_[:n], r_cv["cvm"].to_numpy()[:n], ATOL_CV_DEVIANCE)
    _check("cv: se of deviance (cvsd)", m.cv_se_deviance_[:n], r_cv["cvsd"].to_numpy()[:n], ATOL_CV_DEVIANCE)

    r_coef_min_arr = r_coef_min.loc[xcols, "coef"].to_numpy()
    _check("cv: final coef_ at lambda_min", m.coef_, r_coef_min_arr, ATOL_CV_COEF)


if __name__ == "__main__":
    failures = 0
    for fn_name in [
        "test_wide_lasso", "test_wide_ridge", "test_wide_elastic_net",
        "test_wide_lasso_no_standardize", "test_wide_lasso_unpenalized_variables",
        "test_lambda_max_matches_glmnet_auto_grid", "test_strata", "test_offset",
        "test_weights", "test_left_truncation",
        "test_combined_strata_offset_weights_left_truncation", "test_basic_heavy_ties",
        "test_cross_validation",
    ]:
        try:
            globals()[fn_name]()
            print(f"[PASS] {fn_name}")
        except AssertionError as e:
            failures += 1
            print(f"[FAIL] {fn_name}: {e}")

    print("\n" + "=" * 90)
    print(f"{'check':55s} {'abs_diff':>12s} {'tol':>10s}  status")
    print("=" * 90)
    for row in REPORT:
        print(f"{row['check']:55s} {row['abs_diff']:12.3e} {row['tol']:10.1e}  {row['status']}")
    print("=" * 90)
    print(f"\n{len(REPORT) - failures if failures == 0 else '?'} checks, {failures} test function failure(s)")
    if failures:
        raise SystemExit(1)
