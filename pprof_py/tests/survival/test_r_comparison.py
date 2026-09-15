"""Validation harness: fit `coxph.CoxPH` on the same data R fit in
r_reference/run_all.R, and compare every output R can produce against
what R actually produced (r_reference/results/*.csv) -- not a
hand-derived expectation.

Run standalone (prints a full report) via:
    python tests/test_r_comparison.py
or under pytest, where each `test_*` function is also an individual
assertion-based test case.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import pytest

from pprof_py import CoxPH

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "..", "r_reference", "data")
RESULTS = os.path.join(HERE, "..", "..", "r_reference", "results")

# Tolerances are deliberately tight and were arrived at empirically (see
# docs/VALIDATION_REPORT.md): coefficients and log-likelihoods converge
# to R's answer far tighter than this, so a loosened tolerance here would
# be hiding slack rather than reflecting a real numerical limit.
RTOL_COEF = 1e-5
RTOL_LOGLIK = 1e-6
RTOL_BASELINE = 1e-4
ATOL_RESIDUAL = 1e-4


def _r_coef_table(name):
    return pd.read_csv(os.path.join(RESULTS, f"{name}_coefficients.csv"))


def _r_meta(name):
    return pd.read_csv(os.path.join(RESULTS, f"{name}_meta.csv"))


def _r_baseline(name):
    return pd.read_csv(os.path.join(RESULTS, f"{name}_baseline.csv"))


def _r_residuals(name):
    return pd.read_csv(os.path.join(RESULTS, f"{name}_residuals.csv"))["martingale"].to_numpy()


def _r_baseline_numeric_stratum(name, py_dtype):
    """R's `strata(provider)` labels basehaz() rows like "provider=0",
    not the bare value 0 -- extract the numeric suffix so it can be
    merged against this package's (bare-value) stratum labels."""
    rb = _r_baseline(name)
    rb["stratum"] = rb["strata"].str.extract(r"=(-?\d+)$")[0].astype(py_dtype)
    return rb


def _report_row(label, py_val, r_val, rtol):
    py_val = np.asarray(py_val, dtype=float)
    r_val = np.asarray(r_val, dtype=float)
    abs_diff = np.max(np.abs(py_val - r_val))
    denom = np.maximum(np.abs(r_val), 1e-12)
    rel_diff = np.max(np.abs(py_val - r_val) / denom)
    status = "PASS" if rel_diff < rtol or abs_diff < 1e-10 else "FAIL"
    return dict(check=label, abs_diff=abs_diff, rel_diff=rel_diff, tol=rtol, status=status)


REPORT = []


def _check(label, py_val, r_val, rtol):
    row = _report_row(label, py_val, r_val, rtol)
    REPORT.append(row)
    assert row["status"] == "PASS", (
        f"{label}: max abs diff={row['abs_diff']:.3e}, max rel diff={row['rel_diff']:.3e} "
        f"(tol={rtol:.1e})\npython={py_val}\nr     ={r_val}"
    )


def test_basic():
    d = pd.read_csv(os.path.join(DATA, "basic.csv"))
    m = CoxPH().fit(d[["x1", "x2", "x3"]], duration=d["time"], event=d["event"])
    r = _r_coef_table("basic")
    meta = _r_meta("basic")

    _check("basic: coef", m.coef_, r["coef"], RTOL_COEF)
    _check("basic: se", m.standard_errors_, r["se_coef"], RTOL_COEF)
    _check("basic: z", m.z_scores_, r["z"], RTOL_COEF)
    _check("basic: loglik(beta)", m.log_likelihood_, meta["loglik_beta"], RTOL_LOGLIK)
    _check("basic: loglik(null)", m.log_likelihood_null_, meta["loglik_null"], RTOL_LOGLIK)
    assert m.n_events_ == int(meta["nevent"].iloc[0])
    assert m.n_obs_ == int(meta["n"].iloc[0])

    rb = _r_baseline("basic")
    py_bh = m.baseline_hazard_
    # R's basehaz() emits one row per distinct OBSERVED time (event or
    # censoring), carrying the hazard forward at censoring-only times;
    # this package's baseline_hazard_ emits one row per distinct EVENT
    # time only (the minimal representation of the same step function --
    # see docs/R_COMPATIBILITY.md, question 8). So every Python time must
    # appear in R's table, but not vice versa -- check the inner join
    # recovers every Python row, not that the tables are the same length.
    merged = pd.merge(py_bh, rb, on="time", suffixes=("_py", "_r"))
    assert len(merged) == len(py_bh), "a Python baseline-hazard event time is missing from R's basehaz() output"
    _check("basic: baseline hazard", merged["hazard_py"], merged["hazard_r"], RTOL_BASELINE)

    r_resid = _r_residuals("basic")
    row = _report_row("basic: martingale residuals", m.martingale_residuals_, r_resid, 1e-3)
    row["abs_diff"] = np.max(np.abs(m.martingale_residuals_ - r_resid))
    row["status"] = "PASS" if row["abs_diff"] < ATOL_RESIDUAL else "FAIL"
    REPORT.append(row)
    assert row["status"] == "PASS", row


def test_left_truncation():
    d = pd.read_csv(os.path.join(DATA, "left_truncation.csv"))
    m = CoxPH().fit(d[["x1", "x2"]], start=d["start"], stop=d["stop"], event=d["event"])
    r = _r_coef_table("left_truncation")
    meta = _r_meta("left_truncation")

    _check("left_truncation: coef", m.coef_, r["coef"], RTOL_COEF)
    _check("left_truncation: se", m.standard_errors_, r["se_coef"], RTOL_COEF)
    _check("left_truncation: loglik(beta)", m.log_likelihood_, meta["loglik_beta"], RTOL_LOGLIK)

    rb = _r_baseline("left_truncation")
    merged = pd.merge(m.baseline_hazard_, rb, on="time", suffixes=("_py", "_r"))
    assert len(merged) == len(m.baseline_hazard_)
    _check("left_truncation: baseline hazard", merged["hazard_py"], merged["hazard_r"], RTOL_BASELINE)

    r_resid = _r_residuals("left_truncation")
    diff = np.max(np.abs(m.martingale_residuals_ - r_resid))
    REPORT.append(dict(check="left_truncation: martingale residuals", abs_diff=diff, rel_diff=np.nan,
                        tol=ATOL_RESIDUAL, status="PASS" if diff < ATOL_RESIDUAL else "FAIL"))
    assert diff < ATOL_RESIDUAL


def test_strata():
    d = pd.read_csv(os.path.join(DATA, "strata.csv"))
    m = CoxPH().fit(d[["x1", "x2"]], duration=d["stop"], event=d["event"], strata=d["provider"])
    r = _r_coef_table("strata")
    meta = _r_meta("strata")

    _check("strata: coef", m.coef_, r["coef"], RTOL_COEF)
    _check("strata: se", m.standard_errors_, r["se_coef"], RTOL_COEF)
    _check("strata: loglik(beta)", m.log_likelihood_, meta["loglik_beta"], RTOL_LOGLIK)

    rb = _r_baseline_numeric_stratum("strata", m.baseline_hazard_["stratum"].dtype)
    merged = pd.merge(m.baseline_hazard_, rb, on=["stratum", "time"], suffixes=("_py", "_r"))
    assert len(merged) == len(m.baseline_hazard_)
    _check("strata: baseline hazard", merged["hazard_py"], merged["hazard_r"], RTOL_BASELINE)


def test_offset():
    d = pd.read_csv(os.path.join(DATA, "offset.csv"))
    m = CoxPH().fit(d[["x1", "x2"]], duration=d["stop"], event=d["event"], offset=d["log_exposure"])
    r = _r_coef_table("offset")
    meta = _r_meta("offset")

    _check("offset: coef", m.coef_, r["coef"], RTOL_COEF)
    _check("offset: se", m.standard_errors_, r["se_coef"], RTOL_COEF)
    _check("offset: loglik(beta)", m.log_likelihood_, meta["loglik_beta"], RTOL_LOGLIK)

    rb = _r_baseline("offset")
    merged = pd.merge(m.baseline_hazard_, rb, on="time", suffixes=("_py", "_r"))
    assert len(merged) == len(m.baseline_hazard_)
    _check("offset: baseline hazard", merged["hazard_py"], merged["hazard_r"], RTOL_BASELINE)


def test_weights():
    d = pd.read_csv(os.path.join(DATA, "weights.csv"))
    m = CoxPH().fit(d[["x1", "x2"]], duration=d["stop"], event=d["event"], sample_weight=d["weight"])
    r = _r_coef_table("weights")
    meta = _r_meta("weights")

    _check("weights: coef", m.coef_, r["coef"], RTOL_COEF)
    _check("weights: se (model-based, not robust)", m.standard_errors_, r["se_coef"], RTOL_COEF)
    _check("weights: loglik(beta)", m.log_likelihood_, meta["loglik_beta"], RTOL_LOGLIK)

    rb = _r_baseline("weights")
    merged = pd.merge(m.baseline_hazard_, rb, on="time", suffixes=("_py", "_r"))
    assert len(merged) == len(m.baseline_hazard_)
    _check("weights: baseline hazard", merged["hazard_py"], merged["hazard_r"], RTOL_BASELINE)


def test_combined():
    """Strata + offset + weights + left truncation together -- the exact
    combination the two-stage SMR/SHR workflow needs."""
    d = pd.read_csv(os.path.join(DATA, "combined.csv"))
    m = CoxPH().fit(
        d[["x1", "x2", "x3"]], start=d["start"], stop=d["stop"], event=d["event"],
        strata=d["provider"], offset=d["offset1"], sample_weight=d["weight"],
    )
    r = _r_coef_table("combined")
    meta = _r_meta("combined")

    _check("combined: coef", m.coef_, r["coef"], RTOL_COEF)
    _check("combined: se", m.standard_errors_, r["se_coef"], RTOL_COEF)
    _check("combined: loglik(beta)", m.log_likelihood_, meta["loglik_beta"], RTOL_LOGLIK)

    rb = _r_baseline_numeric_stratum("combined", m.baseline_hazard_["stratum"].dtype)
    merged = pd.merge(m.baseline_hazard_, rb, on=["stratum", "time"], suffixes=("_py", "_r"))
    assert len(merged) == len(m.baseline_hazard_)
    _check("combined: baseline hazard", merged["hazard_py"], merged["hazard_r"], RTOL_BASELINE)

    r_resid = _r_residuals("combined")
    diff = np.max(np.abs(m.martingale_residuals_ - r_resid))
    REPORT.append(dict(check="combined: martingale residuals", abs_diff=diff, rel_diff=np.nan,
                        tol=ATOL_RESIDUAL, status="PASS" if diff < ATOL_RESIDUAL else "FAIL"))
    assert diff < ATOL_RESIDUAL


def test_two_stage_shr_workflow():
    """Reproduces the full two-stage SMR/SHR pattern end to end and
    compares Stage 1's coefficients AND Stage 2's log-likelihood/baseline
    (the actual "observed vs. expected" object of a real SHR model)
    against R."""
    d = pd.read_csv(os.path.join(DATA, "combined.csv"))

    stage1 = CoxPH().fit(
        d[["x1", "x2", "x3"]], start=d["start"], stop=d["stop"], event=d["event"],
        strata=d["provider"], offset=d["offset1"], sample_weight=d["weight"],
    )
    r1 = _r_coef_table("two_stage_1")
    _check("two_stage stage1: coef", stage1.coef_, r1["coef"], RTOL_COEF)

    xbeta = stage1.predict_linear(d[["x1", "x2", "x3"]], offset=d["offset1"])
    r_xbeta = pd.read_csv(os.path.join(RESULTS, "two_stage_xbeta.csv"))["xbeta"].to_numpy()
    _check("two_stage: xbeta (stage1 -> stage2 offset)", xbeta, r_xbeta, 1e-5)

    # Stage 2 of the real SHR/SMR pattern has NO covariates at all --
    # R's `coxph(Surv(...) ~ offset(xbeta))`, offset only. A genuinely
    # empty (n, 0) design matrix is a first-class input here (NumPy's
    # linalg handles 0-column matmul/solve/inv natively), so this is
    # expressed exactly as R expresses it, with no placeholder column.
    empty_X = pd.DataFrame(index=d.index)
    stage2 = CoxPH().fit(
        empty_X, start=d["start"], stop=d["stop"], event=d["event"],
        offset=xbeta, sample_weight=d["weight"],
    )
    meta2 = pd.read_csv(os.path.join(RESULTS, "two_stage_2_meta.csv"))
    _check("two_stage stage2: loglik", stage2.log_likelihood_, meta2["loglik"], RTOL_LOGLIK)

    rb2 = _r_baseline("two_stage_2")
    merged = pd.merge(stage2.baseline_hazard_, rb2, on="time", suffixes=("_py", "_r"))
    assert len(merged) == len(stage2.baseline_hazard_)
    _check("two_stage stage2: baseline hazard", merged["hazard_py"], merged["hazard_r"], RTOL_BASELINE)


def test_two_stage_smr_workflow():
    """The OTHER two-stage acceptance test from Section 34 -- structurally
    different from the SHR test above: stage 1 has no offset/weights at
    all, and stage 2 carries its OWN covariates (not just an offset)
    alongside the stage-1 risk score. Exercises "covariates + offset
    together in a fit whose offset came from a prior fit" as a distinct
    code path from the SHR shape's "offset-only" stage 2.
    """
    d = pd.read_csv(os.path.join(DATA, "smr_two_stage.csv"))

    stage1 = CoxPH().fit(d[["x1", "x2"]], start=d["start"], stop=d["stop"], event=d["event"], strata=d["provider"])
    r1 = _r_coef_table("smr_stage1")
    _check("smr stage1: coef", stage1.coef_, r1["coef"], RTOL_COEF)
    _check("smr stage1: se", stage1.standard_errors_, r1["se_coef"], RTOL_COEF)

    xbeta = stage1.predict_linear(d[["x1", "x2"]])  # no offset in stage 1 for SMR
    r_xbeta = pd.read_csv(os.path.join(RESULTS, "smr_xbeta.csv"))["xbeta"].to_numpy()
    _check("smr: xbeta (stage1 -> stage2 offset)", xbeta, r_xbeta, 1e-5)

    stage2 = CoxPH().fit(d[["z1"]], start=d["start"], stop=d["stop"], event=d["event"], offset=xbeta)
    r2 = _r_coef_table("smr_stage2")
    _check("smr stage2: coef (covariate alongside the offset)", stage2.coef_, r2["coef"], RTOL_COEF)
    _check("smr stage2: se", stage2.standard_errors_, r2["se_coef"], RTOL_COEF)

    rb2 = _r_baseline("smr_stage2")
    merged = pd.merge(stage2.baseline_hazard_, rb2, on="time", suffixes=("_py", "_r"))
    assert len(merged) == len(stage2.baseline_hazard_)
    _check("smr stage2: baseline hazard", merged["hazard_py"], merged["hazard_r"], RTOL_BASELINE)

    r_resid = _r_residuals("smr_stage2")
    diff = np.max(np.abs(stage2.martingale_residuals_ - r_resid))
    REPORT.append(dict(check="smr stage2: martingale residuals", abs_diff=diff, rel_diff=np.nan,
                        tol=ATOL_RESIDUAL, status="PASS" if diff < ATOL_RESIDUAL else "FAIL"))
    assert diff < ATOL_RESIDUAL


def test_efron_basic():
    """Efron ties -- R's own default, distinct from this package's
    Breslow default -- validated on the same heavily-tied dataset as
    test_basic()."""
    d = pd.read_csv(os.path.join(DATA, "basic.csv"))
    m = CoxPH(ties="efron").fit(d[["x1", "x2", "x3"]], duration=d["time"], event=d["event"])
    r = _r_coef_table("basic_efron")
    meta = _r_meta("basic_efron")

    _check("efron/basic: coef", m.coef_, r["coef"], RTOL_COEF)
    _check("efron/basic: se", m.standard_errors_, r["se_coef"], RTOL_COEF)
    _check("efron/basic: loglik(beta)", m.log_likelihood_, meta["loglik_beta"], RTOL_LOGLIK)

    rb = _r_baseline("basic_efron")
    merged = pd.merge(m.baseline_hazard_, rb, on="time", suffixes=("_py", "_r"))
    assert len(merged) == len(m.baseline_hazard_)
    _check("efron/basic: baseline hazard", merged["hazard_py"], merged["hazard_r"], RTOL_BASELINE)

    r_resid = _r_residuals("basic_efron")
    diff = np.max(np.abs(m.martingale_residuals_ - r_resid))
    REPORT.append(dict(check="efron/basic: martingale residuals", abs_diff=diff, rel_diff=np.nan,
                        tol=ATOL_RESIDUAL, status="PASS" if diff < ATOL_RESIDUAL else "FAIL"))
    assert diff < ATOL_RESIDUAL


def test_efron_weights():
    """Efron + weights together -- the formula reconstructed from
    survival's actual agfit4.c source (see docs/R_COMPATIBILITY.md,
    Section 2) rather than a textbook generalization. Also confirms this
    package's standard_errors_ corresponds to R's `naive.var` (what
    summary()'s se(coef) column shows), NOT `var`/vcov() -- which is the
    ROBUST sandwich estimate once weights aren't trivial, a genuine trap
    for anyone comparing programmatically instead of eyeballing summary().
    """
    d = pd.read_csv(os.path.join(DATA, "weights.csv"))
    m = CoxPH(ties="efron").fit(d[["x1", "x2"]], duration=d["stop"], event=d["event"], sample_weight=d["weight"])
    r = _r_coef_table("weights_efron")
    meta = _r_meta("weights_efron")

    _check("efron/weights: coef", m.coef_, r["coef"], RTOL_COEF)
    _check("efron/weights: se (naive/model-based, not vcov's robust)", m.standard_errors_, r["se_coef"], RTOL_COEF)
    _check("efron/weights: loglik(beta)", m.log_likelihood_, meta["loglik_beta"], RTOL_LOGLIK)

    naive_vs_robust = pd.read_csv(os.path.join(RESULTS, "weights_efron_naive_vs_robust.csv"))
    _check("efron/weights: se matches naive.var, NOT var/vcov (robust)", m.standard_errors_, naive_vs_robust["se_naive"], RTOL_COEF)
    # And an explicit negative check: confirm it does NOT match the robust
    # column, so this isn't a tolerance so loose both would pass by accident.
    robust_rel_diff = np.max(np.abs(m.standard_errors_ - naive_vs_robust["se_robust"].to_numpy())
                              / naive_vs_robust["se_robust"].to_numpy())
    assert robust_rel_diff > 0.01, "standard_errors_ should NOT match R's robust variance"

    rb = _r_baseline("weights_efron")
    merged = pd.merge(m.baseline_hazard_, rb, on="time", suffixes=("_py", "_r"))
    assert len(merged) == len(m.baseline_hazard_)
    _check("efron/weights: baseline hazard", merged["hazard_py"], merged["hazard_r"], RTOL_BASELINE)


def test_efron_left_truncation():
    d = pd.read_csv(os.path.join(DATA, "left_truncation.csv"))
    m = CoxPH(ties="efron").fit(d[["x1", "x2"]], start=d["start"], stop=d["stop"], event=d["event"])
    r = _r_coef_table("left_truncation_efron")
    meta = _r_meta("left_truncation_efron")

    _check("efron/left_truncation: coef", m.coef_, r["coef"], RTOL_COEF)
    _check("efron/left_truncation: se", m.standard_errors_, r["se_coef"], RTOL_COEF)
    _check("efron/left_truncation: loglik(beta)", m.log_likelihood_, meta["loglik_beta"], RTOL_LOGLIK)

    rb = _r_baseline("left_truncation_efron")
    merged = pd.merge(m.baseline_hazard_, rb, on="time", suffixes=("_py", "_r"))
    assert len(merged) == len(m.baseline_hazard_)
    _check("efron/left_truncation: baseline hazard", merged["hazard_py"], merged["hazard_r"], RTOL_BASELINE)

    r_resid = _r_residuals("left_truncation_efron")
    diff = np.max(np.abs(m.martingale_residuals_ - r_resid))
    REPORT.append(dict(check="efron/left_truncation: martingale residuals", abs_diff=diff, rel_diff=np.nan,
                        tol=ATOL_RESIDUAL, status="PASS" if diff < ATOL_RESIDUAL else "FAIL"))
    assert diff < ATOL_RESIDUAL


if __name__ == "__main__":
    failures = 0
    for fn_name in [
        "test_basic", "test_left_truncation", "test_strata", "test_offset",
        "test_weights", "test_combined", "test_two_stage_shr_workflow",
        "test_two_stage_smr_workflow", "test_efron_basic", "test_efron_weights",
        "test_efron_left_truncation",
    ]:
        try:
            globals()[fn_name]()
            print(f"[PASS] {fn_name}")
        except AssertionError as e:
            failures += 1
            print(f"[FAIL] {fn_name}: {e}")

    print("\n" + "=" * 100)
    print(f"{'check':45s} {'abs_diff':>12s} {'rel_diff':>12s} {'tol':>10s}  status")
    print("=" * 100)
    for row in REPORT:
        rel = f"{row['rel_diff']:.2e}" if not np.isnan(row.get('rel_diff', np.nan)) else "n/a"
        print(f"{row['check']:45s} {row['abs_diff']:12.3e} {rel:>12s} {row['tol']:10.1e}  {row['status']}")
    print("=" * 100)
    print(f"\n{len(REPORT) - failures if failures==0 else '?'} checks, {failures} test function failure(s)")
    if failures:
        raise SystemExit(1)
