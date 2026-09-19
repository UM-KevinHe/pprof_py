"""Validation harness for Phase 4: competing risks, robust/clustered
variance, and time-dependent covariates against real R output.

Like `test_r_comparison.py`, this reads r_reference/results/*.csv
written by real R runs -- it does NOT run R itself. Until
    python r_reference/generate_phase4_data.py
    Rscript r_reference/run_phase4_competing_risks.R r_reference
    Rscript r_reference/run_phase4_robust.R r_reference
    Rscript r_reference/run_phase4_timedep.R r_reference
have been run (on a machine with R and the `survival` package installed
-- this development environment did not have either), every test here
fails with a file-not-found error, by design, exactly like
`test_r_comparison.py`'s own module docstring already documents for
Phase 1-3. `tests/test_finegray_transform.py`, `test_score_residuals.py`,
`test_robust_variance.py`, and `test_timedep.py` do not depend on this --
they validate against R's own bundled worked examples (transcribed
directly, not regenerated) and cross-reference tools (lifelines,
statsmodels) that ARE available here, so Phase 4 has real validation
today; this file adds a further, direct R check once it can be run.

Run standalone (prints a full report) via:
    python tests/test_phase4_r_comparison.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from pprof_py import CoxPH, CauseSpecificCoxPH, FineGrayPH
from pprof_py.algorithms.survival.finegray import finegray_transform
from pprof_py.data.timedep import build_skeleton, tmerge, UpdateStream

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "r_reference", "data")
RESULTS = os.path.join(HERE, "..", "r_reference", "results")

RTOL_COEF = 1e-5
RTOL_LOGLIK = 1e-6

REPORT = []


def _check(label, py_val, r_val, rtol):
    py_val = np.asarray(py_val, dtype=float)
    r_val = np.asarray(r_val, dtype=float)
    abs_diff = np.max(np.abs(py_val - r_val))
    denom = np.maximum(np.abs(r_val), 1e-12)
    rel_diff = np.max(np.abs(py_val - r_val) / denom)
    status = "PASS" if rel_diff < rtol or abs_diff < 1e-10 else "FAIL"
    REPORT.append(dict(check=label, abs_diff=abs_diff, rel_diff=rel_diff, tol=rtol, status=status))
    assert status == "PASS", (
        f"{label}: max abs diff={abs_diff:.3e}, max rel diff={rel_diff:.3e} (tol={rtol:.1e})\n"
        f"python={py_val}\nr     ={r_val}"
    )


def _r_coef_table(name):
    return pd.read_csv(os.path.join(RESULTS, f"{name}_coefficients.csv"))


def _r_meta(name):
    return pd.read_csv(os.path.join(RESULTS, f"{name}_meta.csv"))


# ---------------------------------------------------------------- competing risks
def _run_competing_risks(csvname, prefix, has_truncation):
    d = pd.read_csv(os.path.join(DATA, csvname))
    X = d[["x1", "x2"]]
    kwargs = dict(start=d["start"], stop=d["stop"]) if has_truncation else dict(duration=d["stop"])

    csc = CauseSpecificCoxPH().fit(X, event=d["event"], causes=[1, 2], **kwargs)
    r_cause1 = _r_coef_table(f"{prefix}_cause1")
    r_cause2 = _r_coef_table(f"{prefix}_cause2")
    _check(f"{prefix}: cause 1 coef", csc[1].coef_, r_cause1["coef"], RTOL_COEF)
    _check(f"{prefix}: cause 1 se", csc[1].standard_errors_, r_cause1["se_coef"], RTOL_COEF)
    _check(f"{prefix}: cause 2 coef", csc[2].coef_, r_cause2["coef"], RTOL_COEF)
    _check(f"{prefix}: cause 2 se", csc[2].standard_errors_, r_cause2["se_coef"], RTOL_COEF)

    # the raw transform, directly against R's own finegray()
    start_arr = d["start"].to_numpy() if has_truncation else np.zeros(len(d))
    transformed = finegray_transform(
        start_arr, d["stop"].to_numpy(), d["event"].to_numpy().astype(float),
        failcode=1.0, id=d["id"].to_numpy(),
    )
    r_transform = pd.read_csv(os.path.join(RESULTS, f"{prefix}_finegray_transform.csv"))
    r_transform = r_transform.sort_values(["id", "fgstart"]).reset_index(drop=True)
    py_transform = pd.DataFrame(
        {"id": d["id"].to_numpy()[transformed.row], "fgstart": transformed.start, "fgstop": transformed.stop}
    ).sort_values(["id", "fgstart"]).reset_index(drop=True)
    assert len(py_transform) == len(r_transform), (
        f"{prefix}: finegray row count differs -- python {len(py_transform)} vs R {len(r_transform)}"
    )
    _check(f"{prefix}: finegray fgstart", py_transform["fgstart"], r_transform["fgstart"], 1e-6)
    _check(f"{prefix}: finegray fgstop", py_transform["fgstop"], r_transform["fgstop"], 1e-6)

    fg = FineGrayPH().fit(X, event=d["event"], failcode=1, id=d["id"], **kwargs)
    r_fg = _r_coef_table(f"{prefix}_finegray_fit")
    _check(f"{prefix}: finegray fit coef", fg.coef_, r_fg["coef"], RTOL_COEF)
    _check(f"{prefix}: finegray fit se", fg.standard_errors_, r_fg["se_coef"], 1e-3)


def test_competing_risks_simple():
    _run_competing_risks("competing_risks_simple.csv", "cr_simple", has_truncation=False)


def test_competing_risks_truncated():
    _run_competing_risks("competing_risks_truncated.csv", "cr_truncated", has_truncation=True)


# ----------------------------------------------------------------- robust variance
def test_robust_strata_truncation_clustering():
    d = pd.read_csv(os.path.join(DATA, "robust_strata_truncation.csv"))
    m = CoxPH().fit(
        d[["x1", "x2"]], start=d["start"], stop=d["stop"], event=d["event"],
        strata=d["strata"], cluster=d["cluster"],
    )
    r = _r_coef_table("robust_strata_truncation")
    r_naive = pd.read_csv(os.path.join(RESULTS, "robust_strata_truncation_naive_se.csv"))
    _check("robust+strata+truncation: coef", m.coef_, r["coef"], RTOL_COEF)
    _check("robust+strata+truncation: robust se", m.standard_errors_, r["se_robust"], 1e-3)
    _check("robust+strata+truncation: naive se", m.naive_covariance_.diagonal() ** 0.5, r_naive["se_naive"], RTOL_COEF)


# ------------------------------------------------------------ time-dependent
def test_timedep_tmerge_and_fit():
    skeleton = pd.read_csv(os.path.join(DATA, "timedep_skeleton.csv"))
    updates = pd.read_csv(os.path.join(DATA, "timedep_updates.csv"))

    base = build_skeleton(id=skeleton["id"].to_numpy(), tstop=skeleton["stop"].to_numpy())
    test1 = tmerge(
        base.id.to_numpy(), base.tstart.to_numpy(), base.tstop.to_numpy(),
        event={"death": UpdateStream(id=skeleton["id"].to_numpy(), time=skeleton["stop"].to_numpy(),
                                      value=skeleton["death"].to_numpy().astype(float))},
    )
    merged = tmerge(
        test1.id.to_numpy(), test1.tstart.to_numpy(), test1.tstop.to_numpy(),
        tdc={"treated": UpdateStream(id=updates["id"].to_numpy(), time=updates["time"].to_numpy())},
        tdc_init={"treated": 0.0},
        event={"death": UpdateStream(id=skeleton["id"].to_numpy(), time=skeleton["stop"].to_numpy(),
                                      value=skeleton["death"].to_numpy().astype(float))},
    )

    r_merged = pd.read_csv(os.path.join(RESULTS, "timedep_merged.csv")).sort_values(["id", "tstart"]).reset_index(drop=True)
    py_merged = merged.sort_values(["id", "tstart"]).reset_index(drop=True)
    assert len(py_merged) == len(r_merged), f"row count differs: python {len(py_merged)} vs R {len(r_merged)}"
    _check("timedep: tstart", py_merged["tstart"], r_merged["tstart"], 1e-6)
    _check("timedep: tstop", py_merged["tstop"], r_merged["tstop"], 1e-6)
    _check("timedep: death", py_merged["death"], r_merged["death"], 1e-9)

    model = CoxPH().fit(
        merged[["treated"]], start=merged["tstart"], stop=merged["tstop"], event=merged["death"],
    )
    r_fit = _r_coef_table("timedep")
    _check("timedep: fit coef", model.coef_, r_fit["coef"], RTOL_COEF)
    _check("timedep: fit se", model.standard_errors_, r_fit["se_coef"], RTOL_COEF)


if __name__ == "__main__":
    import sys

    exit_code = pytest.main([__file__, "-v"])
    print("\n" + "=" * 80)
    print(f"{'check':55s} {'abs diff':>12s} {'rel diff':>12s} {'tol':>10s}  status")
    print("-" * 80)
    for row in REPORT:
        print(f"{row['check']:55s} {row['abs_diff']:12.3e} {row['rel_diff']:12.3e} {row['tol']:10.1e}  {row['status']}")
    sys.exit(exit_code)
